"""ORF Shield - Online Recruitment Fraud Detection (Flask + ALBERT)."""
import os
import secrets
import sqlite3
from functools import wraps

from flask import (Flask, flash, g, redirect, render_template, request,
                   session, url_for)
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash
from wtforms import EmailField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length

from detector import find_signals, load_detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(INSTANCE_DIR, "users.db")


def get_secret_key():
    """Stable secret key (the original regenerated it every start, logging everyone out)."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(INSTANCE_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32))
    with open(path) as fh:
        return fh.read().strip()


app = Flask(__name__)
app.config["SECRET_KEY"] = get_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
CSRFProtect(app)

detector = load_detector(os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "ORF_best_model")))


# ---------------------------------------------------------------- database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                snippet TEXT NOT NULL,
                label TEXT NOT NULL,
                p_fake REAL NOT NULL,
                engine TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )


init_db()


# ---------------------------------------------------------------- forms
class SignInForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign in")


class SignUpForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=80)])
    email = EmailField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, message="Use at least 8 characters.")])
    confirm = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password", message="Passwords do not match.")])
    submit = SubmitField("Create account")


def signin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please sign in first.", "warning")
            return redirect(url_for("signin"))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_globals():
    return {"engine_name": detector.name, "engine_is_model": detector.is_model}


# ---------------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignUpForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        try:
            db().execute(
                "INSERT INTO users (name, username, password) VALUES (?, ?, ?)",
                (form.name.data.strip(), email, generate_password_hash(form.password.data)),
            )
            db().commit()
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "danger")
        else:
            flash("Account created. You can sign in now.", "success")
            return redirect(url_for("signin"))
    return render_template("signup.html", form=form)


@app.route("/signin", methods=["GET", "POST"])
def signin():
    form = SignInForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        row = db().execute("SELECT id, name, password FROM users WHERE username = ?", (email,)).fetchone()
        if row and check_password_hash(row["password"], form.password.data):
            session.clear()
            session.update(user=email, user_id=row["id"], name=row["name"])
            flash(f"Welcome back, {row['name'].split()[0]}!", "success")
            return redirect(url_for("recomendation"))
        flash("Invalid email or password.", "danger")
    return render_template("signin.html", form=form)


@app.route("/prediction/", methods=["GET", "POST"])
@signin_required
def recomendation():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if len(text) < 20:
            flash("Paste a longer job posting (at least 20 characters) for a reliable result.", "warning")
        else:
            result = detector.predict(text)
            result["signals"] = find_signals(text)
            result["percent"] = round(result["p_fake"] * 100, 1)
            result["confidence_pct"] = round(result["confidence"] * 100, 1)
            result["words"] = len(text.split())
            db().execute(
                "INSERT INTO predictions (user_id, snippet, label, p_fake, engine) VALUES (?,?,?,?,?)",
                (session["user_id"], text[:160], result["label"], result["p_fake"], detector.name),
            )
            db().commit()
    return render_template("prediction.html", result=result, text=text)


@app.route("/history")
@signin_required
def history():
    rows = db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 100", (session["user_id"],)
    ).fetchall()
    total = len(rows)
    fake = sum(1 for r in rows if r["label"] == "Fake Job")
    return render_template("history.html", rows=rows, total=total, fake=fake, genuine=total - fake)


@app.route("/signout")
def signout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
