# ORF Shield – Online Recruitment Fraud Detection (ALBERT + Flask)

Rebuilt from `ORF_documentation.docx`: Flask web app with sign-up / sign-in, session management,
real-time ALBERT classification of job postings (Normal vs Fake Job), prediction history and a redesigned dark UI.

## Run
```bash
python -m venv venv && venv\Scripts\activate      # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt        # web app (demo mode)
pip install -r requirements-ml.txt     # optional: ALBERT model + training
python app.py                                      # http://127.0.0.1:5000
```

## Two engines
| Mode | When | Notes |
|------|------|-------|
| **ALBERT (fine-tuned)** | `ORF_best_model/` contains model weights | the real system from the report |
| **Rule-based demo** | no model found | keyword/red-flag scoring so the UI works immediately |

The badge on the home page and the notice on the result card show which engine ran.

## Re-train the lost model
1. Download the *Fake Job Postings* CSV (EMSCAD, Kaggle) as `fake_job_postings.csv`.
2. Install `requirements-ml.txt`, then on a GPU (Colab is fine): `python train_model.py --csv fake_job_postings.csv`
3. The model is written to `ORF_best_model/`; restart the app.

## Structure
```
app.py            Flask routes, auth, history, CSRF
detector.py       ALBERT engine + rule-based fallback + red-flag finder
train_model.py    data prep, balancing, fine-tuning, evaluation (from the notebook)
templates/        base, home, about, signin, signup, prediction, history
static/           css/style.css, js/main.js
instance/         users.db + secret.key (auto-created)
```

## What changed vs. the original
* New responsive UI: animated probability gauge, red-flag list, examples, history dashboard
* CSRF protection, password confirmation (min 8 chars), stable secret key (no logout on restart)
* Prediction history per user; max 512-token truncation
* Training script balances only the **training** split (no SMOTE on token IDs) so evaluation is honest

## Deploy (free live link on Render)
1. Push this repo to GitHub.
2. On https://render.com choose **New > Blueprint**, connect the repo and deploy (`render.yaml` is included).
3. You get a link like `https://orf-shield.onrender.com`.

Notes: the free plan sleeps after inactivity (first load takes ~1 minute) and its disk is temporary, so
accounts and history reset on each redeploy. The hosted app runs in rule-based demo mode; the ALBERT model
needs a larger server than the free tier.
