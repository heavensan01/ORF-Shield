"""Detection engine for the ORF project.

Two engines share one interface:
  * AlbertDetector  - the fine-tuned ALBERT model in ORF_best_model/ (the real system)
  * HeuristicDetector - rule-based fallback so the UI works before the model is retrained
Both return the same dict, so templates do not care which one ran.
"""
import os
import re

CLASS_LABELS = {0: "Normal", 1: "Fake Job"}

# (regex, weight, human readable explanation)
SIGNALS = [
    (r"\b(registration|processing|training|security|application|admin)\s+(fee|deposit|charge)s?\b|\bpay\s+(a\s+)?(fee|deposit)\b|\brefundable\b",
     3.0, "Asks you to pay a fee or deposit"),
    (r"\b(western union|money ?gram|wire transfer|gift cards?|bitcoin|crypto(currency)?|paytm|upi|cash ?app)\b",
     2.5, "Mentions risky payment channels"),
    (r"\b(whatsapp|telegram|signal)\b", 1.8, "Moves the conversation to WhatsApp/Telegram"),
    (r"@(gmail|yahoo|outlook|hotmail|rediffmail)\.", 1.8, "Uses a free personal email instead of a company domain"),
    (r"\bno (prior )?(experience|skills|qualification)s?\s*(needed|required|necessary)?\b|\bfreshers? (can )?apply\b",
     1.2, "Claims no experience or skills are needed"),
    (r"\b(earn|make|income)\b.{0,25}(\$|usd|rs\.?|inr|₹)\s?\d[\d,]*\s*(/|per|a)?\s*(day|daily|week|weekly|hour)",
     2.2, "Promises unusually high daily/weekly earnings"),
    (r"\bguaranteed (income|salary|job|placement|returns?)\b|\b100% (job|placement|guarantee)",
     2.2, "Makes guaranteed-income promises"),
    (r"\b(urgent(ly)?|immediate(ly)? (joining|hiring|start)|limited (slots|seats|vacancies)|apply (now|today|immediately)|hurry)\b",
     1.2, "Creates artificial urgency"),
    (r"\b(ssn|social security|aadhaar|aadhar|passport (copy|number)|bank (account|details)|credit card|debit card|otp|pan card)\b",
     2.5, "Requests sensitive personal or banking details"),
    (r"\b(work from home|work at home|online (typing|data entry|form filling)|copy[- ]paste|like and subscribe)\b",
     1.0, "Typical 'easy online work' scam wording"),
    (r"\b(no interview|without (an )?interview|selected without|direct (selection|joining))\b",
     1.8, "Offers the job without an interview"),
    (r"\b(congratulations|you (have been|are) (selected|shortlisted|chosen))\b",
     1.0, "Claims you were selected without applying"),
    (r"\b(send|forward) (us )?(your )?(money|payment)|advance payment|courier (charges|fee)",
     2.5, "Asks for advance payment"),
]

_COMPILED = [(re.compile(p, re.I | re.S), w, msg) for p, w, msg in SIGNALS]


def find_signals(text: str):
    """Return the human readable scam signals found in the text."""
    found = [msg for rx, _w, msg in _COMPILED if rx.search(text)]
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.45:
        found.append("Excessive use of capital letters")
    if text.count("!") >= 4:
        found.append("Excessive exclamation marks")
    if len(text.split()) < 12:
        found.append("Very short posting with almost no job details")
    return found


class HeuristicDetector:
    name = "Rule-based demo engine"
    is_model = False

    def predict(self, text: str) -> dict:
        score = 0.0
        for rx, w, _ in _COMPILED:
            if rx.search(text):
                score += w
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.45:
            score += 1.0
        if text.count("!") >= 4:
            score += 0.8
        if len(text.split()) < 12:
            score += 1.0
        # squash into 0..1
        p_fake = 1 / (1 + 2.718281828 ** (-(score - 3.0)))
        label_id = 1 if p_fake >= 0.5 else 0
        return {
            "label_id": label_id,
            "label": CLASS_LABELS[label_id],
            "p_fake": round(p_fake, 4),
            "confidence": round(p_fake if label_id else 1 - p_fake, 4),
        }


class AlbertDetector:
    name = "ALBERT (fine-tuned)"
    is_model = True

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=1)[0]
        label_id = int(self.torch.argmax(probs).item())
        return {
            "label_id": label_id,
            "label": CLASS_LABELS.get(label_id, "Unknown"),
            "p_fake": round(float(probs[1]), 4),
            "confidence": round(float(probs[label_id]), 4),
        }


def load_detector(model_dir: str):
    """Use the trained ALBERT model if present, otherwise fall back to the rule engine."""
    has_weights = os.path.isdir(model_dir) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(model_dir)
    )
    if has_weights:
        try:
            return AlbertDetector(model_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[ORF] Could not load model from {model_dir}: {exc}\n[ORF] Falling back to rule-based engine.")
    else:
        print(f"[ORF] No trained model found in '{model_dir}/'. Running in rule-based demo mode.")
    return HeuristicDetector()
