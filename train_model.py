"""Re-train the ALBERT fake-job classifier and save it to ORF_best_model/.

Rebuilt from the 'Source Code' chapter of ORF_documentation.docx.
Run on a GPU machine (e.g. Google Colab):

    pip install -r requirements.txt
    python train_model.py --csv fake_job_postings.csv

Dataset: EMSCAD 'Fake Job Postings' (Kaggle) - columns used: description, fraudulent.
Add --extra other_postings.csv (columns: text,label) to merge more sources as in the paper.

NOTE on class balance: the original notebook applied SMOTE to raw token-ID vectors. Token IDs are
categorical, so interpolating between them creates meaningless 'tokens'. Here the TRAINING split is
balanced by random over-sampling of real minority postings, and the validation split is left untouched
so metrics are honest.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--extra", default=None, help="optional CSV with text,label columns")
p.add_argument("--out", default="ORF_best_model")
p.add_argument("--model", default="albert-base-v1")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=4e-5)
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--majority_cap", type=int, default=10000, help="down-sample genuine posts to this many")
args = p.parse_args()

# 1. load + rename (description -> text, fraudulent -> label)
df = pd.read_csv(args.csv)[["description", "fraudulent"]].rename(columns={"description": "text", "fraudulent": "label"})
if args.extra:
    df = pd.concat([df, pd.read_csv(args.extra)[["text", "label"]]], ignore_index=True)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
df["text"] = df["text"].astype(str)
print("Class counts:\n", df.label.value_counts())

# 2. down-sample majority (17014 -> 10000 in the paper)
maj, mino = df[df.label == 0], df[df.label == 1]
if len(maj) > args.majority_cap:
    maj = resample(maj, replace=False, n_samples=args.majority_cap, random_state=42)
data = pd.concat([maj, mino]).sample(frac=1, random_state=42)

# 3. split FIRST (80/20, stratified), then balance only the training part
train_df, val_df = train_test_split(data, test_size=0.2, stratify=data.label, random_state=42)
tr_maj, tr_min = train_df[train_df.label == 0], train_df[train_df.label == 1]
tr_min_up = resample(tr_min, replace=True, n_samples=len(tr_maj), random_state=42)
train_df = pd.concat([tr_maj, tr_min_up]).sample(frac=1, random_state=42)
print(f"train={len(train_df)}  val={len(val_df)}")

tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)


class TextDS(Dataset):
    def __init__(self, frame):
        self.enc = tok(frame.text.tolist(), truncation=True, padding="max_length", max_length=args.max_len)
        self.labels = frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


train_ds, val_ds = TextDS(train_df), TextDS(val_df)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)


def metrics(pred):
    y, yhat = pred.label_ids, pred.predictions.argmax(-1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, yhat), "precision": pr, "recall": rc, "f1": f1}


targs = dict(output_dir="./results", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
             per_device_eval_batch_size=args.batch * 2, warmup_steps=500, weight_decay=0.01,
             logging_steps=50, save_strategy="epoch", load_best_model_at_end=True,
             metric_for_best_model="f1", learning_rate=args.lr, fp16=torch.cuda.is_available(),
             report_to="none")
try:
    targs = TrainingArguments(eval_strategy="epoch", **targs)       # transformers >= 4.41
except TypeError:
    targs = TrainingArguments(evaluation_strategy="epoch", **targs)  # older versions

trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=metrics)
trainer.train()
print("Final evaluation:", trainer.evaluate())

# 4. confusion matrix + classification report
pred = trainer.predict(val_ds)
yhat = np.argmax(pred.predictions, axis=1)
print("Confusion matrix:\n", confusion_matrix(pred.label_ids, yhat))
print(classification_report(pred.label_ids, yhat, target_names=["Normal", "Fake Job"]))

# 5. save -> the Flask app loads this folder
os.makedirs(args.out, exist_ok=True)
trainer.save_model(args.out)
tok.save_pretrained(args.out)
print(f"Saved model to ./{args.out}")
