from pattern_recognition_model.data import ds

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# Small English subsample so it trains in reasonable time
ds = ds.shuffle(seed=42).select(range(30000))
ds = ds.filter(lambda x: x["language"] == "en")
ds = ds.select(range(min(5000, len(ds))))

# Keep what we need; make the label an int called "label"
ds = ds.map(lambda x: {"label": int(x["prompt_adversarial"])})
ds = ds.select_columns(["prompt", "label", "source", "attack_technique"])
ds = ds.train_test_split(test_size=0.2, seed=42)

print(ds)
print("attack share:", np.mean(ds["train"]["label"]))

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def prep(batch):
    return tok(batch["prompt"], truncation=True, max_length=128)


ds = ds.map(prep, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


args = TrainingArguments(
    output_dir="out",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
)
trainer.train()
