
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
import torch
import json

# Load data
with open("data/train.jsonl") as f:
    data = [json.loads(line) for line in f]

# Get all unique place names
labels = sorted(list({ex["name"] for ex in data}))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# Prepare dataset for classification
def prep(ex):
    return {
        "text": ex["desc"],
        "label": label2id[ex["name"]]
    }

ds = Dataset.from_list([prep(ex) for ex in data])

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tok(batch):
    enc = tokenizer(
        batch["text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    enc["labels"] = batch["label"]
    return enc

ds = ds.map(tok, batched=True)

model = BertForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="geo_peft",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()
model.save_pretrained("geo_peft")
tokenizer.save_pretrained("geo_peft")
