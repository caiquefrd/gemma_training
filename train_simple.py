
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

examples = [prep(ex) for ex in data]

# Validation split (80% train, 20% val)
from sklearn.model_selection import train_test_split
train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)

train_ds = Dataset.from_list(train_examples)
val_ds = Dataset.from_list(val_examples)

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

train_ds = train_ds.map(tok, batched=True, remove_columns=["text", "label"])
val_ds = val_ds.map(tok, batched=True, remove_columns=["text", "label"])

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
    # evaluation_strategy="epoch",
    # load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
model.save_pretrained("geocoding")
tokenizer.save_pretrained("geocoding")
