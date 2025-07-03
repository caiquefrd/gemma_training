from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load dataset
ds = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
    cache_dir="/content/cache"
)


def prep(ex):
    p = f"Place: {ex['name']}\n"
    o = f"Latitude: {ex['lat']}, Longitude: {ex['lon']}"
    return {"text": p + o}

ds = ds["train"].map(prep)

# 2. Load tokenizer & model with 4-bit quantization
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map={"": device},  # ensures proper fallback
)


# 3. Apply LoRA config
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 4. Tokenize dataset
def tok(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = model_inputs["input_ids"].clone()
    model_inputs["labels"] = labels
    return model_inputs

ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

# 5. Training
training_args = TrainingArguments(
    output_dir="geo_peft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    bf16=not torch.cuda.is_available(),
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,
)

print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()
model.save_pretrained("geo_peft")
tokenizer.save_pretrained("geo_peft")
