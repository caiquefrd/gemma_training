
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Path to trained model
model_path = "./geocoding"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load id2label mapping
id2label = model.config.id2label

# Example description to classify
desc = "Importante polo tecnológico e industrial do Vale do Paraíba, conhecido por abrigar centros de pesquisa como o INPE e empresas do setor aeroespacial."

inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
	outputs = model(**inputs)
	pred = torch.argmax(outputs.logits, dim=1).item()

place_name = id2label[str(pred)] if str(pred) in id2label else id2label[pred]
print(f"Predicted place: {place_name}")
