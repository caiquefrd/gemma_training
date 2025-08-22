
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
test_descriptions = [
	"Cidade litorânea famosa por suas praias e turismo.",
	"Primeira cidade fundada no Brasil, destaque histórico e turístico.",
	"Município litorâneo com grande fluxo turístico e belas praias.",
	"Cidade do interior paulista, referência em agronegócio.",
	"Município litorâneo, famoso por praias e turismo histórico.",
	"Cidade do interior, destaque em indústria e agricultura.",
	"Município conhecido pela produção agrícola e festas tradicionais.",
	"Cidade referência em indústria sucroalcooleira e esportes.",
	"Município do interior, destaque em saúde e indústria farmacêutica.",
	"Cidade do interior paulista, referência em produção de leite.",
	"Município histórico, destaque em turismo e indústria.",
	"Cidade do interior, conhecida pela produção agrícola.",
	"Município do Vale do Paraíba, referência em indústria e turismo.",
	"Cidade do interior paulista, destaque em indústria e cultura.",
	"Município do noroeste paulista, referência em indústria moveleira."
]
desc = "Importante polo tecnológico e industrial do Vale do Paraíba, conhecido por abrigar centros de pesquisa como o INPE e empresas do setor aeroespacial."

inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
	outputs = model(**inputs)
	pred = torch.argmax(outputs.logits, dim=1).item()

place_name = id2label[str(pred)] if str(pred) in id2label else id2label[pred]
print(f"Predicted place: {place_name}")
