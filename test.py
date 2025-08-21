from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Caminho do adaptador LoRA
peft_model_path = "./geo_peft"

# 1. Carrega a config do adaptador para saber o modelo base original
config = PeftConfig.from_pretrained(peft_model_path)

# 2. Carrega o modelo base original
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")

# 3. Aplica o adaptador LoRA ao modelo base
model = PeftModel.from_pretrained(base_model, peft_model_path)

# 4. Carrega o tokenizer original
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 5. Define o prompt
prompt = "Place: São José dos Campos\nLatitude: \nLongitude:"

# 6. Geração do texto
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
