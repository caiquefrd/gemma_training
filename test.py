from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("geo_peft")
model = AutoModelForCausalLM.from_pretrained("geo_peft", device_map="auto")

prompt = "Place: Jacareí\nLatitude:"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.decode(out[0]))
