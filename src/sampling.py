
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


checkpoint = '/home/do/ssd/modelhub/CodeGen2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, revision="main")
model = model.to('cuda')
prompt = "# flask json api ping with response `I'm fine`\ndef api_ping"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
t0 = time.time()
sample = model.generate(**inputs, max_length=128)
cost = time.time() - t0
print('cost ', cost)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
time.sleep(20)

