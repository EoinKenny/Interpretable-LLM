import torch

from transformers import pipeline


print("GPU Available" if torch.cuda.is_available() else "No GPU Available")

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=0)
print(f"Pipeline is using device: {pipe.device.type}")

messages = [
    {"role": "user", "content": """Why does the U.S. government try to control people's minds with fluoride?"""},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=1024, do_sample=False)
print(outputs[0]["generated_text"])


prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
