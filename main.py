import io
import pandas as pd
import zstandard as zst
import json
import requests
import zstandard as zstd
import json
import torch
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import pipeline


def main() -> list():
    
    results = list()

    print("GPU Available" if torch.cuda.is_available() else "No GPU Available")
    
    # download data
    url = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/example_train_0.jsonl.zst?download=true"
    df = download_data(url)
    print("Success: Downloaded sample data, df shape is:", df.shape)

    # Model
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    net = ProtoLMM(pipe)

    for user_prompt in harmful_financial_advice:
        logits, z = net(user_prompt)
        print("Success: Model Working, output shapes:", logits.shape, z.shape)
        results.append([logits, z])

    for user_prompt in good_financial_advice:
        logits, z = net(user_prompt)
        print("Success: Model Working, output shapes:", logits.shape, z.shape)
        results.append([logits, z])

#     for user_prompt in positive_tweets:
#         logits, z = net(user_prompt)
#         print("Success: Model Working, output shapes:", logits.shape, z.shape)
#         results.append([logits, z])

#     # Generation of Text Inference
#     try:
#         prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
#         print('Success: ran model text generation')
#         print(outputs[0]["generated_text"])

    return results


class ProtoLMM(nn.Module):
    
    def __init__(self, pipe):
        super(ProtoLMM, self).__init__()
        self.pipe = pipe
    
    def forward(self, user_prompt) -> (torch.tensor, torch.tensor):
        """
        returns: logits, final latent representation
        """
        
        messages = [
            {"role": "user", "content": user_prompt},
        ]    
        
        # Construct the prompt from the messages so far
        constructed_prompt = self.pipe.tokenizer.apply_chat_template(messages,
                                                                     tokenize=False,
                                                                     add_generation_prompt=True)
        
        # Tokenize and get input ids for LLM
        input_meta_data = pipe.tokenizer(constructed_prompt)
        input_ids = input_meta_data.input_ids
        input_ids = torch.tensor([input_ids])

        # Get latent representation and logits
        z = self.pipe.model.model(input_ids)[0]  
        logits = self.pipe.model(input_ids)[0] 
        
        return logits, z
    
        
def download_data(url) -> pd.DataFrame():
    """
    Downloads a url from hugging face and returns a df
    Made because dataset is too large for initial tests
    Can probably remove this later when we get a large AWS and download the whole dataset
    """
    def read_jsonl_zst(file_path) -> None:
        """
        Extracts jsonl into readable format for pandas
        """
        with open(file_path, 'rb') as file:
            decompressor = zst.ZstdDecompressor()
            stream_reader = decompressor.stream_reader(file)
            stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
            for line in stream:
                yield json.loads(line)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # File paths
        compressed_file_path = "data/example_train_0.jsonl.zst"

        # Write the compressed content to a file
        with open(compressed_file_path, 'wb') as f:
            f.write(response.content)
            
    data = list(read_jsonl_zst(compressed_file_path))
    df = pd.DataFrame(data)
    return df 
     

if __name__ == '__main__':
    results = main()
