import torch
import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from funcs import make_prompt, pre_process_df, extract_answer


def main():
        
    access_token='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'
    model_type = 'google/gemma-2-9b-it'
    device = 'auto'  # 'auto'
    
    tokenizer = AutoTokenizer.from_pretrained(model_type,
                                              token=access_token,
                                              )
    model = AutoModelForCausalLM.from_pretrained(model_type,
                                                 device_map=device,
                                                 token=access_token,
                                                 torch_dtype=torch.float16,  # Use half precision if possible
                                                 )    
    
    for dataset in ['imdb', 'blogs', 'emails']:
    
        df = pd.read_csv('datasets/'+dataset+'.csv')
        df = pre_process_df(df, dataset)
        
        # Set pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        df_result = pd.DataFrame(columns=['label', 'prediction', 'llm_response', 'text', 'index'])
        
        # Generate responses
        with torch.no_grad():

            if dataset == 'imdb': 
                temp_df = df.sample(500, random_state=42)
            elif dataset == 'blogs': 
                temp_df = df.sample(500, random_state=42)
            elif dataset == 'emails': 
                temp_df = df.sample(500, random_state=42)
            else:
                raise TypeError('Wrong dataset name')

            for idx, row in temp_df.iterrows():
                    
                text  = row.text
                label = row.label
                
                messages = make_prompt(text, dataset)
                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
                
                # Create attention mask
                attention_mask = encodeds.ne(tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else torch.ones_like(encodeds).long()
                
                encodeds = encodeds.to(model.device)
                attention_mask = attention_mask.to(model.device)
                
                try:
                    generated_ids = model.generate(
                        encodeds,
                        attention_mask=attention_mask,
                        max_new_tokens=3000,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    response = tokenizer.batch_decode(generated_ids)[0]
                    prediction = extract_answer(response.split('<end_of_turn>')[-2])
                    assert len(response.split('<end_of_turn>')) == 3  # make sure there is a response from the LLM
                    
                except Exception as e:
                    print(f"Error during generation for {dataset}, row {idx}: {str(e)}")
                    print(response)
                    response = "Error during generation"
                    prediction = None
                                                
                # Create a list with these values
                data = [label, prediction, response, text, idx]
                df_result.loc[len(df_result)] = data
                df_result.to_csv('data/zero_shot_results_'+dataset+'.csv')
                print(dataset, idx, label, prediction)


if __name__ == '__main__':
    main()
