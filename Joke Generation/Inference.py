# Preliminaries
import os
import numpy as np 
import pandas as pd

#transformers
from transformers import GPT2LMHeadModel

# Pytorch
import torch
import torch.nn as nn

#warnings
import warnings
warnings.filterwarnings('ignore')

# My Module
import config

# HElper Function
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

# Model Loading
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>'}
num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer)) 

#loading Model State
models_path = "/kaggle/input/fine-tuning-open-gp-2/trained_models/gpt2_medium_joker_0.pt" # ADD PATH TO YOUR SAVED MODEL HERE
model.load_state_dict(torch.load(models_path))

device='cuda'
model.to(device)

def predict(length_of_joke,number_of_jokes):
    joke_num = 0
    model.eval()
    with torch.no_grad():
        for joke_idx in range(number_of_jokes):
        
            joke_finished = False

            cur_ids = torch.tensor(config.Tokenizer.encode('JOKE')).unsqueeze(0).to(device)

            for i in range(length_of_joke):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in config.Tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = config.Tokenizer.decode(output_list)

                print(output_text+'\n')

# Start Predicting
predict(64,5)