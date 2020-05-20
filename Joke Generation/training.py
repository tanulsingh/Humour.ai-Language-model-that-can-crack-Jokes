'''
This file containing the training code for Joke Generation Model
'''
# Preliminaries
import os
import pandas as pd
import numpy as np

#Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

# Transformers
from transformers import GPT2LMHeadModel
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup

#Warnings
import warnings
warnings.filterwarnings('ignore')

# MyModule
import config

# INITIALIZING MODEL AND ADDING THE PAD TOKEN
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>'}
num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer))


# Dataset
class Jokesdataset(Dataset):
    '''
    This class builds the custom dataset for Dataloader
    '''
  def __init__(self,data,tokenizer):
    self.data = data
    self.tokenizer = tokenizer
    self.eos_tok = "<|endoftext|>"
    #Adding JOKE: at the start and EOS TOKEN at end
    self.data['Joke'] = self.data['Joke'].apply(lambda x: "JOKE:" + str(x) + self.eos_tok)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    joke = self.data.iloc[idx,1]
    
    inputs = self.tokenizer.encode_plus(
            joke,
            None,
            add_special_tokens = True,
            max_length = config.MAX_LEN,
            pad_to_max_length = True
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    return {'ids':torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'target':torch.tensor(ids,dtype=torch.long)}


# Training Function

def train_fn(data_loader, model, optimizer, device, scheduler,epoch):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        labels = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device,dtype=torch.long)
          
        optimizer.zero_grad()
        outputs = model(
            input_ids =ids,
            attention_mask=mask,
            labels = labels
        )

        loss, logits = outputs[:2]                        
        loss.backward()

        optimizer.step()
        if scheduler is not None:
                scheduler.step()

        if (bi+1) % 500 == 0:
            print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, config.EPOCHS, bi+1,len(data_loader), loss.item()))

device = 'cuda' # Selecting Device

def run():
  jokes = pd.read_csv(config.TRAIN_PATH) #add the path to your Dataset in config File

  jokes_dataset = Jokesdataset(jokes,config.Tokenizer)
  jokes_dataloader = DataLoader(jokes_dataset,
                                batch_size=config.BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)
  
  model.to(device)

  num_train_steps = int(len(jokes_dataloader) / config.BATCH_SIZE * config.EPOCHS)

  optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
  scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)

  for epoch in range(config.EPOCHS):
        print(f"EPOCH {epoch+1} started" + '=' * 30)
        train_fn(jokes_dataloader, model, optimizer, device, scheduler,epoch=epoch)
        
        models_folder = config.MODEL_FOLDER 
        if not os.path.exists(models_folder):
          os.mkdir(models_folder)
        # Saving Model after each Epoch
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_joke_generator{epoch}.pt"))


# BEGINNING TRAINING
run()


