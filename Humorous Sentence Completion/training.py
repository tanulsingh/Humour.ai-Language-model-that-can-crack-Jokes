# Preliminaries
import os
import pandas as pd
import numpy as np

#Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

#Transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup

#Warning
import warnings
warnings.filterwarnings('ignore')

#Mymodule
import config

# Processing Data
def process_jokes(raw_fp):
    df = pd.read_csv(raw_fp)

    # Append token at the end of each joke to indicate the end of a joke

    what_jokes = df[df.Joke.str.lower().str.startswith("what")].Joke.str.split("?")
    how_jokes = df[df.Joke.str.lower().str.startswith("how")].Joke.str.split("?")
    why_jokes = df[df.Joke.str.lower().str.startswith("why")].Joke.str.split("?")
    when_jokes = df[df.Joke.str.lower().str.startswith("when")].Joke.str.split("?")
    where_jokes = df[df.Joke.str.lower().str.startswith("where")].Joke.str.split("?")

    jokes = []
    for joke_ in [what_jokes, how_jokes, why_jokes, when_jokes, where_jokes]:
        joke_df_ = pd.DataFrame(joke_.values.tolist()).iloc[:, :2].dropna()
        joke_df_.columns = ["questions", "answer"]
        jokes.append(joke_df_)

    jokes_df = pd.concat(jokes)
    jokes_df = (
        jokes_df[~(jokes_df.answer.isin([""]))].drop_duplicates().reset_index(drop=True)
    )

    riddle_jokes_list = (
        "<soq> " + jokes_df.questions + " <eoq> " + jokes_df.answer + " <|endoftext|>"
    ).values.tolist()
    riddle_jokes = "\n".join(riddle_jokes_list)

    return riddle_jokes_list


# Creating Custom DataSet

class Jokesdataset(Dataset):
  def __init__(self,data,tokenizer):
    self.data = data
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    joke = self.data[idx]
  
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


# Initializing Model and adding our special Tokens to model vocab

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>','bos_token':'<soq>','sep_token':'<eoq>'}
num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer))

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

        if (bi+1) % 100 == 0:
           print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, config.EPOCHS, bi+1,len(data_loader), loss.item()))


#ENGINE

def run():
  joke_list = process_jokes(config.TRAIN_PATH)
  
  jokes_dataset = Jokesdataset(joke_list,config.Tokenizer)
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
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_3.pt"))


# Begin Training
run()