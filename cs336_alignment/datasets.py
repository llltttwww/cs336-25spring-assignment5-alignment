import os
os.environ["HF_HOME"]="/workspace/home/luotianwei/hf_cache"
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from cs336_alignment.post_training_utils import tokenize_prompt_and_output

import wandb

R1_ZERO_PROMPT = (
    "A conversation between User and Assistant. "
    "The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind "
    "and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

class MathSFTDataset(Dataset):
    def __init__(self,path,tokenizer,max_size=None):
        self.data=[]
    
        with open(path,'r') as f:
            for line in f:
                item=json.loads(line)
                problem=item["prompt"]
                response=item["response"]
                problem=R1_ZERO_PROMPT.format(question=problem.strip())
                self.data.append((problem,response))
        
        if max_size:
            self.data=self.data[:max_size]
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
class MathDataset(Dataset):
    def __init__(self,path,tokenizer=None,max_size=None):
        self.data=[]
    
        with open(path,'r') as f:
            for line in f:
                item=json.loads(line)
                self.data.append((item["problem"],item["answer"]))
        
        if max_size:
            self.data=self.data[:max_size]
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    
def make_sft_collate_fn(tokenizer):
    def collate_fn(batch):
        prompts,responses=zip(*batch)
        return tokenize_prompt_and_output(prompts,responses,tokenizer,padding_side="right")
    return collate_fn

