import os
os.environ["HF_HOME"]="/workspace/home/luotianwei/hf_cache"
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,get_scheduler
from unittest.mock import patch
from vllm import LLM,SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
from cs336_alignment.datasets import MathSFTDataset,MathDataset,make_sft_collate_fn
from tqdm import tqdm
# import CrossEntropyloss from torch
from torch.nn import CrossEntropyLoss
import random

from vllm_utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    eval_vllm_model,
    evaluate_vllm
)


MAIN_MODEL_CUDA="cuda:3"
VLLM_MODEL_CUDA="cuda:4"

from post_training_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    r1_zero_reward_fn,
    log_generations,
    log_val_metrics
)   
    

def run_sft_with_vllm(
    train_path,
    val_path,
    model_id,
    micro_batch_size,
    lr,
    num_epochs,
    grad_accum_steps,
    eval_every,
    max_train_size=None,
):
    seed=42
    torch.manual_seed(seed)
    random.seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = MathSFTDataset(train_path, tokenizer)
    val_dataset = MathDataset(val_path, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=make_sft_collate_fn(tokenizer),
        drop_last=True,
    )

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": MAIN_MODEL_CUDA}
    )
    
    policy_model=torch.compile(policy_model)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-8)

    total_training_steps = (len(train_dataset) * num_epochs) // (grad_accum_steps* micro_batch_size)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    llm = init_vllm(model_id, device=VLLM_MODEL_CUDA, seed=42)

    wandb.init(project="cs336_assignment5_sft_MATH")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_iter_step = 0
    train_step=0
    eval_step = 0

    for epoch in range(num_epochs):
        policy_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch in pbar:
            # 不 squeeze，直接上 GPU
            batch = {k: v.to(MAIN_MODEL_CUDA) for k, v in batch.items()}
            attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).long()
            out = get_response_log_probs(policy_model, batch["input_ids"], batch["labels"],return_token_entropy=True)
            valid_token_entropy=out['token_entropy']*batch['response_mask']
            mean_token_entropy=valid_token_entropy.sum()/ batch['response_mask'].sum()
            ## original
            loss, _ = sft_microbatch_train_step(
                out["log_probs"], batch["response_mask"], grad_accum_steps,normalize_constant=1.0
            )
            ##
            # ### modified
            # logits=policy_model(batch["input_ids"],attention_mask=attention_mask).logits
            # loss,_=sft_microbatch_train_step(logits,batch["labels"],batch["response_mask"],grad_accum_steps)
            # ###
            if (train_iter_step + 1) % grad_accum_steps == 0:
                train_step+=1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": loss.item()*grad_accum_steps,
                    "train/grad_norm": grad_norm.item(),
                    "train/valid_tokens": batch["response_mask"].sum().item(),
                    "train/mean_token_entropy":mean_token_entropy.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train_step": train_step,
                })
                
                if train_step%eval_every==0:
                    eval_step+=1
                    eval_vllm_model(policy_model,tokenizer,llm,val_dataset,eval_step,r1_zero_reward_fn)
                    

            train_iter_step += 1
                
                
if __name__=='__main__':
    run_sft_with_vllm(
        train_path='/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/data/MATH/sft_new.jsonl',
        val_path='/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/data/MATH/extracted_validation.jsonl',
        model_id="Qwen/Qwen2.5-Math-1.5B",
        micro_batch_size=8,
        lr=5e-5,
        num_epochs=20,
        grad_accum_steps=8,
        eval_every=3,
        max_train_size=None,
    )