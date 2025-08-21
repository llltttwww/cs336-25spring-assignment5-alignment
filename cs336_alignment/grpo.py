import os
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/workspace/home/luotianwei/hf_cache')
os.environ['CUDA_VISIBLE_DEVICES']="1,2"

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import json, random, math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)

import numpy as np
import wandb
import typer
from tqdm import trange

# === 你已实现的工具函数 ===
from cs336_alignment.utils import r1_zero_reward_fn,load_jsonl
from cs336_alignment.post_training_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_group_normalize_rewards,
    compute_policy_gradient_loss,
    masked_mean,
    grpo_microbatch_train_step,
    log_val_metrics,
    compute_old_log_probs_streaming
)
from cs336_alignment.vllm_utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    batched_generate_vllm,
    eval_vllm_model
)


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

from cs336_alignment.datasets import MathDataset

def format_r1_batch(qs: List[str]) -> List[str]:
    return [R1_ZERO_PROMPT.format(question=q.strip()) for q in qs]

# -------------------------
# 配置
# -------------------------
@dataclass
class Config:
    # data
    train_path: str = "/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/data/MATH/extracted_train.jsonl"
    val_path: str   = "/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/data/MATH/extracted_validation.jsonl"
    val_size: int   = 1024

    # model
    model_id: str = "/workspace/home/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/"
    dtype: str = "bfloat16"
    attn_impl: str = "flash_attention_2"
    pad_side: Literal["left", "right"] = "right"

    # optimization
    n_grpo_steps: int = 600
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    gradient_accumulation_steps: int = 128
    grad_clip: float = 1.0

    # rollout
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    top_p: float = 1.0
    old_logprob_stream_batch_size: int = 16

    # training schedule (on-policy)
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256

    # loss
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip"
    use_std_normalization: bool = True
    advantage_eps: float = 1e-6
    cliprange: float = 0.2  # only used for grpo_clip off-policy

    # logging
    project: str = "cs336-grpo"
    run_name: Optional[str] = "standard_setting"
    eval_every: int = 10
    seed: int = 1234
    max_eval_size: int=1000
    eval_baseline: bool=True

app = typer.Typer(add_completion=False)

# -------------------------
# 主训练循环
# -------------------------
@app.command()
def train(
    cfg_path: Optional[str] = typer.Option(None, help="可选：加载 JSON 配置文件"),
    override: Optional[str] = typer.Option(
        None, "--override", "-o",
        help='JSON 字符串覆盖配置，如：\'{"learning_rate":5e-5,"loss_type":"reinforce_with_baseline"}\'',
    ),
):
    # ---- 配置合并 ----
    cfg = Config()
    if cfg_path:
        cfg_dict = json.loads(Path(cfg_path).read_text())
        cfg = Config(**{**cfg.__dict__, **cfg_dict})

    # 解析 --override
    if override:
        try:
            upd = json.loads(override)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"--override 必须是合法 JSON：{e}")
        for k, v in upd.items():
            if not hasattr(cfg, k):
                raise typer.BadParameter(f"未知配置项：{k}")
            setattr(cfg, k, v)

    # ---- 派生量检查 ----
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0
    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    n_train_steps_per_rollout_batch=cfg.rollout_batch_size*cfg.epochs_per_rollout_batch // cfg.train_batch_size
    assert cfg.rollout_batch_size % cfg.group_size == 0
    n_prompts_per_rollout = cfg.rollout_batch_size // cfg.group_size

    # ---- 随机种子 ----
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # ---- HF 模型 (训练用) ----
    dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)[cfg.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_impl,
        device_map="cuda:0",   # 主卡用于训练
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.pad_side

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # ---- vLLM 模型 (rollout 用) ----
    llm = init_vllm(cfg.model_id, device="cuda:1", seed=cfg.seed)

    # ---- 数据 ----
    train_prompts, train_gts = load_jsonl(Path(cfg.train_path))
    val_prompts, val_gts = load_jsonl(Path(cfg.val_path), limit=cfg.val_size)
    # val_dataset = list(zip(val_prompts, val_gts))  # 包装成简单 dataset
    val_dataset=MathDataset(cfg.val_path)

    # ---- wandb ----
    wandb.init(project=cfg.project, name=cfg.run_name, config=cfg.__dict__)
    # wandb.define_metric("train/*", step_metric="train/global_train_step")
    # wandb.define_metric("val/*",   step_metric="val/eval_step")
    wandb.define_metric("global_train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="global_train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    step_bar = trange(cfg.n_grpo_steps, desc="GRPO")
    
    if cfg.eval_baseline:
        print('Evaluating baseline model...')
        model.eval()
        eval_vllm_model(
            policy_model=model,
            tokenizer=tokenizer,
            llm=llm,
            val_dataset=val_dataset,
            eval_step=0,
            reward_fn=r1_zero_reward_fn,
        )
        model.train()

    global_train_step=0
    for global_step in step_bar:
        # === 采样一个 rollout batch 的 prompts ===
        idxs = random.sample(range(len(train_prompts)), n_prompts_per_rollout)
        prompts_batch = [train_prompts[i] for i in idxs]
        prompts_batch = format_r1_batch(prompts_batch)
        gts_batch = [train_gts[i] for i in idxs]

        # === vLLM rollout ===
        model.eval()
        load_policy_into_vllm_instance(model, llm)  # 同步最新参数到 vLLM
        responses = batched_generate_vllm(
            llm=llm,
            prompts=prompts_batch,
            group_size=cfg.group_size,
            max_new_tokens=cfg.sampling_max_tokens,
            min_new_tokens=cfg.sampling_min_tokens,
            temperature=cfg.sampling_temperature,
            top_p=cfg.top_p,
        )

        # 展开 ground truth
        repeated_gts = [gt for gt in gts_batch for _ in range(cfg.group_size)]
        repeated_prompts = [p for p in prompts_batch for _ in range(cfg.group_size)]

        # === 计算 rewards/advantages（分组归一化）===
        advantages, raw_rewards, rew_meta = compute_group_normalize_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_gts,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.use_std_normalization,
        )
        # (B,) -> (B,1) 便于与 token 维广播
        advantages = advantages.to(model.device).unsqueeze(-1)
        raw_rewards = raw_rewards.to(model.device).unsqueeze(-1)

        # # === 计算 old_log_probs + response_mask（用于 off-policy 的 grpo_clip；on-policy也可留档）===
        # with torch.no_grad():
        #     old_log_probs, response_mask = policy_response_log_probs(
        #         model, tokenizer, repeated_prompts, responses, padding_side=cfg.pad_side
        #     )  # shapes: (B,T), (B,T bool)

        tokenized_dict= tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=responses,
            tokenizer=tokenizer,
            padding_side=cfg.pad_side,
        )
        input_ids=tokenized_dict['input_ids'].to(model.device) # (B,T-1)
        labels=tokenized_dict['labels'].to(model.device) #(B,T-1)
        response_mask=tokenized_dict['response_mask'].to(model.device) #(B,T-1)
        # ===== 把 rollout 打包成 "old_rollout" 缓存 =====
        # 注意：prompts/responses 是 python list，其他是 tensor；放到同一 device 便于训练期切片
        old_log_probs = None
        if cfg.loss_type == "grpo_clip":
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            stream_batch_size = getattr(
                cfg, "old_logprob_stream_batch_size",
                max(1, cfg.train_batch_size // cfg.gradient_accumulation_steps)
            )
            old_log_probs = compute_old_log_probs_streaming(
                model=model,
                input_ids=input_ids,
                labels=labels,
                stream_batch_size=stream_batch_size,
                device=model.device,
            )['log_probs']
        old_rollout = {
            "prompts": repeated_prompts,      # list[str]
            "responses": responses,           # list[str]

            # tensors on CPU
            "input_ids":     input_ids,       # (B, T-1) long CPU
            "labels":        labels,          # (B, T-1) long CPU
            "response_mask": response_mask,   # (B, T-1) bool CPU
            "advantages":    advantages.cpu(),# (B, 1)  float CPU
            "raw_rewards":   raw_rewards.cpu(),# (B, 1) float CPU
            "old_log_probs": old_log_probs.cpu() if old_log_probs is not None else None,  # (B, T-1) float CPU
        }
        num_samples = len(old_rollout["prompts"])

        # === 进入 "train_epochs"：用 train_batch_size 从 old_rollout 取 batch 进行更新 ===
        model.train()

        # 1) 索引放 CPU，避免 CPU 张量用 CUDA 索引
        indices = torch.randperm(num_samples)  # CPU

        for epoch_in_batch in range(cfg.epochs_per_rollout_batch):
            if epoch_in_batch > 0:
                indices = torch.randperm(num_samples)

            for start in range(0, num_samples, cfg.train_batch_size):
                end = min(start + cfg.train_batch_size, num_samples)
                batch_idx_cpu = indices[start:end]                    # CPU long tensor
                batch_idx_list = batch_idx_cpu.tolist()

                # 2) 文本依然只是为了调试/日志，训练不需要搬 GPU
                batch_prompts_list   = [old_rollout["prompts"][i]   for i in batch_idx_list]
                batch_responses_list = [old_rollout["responses"][i] for i in batch_idx_list]

                # 3) 大张量保持在 CPU；不要提前 .to(model.device)
                batch_input_ids_cpu      = old_rollout["input_ids"][batch_idx_cpu]         # (B, T-1) CPU
                batch_labels_cpu         = old_rollout["labels"][batch_idx_cpu]            # (B, T-1) CPU
                batch_response_mask_cpu  = old_rollout["response_mask"][batch_idx_cpu]     # (B, T-1) CPU (bool)
                batch_advantages_cpu     = old_rollout["advantages"][batch_idx_cpu]        # (B, 1) CPU
                batch_rewards_cpu        = old_rollout["raw_rewards"][batch_idx_cpu]       # (B, 1) CPU

                # old_log_probs 只有在 grpo_clip 时才有；否则保持 None
                batch_old_log_probs_cpu = None
                if (old_rollout.get("old_log_probs") is not None) and (cfg.loss_type == "grpo_clip"):
                    batch_old_log_probs_cpu = old_rollout["old_log_probs"][batch_idx_cpu]  # (B, T-1) CPU

                # 4) 动态 micro 切分
                batch_size_now = batch_input_ids_cpu.size(0)
                micro_batch_size  = math.ceil(batch_size_now / cfg.gradient_accumulation_steps)
                num_micro_steps   = math.ceil(batch_size_now / micro_batch_size)

                optimizer.zero_grad(set_to_none=True)
                total_loss_value = 0.0

                for micro_step in range(num_micro_steps):
                    micro_start = micro_step * micro_batch_size
                    micro_end   = min(micro_start + micro_batch_size, batch_size_now)

                    # 5) 仅把当前 micro 切片搬到 GPU
                    input_ids_micro      = batch_input_ids_cpu[micro_start:micro_end].to(model.device, non_blocking=True)
                    labels_micro         = batch_labels_cpu[micro_start:micro_end].to(model.device, non_blocking=True)
                    response_mask_micro  = batch_response_mask_cpu[micro_start:micro_end].to(model.device, non_blocking=True)
                    advantages_micro     = batch_advantages_cpu[micro_start:micro_end].to(model.device, non_blocking=True)
                    rewards_micro        = batch_rewards_cpu[micro_start:micro_end].to(model.device, non_blocking=True)
                    old_log_probs_micro  = None
                    if batch_old_log_probs_cpu is not None:
                        old_log_probs_micro = batch_old_log_probs_cpu[micro_start:micro_end].to(model.device, non_blocking=True)

                    # # 建议传入 attention_mask
                    # attention_mask_micro = (input_ids_micro != tokenizer.pad_token_id).long()

                    # 当前策略 log-probs（需要梯度）
                    current_out = get_response_log_probs(
                        model=model,
                        input_ids=input_ids_micro,
                        labels=labels_micro,
                        return_token_entropy=False,
                        # attention_mask=attention_mask_micro,
                    )
                    current_log_probs = current_out["log_probs"]  # (b, T-1)

                    # 6) 选择损失（old_log_probs 仅在 grpo_clip 使用）
                    loss_tensor, _ = grpo_microbatch_train_step(
                        policy_log_probs=current_log_probs,
                        response_mask=response_mask_micro,
                        gradient_accumulation_steps=num_micro_steps,  # 动态缩放
                        loss_type=cfg.loss_type,
                        raw_rewards=rewards_micro if cfg.loss_type == "no_baseline" else None,
                        advantages=advantages_micro if cfg.loss_type in ("reinforce_with_baseline", "grpo_clip") else None,
                        old_log_probs=old_log_probs_micro,
                        cliprange=cfg.cliprange if cfg.loss_type == "grpo_clip" else None,
                    )

                    total_loss_value += float(loss_tensor.detach().cpu())

                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                wandb.log({
                    "train/loss": total_loss_value,
                    "train/reward_mean": float(rew_meta["mean_reward"]),
                    "train/reward_std": float(rew_meta["std_reward"]),
                    "train/epoch_in_rollout": epoch_in_batch,
                    "global_train_step":global_train_step,
                })
                global_train_step+=1

        # === 周期性验证 ===
        if (global_step + 1) % cfg.eval_every == 0:
            model.eval()
            eval_vllm_model(
                policy_model=model,
                tokenizer=tokenizer,
                llm=llm,
                val_dataset=val_dataset,
                eval_step=global_step+1,
                reward_fn=r1_zero_reward_fn,
            )
            model.train()
        
    wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    app()
