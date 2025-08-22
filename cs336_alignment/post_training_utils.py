import os
os.environ['HF_HOME']='/workspace/home/luotianwei/hf_cache'

import torch
from transformers import PreTrainedTokenizer,PreTrainedModel
from torch import nn
import numpy as np
from cs336_alignment.utils import r1_zero_reward_fn
from typing import Literal, Optional
import torch.nn.functional as F
import wandb
from typing import Callable

def tokenize_prompt_and_output(
    prompt_strs,
    output_strs,
    tokenizer: PreTrainedTokenizer,
    padding_side: Optional[Literal["left", "right"]] = None,
):
    """
    Tokenize question (prompt) and target (output), concatenate, and build masks.
    Returns tensors shaped to predict next token (shift by 1), i.e. length max_len - 1.

    Keys in return:
      - input_ids:      (B, T-1)
      - labels:         (B, T-1)
      - response_mask:  (B, T-1)  True only on response token positions in labels
      - attention_mask: (B, T-1)  1 on non-pad positions in input_ids
    """
    assert len(prompt_strs) == len(output_strs)

    # decide padding side
    pad_side = padding_side or getattr(tokenizer, "padding_side", "right")
    assert pad_side in ("left", "right"), f"padding_side must be 'left' or 'right', got {pad_side}"

    # no special tokens; we control formatting ourselves
    prompt_ids_list = tokenizer(prompt_strs,max_length=1024, add_special_tokens=False).input_ids
    output_ids_list = tokenizer(output_strs,max_length=1024, add_special_tokens=False).input_ids

    combined_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    lens = [len(ids) for ids in combined_ids_list]
    max_len = max(lens)

    B = len(combined_ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # 保险：若没定义 pad_token，则临时把 eos 当 pad
        pad_id = tokenizer.eos_token_id
        assert pad_id is not None, "tokenizer needs pad_token_id or eos_token_id"

    # Allocate
    input_ids      = torch.full((B, max_len),     pad_id, dtype=torch.long)
    labels         = torch.full((B, max_len - 1), pad_id, dtype=torch.long)
    response_mask  = torch.zeros((B, max_len - 1), dtype=torch.bool)

    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        total = p_ids + o_ids
        L = len(total)

        if pad_side == "right":
            # tokens at front, pad at tail
            input_ids[i, :L] = torch.tensor(total, dtype=torch.long)
            labels[i, :L-1]  = torch.tensor(total[1:], dtype=torch.long)
            # response positions in labels index: [len(p_ids)-1, L-1)
            start = len(p_ids) - 1
            end   = L - 1
            if start < end:  # 防止空响应
                response_mask[i, start:end] = True
        else:
            # left padding: pad at head, tokens right-aligned
            pad_len = max_len - L
            input_ids[i, pad_len:] = torch.tensor(total, dtype=torch.long)
            labels[i, pad_len:pad_len + L - 1] = torch.tensor(total[1:], dtype=torch.long)
            start = pad_len + len(p_ids) - 1
            end   = pad_len + L - 1
            if start < end:
                response_mask[i, start:end] = True

    # shift inputs to align with labels
    input_ids = input_ids[:, :-1]

    # # attention mask: 1 for non-pad positions in input_ids
    # attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,          # (B, max_len-1)
        "labels": labels,                # (B, max_len-1)
        "response_mask": response_mask,  # (B, max_len-1) bool
        # "attention_mask": attention_mask # (B, max_len-1) long
    }


def compute_entropy(logits:torch.Tensor) ->torch.Tensor:
    p=torch.softmax(logits,dim=-1)
    logp=logits-torch.logsumexp(logits,dim=-1,keepdim=True)
    entropy=-torch.sum(p*logp,dim=-1)
    return entropy

def get_response_log_probs(model:PreTrainedModel,
                           input_ids:torch.Tensor,
                           labels:torch.Tensor,
                           return_token_entropy:bool=False,
                           attention_mask=None,
                           ):
    logits=model(input_ids).logits
    log_probs=nn.functional.log_softmax(logits,dim=-1)
    log_probs=torch.gather(log_probs,dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    return_dict={"log_probs":log_probs}
    if return_token_entropy:
        entropy=compute_entropy(logits)
        return_dict["token_entropy"]=entropy
    return return_dict


def compute_old_log_probs_streaming(
    model: PreTrainedModel,
    input_ids: torch.Tensor,          # (B, T-1), 建议放在 CPU
    labels: torch.Tensor,             # (B, T-1), 建议放在 CPU
    stream_batch_size: int,           # 每次送入 GPU 的“流式 batch 大小”
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    以流式/分块方式在 GPU 上计算 old_log_probs，返回 (B, T-1) 的 CPU 张量。
    仅用于 grpo_clip：在 rollout 结束、参数还未更新时，用 θ_old 计算。
    """
    assert input_ids.dim() == 2 and labels.dim() == 2, "input_ids / labels 应为 (B, T-1)"
    assert input_ids.size() == labels.size(), "input_ids 与 labels 形状必须一致"

    compute_device = device or next(model.parameters()).device

    old_log_prob_chunks: list[torch.Tensor] = []
    total_examples = input_ids.size(0)

    with torch.inference_mode():  # 推理模式，省内存/更快
        for start_index in range(0, total_examples, stream_batch_size):
            end_index = min(start_index + stream_batch_size, total_examples)

            input_ids_chunk = input_ids[start_index:end_index].to(compute_device, non_blocking=True)
            labels_chunk = labels[start_index:end_index].to(compute_device, non_blocking=True)

            # 不需要 KV cache，进一步降低显存峰值
            logits_chunk = model(
                input_ids_chunk,
                use_cache=False,
            ).logits

            token_log_probs_chunk = torch.log_softmax(logits_chunk, dim=-1)
            selected_log_probs_chunk = torch.gather(
                token_log_probs_chunk,
                dim=-1,
                index=labels_chunk.unsqueeze(-1)
            ).squeeze(-1)  # (chunk_size, T-1)

            old_log_prob_chunks.append(selected_log_probs_chunk.cpu())

            # 及时释放中间张量，降低显存峰值
            del logits_chunk, token_log_probs_chunk
            del input_ids_chunk, labels_chunk
            del selected_log_probs_chunk

    log_probs=torch.cat(old_log_prob_chunks, dim=0)  # (B, T-1) on CPU

    return_dict={"log_probs":log_probs}
    return return_dict

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim)
    return summed / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask:torch.Tensor,
    gradient_accumulation_steps:int,
    normalize_constant:float=1.0,
) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
    
    loss=masked_normalize(policy_log_probs,response_mask,normalize_constant,dim=-1)
    loss=-loss.mean()/gradient_accumulation_steps
    loss.backward()
    
    meta_data={
        "loss":loss.detach(),
        "log_probs":policy_log_probs.detach(),
        "response_mask":response_mask.detach(),
    }
    
    return loss,meta_data


def log_generations(model, tokenizer, prompts, ground_truths, reward_fn,
                    max_new_tokens=1024, temperature=1.0, top_p=1.0,
                    log_file="log.txt"):
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    # 奖励
    rewards = [reward_fn(p, r, gt) for p, r, gt in zip(prompts, responses, ground_truths)]

    # 新增三个平均 reward
    avg_format_reward = np.mean([rw["format_reward"] for rw in rewards])
    avg_answer_reward = np.mean([rw["answer_reward"] for rw in rewards])
    avg_reward        = np.mean([rw["reward"] for rw in rewards])

    # token 熵
    with torch.no_grad():
        logits = model(**inputs).logits
        entropy = compute_entropy(logits).float()
        avg_entropy = entropy.mean(dim=-1).cpu().numpy()

    # 长度统计
    lengths = [len(tokenizer.encode(r)) for r in responses]
    correct_mask = [rw["reward"] > 0 for rw in rewards]
    avg_len = np.mean(lengths)
    avg_len_correct = np.mean([l for l, c in zip(lengths, correct_mask) if c]) if any(correct_mask) else 0
    avg_len_incorrect = np.mean([l for l, c in zip(lengths, correct_mask) if not c]) if not all(correct_mask) else 0

    # 记录到 log.txt
    with open(log_file, "a", encoding="utf-8") as f:
        for p, r, gt, rw, ent, l in zip(prompts, responses, ground_truths, rewards, avg_entropy, lengths):
            f.write(f"Prompt: {p}\n")
            f.write(f"Response: {r}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write(f"Reward: {rw}\n")
            f.write(f"Avg Token Entropy: {ent:.4f}\n")
            f.write(f"Response Length: {l}\n")
            f.write("-" * 50 + "\n")
        
        f.write("Overall Stats:\n")
        f.write(f"Avg Token Entropy: {np.mean(avg_entropy):.4f}\n")
        f.write(f"Avg Response Length: {avg_len:.2f}\n")
        f.write(f"Avg Response Length (Correct): {avg_len_correct:.2f}\n")
        f.write(f"Avg Response Length (Incorrect): {avg_len_incorrect:.2f}\n")
        f.write(f"Avg Format Reward: {avg_format_reward:.4f}\n")
        f.write(f"Avg Answer Reward: {avg_answer_reward:.4f}\n")
        f.write(f"Avg Reward: {avg_reward:.4f}\n")
        f.write("=" * 50 + "\n")

    return {
        "avg_token_entropy": float(np.mean(avg_entropy)),
        "avg_response_length": float(avg_len),
        "avg_response_length_correct": float(avg_len_correct),
        "avg_response_length_incorrect": float(avg_len_incorrect),
        "avg_format_reward": float(avg_format_reward),
        "avg_answer_reward": float(avg_answer_reward),
        "avg_reward": float(avg_reward),
    }
    
def log_val_metrics(
    model,
    tokenizer,
    prompts,
    responses,
    ground_truths,
    reward_fn,
    log_file="log.txt",
    log_to_wandb=True,                 # 新增：是否写入 wandb
    wandb_prefix="val/",               # 新增：指标前缀
    log_per_sample_table=True          # 新增：是否上传逐样本表
):
    """
    根据已有 responses 计算奖励、熵、长度等统计信息，写入日志文件，并可选记录到 wandb。
    """
    model.eval()

    # 奖励
    rewards = [reward_fn(r, gt) for p, r, gt in zip(prompts, responses, ground_truths)]

    # 三个平均 reward
    avg_format_reward = np.mean([rw["format_reward"] for rw in rewards]) if rewards else 0.0
    avg_answer_reward = np.mean([rw["answer_reward"] for rw in rewards]) if rewards else 0.0
    avg_reward        = np.mean([rw["reward"] for rw in rewards]) if rewards else 0.0

    # 长度统计
    lengths = [len(tokenizer.encode(r)) for r in responses]
    correct_mask = [rw["reward"] > 0 for rw in rewards]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    avg_len_correct = float(np.mean([l for l, c in zip(lengths, correct_mask) if c])) if any(correct_mask) else 0.0
    avg_len_incorrect = float(np.mean([l for l, c in zip(lengths, correct_mask) if not c])) if not all(correct_mask) else 0.0

    # 记录到 log.txt
    with open(log_file, "a", encoding="utf-8") as f:
        for p, r, gt, rw, l in zip(prompts, responses, ground_truths, rewards, lengths):
            f.write(f"Prompt: {p}\n")
            f.write(f"Response: {r}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write(f"Reward: {rw}\n")
            f.write(f"Response Length: {l}\n")
            f.write("-" * 50 + "\n")
        
        f.write("Overall Stats:\n")
        f.write(f"Avg Response Length: {avg_len:.2f}\n")
        f.write(f"Avg Response Length (Correct): {avg_len_correct:.2f}\n")
        f.write(f"Avg Response Length (Incorrect): {avg_len_incorrect:.2f}\n")
        f.write(f"Avg Format Reward: {avg_format_reward:.4f}\n")
        f.write(f"Avg Answer Reward: {avg_answer_reward:.4f}\n")
        f.write(f"Avg Reward: {avg_reward:.4f}\n")
        f.write("=" * 50 + "\n")

    # === 记录到 wandb（整体统计 + 可选逐样本表） ===
    metrics = {
        f"{wandb_prefix}avg_response_length": avg_len,
        f"{wandb_prefix}avg_response_length_correct": avg_len_correct,
        f"{wandb_prefix}avg_response_length_incorrect": avg_len_incorrect,
        f"{wandb_prefix}avg_format_reward": float(avg_format_reward),
        f"{wandb_prefix}avg_answer_reward": float(avg_answer_reward),
        f"{wandb_prefix}avg_reward": float(avg_reward),
        f"{wandb_prefix}num_samples": len(responses),
    }

    if log_to_wandb and wandb.run is not None:
        # 整体指标
        wandb.log(metrics)

        # 逐样本表（可选）
        if log_per_sample_table and len(responses) > 0:
            table = wandb.Table(columns=[
                "prompt", "response", "ground_truth",
                "format_reward", "answer_reward", "reward",
                "response_length", "is_correct"
            ])
            for p, r, gt, rw, l, c in zip(
                prompts, responses, ground_truths, rewards,lengths, correct_mask
            ):
                table.add_data(
                    p, r, gt,
                    float(rw["format_reward"]), float(rw["answer_reward"]), float(rw["reward"]),
                    int(l), bool(c)
                )
            wandb.log({f"{wandb_prefix}samples": table})

            # 也可以顺便打几个直方图，便于观测分布（可选）
            wandb.log({
                f"{wandb_prefix}response_length_hist": wandb.Histogram(lengths),
                f"{wandb_prefix}reward_hist": wandb.Histogram([float(rw["reward"]) for rw in rewards]),
            })

    return {
        "avg_response_length": float(avg_len),
        "avg_response_length_correct": float(avg_len_correct),
        "avg_response_length_incorrect": float(avg_len_incorrect),
        "avg_format_reward": float(avg_format_reward),
        "avg_answer_reward": float(avg_answer_reward),
        "avg_reward": float(avg_reward),
    }
    
def compute_group_normalize_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """    
    rewards=[]
    for r,git in zip(rollout_responses,repeated_ground_truths):
        rw=reward_fn(r,git)
        rewards.append(rw["reward"])
    
    rewards=torch.tensor(rewards,dtype=torch.float32)
    rewards = rewards.view(-1, group_size)
    rewards_mean = rewards.mean(dim=1, keepdim=True)
    rewards_std = rewards.std(dim=1, keepdim=True) + advantage_eps
    advantages = rewards - rewards_mean
    if normalize_by_std:
        advantages /= rewards_std
    advantages = advantages.flatten()
    
    rewards=rewards.flatten()
    mean_reward=rewards.mean()
    std_reward= rewards.std()
    
    metadata={"mean_reward":mean_reward,"std_reward":std_reward}
    
    return advantages,rewards,metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    loss=-raw_rewards_or_advantages*policy_log_probs
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    log_probs_ratio=policy_log_probs-old_log_probs
    log_probs_ratio=torch.exp(log_probs_ratio)
    
    unclipped_term=log_probs_ratio*advantages
    clipped_term=log_probs_ratio*advantages
    if cliprange is not None:
        clipped_term=torch.clamp(log_probs_ratio,1-cliprange,1+cliprange)*advantages
    
    loss=-torch.min(unclipped_term,clipped_term)
    
    was_clipped=clipped_term<unclipped_term
    
    metadata={"was_clipped":was_clipped}
    
    return loss,metadata    

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, metadata

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor=tensor*mask
    masked_sum= masked_tensor.sum(dim=dim)
    masked_count = mask.sum(dim=dim)
    return masked_sum / masked_count
    

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    
    loss,metadata=compute_policy_gradient_loss(policy_log_probs,loss_type,raw_rewards,advantages,old_log_probs,cliprange)
    loss = masked_mean(loss, response_mask, dim=-1)  # average over sequence length
    # loss=masked_normalize(loss,response_mask,normalize_constant=loss.shape[1],dim=-1)
    loss=loss.mean() / gradient_accumulation_steps  # average over batch and normalize by accumulation steps
    loss.backward()
    return loss,metadata


if __name__=='__main__':
    from transformers import AutoModelForCausalLM,AutoTokenizer
    
    model=AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-Math-1.5B',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    ).to("cuda")
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    prompts= [
        "Hello, my name is",
        "The president of the United State is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    log_generations(
        model,
        tokenizer,
        prompts,
        ground_truths=["John HH", "Joe Biden", "Paris bb", "bright  aa"],
        reward_fn=r1_zero_reward_fn
    )