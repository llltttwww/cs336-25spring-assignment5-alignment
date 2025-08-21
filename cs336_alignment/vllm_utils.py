import os
os.environ["HF_HOME"]="/workspace/home/luotianwei/hf_cache"
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
from vllm import LLM,SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
from cs336_alignment.datasets import MathSFTDataset,make_sft_collate_fn
from tqdm import tqdm
# import CrossEntropyloss from torch
from torch.nn import CrossEntropyLoss
from post_training_utils import log_val_metrics
from typing import Callable
import random
from pydantic import BaseModel
from cs336_alignment.utils import ordered_filename
from typing import List

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

def init_vllm(model_id:str,device:str,seed:int,gpu_memory_utilization:float=0.85):
    vllm_set_random_seed(seed)
    world_size_patch=patch("torch.distributed.get_world_size",return_value=1)
    profiling_patch=patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch,profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
def load_policy_into_vllm_instance(policy:PreTrainedModel,llm:LLM):
    state_dict=policy.state_dict()
    
    # Handle compiled models by stripping _orig_mod prefix
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()
        }
    
    llm_model=llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
    
def eval_vllm_model(policy_model:PreTrainedModel,tokenizer:AutoTokenizer,llm:LLM,val_dataset:Dataset,eval_step:int,reward_fn: Callable):
    policy_model.eval()
    load_policy_into_vllm_instance(policy_model, llm)
    
    indices = random.sample(range(len(val_dataset)), 1000)
    # indices=range(len(val_dataset))
    val_samples = [val_dataset.data[i] for i in indices]
    prompts = [R1_ZERO_PROMPT.format(question=p.strip()) for p, _ in val_samples]
    ground_truths = [r for _, r in val_samples]
    evaluate_vllm(llm,reward_fn,prompts,ground_truths)
    sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    stop=["</answer>"],
    include_stop_str_in_output=True
    )
    vllm_outputs = llm.generate(
        prompts,
        sampling_params
    )
    responses = [o.outputs[0].text for o in vllm_outputs]

    stats = log_val_metrics(
        policy_model, tokenizer, prompts, responses,
        ground_truths, reward_fn=reward_fn
    )
    log_dict = {f"eval/{k}": v for k, v in stats.items()}
    log_dict["eval_step"] = eval_step
    wandb.log(log_dict)
    eval_step += 1



class EvalMetrics(BaseModel):
    n_examples: int
    n_format_correct: int
    n_format_incorrect: int
    n_answer_correct: int
    n_answer_incorrect: int
    n_correct: int
    n_incorrect: int
    format_accuracy: float
    answer_accuracy: float
    accuracy: float


class EvalResult(BaseModel):
    prompt: str
    completion: str
    ground_truth: str
    rewards: dict[str, float]
    

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams | None = None,
    out_dir: str | None = "eval_vllm_counterpart",
    out_file: str | None = None,
    write: bool = True,
    min_tokens: int = 0,
) -> tuple[list[EvalResult], EvalMetrics]:
    """
    Eval LM on prompts, compute eval metrics, optionally serialize to disk, return evaluation results.
    """
    sampling_params = eval_sampling_params or SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = vllm_model.generate(
        prompts,
        sampling_params,
    )

    os.makedirs(out_dir, exist_ok=True)
    filename = out_file or f"{ordered_filename('eval')}.jsonl"
    outpath = os.path.join(out_dir, filename)

    results = []

    metrics = EvalMetrics(
        n_examples=len(prompts),
        n_format_correct=0,
        n_format_incorrect=0,
        n_answer_correct=0,
        n_answer_incorrect=0,
        n_correct=0,
        n_incorrect=0,
        format_accuracy=0.0,
        answer_accuracy=0.0,
        accuracy=0.0,
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text
        ground_truth = ground_truths[i]
        rewards = reward_fn(completion, ground_truth)

        format_correct = rewards["format_reward"]
        answer_correct = rewards["answer_reward"]
        correct = rewards["reward"]

        metrics.n_format_correct += int(format_correct)
        metrics.n_format_incorrect += int(not format_correct)
        metrics.n_answer_correct += int(answer_correct)
        metrics.n_answer_incorrect += int(not answer_correct)
        metrics.n_correct += int(correct)
        metrics.n_incorrect += int(not correct)

        result = EvalResult(
            prompt=prompt,
            completion=completion,
            ground_truth=ground_truth,
            rewards=rewards,
        )

        results.append(result)

    metrics.format_accuracy = metrics.n_format_correct / metrics.n_examples
    metrics.answer_accuracy = metrics.n_answer_correct / metrics.n_examples
    metrics.accuracy = metrics.n_correct / metrics.n_examples

    if write:
        with open(outpath, "w") as f:
            f.write(metrics.model_dump_json() + "\n")
            f.write("\n".join([result.model_dump_json() for result in results]) + "\n")

    return results, metrics



def batched_generate_vllm(
    llm: LLM,
    prompts: List[str],
    group_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    # repeat prompts for group rollouts
    repeated_prompts = [p for p in prompts for _ in range(group_size)]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_new_tokens,
        max_tokens=max_new_tokens,
        stop=["</answer>"],  # R1 Zero style
        include_stop_str_in_output=True
    )
    outputs = llm.generate(repeated_prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    return responses