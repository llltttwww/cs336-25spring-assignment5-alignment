import os 
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'

from cs336_alignment.utils import r1_zero_reward_fn

import json
import os
from typing import Callable,List

from vllm import LLM,SamplingParams

DATA_PATH='/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/data/MATH/validation.jsonl'
OUTPUT_PATH='/workspace/home/luotianwei/cs336/cs336-25spring-assignment5-alignment/output/test_results/qwen2.5-1.5b-base/validation.jsonl'

R1_ZERO_PREFIX = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

def r1_zero_prompt(problem:str) -> str:
    return  R1_ZERO_PREFIX.format(question=problem)

def evaluate_vllm(
    vllm_model:LLM,
    reward_fn:Callable,
    prompts:List[str],
    eval_sampling_params:SamplingParams,
    references:List[str],
) -> None:
    outputs=vllm_model.generate(prompts,eval_sampling_params)
    total_format_reward=0.0
    total_answer_reward=0.0
    total_reward=0.0
    results=[]
    for i,output in enumerate(outputs):
        prompt=output.prompt
        generated_text=output.outputs[0].text
        reward_dict=reward_fn(generated_text,references)
        format_reward= reward_dict["format_reward"]
        answer_reward= reward_dict["answer_reward"]
        reward=reward_dict["reward"]
        
        results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "reward": reward,
            }
        )
        total_format_reward+=format_reward
        total_answer_reward+=answer_reward
        total_reward+=reward
        
    mean_format_reward=total_format_reward/len(outputs)
    mean_answer_reward=total_answer_reward/len(outputs)
    mean_reward=total_reward/len(outputs)
    
    with open(OUTPUT_PATH,"w") as f:
        json.dump(results,f,indent=2)
    
    print(f'Mean format reward: {mean_format_reward}')
    print(f'Mean answer reward: {mean_answer_reward}')
    print(f'Mean reward: {mean_reward}')
    print(f'Saved to {OUTPUT_PATH}')
    

if __name__=='__main__':
    promblems,references=[],[]
    with open(DATA_PATH,"r") as f:
        for line in f:
            obj=json.loads(line)
            promblems.append(r1_zero_prompt(obj['problem']))
            references.append(obj['solution'])
            
    llm=LLM(model='Qwen/Qwen2.5-Math-1.5B')
    
    sampling_params=SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=promblems,
        eval_sampling_params=sampling_params,
        references=references
    )
    



# from vllm import LLM, SamplingParams
# import os 


# os.environ["HF_HOME"]="/workspace/home/luotianwei/hf_cache"

# prompts= [
#     "Hello, my name is",
#     "The president of the United State is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# sampling_params=SamplingParams(
#     temperature=1.0,
#     top_p=1.0,
#     max_tokens=1024,
#     stop=["\n"]
# )

# llm=LLM(model="Qwen/Qwen2.5-Math-1.5B")

# outputs=llm.generate(prompts,sampling_params)

# for output in outputs:
#     prompt=output.prompt
#     generated_text=output.outputs[0].text
#     print(f"Prompt:{prompt!r},Generated text:{generated_text!r}")