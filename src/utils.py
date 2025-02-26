import json
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_json_lines(path):
    with open(path, "r") as f:
        data = [json.loads(d) for d in f]
    return data

def write_json(path, content):
    with open(path, "w") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

def write_json_lines(path, content):
    with open(path, "w") as f:
        for d in content:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def trajectory_to_response(trajectory):
    response = ""
    for action in trajectory:
        response += action
    return response

def apply_chat_template(prompt, response, tokenizer, add_bos=True):
    # if tokenizer == None:
    #     prompt_with_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n"
    # else:
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}]
    #     prompt_with_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # if isinstance(response, str):
    #     prompt_with_template += response
    # elif isinstance(response, list):
    #     prompt_with_template += trajectory_to_response(response)
    # else:
    #     print(response)
    #     assert False, "wrong response type"
    # return prompt_with_template

    system_prompt = """You are a helpful assistant capable of multi-step reasoning. 
If you are provided with a task that can benefit from in-depth thinking, e.g., math, coding, and logical reasoning, you should solve the problem step by step and eventually give your answer.
Use <|Reasoning_step|> and <|/Reasoning_step|> to mark the start and end of one step of reasoning, and wrap your final answer with <|Output|> and <|/Output|> after sufficient reasoning steps.
"""
    prompt_with_template = "<|begin_of_text|>" if add_bos else ""
    # prompt_with_template += "<|start_header_id|>system<|end_header_id|>\n\n"+system_prompt+"<|eot_id|>"
    prompt_with_template += "<|start_header_id|>user<|end_header_id|>\n\n"+prompt+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # prompt_with_template += "<|Reasoning_step|>"
    if isinstance(response, str):
        prompt_with_template += response
    elif isinstance(response, list):
        prompt_with_template += trajectory_to_response(response)
    else:
        print(response)
        assert False, "wrong response type"
    return prompt_with_template


def load_actor_model(path):
    actor_tokenizer = AutoTokenizer.from_pretrained(path)
    actor_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return actor_model, actor_tokenizer
