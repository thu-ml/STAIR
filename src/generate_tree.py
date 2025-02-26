from mcts import mcts
from mcts_node import mcts_node
import json
from tqdm import tqdm
from datetime import datetime
import argparse
from omegaconf import OmegaConf
from config import MCTSBaseConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from final_orm import evaluate
from utils import read_json, write_json, apply_chat_template, load_actor_model
from diskcache import Cache
import threading
from openai import OpenAI
import openai
import os
import time
import logging

def try_vllm(actor, tokenizer, node, config, terminators, retry_num=0):
    try:
        completion = actor.completions.create(
            model="actor",
            # no bos since vllm will automatically generate bos
            prompt=apply_chat_template(node.prompt, node.trajectory, tokenizer, add_bos=False),
            echo=False,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={
                "stop_token_ids": terminators,
                "top_k": config.top_k,
                "skip_special_tokens": False,
                "include_stop_str_in_output": True
            }
        )
        return completion.choices[0].text
    except Exception as e:
        logging.warning("VLLM ERROR: " + str(e))
        retry_num += 1
        time.sleep(2**retry_num)
        logging.warning(node.prompt + " RETRY!!!!!! RETRY TIME: " + str(retry_num))
        return try_vllm(actor, tokenizer, node, config, terminators, retry_num)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="../config/tree_generate_test.yaml")
    args = args.parse_args()
    return args

def generate_action(actor, tokenizer, node:mcts_node, config):
    terminators = [
        tokenizer.eos_token_id
    ] + [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in config.stop_tokens]
    if config.generate_mode == "local":
        prompt = apply_chat_template(node.prompt, node.trajectory, tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(actor.device)

        with torch.no_grad():
            outputs = actor.generate(
                input_ids,
                max_new_tokens=config.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            ).cpu()
        response = outputs[0][input_ids.shape[-1]:]
        if config.p_average_strategy == "uniform":
            return tokenizer.decode(response, skip_special_tokens=False), 1
        with torch.no_grad():
            second_outputs = actor(outputs.to(actor.device)).cpu()
        logits = second_outputs.logits[0][input_ids.shape[-1]-1:-1] / config.temperature
        probs = torch.softmax(logits, dim=-1)
        prob = probs[torch.arange(response.size(0)), response]
        if config.p_average_strategy == "geometric":
            p = torch.exp(torch.sum(torch.log(prob)) / len(prob))
        elif config.p_average_strategy == "arithmetic":
            p = torch.sum(prob) / len(prob)
        return tokenizer.decode(response, skip_special_tokens=False), float(p)
    else:
        return try_vllm(actor, tokenizer, node, config, terminators), 1.0

# for outcome -1<=reward<=1; for safe-constraint -2k1-k2<=reward<=2k1+k2
def _get_reward(prompt, whole_answer, ground_truth, question_type, config, retry_num=0):
    try:
        # You can define your own evaluate orm. Should return safe_score, helpful_score (-1 <= both of them <= 1)
        mode = config.mode
        safe_score, helpful_score = evaluate(mode, prompt, whole_answer, ground_truth, question_type)
        if mode == "outcome":
            if question_type == "safety":
                return safe_score
            else:
                return helpful_score
        elif mode == "safe-constraint":
            return (config.k1 + config.k2) * safe_score + config.k1 * safe_score * helpful_score
        else:
            print("WRONG MODE:", mode)
            exit(-1)
    except Exception as e:
        logging.warning("ORM ERROR:" + str(e))
        retry_num += 1
        time.sleep(2**retry_num)
        logging.warning(prompt + " RETRY!!!!!! RETRY TIME: " + str(retry_num))
        return _get_reward(prompt, whole_answer, ground_truth, question_type, config, retry_num)

def get_reward(node:mcts_node, ground_truth, question_type, config):
    answer = ""
    mode = config.mode
    for action in node.trajectory:
        answer += action
    return _get_reward(node.prompt, answer, ground_truth, question_type, config)

def rollout_and_get_reward(actor, tokenizer, node:mcts_node, ground_truth, config, question_type):
    terminators = [
        tokenizer.eos_token_id
    ] + [tokenizer.convert_tokens_to_ids(end_token) for end_token in config.end_tokens]

    if config.generate_mode == "local":
        prompt = apply_chat_template(node.prompt, node.trajectory, tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(actor.device)

        with torch.no_grad():
            outputs = actor.generate(
                input_ids,
                eos_token_id=terminators,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_tokens*config.max_depth,
            ).cpu()
        response = outputs[0][input_ids.shape[-1]:]
        answer = ""
        for action in node.trajectory:
            answer += action
        rollout_action = tokenizer.decode(response, skip_special_tokens=False)
        answer += rollout_action
    else:
        rollout_action = try_vllm(actor, tokenizer, node, config, terminators)
        answer = ""
        for action in node.trajectory:
            answer += action
        answer += rollout_action

    return _get_reward(node.prompt, answer, ground_truth, question_type, config)


def thread_function(prompts_data, config, worker_order):
    logging.info("THREAD " + str(worker_order) +" BEGIN")

    if config.generate_mode == "local":
        actor_model, actor_tokenizer = load_actor_model(config.actor_model_dir)
        actor_model.eval()
    else:
        openai_api_key = "EMPTY"
        openai_api_base = config.server_url
        actor_model = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        actor_tokenizer = AutoTokenizer.from_pretrained(config.actor_model_dir)

    if config.use_cache:
        folder_path = os.path.join(config.cache_dir, str(worker_order))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cache = Cache(folder_path)

    output_content = []
    for prompt_index, prompt in enumerate(prompts_data):
        if config.use_cache and prompt["question"] in cache:
            mct_data = cache[prompt["question"]]
            output_content.append(mct_data)
            continue
        question = prompt["question"]
        ground_truth = prompt["ground_truth"] if "ground_truth" in prompt.keys() else None
        question_type = prompt["type"]

        root_node = mcts_node(None, "", 1)
        root_node.set_param(config.c, config.max_depth, question, 0, [])
        mct = mcts(root_node)

        for iteration in tqdm(range(config.iterations), desc=f"Worker {worker_order} Prompt {prompt_index} Iter", ncols=100):
            # Select until reach a leaf node
            selected_node = mct.select(config.able_to_reselected, score_type=config.score_type)
            if selected_node == None:
                break
            if selected_node.visited_times != 0 and not selected_node.is_terminal and selected_node.depth < selected_node.max_depth:
                # Expand and select one child
                children = []
                for _ in range(config.generate_samples_number):
                    action, p = generate_action(actor_model, actor_tokenizer, selected_node, config)
                    if action not in children:
                        mct.add_node(selected_node, action, p)
                        children.append(action)
                selected_node = mct.select(config.able_to_reselected, selected_node, config.score_type)

            # Evaluate
            if selected_node.visited_times != 0:
                # Has already visited before
                is_terminal = selected_node.is_terminal
                if selected_node.reward != None:
                    feedback = selected_node.reward
                    feedback_type = "reward"
                else:
                    feedback = rollout_and_get_reward(actor_model, actor_tokenizer, selected_node, ground_truth, config, question_type)
                    feedback_type = "rollout"
            else: # First time encounter
                is_terminal = False
                for end_token in config.end_tokens:
                    if selected_node.trajectory != [] and end_token in selected_node.trajectory[-1]:
                        is_terminal = True
                        break
                if is_terminal:
                    feedback = get_reward(selected_node, ground_truth, question_type, config)
                    feedback_type = "reward"
                else:
                    feedback = rollout_and_get_reward(actor_model, actor_tokenizer, selected_node, ground_truth, config, question_type)
                    feedback_type = "rollout"
            # Backprapagation
            mct.update(selected_node, feedback, is_terminal, feedback_type)

        if config.visit_all_node: # In this stage, we won't visit already visited node
            selected_node = mct.select(False, score_type=config.score_type)
            pbar = tqdm(desc=f"Worker {worker_order} Prompt {prompt_index} Iter Left Clear")
            while selected_node != None:
                # Evaluate
                if selected_node.visited_times > 0: # For node already visited but still can expand
                    mct.update(selected_node, -100, selected_node.is_terminal, "refresh") # We won't expand in this stage
                    selected_node = mct.select(False, score_type=config.score_type)
                    pbar.update(1)
                    continue
                # For node not visited before
                is_terminal = False
                for end_token in config.end_tokens:
                    if selected_node.trajectory != [] and end_token in selected_node.trajectory[-1]:
                        is_terminal = True
                        break
                if is_terminal:
                    feedback = get_reward(selected_node, ground_truth, question_type, config)
                    feedback_type = "reward"
                else:
                    feedback = rollout_and_get_reward(actor_model, actor_tokenizer, selected_node, ground_truth, config, question_type)
                    feedback_type = "rollout"
                # Backprapagation
                mct.update(selected_node, feedback, is_terminal, feedback_type)
                selected_node = mct.select(False, score_type=config.score_type)
                pbar.update(1)
            pbar.close()

        mct_data = mct.show_tree()
        if config.use_cache:
            cache[prompt["question"]] = mct_data
        output_content.append(mct_data)

    write_json(os.path.join(config.output_path, str(worker_order)+".json"), output_content)
    logging.info("THREAD " + str(worker_order) +" END")


def main():
    args = parse_args()
    config = OmegaConf.structured(MCTSBaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    logging.basicConfig(filename=config.log_file, level=logging.INFO)
    logging.info("CONFIG IS:"+str(config))

    prompts_data = read_json(config.train_prompt_path)
    logging.info("PROMPT DATA LOADED")

    threads = []
    for i in range(config.worker_num):
        prompts_data_for_worker = prompts_data[min(i*config.worker_prompt_num,len(prompts_data)):min((i+1)*config.worker_prompt_num, len(prompts_data))]
        thread = threading.Thread(target=thread_function, args=(prompts_data_for_worker, config, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
