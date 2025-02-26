from utils import read_json, write_json_lines
import argparse
import torch
import random
from transformers import AutoTokenizer
from utils import apply_chat_template, trajectory_to_response
from tqdm import tqdm
import queue
import threading

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=42, help="Random seed")
    args.add_argument('--mcts_path', type=str, default="../mct_data/mct_data.json", help="MCT data path")
    args.add_argument('--traindata_type', type=str, default='dpo', help="Choose from sft and dpo")
    args.add_argument('--v_threshold', type=float, default=0.5, help="Min delta v between chosen and rejected in dpo pairs. Only activate when dpo")
    args.add_argument('--actor_terminal_only', type=bool, default=False, help="Whether to use only full trajectories to construct dpo pairs.")
    args.add_argument('--actor_data_path', type=str, default="../actor_train_data/actor_data.jsonl", help="Path to write actor traindata")
    args.add_argument('--tokenizer_path', type=str, default="../actor", help="Tokenizer path")
    args.add_argument('--max_tokens', type=int, default=4096, help="Max tokens for node to be in actor traindata")
    args.add_argument('--max_child_num', type=int, default=4, help="Max children number for each node in MCT")
    args.add_argument('--min_visit_times', type=int, default=1, help="Min visit times for non-terminal node to be in dpo pairs / sft. Only activate when dpo and not terminal only / sft")
    args.add_argument('--max_gb_ratio', type=int, default=4, help="Max ratio between good and bad answers when construct pairs. Only activate when dpo and terminal only")
    args.add_argument('--good_value_threshold', type=float, default=0.8, help="Threshold for node to be a good node.")
    args.add_argument('--max_sft_label', type=int, default=10, help="Max sft for each prompt. Only activate when sft")
    args = args.parse_args()
    return args

def generate_data(mcts_data, args, save_place):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    train_data = []
    
    if args.traindata_type == "dpo":
        for d in tqdm(mcts_data):
            tags = queue.Queue()
            tags.put("0")
            prompt = d["0"]["prompt"]
            if args.actor_terminal_only:
                right_answer_info = []
                wrong_answer_info = []
                while not tags.empty():
                    tag = tags.get()
                    for i in range(args.max_child_num):
                        idx = tag+"."+str(i)
                        if idx in d.keys():
                            response = trajectory_to_response(d[idx]["trajectory"])
                            prompt_with_template = apply_chat_template(prompt, response, tokenizer)
                            tensor = tokenizer(prompt_with_template, add_special_tokens=False, return_tensors="pt")
                            length = len(tensor["input_ids"][0])
                            if length <= args.max_tokens:
                                tags.put(idx)
                                if d[idx]["is_terminal"]:
                                    reward = d[idx]["reward"]
                                    if reward >= args.good_value_threshold:
                                        right_answer_info.append((response, d[idx]["visited_times"], reward))
                                    elif reward <= args.good_value_threshold - args.v_threshold:
                                        wrong_answer_info.append((response, d[idx]["visited_times"], reward))
                if len(right_answer_info) == 0 or len(wrong_answer_info) == 0:
                    continue
                random.shuffle(right_answer_info)
                random.shuffle(wrong_answer_info)
                if len(right_answer_info) < len(wrong_answer_info):
                    right_answer_info *= min(int(len(wrong_answer_info)/len(right_answer_info))+1,args.max_gb_ratio)
                else:
                    wrong_answer_info *= min(int(len(right_answer_info)/len(wrong_answer_info))+1,args.max_gb_ratio)
                for i in range(min(len(right_answer_info), len(wrong_answer_info))):
                    train_data.append({
                        "prompt": prompt,
                        "chosen": right_answer_info[i][0],
                        "rejected": wrong_answer_info[i][0],
                        "chosen_score": right_answer_info[i][2],
                        "rejected_score": wrong_answer_info[i][2],
                        "chosen_is_terminal": True,
                        "rejected_is_terminal": True,
                        "chosen_visited_times": right_answer_info[i][1],
                        "rejected_visited_times": wrong_answer_info[i][1]
                    })

            else:
                while not tags.empty():
                    tag = tags.get()
                    responses = []
                    values = []
                    is_terminal = []
                    visited_times = []
                    for i in range(args.max_child_num):
                        idx = tag+"."+str(i)
                        if idx in d.keys():
                            value = d[idx]["true_value"]
                            response = trajectory_to_response(d[idx]["trajectory"])
                            prompt_with_template = apply_chat_template(prompt, response, tokenizer)
                            tensor = tokenizer(prompt_with_template, add_special_tokens=False, return_tensors="pt")
                            length = len(tensor["input_ids"][0])
                            if length <= args.max_tokens:
                                tags.put(idx)
                                if d[idx]["is_terminal"] or d[idx]["visited_times"] >= args.min_visit_times:
                                    responses.append(response)
                                    values.append(value)
                                    is_terminal.append(d[idx]["is_terminal"])
                                    visited_times.append(d[idx]["visited_times"])
                    _len = len(responses)
                    for i in range(_len):
                        for j in range(_len):
                            if values[i] - values[j] > args.v_threshold and values[i] >= args.good_value_threshold:
                                train_data.append({
                                    "prompt": prompt,
                                    "chosen": responses[i],
                                    "rejected": responses[j],
                                    "chosen_score": values[i],
                                    "rejected_score": values[j],
                                    "chosen_is_terminal": is_terminal[i],
                                    "rejected_is_terminal": is_terminal[j],
                                    "chosen_visited_times": visited_times[i],
                                    "rejected_visited_times": visited_times[j]
                                })

    elif args.traindata_type == "sft":
        for d in tqdm(mcts_data):
            prompt_train_data = []
            tags = queue.Queue()
            tags.put("0")
            prompt = d["0"]["prompt"]
            while not tags.empty():
                tag = tags.get()
                has_child = False
                for i in range(args.max_child_num):
                    idx = tag+"."+str(i)
                    if idx in d.keys():
                        response = trajectory_to_response(d[idx]["trajectory"])
                        prompt_with_template = apply_chat_template(prompt, response, tokenizer)
                        tensor = tokenizer(prompt_with_template, add_special_tokens=False, return_tensors="pt")
                        length = len(tensor["input_ids"][0])
                        if length <= args.max_tokens:
                            tags.put(idx)
                        has_child = True
                if not has_child and (not args.actor_terminal_only or d[tag]["is_terminal"]):
                    if d[tag]["true_value"] >= args.good_value_threshold and (d[tag]["visited_times"] >= args.min_visit_times or d[tag]["is_terminal"]):
                        prompt_train_data.append({
                            "prompt": prompt,
                            "chosen": trajectory_to_response(d[tag]["trajectory"]),
                            "chosen_score": d[tag]["true_value"],
                            "chosen_is_terminal": d[tag]["is_terminal"],
                            "chosen_visited_times": d[tag]["visited_times"]
                        })
            random.shuffle(prompt_train_data)
            train_data += prompt_train_data[:min(len(prompt_train_data), args.max_sft_label)]

    else:
        print("WRONG TYPE:", args.traindata_type)
        exit(-1)

    save_place += train_data

def main():
    args = parse_args()
    print(args)
    random.seed(args.seed)
    mcts_data = read_json(args.mcts_path)
    print("MCT_LEN:", len(mcts_data))

    actor_partial_data = [[] for _ in range(200)]
    worker_prompt_num = (len(mcts_data) - 1) // 200 + 1
    threads = []
    for i in range(200):
        mcts_data_for_worker = mcts_data[min(i*worker_prompt_num,len(mcts_data)):min((i+1)*worker_prompt_num,len(mcts_data))]
        thread = threading.Thread(target=generate_data, args=(mcts_data_for_worker, args, actor_partial_data[i]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    actor_data = []
    for actor_partial_d in actor_partial_data:
        actor_data += actor_partial_d

    if args.traindata_type == "dpo":
        print("ACTOR_PAIRS_CNT:", len(actor_data))
        terminal_terminal_cnt = 0
        terminal_non_terminal_cnt = 0
        non_terminal_non_terminal_cnt = 0
        for d in actor_data:
            if d["chosen_is_terminal"] and d["rejected_is_terminal"]:
                terminal_terminal_cnt += 1
            elif not d["chosen_is_terminal"] and not d["rejected_is_terminal"]:
                non_terminal_non_terminal_cnt += 1
            else:
                terminal_non_terminal_cnt += 1
        print("Terminal Terminal CNT:", terminal_terminal_cnt)
        print("Terminal Non-Terminal CNT:", terminal_non_terminal_cnt)
        print("Non-Terminal Non-Terminal CNT:", non_terminal_non_terminal_cnt)
    else:
        print("ACTOR_SFT_CNT:", len(actor_data))
        terminal_cnt = 0
        non_terminal_cnt = 0
        for d in actor_data:
            if d["chosen_is_terminal"]:
                terminal_cnt += 1
            else:
                non_terminal_cnt += 1
        print("Terminal CNT:", terminal_cnt)
        print("Non-Terminal CNT:", non_terminal_cnt)

    write_json_lines(args.actor_data_path, actor_data)

if __name__ == '__main__':
    main()
