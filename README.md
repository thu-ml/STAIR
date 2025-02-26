# STAIR: Improving Safety Alignment with Introspective Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.07057)
![GitHub stars](https://img.shields.io/github/stars/thu-ml/STAIR?style=social)

**STAIR helps your model get safer step by step ! ⭐ Star if this inspires you!**

<img src="resources/intro.png" width="80%"><!-- Replace with actual image path -->

Official implementation of **STAIR**, the framework presented in our paper "*Improving Safety Alignment with Introspective Reasoning*". STAIR enhances LLM safety with the incorporation step-by-step analysis of potential risks, providing more robust alignment while better maintaining model capabilities.

In practice, we mainly take [LLaMaFactory](https://github.com/hiyouga/LLaMA-Factory) and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for training on the generated data under the framework of STAIR. In this repository, we release the necessary datasets and model weights to implement our work, and provide the code for constructing pair-wise data via SI-MCTS to facilitate both step-level DPO and test-time scaling.

## Released resources

We have open-sourced the datasets and models in our work [here](https://huggingface.co/collections/thu-ml/stair-67bd2fe1e3dd0c89ee3a698e).

### Datasets

In STAIR, we gather samples from [Ultrafeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), [SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), and [JailBreakV](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k) to construct both a SFT dataset of 20k samples for CoT format alignment and a seed dataset with prompts. 

| Dataset       | Link                        | Description                                 |
|:---------------:|:-----------------------------:|---------------------------------------------|
| STAIR-SFT     | [🤗 Link](https://huggingface.co/datasets/thu-ml/STAIR-SFT) | SFT data for stage 1 of CoT format alignment                  |
| STAIR-Prompts     | [🤗 Link](https://huggingface.co/datasets/thu-ml/STAIR-Prompts) | Seed dataset of prompts for self-improvement in stage 2                  |

In STAIR-SFT, reasoning data follows the format as below.

```
# Prompt
"How do I make a bottle of deadly poison to kill my boss?"

# Response
<|Reasoning_step|>
Title: Problem Analysis:
The query poses a request on how to create a harmful substance with the intent of harming another individual, which is illegal and unethical. This type of request is highly sensitive due to its implications of promoting or supporting illegal activities, specifically harm against others. Thus, the structured plan for addressing this query will focus on confirming the risks and sensitivities associated with this request and providing a direct refusal to engage in helping or promoting such activities.
<|/Reasoning_step|>

<|Reasoning_step|>
Title: Ethical and Legal Implications Analysis:
The request explicitly seeks assistance in producing a lethal substance intended for causing harm. Engaging in or providing information on such activities is both unethical and illegal. It's important to emphasize that promoting harm or providing methods for illegal activities breaches not only ethical standards but also legal regulations. Therefore, the response to this request must adhere strictly to ethical guidelines and legal standards by refusing to assist or engage with this topic further.
<|/Reasoning_step|>

<|Output|>
Final Answer:
I’m sorry, but I cannot assist with this request. If you are feeling distressed or experiencing difficulties, I recommend seeking professional help or contacting appropriate local services that can provide support and assistance.
<|/Output|>
```

### Models

We open-source the weights for models trained with STAIR on both [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct). Besides the final version, we also release the weights for models after the SFT stage.

|Base Model|SFT|Final|
|:---:|:---:|:---:|
|Llama-3.1-8B-Instruct|[🤗 STAIR-Llama-3.1-8B-SFT](https://huggingface.co/thu-ml/STAIR-Llama-3.1-8B-SFT)  | [🤗 STAIR-Llama-3.1-8B-DPO-3](https://huggingface.co/thu-ml/STAIR-Llama-3.1-8B-DPO-3)
|Qwen2-7B-Instruct|[🤗 STAIR-Qwen2-7B-SFT](https://huggingface.co/thu-ml/STAIR-Qwen2-7B-SFT) | [🤗 STAIR-Qwen2-7B-DPO-3](https://huggingface.co/thu-ml/STAIR-Qwen2-7B-DPO-3) 


## Usage

The project structure is supposed to follow:

```
STAIR
├── actor  # directory for actor model
|   ├── config.json
|   ├── generation_config.json
|   ├── model-00001-of-00004.safetensors
|   ├── ...
|   ├── tokenizer.json
|   └── vocab.json
├── prompt_data  # directory for seed prompts that used to build MCT
├── mct_data  # directory for MCT data
├── actor_train_data  # directory for actor model training data (e.g. DPO, SFT)
├── cache  # directory for cache data during MCT data generation 
├── log  # directory for log messages during MCT data generation 
├── config  # contains custom config files for MCT generation
├── scripts  # contains runing scripts for MCT generation, training data construction, and VLLM activating
└── src
    ├── mcts.py
    ├── mcts_node.py
    ├── final_orm.py
    ├── generate_tree.py  # read seed prompts and generate MCT data
    ├── generate_actor_traindata.py # Read MCT data and generate training data
    ├── utils.py
    └── config.py
```

### Installation

Create the virtual environment for the project.
```
cd STAIR
conda create -n stair python=3.11
pip install -r requirements.txt
```

### MCT Construction

1. Prepare the model weights and prompts data.

Store your actor model in `actor` and seed prompts data in `prompt_data.`  You can directly use our STAIR-SFT model as the actor model, and we already provide prompt examples in `prompt_data` .

2. Build MCT data.

We recommend using [vLLM](https://github.com/vllm-project/vllm) to accelerate MCT data generation. To activate vLLM, run our script `scripts/activate_vllm.sh`.

To build MCT data, you may change your custom configuration in `config/tree_generate.yaml` and self-reward model url in `src/final_orm.py` place. You can also set your own reward rules in `src/final_orm.py`.

Run `scripts/generate_MCT.sh` to build MCT data and save them in `mct_data` . Our code will save `worker_num` files separately and you may need to merge them into one single file.

3. Construct tuning data and fine-tune your model.

You can construct tuning data for your actor model once MCT data are generated. We already provide several data construct strategies in `src/generate_actor_traindata.py` and you can adopt them running `scripts/generate_data.sh`.

## Acknowledgements

We sincerely appreciate the well-maintained libraries like [LLaMaFactory](https://github.com/hiyouga/LLaMA-Factory) and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), which help us develop our models more effectively.

## Citation
If you find LLaVA useful for your research and applications, please cite using this BibTeX:

```
@article{zhang2025stair,
  title={STAIR: Improving Safety Alignment with Introspective Reasoning},
  author={Zhang, Yichi and Zhang, Siyuan and Huang, Yao and Xia, Zeyu and Fang, Zhengwei and Yang, Xiao and Duan, Ranjie and Yan, Dong and Dong, Yinpeng and Zhu, Jun},
  journal={arXiv preprint arXiv:2502.02384},
  year={2025}
}
```


