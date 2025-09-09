# DEPRECATED - PLEASE USE THE Activated LoRA (aLoRA) IMPLEMENTATION IN HUGGINGFACE PEFT

PLEASE USE NEWER PEFT IMPLEMENTATION, see https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md#activated-lora-alora

Activated LoRA (aLoRA) is a new low rank adapter architecture that allows for reusing existing base model KV cache for more efficient inference, unlike standard LoRA models. As a result, aLoRA models can be quickly invoked as-needed for specialized tasks during (long) flows where the base model is primarily used, avoiding potentially expensive prefill costs in terms of latency, throughput, and GPU memory. See the whitepaper for a detailed discussion of these advantages and how they scale with context length, number of aLoRAs used, etc. 

This repo contains source code necessary to both train and do inference with aLoRA models.

Whitepaper: [`Activated LoRA: Fine-tuned LLMs for Intrinsics (arXiv)`](https://arxiv.org/abs/2504.12397) for a detailed description of the method and some results.

IBM Research Blogpost: [`Inference Friendly aLoRAs`](https://research.ibm.com/blog/inference-friendly-aloras-lora)


---
## Installation

```bash
pip install alora
```


---

## Source Code
The main implementation can be found in:

**Source directory:** [`alora/`](alora/)

---
## General usage

This repo implements aLoRA using the [Huggingface PEFT library](https://huggingface.co/docs/peft/en/index). 

In so doing, it introduces aLoRA specific classes that subclass relevant PEFT classes, allowing for as much functionality from PEFT to be carried over as possible. Throughout, the goal is to enable seamless integration of these aLoRA classes into preexisting LoRA training pipelines as much as possible (see **Important notes** and **Limitations** below).

**Limitations** The aLoRA architecture---since it seeks to re-use base model cache---only is supported with CausalLM models, and adapters must *only* be applied to the attention modules, i.e. the queries, keys, and values (e.g.[`q_proj`, `k_proj`, `v_proj`]).

**Important notes** While aLoRA uses low-rank adaptation of the weight matrices just like LoRA, since the usage of the weights is different in the architecture, models trained as LoRAs will not work if run as aLoRAs, and vice versa. Similarly, hyperparameter settings will not carry over between aLoRA and LoRA, indeed **aLoRA will typically need higher rank, e.g. `r=32`**.


---
## Getting Started

For inference with existing aLoRA models (e.g. from Huggingface): see **Inference Example** below.

For training models: see **Training Example** below to get started with creating aLoRA models and training them with `SFTTrainer` from Huggingface.

---

## Inference Example

A simple inference script is available for a trained **Uncertainty Quantification aLoRA** [Granite 3.2 8B Instruct - Uncertainty aLoRA](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty), showing how to switch between the base model and the aLoRA while reusing the **base model kV cache** and using **Hugging Face libraries** for generation:

**Inference Example Script location:** [`experiments/inference_example.py`](experiments/inference_example.py)

In this script, a user question "What is IBM Research?" is asked, the base model (Granite 3.2 8b instruct) is invoked to answer the query, and the Uncertainty aLoRA is then invoked to generate a certainty score for the answer of the base model (see [Granite 3.2 8B Instruct - Uncertainty aLoRA](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty) for an explanation of how this certainty score is defined).

The example in the script with KV cache reuse can be visualized as follows. 1) The base model prefills the question and any supporting documents (e.g. if in a RAG system) and generates an answer based off of that KV cache plus any previously existing KV cache for the context. 2) The Uncertainty Quantification aLoRA is invoked, the invocation string (instruction) is prefilled, and the aLoRA model can generate a response using all available KV cache (unlike LoRA, no need to redo prefill of the vast majority of the context!). 

![image](https://github.ibm.com/Kristjan-H-Greenewald/activated-lora/assets/142635/522074f6-9771-484d-a69f-cd39ef391c2d)

### Bare minimum inference commands (w/o cache reuse)
For simplicity and to make the commands clear, we also show the simplest possible inference. In its most basic form, inference with an aLoRA model can be done as follows. Note that the aLoRA model classes are used explicitly, and the invocation sequence here gets the one the model was trained with (saved in the aLoRA config). The `INVOCATION_SEQUENCE` is appended to the input, tokenized, and its token length computed by `tokenize_alora`. `alora_offsets` passes this (length-1) to the aLoRA model, giving it the necessary location to turn on the adapter weights in the token sequence.
```python
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
from alora.tokenize_alora import tokenize_alora

BASE_MODEL="BASE_MODEL_LOCATION"
ALORA_NAME="ALORA_ADAPTER_LOCATION"


model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map = 'auto')
model_alora = aLoRAPeftModelForCausalLM.from_pretrained(model_base,ALORA_NAME)
INVOCATION_SEQUENCE = model_alora.peft_config.invocation_string

inputs, alora_offsets = tokenize_alora(tokenizer,input_string + "\n", INVOCATION_SEQUENCE)
out_gen = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=200, alora_offsets=alora_offsets)
```
The above simple example does not reuse the KV cache from the base model, see the inference script for details on how to do that.

---

## Training Example
To train an **Activated LoRA (aLoRA)**, use the following script as a guide:

```bash
python train_scripts/basic_finetune_example.py --adapter aLoRA
```

**Script location:** [`train_scripts/finetune_alora_example.py`](train_scripts/basic_finetune_example.py)

The key part of the code (for the aLoRA architecture) is here:
```python
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig

INVOCATION_PROMPT = "<|start_of_role|>certainty<|end_of_role|>" # feel free to use any desired string or prompt so long as you are consistent
...
peft_config = aLoraConfig(
  r=32,
  lora_alpha=32,
  lora_dropout=0.05,
  bias="none",
  task_type="CAUSAL_LM",
  invocation_string=INVOCATION_PROMPT,
  target_modules=["q_proj","k_proj", "v_proj"],
)
response_tokens = tokenizer(INVOCATION_PROMPT, return_tensors="pt", add_special_tokens=False)
response_token_ids = response_tokens['input_ids']
# Create the aLoRA model, including tokenized invocation string to ensure weights are activated when needed.
peft_model = aLoRAPeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
...
# continue to train with SFTTrainer...
```

This script runs on a very small example JSONL data file [`train_scripts/example_data.jsonl`](train_scripts/example_data.jsonl)

Note that this code includes standard LoRA training for comparison, it can be called with 
```bash
python train_scripts/basic_finetune_example.py --adapter LoRA
```



**Key points**

* aLoRA will need larger rank than a corresponding LoRA model, often rank 32 works well.
* An invocation string must be defined, the adapted weights are activated *one token after this sequence begins*. As such, it must be included in the input strings in your training data, or added in during the training script.
* The invocation string can simply be the standard generation prompt, when searching the string, the code looks for the last use of the invocation sequence in the string.
* It is a good idea to have the invocation string begin and end with special tokens, to ensure that it is always tokenized as the same token sequence when encountered.
* The invocation string is saved in the aLoraConfig, which is included as a file in the model save directory when saving. This allows for recovery when loading the model later.
* For now, the invocation sequence must be tokenized and passed to the aLoRA model prior to training. This is not necessary at inference time.
  


---
## Training with Saving Callback

An expanded training script with a save model callback is at [`train_scripts/finetune_example_callback.py`](train_scripts/finetune_example_callback.py)

**Behavior** This callback saves the model whenever the loss on the provided validation data is best so far. This can be used to revert to back to the model with the best validation loss. The frequency of checking the validation loss can be set by adjusting the standard arguments to SFTTrainer.




## vLLM

vLLM support coming soon.







