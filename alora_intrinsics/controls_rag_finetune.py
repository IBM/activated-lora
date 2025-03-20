import click, os, torch
import numpy as np
#from lakehouse.assets.config import DMF_MODEL_CACHE
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
#alora
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM as aLoRAPeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
# standard lora
from peft import PeftModelForCausalLM, LoraConfig
import json
from alora_intrinsics.alora.multi_collator import DataCollatorForCompletionOnlyLM_Multi
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainerCallback


DATA_PATH = os.getenv("HF_DATASETS_CACHE")
MODEL_NAME = "ibm-granite/granite-3.2-8b-instruct"

#Universal start sequence (to turn on aLoRA)
INVOCATION_PROMPT = "<|start_of_role|>"
#Complete set of possible start sequences (for completion-only collator)
INVOCATION_PROMPT_SET = ['<|start_of_role|>assistant {"length": "'+ lngth + '"}<|end_of_role|>' for lngth in ["long","medium","short"]]
# SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
# HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/chat_template_dump_with_controls_0.4"
DATASET_FILES = ["train_raft.jsonl", "val_raft.jsonl"]
SAVE_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora"


def get_datasets():
    datasets = []
    for ds in DATASET_FILES:
        if ds[-1] == "n": #json

            file = open(DATASET_PATH + '/' + ds)
            data = json.load(file)


        else: #jsonl
            file = open(DATASET_PATH +'/' +  ds)
            data = {"conversations":[(json.loads(line)) for line in file]}#,"documents":[(json.loads(line))["documents"] for line in file]}
        datasets.append(data)
    return datasets
def process_datasets(datasets,tokenizer,max_rows):
    proc_datasets = []

    for ds in datasets:
        inputs = []
        targets = []
        add = ""



        max_rs = max_rows
    
        
        for i in range(0,min(len(ds["conversations"]),max_rs)):
            convo = ds["conversations"][i]["messages"]
            docs = ds["conversations"][i]["documents"]
            lngth = ds["conversations"][i]["controls"]["length"]
            if convo[0]["role"] != "system": #Optionally replace default system prompt. The Granite 3.1 chat template inserts a system prompt with today's date by default. 
                # If a system prompt is not needed, it will need to be manually removed from the `string' below.
                convo = [{"role":"system", "content": ""}] +convo#"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
                string = tokenizer.apply_chat_template(conversation =convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
                string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
                string = string[len(string_to_remove):]
            else:
                string = tokenizer.apply_chat_template(conversation=convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
                part1rest = string.split('<|start_of_role|>documents<|end_of_role|>')
                part23 = part1rest[1].split('<|end_of_text|>')
                string = part1rest[0] + part1rest[1][len(part23[0])+1:] + '<|start_of_role|>documents<|end_of_role|>' + part23[0] + '<|end_of_text|>'
                #print(string)
                #print(docstr)
            # Append invocation sequence here.  Doing manually to ensure consistency with data collator
            if lngth == "long":
                ix = 0
            elif lngth == "medium":
                ix = 1
            else: #short
                ix = 2
            inputs.append(string + INVOCATION_PROMPT_SET[ix]) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )

            # Targets (that aLoRA is meant to learn to generate)
            targets.append(convo[-1]["content"].split('Reference(s)')[0] + '<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>')
        proc_dict = dict()
        proc_dict['input'] = inputs
        proc_dict['target'] = targets


        print(ds["conversations"][0])
        print(inputs[0])
        print(targets[0])

        proc_datasets.append(Dataset.from_dict(proc_dict))
    return proc_datasets

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"{example['input'][i]}{example['target'][i]}"

        output_texts.append(text)
    return output_texts
from transformers import TrainerCallback

#class SaveModelCallback(TrainerCallback):
#    def on_save(self, args, state, control, **kwargs):
#        """Ensure tied weights are assigned before saving."""
#        trainer = kwargs["trainer"]
#        if getattr(trainer.model, "tie_weights", None):
#            trainer.model.tie_weights()  # Fix weight sharing before savingi
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float("inf")  # Track best loss

    def on_evaluate(self, args, state, control, **kwargs):
        """Save the best model manually during evaluation."""

        model = kwargs["model"]
        metrics = kwargs["metrics"]
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss  # Update best loss
          

            # Ensure tied weights are applied before saving
            #if getattr(trainer.model, "tie_weights", None):
            #    trainer.model.tie_weights()

            # Manually save best model
            model.save_pretrained(args.output_dir)


@click.command()
@click.option('--adapter', type=click.STRING, help='adapter, LoRA or aLoRA')
@click.option('--int_name', type=click.STRING, help='dataset')
def SFT_data(int_name,adapter):

    data = get_datasets()






       
   
    model_name = MODEL_NAME

    token = os.getenv("HF_MISTRAL_TOKEN")
    model_dir = model_name #os.path.join(DMF_MODEL_CACHE, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side='right',trust_remote_code=True,token=token)
        
    model_base = AutoModelForCausalLM.from_pretrained(model_dir,device_map = 'auto', use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
#    model_base.config.pad_token_id = model_base.config.eos_token_id
    tokenizer.add_special_tokens = False
    datasets = process_datasets(data,tokenizer,max_rows = 400000)

    train_dataset = concatenate_datasets(datasets[0:1])#train_dataset
    subsample_size = 40000
    train_dataset = train_dataset.shuffle(seed=42).select(range(min(len(train_dataset),subsample_size)))
    val_dataset = datasets[1]
    print(model_base)
    
    collator = DataCollatorForCompletionOnlyLM_Multi(INVOCATION_PROMPT_SET, tokenizer=tokenizer)
    
    prefix = "mar19_1"
    if adapter != 'LoRA': # aLoRA model
        peft_config = aLoraConfig(
            r=128,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj", "v_proj"],#Can only do q, k, v layers (for now).
            #layers_to_transform=[38,39]
        )
        response_tokens = tokenizer(INVOCATION_PROMPT, return_tensors="pt", add_special_tokens=False)
        response_token_ids = response_tokens['input_ids']
        peft_model = aLoRAPeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
        #tmp_dir = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp"
        sft_args = SFTConfig(output_dir=SAVE_PATH + f"/{prefix}_8bsft_Control_alora_sz32"+ int_name,
                evaluation_strategy = "steps",
                eval_steps=300,
                dataset_kwargs={"add_special_tokens":False},num_train_epochs=3,learning_rate=6e-7*5*10/5,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_args,
            formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=[SaveBestModelCallback()]
        #,
        )
        trainer.train()
        #load from best
        #peft_best = aLoRAPeftModelForCausalLM.from_pretrained(model_base,tmp_dir + '/adapter')
        peft_model.save_pretrained(SAVE_PATH + f"/{prefix}_8bsft_Control_alora_sz32_last"+ int_name)



        #####################################################################
        #####################################################################
    else: #standard LoRA. THESE HYPERPARAMETERS ARE NOT TUNED
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj",  "v_proj"],
            #layers_to_transform=[38,39]
        )
        peft_model = PeftModelForCausalLM(model_base, peft_config)
        
        #tmp_dir = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp"
        sft_args = SFTConfig(output_dir=SAVE_PATH + f"/{prefix}_8bsft_Control_standard_lora_sz6"+ int_name,
           #     evaluation_strategy = "steps",
          #      eval_steps=300,
                dataset_kwargs={"add_special_tokens":False},num_train_epochs=3,learning_rate=6e-7*5*10/5*2*2*5/32,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=train_dataset,
         #   eval_dataset=val_dataset,
            args=sft_args,
            formatting_func=formatting_prompts_func,
        data_collator=collator,
        #callbacks=[SaveBestModelCallback()]
        #,
        )
        trainer.train()
        #load from best
        #peft_best = PeftModelForCausalLM.from_pretrained(model_base,tmp_dir + '/adapter')







       # trainer = SFTTrainer(
           # peft_model,
          #  train_dataset=merged_dataset,
         #   args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=1,learning_rate=6e-7,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
        #    formatting_func=formatting_prompts_func,
        #data_collator=collator
        #,
        #)
        #trainer.train()
    
        peft_model.save_pretrained(SAVE_PATH + f"/{prefix}_8bsft_Control_standard_lora_sz6_last"+ int_name)
        

 
    
    


if __name__ == "__main__":
   
    SFT_data()
