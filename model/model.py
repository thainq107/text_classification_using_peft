from transformers import AutoConfig, AutoModelForSeq2SeqLM
from peft import (
    get_peft_model,
    TaskType, 
    LoraConfig, 
    PrefixTuningConfig,
    PromptTuningConfig,
)

def load_model(model_args):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task="text-generation"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    return model

def prepare_lora_model(lora_args, model):
    # creating model
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha, 
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias=lora_args.lora_bias
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model

def prepare_prefix_model(prefix_args, model):
    # creating model
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=prefix_args.prefix_num_virtual_tokens
    )
    prefix_model = get_peft_model(model, peft_config)
    return prefix_model

def prepare_prompt_model(prefix_args, model):
    # creating model
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=prefix_args.prompt_num_virtual_tokens
    )
    prefix_model = get_peft_model(model, peft_config)
    return prefix_model