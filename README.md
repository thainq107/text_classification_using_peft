# Text Classification using Parameter-Efficient Fine-Tuning Methods (PEFT-Huggingface)

## Dependencies
- Python 3.10
- [PyTorch](https://github.com/pytorch/pytorch) 2.0 +
  ```
  pip install -r requirements.txt
  ```
## Dataset
  [carblacac/twitter-sentiment-analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)

## LoRA (Low-Rank Adaptation)
### Training
  ```
    python run_fine_tuning_peft.py \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --model_name_or_path google/flan-t5-xl \
        --use_lora True \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```

### Predict
  Load model from huggingface repository
  ```
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "thainq107/flan-t5-xl-twitter-sentiment-analysis-lora"

    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    inputs = tokenizer("I hate you:", return_tensors="pt")
    outputs = model.generate(**inputs)
    tokenizer.batch_decode(outputs, skip_special_tokens=True)
  ```

## Prefix-Tuning
### Training
  ```
    python run_fine_tuning_peft.py \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --model_name_or_path google/flan-t5-xl \
        --use_prefix True \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```

## Prompt-Tuning
### Training
  ```
    python run_fine_tuning_peft.py \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --model_name_or_path google/flan-t5-xl \
        --use_prompt True \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```
