import pandas as pd
from datasets import load_dataset

def get_dataset(data_args):
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs[:100]"
    )
    return dataset



def apply_dpo_template(sample, tokenizer):

    prompt_message = [sample["chosen"][-2]]
    
    sample["chosen_final"] = sample["chosen"][-1]["content"] + "\n"
    sample["rejected_final"] = sample["rejected"][-1]["content"] + "\n"
    sample["prompt_final"] = tokenizer.apply_chat_template(
      prompt_message, tokenize=False, add_generation_prompt=True
    )
    
    return sample