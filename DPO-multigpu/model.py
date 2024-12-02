from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import torch, os

def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
def create_and_prepare_model(model_args):
    bnb_config = None
    
    local_rank = os.getenv("LOCAL_RANK")
    device_string = "cuda:" + str(local_rank)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_args["MODEL_NAME"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={'':device_string},
        quantization_config=bnb_config,
        is_trainable=True,
    )
    
    model.config.use_cache = False
    
    ref_model = AutoPeftModelForCausalLM.from_pretrained(
        model_args["MODEL_NAME"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={'':device_string},
        quantization_config=bnb_config,
        is_trainable=False,
    )
    
    ref_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args["MODEL_NAME"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # peft_config = LoraConfig(
    #         lora_alpha=16,
    #         lora_dropout=0.1,
    #         r=64,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #         target_modules=args.lora_target_modules.split(",")
    #         if args.lora_target_modules != "all-linear"
    #         else args.lora_target_modules,
    #     )
    return tokenizer, model, ref_model