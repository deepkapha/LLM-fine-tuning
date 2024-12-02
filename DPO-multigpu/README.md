Prerequiste:
- Finetune SFT model and push to hub
    - You can simply run the ipynb file i.e. sft/SFT.ipynb
    - Make sure we upload the model to hub. And pass the hub_name into DPO's config.

Steps:

- Install requirements:
    pip install -r requirements.txt

- Login to HuggingFace:
    On Terminal
    - huggingface-cli login

    Now pass your huggingface token

- Change the required params from config.py

- Finally run the main script:
    accelerate launch --config_file deepspeed.yaml train.py