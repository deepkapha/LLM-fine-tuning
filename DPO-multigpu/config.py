model_args = {
    "MODEL_NAME":"hiiamsid/sft_model"
}

data_args = {
    "path":"/workspace/split_0.jsonl"
}

train_args = {
    "output_dir":"/content/drive/MyDrive/deepkapha/dpo_model",
    "per_device_train_batch_size":2,
    "gradient_accumulation_steps":4,
    "optim":"rmsprop",
    "evaluation_strategy":"no",
    "save_strategy":"no",
    "logging_steps":10,
    "learning_rate":2e-4,
    "warmup_ratio":0.03,
    "lr_scheduler_type":"constant",
    "epochs":1,
    "output_dir": "models",
    "hub_dir": "hiiamsid/dpo_new"
}