from trl import DPOConfig, DPOTrainer

def get_trainer(training_args, model, ref_model, dpo_dataset, tokenizer, peft_config):
    training_arguments = DPOConfig(
        output_dir=training_args["output_dir"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
        optim=training_args["optim"],
        logging_steps=training_args["logging_steps"],
        learning_rate=training_args["learning_rate"],
        fp16=True,
        warmup_ratio=training_args["warmup_ratio"],
        group_by_length=False,
        lr_scheduler_type=training_args["lr_scheduler_type"],
        gradient_checkpointing=True,
        report_to="none",
        num_train_epochs=training_args["epochs"],
        evaluation_strategy = training_args["evaluation_strategy"],
        save_strategy= training_args["save_strategy"]
    )

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_arguments,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        beta=0.1, 
        max_prompt_length=512,
        max_length=1024,
        peft_config = peft_config
    )

    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    return trainer