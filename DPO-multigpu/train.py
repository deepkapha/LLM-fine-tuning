from config import model_args, data_args, train_args
from transformers import set_seed
from model import create_and_prepare_model
from data import get_dataset, apply_dpo_template
from trainer import get_trainer

def main(model_args, data_args, train_args):
    set_seed(42)
    tokenizer, model, ref_model = create_and_prepare_model(model_args)
    dataset = get_dataset(data_args)
    dataset = dataset.map(apply_dpo_template, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['prompt', 'prompt_id', 'chosen', 'rejected', 'messages', 'score_chosen', 'score_rejected'])

    dataset = dataset.rename_column("chosen_final", "chosen")
    dataset = dataset.rename_column("rejected_final", "rejected")
    dataset = dataset.rename_column("prompt_final", "prompt")

    trainer = get_trainer(train_args, model, ref_model, dataset, tokenizer, None)

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
    print("Saving Model")
    trainer.save_model(train_args["output_dir"])
    print("Saved")
    print("pushing model to hub")
    trainer.push_to_hub(train_args["hub_dir"])
    print("Pushed")

if __name__ == "__main__":
    main(model_args, data_args, train_args)

    