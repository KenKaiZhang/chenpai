import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel


def load_and_format_data(tokenizer, data_file_path):

    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenizer=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    try:
        print(f"Formatting data from {data_file_path}...")
        dataset = Dataset.from_json(data_file_path)
        formatted_dataset = dataset.map(
            format_prompts, batched=True, remove_columns=["messages"]
        )
        print("Successfully formatted data.")
        return formatted_dataset
    except Exception as e:
        raise Exception(f"Failed to format data from {data_file_path}: {e}")


def save_and_merge_model(trainer, model, config):
    print(f"Saving LoRA adapters to {config.OUTPUT_DIR}...")
    trainer.save_model(config.OUTPUT_DIR)
    print("LoRA adapters saved.")

    try:
        print("Merging LoRA adapters and saving model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_full, tokenizer_full = FastLanguageModel.from_pretrained(
            model_name=config.MODEL_NAME,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=config.DTYPE,
            load_in_4bit=False,
        )

        if tokenizer_full.pad_token is None:
            tokenizer_full.pad_token = tokenizer_full.eos_token

        # Get the peft config from the trained model
        peft_config = model.config

        # Load the fine-tuned adapters onto the full model
        model_full = FastLanguageModel.get_peft_model(model_full, peft_config)
        model_full.load_adapter(config.OUTPUT_DIR, adapter_name="default")
        model_full.merge_and_unload()
        print("LoRA adapters merged.")

        merged_model_output_dir = os.path.join(config.OUTPUT_DIR, "merged_model")
        os.makedirs(merged_model_output_dir, exist_ok=True)
        model_full.save_pretrained(merged_model_output_dir)
        tokenizer_full.save_pretrained(merged_model_output_dir)
        print(f"Model saved to {merged_model_output_dir}.")

    except Exception as e:
        raise Exception(f"Failed to merge and save full model: {e}")
