import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


def setup_model_and_tokenizer(config):

    try:
        print(f"Loading model: {config.MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.MODEL_NAME,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=config.DTYPE,
            load_in_4bit=config.LOAD_IN_4BIT,
        )

        # Llama3 needs a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Configuring LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.LORA_R,
            target_modules=config.LORA_TARGET_MODULES,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias=config.LORA_BIAS,
            use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
            random_state=config.SEED,
            max_seq_length=config.MAX_SEQ_LENGTH,
        )

        print("Model and tokenizer loaded and configured.")
        return model, tokenizer

    except Exception as e:
        raise Exception(f"Failed to load setup model and tokenizer: {e}")


def setup_sft_trainer(model, tokenizer, train_dataset, config):
    try:
        print("Setting up SFTTrainer...")
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            fp16=config.FP16,
            bf16=config.BF16,
            logging_steps=config.LOGGING_STEPS,
            optim=config.OPTIM,
            seed=config.SEED,
            max_grad_norm=config.MAX_GRAD_NORM,
            report_to="none",
        )

        print("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=config.MAX_SEQ_LENGTH,
            args=training_args,
            packing=False,
        )

        print("SFTTrainer setup complete.")
        return trainer

    except Exception as e:
        raise Exception(f"Failed to setup SFT trainer: {e}")
