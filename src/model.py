# src/inference.py

import torch
import os
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


def load_fine_tuned_model(model_name, output_dir, max_seq_length, dtype, load_in_4bit):
    try:
        merged_model_path = os.path.join(output_dir, "merged_model")
        print(f"Loading model from {merged_model_path}...")

        if os.path.exists(merged_model_path) and os.path.isdir(merged_model_path):
            model = FastLanguageModel.from_pretrained(
                merged_model_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        else:
            print(f"Merged model not found at {merged_model_path}.")
            print(f"Loading base model and adapters from {output_dir}.")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            model.load_adapter(output_dir, adapter_name="default")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        print(f"Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        raise Exception(f"Failed to load fine tuned model: {e}")


def generate_response(
    model, tokenizer, user_question, max_new_tokens=512, temperature=0.7, top_p=0.9
):
    try:
        messages = [{"role": "user", "content": user_question}]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        return generated_text
    except Exception as e:
        raise Exception(f"Model failed to generate response: {e}")
