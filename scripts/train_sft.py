import os
from huggingface_hub import login

from config.training_config import TrainingConfig
from src.model import load_fine_tuned_model, generate_response
from src.setup import setup_model_and_tokenizer, setup_sft_trainer
from src.utils import load_and_format_data, save_and_merge_model


def main():
    
    try:
        # 1. Login to Hugging Face
        if "HF_TOKEN" not in os.environ:
            print("Hugging Face token not found in environment variables. Please log in.")
            login()
        else:
            print("Hugging Face token found in environment variables. Proceeding...")

        # 2. Load Configuration
        config = TrainingConfig()

        # Create output directory if it doesn't exist
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # 3. Setup Model and Tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)

        # 4. Prepare Dataset
        train_dataset = load_and_format_data(tokenizer, config.DATA_FILE_PATH)

        # 5. Setup Trainer
        trainer = setup_sft_trainer(model, tokenizer, train_dataset, config)

        # 6. Train the model
        print("\n--- Starting Model Training ---")
        trainer.train()
        print("--- Model Training Complete ---")

        # 7. Save the Fine-tuned Model (LoRA adapters and optionally merged full model)
        save_and_merge_model(trainer, model, tokenizer, config)
    
    except Exception as e:
        print(f"\n--- Error during setup ---")
        print(e)
        return 
    
    # 8. Test Inference (Optional, after training)
    print("\n--- Running a quick inference test ---")
    try:
        test_model, test_tokenizer = load_fine_tuned_model(
            config.MODEL_NAME,
            config.OUTPUT_DIR,
            config.MAX_SEQ_LENGTH,
            config.DTYPE,
            config.LOAD_IN_4BIT # Load in 4-bit for inference
        )
        test_question = "What is your primary focus when approaching a new project?"
        response = generate_response(test_model, test_tokenizer, test_question)
        print(f"\nUser Question: {test_question}")
        print(f"Bot Response: {response}")

        test_question_2 = "Tell me about a time you faced a significant challenge and how you overcame it."
        response_2 = generate_response(test_model, test_tokenizer, test_question_2)
        print(f"\nUser Question: {test_question_2}")
        print(f"Bot Response: {response_2}")

    except Exception as e:
        print(f"--- Error during inference test ---")
        print(e)
        return


if __name__ == "__main__":
    main()