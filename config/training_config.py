import torch

class TrainingConfig:
    # Model parameters
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    MAX_SEQ_LENGTH = 2048 # Adjust based on GPU memory and desired context length
    DTYPE = None # None for auto detection. Use torch.float16 for T4/V100, BF16 for A100.
    LOAD_IN_4BIT = True # Use 4bit quantization (QLoRA)

    # LoRA parameters
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_BIAS = "none"
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    USE_GRADIENT_CHECKPOINTING = "unsloth"

    # Training parameters
    OUTPUT_DIR = "/app/output/llama-3-8b-personal-bot" # Inside container
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 10
    NUM_TRAIN_EPOCHS = 3
    LEARNING_RATE = 2e-4
    LOGGING_STEPS = 1
    OPTIM = "adamw_8bit"
    SEED = 42
    MAX_GRAD_NORM = 0.3
    # Use FP16 if BF16 is not supported by your GPU (e.g., T4)
    FP16 = not torch.cuda.is_bf16_supported()
    # Use BF16 if your GPU supports it (e.g., A100)
    BF16 = torch.cuda.is_bf16_supported()

    # Data paths (inside container)
    DATA_FILE_PATH = "/app/data/my_personal_data.jsonl"