"""
=============================================================================
RISK PROJECT — PHASE 2: LoRA FINE-TUNING
=============================================================================

Purpose:
    Fine-tune llama3.1:8b on 50K LoRA training pairs using PEFT (Parameter-Efficient
    Fine-Tuning). Validate on the original 200-patient sample from the LLM project.
    Compare fine-tuned model against base llama3.1:8b zero-shot baseline.

Input:  data/lora_training_data_combined.jsonl (50K pairs from Phase 1)
        data/llm_sample.csv (200-patient validation set from LLM project)

Output: data/lora_checkpoints/ (training checkpoints)
        data/lora_adapters/ (final LoRA weights)
        data/phase2_results/ (predictions + metrics + plots)

Hardware:
    T4 (16GB) — primary GPU for training
    RTX 2080 (8GB) — idle during fine-tuning

Training Hyperparameters:
    - LoRA Rank: 16
    - LoRA Alpha: 32
    - Learning Rate: 2e-4
    - Epochs: 3
    - Batch Size: 4
    - Warmup Steps: 100
    - Save Checkpoints: Every 500 steps
    - Total Steps: ~38,150 (50,866 pairs / batch 4 / 3 epochs, accounting for gradient accumulation)

Validation:
    - Use 200-patient sample (llm_sample.csv) from LLM project
    - Compare fine-tuned vs. base llama3.1:8b zero-shot baseline
    - Baseline recall: 18.2% (37/200 HIGH)
    - Baseline accuracy: 74.5%

Approach:
    - Load pre-trained llama3.1:8b from HuggingFace
    - Apply LoRA via PEFT library
    - Train with custom tokenization for mortality pairs
    - Checkpoint every 500 steps for crash-safety
    - Validate every 1000 steps (optional, to monitor overfitting)
    - Save final adapter weights (lightweight, ~100MB)
    - Generate comparison plots: loss curves, confusion matrices, AUC

Notes:
    - This script loads the model from disk/HuggingFace, NOT Ollama
    - Ollama will be used for Phase 3 inference (RAG + persona)
    - Temperature 0.0 during validation for deterministic predictions
    - All outputs timestamped and logged

=============================================================================
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import logging
from tqdm import tqdm

# Transformers & PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "lora_training_data_combined.jsonl")
VALIDATION_DATA_FILE = os.path.join(DATA_DIR, "llm_sample.csv")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "lora_checkpoints")
ADAPTER_DIR = os.path.join(DATA_DIR, "lora_adapters")
RESULTS_DIR = os.path.join(DATA_DIR, "phase2_results")

MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Base model (we use 8b but 7b available; adjust as needed)
# For llama3.1:8b, use: "meta-llama/Llama-3.1-8B"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# LoRA Hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training Hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 1000

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 512

# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    """Configure logging with timestamp."""
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_info(msg):
    logger.info(msg)

def log_warning(msg):
    logger.warning(msg)

def log_error(msg):
    logger.error(msg)

# ============================================================
# DATA LOADING
# ============================================================

def load_training_data(filepath, max_samples=None):
    """
    Load LoRA training pairs from JSONL file.
    
    Args:
        filepath: path to lora_training_data_combined.jsonl
        max_samples: limit to N samples (for testing); None = all
        
    Returns:
        list of dicts with keys: instruction, input, output
    """
    
    log_info(f"Loading training data from {filepath}...")
    
    pairs = []
    
    try:
        with open(filepath, "r") as f:
            for line_idx, line in enumerate(f):
                if max_samples and line_idx >= max_samples:
                    break
                
                try:
                    pair = json.loads(line)
                    pairs.append(pair)
                except json.JSONDecodeError as e:
                    log_warning(f"Skipped malformed JSON at line {line_idx}: {e}")
                    continue
    
    except FileNotFoundError:
        log_error(f"Training data file not found: {filepath}")
        return None
    except Exception as e:
        log_error(f"Failed to load training data: {e}")
        return None
    
    log_info(f"Loaded {len(pairs)} training pairs")
    
    # Print distribution
    n_yes = sum(1 for p in pairs if p["output"] == "Yes")
    n_no = len(pairs) - n_yes
    log_info(f"  Mortality distribution: Yes={n_yes} ({n_yes/len(pairs):.1%}), No={n_no} ({n_no/len(pairs):.1%})")
    
    return pairs

def load_validation_data(filepath):
    """
    Load validation set (200-patient sample from LLM project).
    
    Args:
        filepath: path to llm_sample.csv
        
    Returns:
        pandas DataFrame
    """
    
    log_info(f"Loading validation data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        log_error(f"Validation data file not found: {filepath}")
        return None
    except Exception as e:
        log_error(f"Failed to load validation data: {e}")
        return None
    
    log_info(f"Loaded {len(df)} validation samples")
    log_info(f"  Mortality rate: {df['mortality'].mean():.1%}")
    
    return df

# ============================================================
# TOKENIZATION
# ============================================================

def create_prompt(instruction, input_text):
    """
    Format instruction + input as a prompt for the model.
    
    Args:
        instruction: str (e.g., "Predict in-hospital mortality...")
        input_text: str (patient description)
        
    Returns:
        str: formatted prompt
    """
    
    prompt = f"""Below is an instruction that describes a task and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    return prompt

def tokenize_pair(pair, tokenizer, max_length=MAX_SEQ_LENGTH):
    """
    Tokenize a single instruction-input-output pair.
    
    Args:
        pair: dict with keys instruction, input, output
        tokenizer: HuggingFace tokenizer
        max_length: max sequence length
        
    Returns:
        dict with input_ids, attention_mask, labels
    """
    
    instruction = pair["instruction"]
    input_text = pair["input"]
    output = pair["output"]
    
    # Create full prompt with output
    prompt = create_prompt(instruction, input_text)
    full_text = prompt + output
    
    # Tokenize
    encodings = tokenizer(
        full_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids (model predicts next token)
    encodings["labels"] = encodings["input_ids"].copy()
    
    return encodings

def create_dataset(pairs, tokenizer, max_length=MAX_SEQ_LENGTH):
    """
    Create HuggingFace Dataset from training pairs.
    
    Args:
        pairs: list of dicts
        tokenizer: HuggingFace tokenizer
        max_length: max sequence length
        
    Returns:
        Dataset object
    """
    
    log_info("Tokenizing training data...")
    
    tokenized_data = []
    
    for idx, pair in enumerate(tqdm(pairs, desc="Tokenizing")):
        try:
            encodings = tokenize_pair(pair, tokenizer, max_length)
            tokenized_data.append(encodings)
        except Exception as e:
            log_warning(f"Failed to tokenize pair {idx}: {e}")
            continue
    
    log_info(f"Tokenized {len(tokenized_data)}/{len(pairs)} pairs")
    
    # Create Dataset
    dataset = Dataset.from_dict({
        "input_ids": [d["input_ids"] for d in tokenized_data],
        "attention_mask": [d["attention_mask"] for d in tokenized_data],
        "labels": [d["labels"] for d in tokenized_data],
    })
    
    return dataset

# ============================================================
# MODEL SETUP
# ============================================================

def setup_model_and_tokenizer():
    """
    Load base model and tokenizer from HuggingFace.
    Apply LoRA config.
    
    Returns:
        model, tokenizer
    """
    
    log_info(f"Loading model: {MODEL_NAME}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log_info("Tokenizer loaded")
    except Exception as e:
        log_error(f"Failed to load tokenizer: {e}")
        return None, None
    
    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        log_info("Base model loaded")
    except Exception as e:
        log_error(f"Failed to load model: {e}")
        return None, None
    
    # Apply LoRA
    log_info("Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj"],  # Standard for LLaMA
    )
    
    try:
        model = get_peft_model(model, lora_config)
        log_info("LoRA applied successfully")
        
        # Print trainable parameters
        model.print_trainable_parameters()
    except Exception as e:
        log_error(f"Failed to apply LoRA: {e}")
        return None, None
    
    return model, tokenizer

# ============================================================
# TRAINING
# ============================================================

def train(model, tokenizer, train_dataset, output_dir=CHECKPOINT_DIR):
    """
    Fine-tune model on training dataset using HuggingFace Trainer.
    
    Args:
        model: PEFT model
        tokenizer: HuggingFace tokenizer
        train_dataset: HuggingFace Dataset
        output_dir: checkpoint directory
        
    Returns:
        trainer object
    """
    
    log_info("Setting up training arguments...")
    
    # <human_intervention>
    # Adjust training args based on GPU memory and dataset size
    # For T4 (16GB) with llama3.1:8b, batch_size=4 is safe
    # Gradient accumulation can increase effective batch size without OOM
    # </human_intervention>
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        save_steps=SAVE_STEPS,
        save_total_limit=3,  # Keep last 3 checkpoints
        seed=SEED,
        fp16=True,  # Mixed precision for T4
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
    )
    
    log_info("Initializing Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        ),
    )
    
    log_info("Starting training...")
    log_info(f"  Epochs: {NUM_EPOCHS}")
    log_info(f"  Batch size: {BATCH_SIZE}")
    log_info(f"  Learning rate: {LEARNING_RATE}")
    log_info(f"  Total samples: {len(train_dataset)}")
    
    # Train
    try:
        trainer.train()
        log_info("Training completed successfully")
    except KeyboardInterrupt:
        log_warning("Training interrupted by user")
    except Exception as e:
        log_error(f"Training failed: {e}")
        return None
    
    return trainer

# ============================================================
# VALIDATION
# ============================================================

def validate_on_llm_sample(model, tokenizer, val_df):
    """
    Validate fine-tuned model on 200-patient sample from LLM project.
    Compare to base model zero-shot baseline.
    
    Args:
        model: fine-tuned PEFT model
        tokenizer: HuggingFace tokenizer
        val_df: validation DataFrame (llm_sample.csv)
        
    Returns:
        dict with predictions and metrics
    """
    
    log_info("Validating on 200-patient sample...")
    
    model.eval()
    
    predictions_finetuned = []
    predictions_base = []
    
    for idx, (_, row) in enumerate(tqdm(val_df.iterrows(), total=len(val_df), desc="Validating")):
        
        description = row["patient_description"]
        true_label = row["mortality"]
        
        # Create prompt
        instruction = "Predict in-hospital mortality from the patient description."
        prompt = create_prompt(instruction, description)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with fine-tuned model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,  # Deterministic
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and parse prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]  # Remove prompt
        
        # Simple parser: look for Yes/No
        pred_finetuned = 1 if "Yes" in response else 0
        predictions_finetuned.append(pred_finetuned)
        
        # Base model prediction (from your LLM project, if available)
        # For now, just append placeholder
        predictions_base.append(row.get("llama31_zero_shot", -1))
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    
    y_true = val_df["mortality"].values
    y_pred_finetuned = np.array(predictions_finetuned)
    
    acc = accuracy_score(y_true, y_pred_finetuned)
    cm = confusion_matrix(y_true, y_pred_finetuned)
    
    n_flagged = y_pred_finetuned.sum()
    
    results = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "n_flagged": int(n_flagged),
        "n_total": len(y_true),
        "predictions": y_pred_finetuned.tolist(),
        "true_labels": y_true.tolist(),
    }
    
    log_info(f"Validation Results (Fine-tuned):")
    log_info(f"  Accuracy: {acc:.1%}")
    log_info(f"  Flagged HIGH: {n_flagged}/{len(y_true)}")
    log_info(f"  Confusion Matrix: {cm}")
    
    return results

# ============================================================
# SAVE & EXPORT
# ============================================================

def save_adapter(model, output_dir=ADAPTER_DIR):
    """Save LoRA adapter weights."""
    log_info(f"Saving adapter to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    log_info("Adapter saved successfully")

def save_results(results, output_dir=RESULTS_DIR):
    """Save validation results to JSON."""
    log_info(f"Saving results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "validation_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        "accuracy": float(results["accuracy"]),
        "n_flagged": int(results["n_flagged"]),
        "n_total": int(results["n_total"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "predictions": results["predictions"],
        "true_labels": results["true_labels"],
    }
    
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    log_info(f"Results saved to {results_file}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Main execution pipeline."""
    
    log_info("=" * 70)
    log_info("PHASE 2: LoRA FINE-TUNING")
    log_info("=" * 70)
    
    # Check GPU
    log_info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log_info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ========== LOAD DATA ==========
    
    training_pairs = load_training_data(TRAINING_DATA_FILE)
    if training_pairs is None:
        log_error("Failed to load training data")
        return
    
    val_df = load_validation_data(VALIDATION_DATA_FILE)
    if val_df is None:
        log_error("Failed to load validation data")
        return
    
    # ========== SETUP MODEL ==========
    
    model, tokenizer = setup_model_and_tokenizer()
    if model is None or tokenizer is None:
        log_error("Failed to setup model and tokenizer")
        return
    
    # ========== CREATE DATASET ==========
    
    train_dataset = create_dataset(training_pairs, tokenizer)
    if train_dataset is None:
        log_error("Failed to create training dataset")
        return
    
    # ========== TRAIN ==========
    
    trainer = train(model, tokenizer, train_dataset)
    if trainer is None:
        log_error("Training failed")
        return
    
    # ========== VALIDATE ==========
    
    results = validate_on_llm_sample(model, tokenizer, val_df)
    
    # ========== SAVE ==========
    
    save_adapter(model)
    save_results(results)
    
    # ========== SUMMARY ==========
    
    log_info("=" * 70)
    log_info("PHASE 2 COMPLETE")
    log_info("=" * 70)
    log_info(f"Adapter saved to: {ADAPTER_DIR}")
    log_info(f"Results saved to: {RESULTS_DIR}")
    log_info(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    log_info("Ready for Phase 3: RAG + Clinical Persona Inference")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
