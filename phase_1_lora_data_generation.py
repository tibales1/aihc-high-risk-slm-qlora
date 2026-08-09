"""
=============================================================================
RISK PROJECT — PHASE 1: LoRA TRAINING DATA GENERATION
=============================================================================

Purpose:
    Load the full MIMIC-III mortality cohort (50,866 admissions).
    Convert structured features to natural language patient descriptions.
    Format as LoRA training pairs with ground-truth mortality labels.
    Checkpoint every 5K admissions for crash safety.

Input:  data/features_mortality.csv (from ML/DL mortality project)
Output: data/lora_training_data/ (checkpointed JSON files)
        data/lora_training_data_combined.jsonl (final concatenated pairs)

Hardware:
    CPU only (no GPU needed for this phase).
    Phase 2 will fine-tune on T4 using this data.

Approach:
    - Pure Python data wrangling (no LLM calls)
    - Procedural loops with explicit error handling
    - Checkpoint after every 5K rows for resumability
    - Instruction diversity via cycling through prompts
    - Yes/No binary output for mortality (ground truth from data)

Working Rhythm:
    - Load cohort
    - Loop through admissions in 5K batches
    - Generate descriptions for each (procedural, no list comp)
    - Format as LoRA pairs with cycled instructions
    - Save checkpoint (descriptions + pairs)
    - Log progress and errors
    - Combine all checkpoints into final JSONL

=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "lora_training_data")
OUTPUT_FILE = os.path.join(DATA_DIR, "lora_training_data_combined.jsonl")

FEATURES_FILE = os.path.join(DATA_DIR, "features_mortality.csv")

BATCH_SIZE = 5000  # Checkpoint every 5K admissions
RANDOM_SEED = 42

# LoRA instruction prompts (will be cycled)
INSTRUCTIONS = [
    "Predict in-hospital mortality from the patient description.",
    "Based on this patient profile, assess mortality risk.",
    "Determine whether this patient is at high mortality risk.",
    "Will this patient survive to discharge?",
    "Evaluate the patient's likelihood of in-hospital death.",
    "Assess if this patient will experience in-hospital mortality.",
    "Predict the patient's survival outcome.",
    "Determine mortality risk for this patient.",
]

# ============================================================
# LOGGING
# ============================================================

def log_message(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def log_error(msg):
    """Log an error message."""
    log_message(msg, level="ERROR")

def log_warning(msg):
    """Log a warning message."""
    log_message(msg, level="WARNING")

# ============================================================
# BUILD PATIENT DESCRIPTION (from LLM tutorial)
# ============================================================

def build_patient_description(row):
    """
    Convert a feature row into a natural language patient description.
    
    Adapted from: llom_project.ipynb Phase 0
    
    Args:
        row: pandas Series with columns from features_mortality.csv
        
    Returns:
        str: clinical-style patient description
    """
    
    try:
        # Gender
        gender = "male" if row.get("gender_male", 0) == 1 else "female"
        
        # Age
        age = int(row["age_years"])
        
        # Insurance
        if row.get("ins_Medicaid", 0) == 1:
            insurance = "Medicaid"
        elif row.get("ins_Medicare", 0) == 1:
            insurance = "Medicare"
        elif row.get("ins_Private", 0) == 1:
            insurance = "private insurance"
        elif row.get("ins_Self Pay", 0) == 1:
            insurance = "self-pay"
        else:
            insurance = "government insurance"
        
        # Ethnicity
        if row.get("eth_WHITE", 0) == 1:
            ethnicity = "White"
        elif row.get("eth_OTHER_UNKNOWN", 0) == 1:
            ethnicity = "other/unknown ethnicity"
        else:
            ethnicity = "Black"
        
        # Length of stay
        los = row["los_days"]
        if los < 1:
            los_text = "less than 1 day"
        else:
            los_text = f"{los:.1f} days"
        
        # Prior admissions
        n_prior = int(row["n_prior_admissions"])
        if n_prior == 0:
            prior_text = "no prior hospital admissions"
        elif n_prior == 1:
            prior_text = "1 prior hospital admission"
        else:
            prior_text = f"{n_prior} prior hospital admissions"
        
        # Diagnosis count
        n_dx = int(row["n_diagnoses"])
        
        # Substance use
        substances = []
        if row.get("has_alcohol", 0) == 1:
            substances.append("alcohol use disorder")
        if row.get("has_opioid", 0) == 1:
            substances.append("opioid use")
        if row.get("has_cocaine", 0) == 1:
            substances.append("cocaine use")
        if row.get("has_cannabis", 0) == 1:
            substances.append("cannabis use")
        if row.get("has_sedative", 0) == 1:
            substances.append("sedative use")
        if row.get("has_other_sub", 0) == 1:
            substances.append("other substance use")
        
        if substances:
            substance_text = "Substance use history includes: " + ", ".join(substances) + "."
        else:
            substance_text = "No documented substance use disorders."
        
        # Assemble description
        description = (
            f"Patient is a {age}-year-old {ethnicity} {gender} admitted to the ICU "
            f"with {insurance}. The patient has {prior_text} and {n_dx} diagnosis "
            f"codes on the current admission. Length of ICU stay is {los_text}. "
            f"{substance_text}"
        )
        
        return description
    
    except Exception as e:
        log_error(f"Failed to build description for row: {e}")
        return None

# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def get_checkpoint_path(batch_num):
    """Get the file path for a checkpoint."""
    return os.path.join(CHECKPOINT_DIR, f"batch_{batch_num:05d}.json")

def save_checkpoint(batch_num, descriptions, lora_pairs, batch_start, batch_end):
    """
    Save a batch checkpoint.
    
    Args:
        batch_num: batch number (0, 1, 2, ...)
        descriptions: list of {hadm_id, description}
        lora_pairs: list of {instruction, input, output}
        batch_start: starting row index
        batch_end: ending row index (exclusive)
    """
    
    checkpoint_data = {
        "batch_num": batch_num,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "count": len(lora_pairs),
        "timestamp": datetime.now().isoformat(),
        "descriptions": descriptions,
        "lora_pairs": lora_pairs,
    }
    
    checkpoint_path = get_checkpoint_path(batch_num)
    
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        log_message(f"Saved checkpoint {batch_num} ({batch_start}-{batch_end}, {len(lora_pairs)} pairs)")
    except Exception as e:
        log_error(f"Failed to save checkpoint {batch_num}: {e}")
        raise

def load_checkpoint(batch_num):
    """
    Load a checkpoint if it exists.
    
    Returns:
        tuple (descriptions, lora_pairs) or (None, None) if not found
    """
    
    checkpoint_path = get_checkpoint_path(batch_num)
    
    if not os.path.exists(checkpoint_path):
        return None, None
    
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        log_message(f"Loaded checkpoint {batch_num} ({data['count']} pairs)")
        return data["descriptions"], data["lora_pairs"]
    except Exception as e:
        log_error(f"Failed to load checkpoint {batch_num}: {e}")
        return None, None

def find_last_checkpoint():
    """
    Find the highest checkpoint number already saved.
    
    Returns:
        int: last checkpoint number, or -1 if no checkpoints exist
    """
    
    if not os.path.exists(CHECKPOINT_DIR):
        return -1
    
    checkpoint_files = []
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.startswith("batch_") and filename.endswith(".json"):
            try:
                batch_num = int(filename.split("_")[1].replace(".json", ""))
                checkpoint_files.append(batch_num)
            except:
                pass
    
    if not checkpoint_files:
        return -1
    
    return max(checkpoint_files)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Main execution pipeline."""
    
    log_message("=" * 70)
    log_message("PHASE 1: LoRA TRAINING DATA GENERATION")
    log_message("=" * 70)
    
    # ========== SETUP ==========
    
    log_message("Setting up directories and configuration...")
    
    # Create checkpoint directory
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        log_message(f"Created checkpoint directory: {CHECKPOINT_DIR}")
    
    # ========== LOAD DATA ==========
    
    log_message(f"Loading features from {FEATURES_FILE}...")
    
    if not os.path.exists(FEATURES_FILE):
        log_error(f"Features file not found: {FEATURES_FILE}")
        log_message("Make sure features_mortality.csv is in the data/ directory")
        return
    
    try:
        df = pd.read_csv(FEATURES_FILE)
    except Exception as e:
        log_error(f"Failed to load features file: {e}")
        return
    
    n_total = len(df)
    n_mortality = df["mortality"].sum()
    mortality_rate = n_mortality / n_total if n_total > 0 else 0
    
    log_message(f"Loaded {n_total} admissions")
    log_message(f"  Deaths: {n_mortality}")
    log_message(f"  Survived: {n_total - n_mortality}")
    log_message(f"  Mortality rate: {mortality_rate:.1%}")
    
    # ========== CHECK FOR RESUMABLE STATE ==========
    
    last_checkpoint = find_last_checkpoint()
    
    if last_checkpoint >= 0:
        resume_from_batch = last_checkpoint + 1
        resume_from_row = resume_from_batch * BATCH_SIZE
        log_message(f"Found checkpoint {last_checkpoint}. Resuming from batch {resume_from_batch} (row {resume_from_row})...")
    else:
        resume_from_batch = 0
        resume_from_row = 0
        log_message("No checkpoints found. Starting from beginning.")
    
    # ========== PROCESS BATCHES ==========
    
    log_message(f"Processing {n_total} admissions in batches of {BATCH_SIZE}...")
    
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    log_message(f"Total batches: {n_batches}")
    
    n_successfully_generated = 0
    n_failed_descriptions = 0
    n_skipped_batches = 0
    
    instruction_index = 0
    
    for batch_num in range(n_batches):
        
        # Skip already-checkpointed batches
        if batch_num < resume_from_batch:
            n_skipped_batches += 1
            continue
        
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        batch_size_actual = batch_end - batch_start
        
        log_message(f"Batch {batch_num}/{n_batches-1} (rows {batch_start}-{batch_end})...")
        
        # Extract batch
        batch_df = df.iloc[batch_start:batch_end].copy()
        
        # Initialize batch containers
        descriptions_list = []
        lora_pairs_list = []
        
        # Process each row in the batch
        for row_idx, (_, row) in enumerate(batch_df.iterrows()):
            
            # <human_intervention>
            # Error handling: skip rows with missing critical fields
            try:
                if pd.isna(row.get("age_years")) or pd.isna(row.get("los_days")):
                    n_failed_descriptions += 1
                    continue
            except:
                n_failed_descriptions += 1
                continue
            # </human_intervention>
            
            # Build description
            description = build_patient_description(row)
            
            if description is None:
                n_failed_descriptions += 1
                continue
            
            # Store description
            descriptions_list.append({
                "hadm_id": row.get("hadm_id", "unknown"),
                "description": description
            })
            
            # Get mortality label
            mortality_label = int(row["mortality"])
            output_label = "Yes" if mortality_label == 1 else "No"
            
            # Cycle through instructions for diversity
            instruction = INSTRUCTIONS[instruction_index % len(INSTRUCTIONS)]
            instruction_index += 1
            
            # Create LoRA pair
            lora_pair = {
                "instruction": instruction,
                "input": description,
                "output": output_label
            }
            lora_pairs_list.append(lora_pair)
            
            n_successfully_generated += 1
        
        # Save checkpoint
        try:
            save_checkpoint(batch_num, descriptions_list, lora_pairs_list, batch_start, batch_end)
        except Exception as e:
            log_error(f"Failed to save checkpoint {batch_num}. Aborting.")
            return
    
    # ========== SUMMARY ==========
    
    log_message("=" * 70)
    log_message("BATCH PROCESSING COMPLETE")
    log_message("=" * 70)
    log_message(f"Successfully generated: {n_successfully_generated} pairs")
    log_message(f"Failed descriptions: {n_failed_descriptions}")
    log_message(f"Skipped batches (already done): {n_skipped_batches}")
    
    # ========== COMBINE CHECKPOINTS ==========
    
    log_message(f"Combining checkpoints into single JSONL file...")
    
    all_pairs = []
    combined_descriptions = []
    
    for batch_num in range(n_batches):
        descriptions, lora_pairs = load_checkpoint(batch_num)
        
        if lora_pairs is None:
            log_warning(f"Checkpoint {batch_num} not found during combine step. Skipping.")
            continue
        
        combined_descriptions.extend(descriptions)
        all_pairs.extend(lora_pairs)
    
    log_message(f"Loaded {len(all_pairs)} pairs from checkpoints")
    
    # Write combined JSONL (one pair per line)
    try:
        with open(OUTPUT_FILE, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")
        log_message(f"Saved {len(all_pairs)} LoRA pairs to {OUTPUT_FILE}")
    except Exception as e:
        log_error(f"Failed to write combined JSONL: {e}")
        return
    
    # ========== FINAL REPORT ==========
    
    log_message("=" * 70)
    log_message("PHASE 1 COMPLETE")
    log_message("=" * 70)
    log_message(f"Output file: {OUTPUT_FILE}")
    log_message(f"Checkpoint directory: {CHECKPOINT_DIR}")
    log_message(f"Total LoRA pairs: {len(all_pairs)}")
    
    # Compute mortality distribution in pairs
    n_yes = sum(1 for p in all_pairs if p["output"] == "Yes")
    n_no = len(all_pairs) - n_yes
    log_message(f"Mortality distribution in pairs:")
    log_message(f"  Yes (died): {n_yes} ({n_yes/len(all_pairs):.1%})")
    log_message(f"  No (survived): {n_no} ({n_no/len(all_pairs):.1%})")
    
    log_message("Ready for Phase 2: LoRA fine-tuning on T4")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
