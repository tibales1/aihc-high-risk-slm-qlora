# PHASE 1: LoRA TRAINING DATA GENERATION

## Overview

**What it does:**
- Loads `features_mortality.csv` (50,866 admissions)
- Converts each admission to natural language patient description
- Formats as LoRA training pairs: `{instruction, input (description), output (Yes/No)}`
- Checkpoints every 5K admissions for crash safety
- Combines all checkpoints into final JSONL file

**Hardware required:**
- CPU only (no GPU needed)
- ~2–4 hours runtime on i9-12900KF

**Output:**
- `data/lora_training_data_combined.jsonl` — final training file (~50K lines, one pair per line)
- `data/lora_training_data/` — checkpoint JSON files (batch_00000.json, batch_00001.json, etc.)

---

## Setup

### 1. Prepare Data Directory

```bash
# On your Windows machine, in your project directory:
mkdir -p data
# Ensure features_mortality.csv is in data/
# (from your ML/DL mortality project)
```

### 2. Install Python Dependencies

```bash
pip install pandas numpy
```

(No special GPU packages needed for Phase 1.)

### 3. Verify Features File

```python
import pandas as pd

df = pd.read_csv("data/features_mortality.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Mortality rate: {df['mortality'].mean():.1%}")
```

Should show:
- ~50,866 rows
- Columns including: `age_years`, `los_days`, `mortality`, `hadm_id`, insurance/ethnicity flags, substance use flags
- Mortality rate ~11.4%

---

## Running Phase 1

### Option A: First Run (from scratch)

```bash
# Navigate to your project directory
cd C:\path\to\your\project

# Run Phase 1
python phase_1_lora_data_generation.py
```

**Expected output:**
```
[2026-07-22 17:15:30] [INFO] ======================================================================
[2026-07-22 17:15:30] [INFO] PHASE 1: LoRA TRAINING DATA GENERATION
[2026-07-22 17:15:30] [INFO] ======================================================================
[2026-07-22 17:15:30] [INFO] Setting up directories and configuration...
[2026-07-22 17:15:30] [INFO] Created checkpoint directory: data/lora_training_data
[2026-07-22 17:15:30] [INFO] Loading features from data/features_mortality.csv...
[2026-07-22 17:15:32] [INFO] Loaded 50866 admissions
[2026-07-22 17:15:32] [INFO]   Deaths: 5798
[2026-07-22 17:15:32] [INFO]   Survived: 45068
[2026-07-22 17:15:32] [INFO]   Mortality rate: 11.4%
[2026-07-22 17:15:32] [INFO] No checkpoints found. Starting from beginning.
[2026-07-22 17:15:32] [INFO] Processing 50866 admissions in batches of 5000...
[2026-07-22 17:15:32] [INFO] Total batches: 11
[2026-07-22 17:15:32] [INFO] Batch 0/10 (rows 0-5000)...
  ...
[2026-07-22 17:XX:XX] [INFO] Saved checkpoint 10 (50000-50866, 866 pairs)
```

Expect ~30–60 seconds per batch. Total ~10–15 batches = ~2–4 hours.

### Option B: Resume from Checkpoint

If the script crashes or you interrupt it, re-run the same command:

```bash
python phase_1_lora_data_generation.py
```

It will automatically:
1. Detect the last completed checkpoint
2. Skip already-processed batches
3. Resume from the next batch
4. Combine all checkpoints at the end

**Example resume output:**
```
[2026-07-22 18:00:00] [INFO] Found checkpoint 5. Resuming from batch 6 (row 30000)...
[2026-07-22 18:00:00] [INFO] Skipped batches (already done): 6
[2026-07-22 18:00:00] [INFO] Batch 6/10 (rows 30000-35000)...
```

---

## Validation & Inspection

### 1. Check Checkpoint Structure

```bash
# List all checkpoints
dir data\lora_training_data\

# Inspect a single checkpoint (e.g., batch_00000.json)
# It's a JSON file with:
# {
#   "batch_num": 0,
#   "batch_start": 0,
#   "batch_end": 5000,
#   "count": 5000,
#   "timestamp": "...",
#   "descriptions": [{hadm_id, description}, ...],
#   "lora_pairs": [{instruction, input, output}, ...]
# }
```

### 2. Inspect Final JSONL

```python
import json

# Read a few lines from the combined file
with open("data/lora_training_data_combined.jsonl", "r") as f:
    for i in range(5):
        line = f.readline()
        pair = json.loads(line)
        print(f"Pair {i}:")
        print(f"  Instruction: {pair['instruction']}")
        print(f"  Input (first 100 chars): {pair['input'][:100]}...")
        print(f"  Output: {pair['output']}")
        print()
```

**Example output:**
```
Pair 0:
  Instruction: Predict in-hospital mortality from the patient description.
  Input (first 100 chars): [redacted - MIMIC-derived record; see the template in phase_1_lora_data_generation.py]
  Output: No

Pair 1:
  Instruction: Based on this patient profile, assess mortality risk.
  Input (first 100 chars): [redacted - MIMIC-derived record; see the template in phase_1_lora_data_generation.py]
  Output: Yes
```

### 3. Verify Mortality Distribution

```python
import json

yes_count = 0
no_count = 0

with open("data/lora_training_data_combined.jsonl", "r") as f:
    for line in f:
        pair = json.loads(line)
        if pair["output"] == "Yes":
            yes_count += 1
        else:
            no_count += 1

total = yes_count + no_count
print(f"Total pairs: {total}")
print(f"Yes (died): {yes_count} ({yes_count/total:.1%})")
print(f"No (survived): {no_count} ({no_count/total:.1%})")
```

Should match the original cohort mortality rate (~11.4%).

### 4. Count Instruction Diversity

```python
import json
from collections import Counter

instructions = []

with open("data/lora_training_data_combined.jsonl", "r") as f:
    for line in f:
        pair = json.loads(line)
        instructions.append(pair["instruction"])

instruction_counts = Counter(instructions)
for instr, count in sorted(instruction_counts.items(), key=lambda x: -x[1]):
    print(f"{count:5d} | {instr}")
```

Should show roughly equal distribution across 8 instructions (~6-7K each).

---

## Expected Metrics

| Metric | Expected Value |
|--------|----------------|
| Total admissions | 50,866 |
| Total pairs generated | ~50,850–50,866 (depends on missing data) |
| Mortality rate (Yes) | 11.4% (~5,798 pairs) |
| Survival rate (No) | 88.6% (~45,068 pairs) |
| Unique instructions | 8 (evenly distributed) |
| Checkpoint files | 11 (batches 0–10) |
| Final JSONL size | ~150–200 MB |

---

## Troubleshooting

### Issue: "Features file not found"

**Solution:** Make sure `features_mortality.csv` is in the `data/` directory.

```bash
# Verify the file exists
dir data\features_mortality.csv
```

### Issue: Script crashes mid-batch

**Solution:** Re-run the script. It will resume from the last successful checkpoint.

```bash
python phase_1_lora_data_generation.py
```

The checkpoint system ensures no duplicate pairs are generated.

### Issue: Out of memory

**Solution:** Phase 1 uses very little memory (~500MB for batch). If you run out:
- Close other programs
- Reduce `BATCH_SIZE` in the script (line 50) to 2000 instead of 5000

### Issue: Slow performance

**Solution:** This is normal. Average ~30–60 seconds per 5K batch on a modern CPU. Total ~2–4 hours is expected.

To monitor progress:
```bash
# Watch checkpoints accumulate
dir data\lora_training_data\ | find "batch_"
```

---

## Next Steps (Phase 2)

Once Phase 1 completes and you have `data/lora_training_data_combined.jsonl`:

1. **Phase 2: LoRA Fine-tuning**
   - Load the JSONL file
   - Fine-tune `llama3.1:8b` on T4 using LoRA
   - Checkpoint every N steps
   - Evaluate on held-out test set

2. **Baseline comparison** (sanity check)
   - Run base `llama3.1:8b` on 200-patient sample (zero-shot)
   - Compare to fine-tuned version
   - Should see improvement over zero-shot 18.2% recall

---

## Code Notes

- **Procedural style:** Explicit loops, no list comprehensions, no lambdas
- **Human interventions:** Marked with `<human_intervention>` tags for error handling
- **Checkpointing:** Automatic resumability—no need to restart from scratch
- **Instruction cycling:** Instructions are cycled in order for training diversity
- **Error handling:** Missing data rows are skipped with logged warnings

---

## File Outputs

After Phase 1 completes:

```
data/
├── features_mortality.csv                    (input)
├── lora_training_data_combined.jsonl         (final output)
└── lora_training_data/
    ├── batch_00000.json
    ├── batch_00001.json
    ├── batch_00002.json
    └── ... (up to batch_00010.json)
```

Each batch JSON contains:
- `descriptions`: List of {hadm_id, description}
- `lora_pairs`: List of {instruction, input, output}

The combined JSONL is ready for Phase 2 fine-tuning.
