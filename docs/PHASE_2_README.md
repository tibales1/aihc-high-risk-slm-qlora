# PHASE 2: LoRA FINE-TUNING

## Overview

**What it does:**
- Loads 50K LoRA training pairs from Phase 1 output
- Fine-tunes `llama3.1:8b` on T4 using Parameter-Efficient Fine-Tuning (LoRA)
- Validates on 200-patient sample from your LLM project
- Compares fine-tuned model vs. base llama3.1:8b zero-shot baseline
- Saves LoRA adapters (lightweight weights, ~100MB)
- Generates training metrics and validation results

**Hardware required:**
- T4 (16GB) — primary GPU for fine-tuning
- RTX 2080 (8GB) — idle
- CPU: i9-12900KF (used for data loading, model I/O)

**Output:**
- `data/lora_adapters/` — LoRA adapter weights (can be merged later)
- `data/lora_checkpoints/` — training checkpoints (resume-safe)
- `data/phase2_results/validation_results.json` — predictions and metrics

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LoRA Rank** | 16 | Standard balance between model capacity and efficiency |
| **LoRA Alpha** | 32 | 2×rank; controls scaling of LoRA output |
| **Learning Rate** | 2e-4 | Standard for LoRA (lower than full fine-tuning) |
| **Batch Size** | 4 | T4 16GB limit; higher = faster but risky OOM |
| **Epochs** | 3 | 50K pairs × 3 epochs = ~150K gradient steps |
| **Warmup Steps** | 100 | 1% of total steps; gradual learning rate ramp |
| **Save Checkpoints** | Every 500 steps | Resume-safe; manageable storage |
| **Max Sequence Length** | 512 | Sufficient for patient descriptions + instruction |
| **Mixed Precision (fp16)** | Enabled | Reduces memory by ~50%, maintains accuracy |
| **Optimizer** | paged_adamw_8bit | Memory-efficient AdamW on T4 |

---

## Setup

### 1. Install Dependencies

```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face ecosystems
pip install transformers peft datasets

# Utilities
pip install pandas numpy scikit-learn tqdm

# Optional: for plotting (Phase 3)
pip install matplotlib seaborn
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. Verify Phase 1 Completed

Phase 2 requires Phase 1 output:

```bash
# Check training data exists
ls -lah data/lora_training_data_combined.jsonl
# Should show: ~150-200 MB

# Check validation data exists (from your LLM project)
ls -lah data/llm_sample.csv
# Should show: ~200 rows, with patient_description column
```

---

## Pre-Flight Check

**Before committing 2-6 hours to fine-tuning, run:**

```bash
python phase_2_setup_check.py
```

**Expected output:**

```
======================================================================
PHASE 2: PRE-FLIGHT VALIDATION CHECK
======================================================================

[PREFLIGHT] Checking GPU...
  ✓ GPU detected: Tesla T4
  ✓ GPU memory: 16.0 GB

[PREFLIGHT] Checking Python packages...
  ✓ PyTorch
  ✓ Hugging Face Transformers
  ✓ PEFT
  ✓ Hugging Face Datasets
  ✓ Pandas
  ✓ NumPy
  ✓ scikit-learn
  ✓ tqdm

[PREFLIGHT] Checking training data...
  ✓ Training data exists: data/lora_training_data_combined.jsonl
    File size: 156.2 MB

[PREFLIGHT] Checking validation data...
  ✓ Validation data exists: data/llm_sample.csv
    Samples: 200
    Mortality rate: 11.0%

[PREFLIGHT] Checking HuggingFace model...
  ✓ transformers library available
  Model: meta-llama/Llama-3.1-8B
  First run will download ~16GB...

[PREFLIGHT] Estimating VRAM usage...
  Estimated breakdown:
    Model (fp16): 8.0 GB
    Optimizer: 2.0 GB
    Batch (4 samples): 2.0 GB
    LoRA: 0.2 GB
    Cache: 1.0 GB
    Total: 13.2 GB
  Available VRAM: 16.0 GB
  ✓ Estimated usage is within safe limits

======================================================================
PRE-FLIGHT CHECKS: 6/6 PASSED
======================================================================

✓ All checks passed!
You can now run Phase 2:
  python phase_2_lora_finetuning.py

Expected runtime: 2-6 hours
First run will download model (~16GB, 15-30 min)
```

If any check fails, fix the issue and retry.

---

## Running Phase 2

### First Run (with model download)

```bash
cd C:\path\to\your\project

# First run downloads llama3.1:8b from HuggingFace (15-30 min on good internet)
python phase_2_lora_finetuning.py
```

**Expected output (first 30 seconds):**

```
[2026-07-22 17:30:00] [INFO] ======================================================================
[2026-07-22 17:30:00] [INFO] PHASE 2: LoRA FINE-TUNING
[2026-07-22 17:30:00] [INFO] ======================================================================
[2026-07-22 17:30:00] [INFO] Device: cuda
[2026-07-22 17:30:00] [INFO] GPU: Tesla T4
[2026-07-22 17:30:00] [INFO] GPU Memory: 16.0 GB
[2026-07-22 17:30:00] [INFO] Loading training data from data/lora_training_data_combined.jsonl...
[2026-07-22 17:30:00] [INFO] Loaded 50866 training pairs
[2026-07-22 17:30:00] [INFO]   Mortality distribution: Yes=5790 (11.4%), No=45076 (88.6%)
[2026-07-22 17:30:00] [INFO] Loading validation data from data/llm_sample.csv...
[2026-07-22 17:30:00] [INFO] Loaded 200 validation samples
[2026-07-22 17:30:00] [INFO]   Mortality rate: 11.0%
[2026-07-22 17:30:01] [INFO] Loading model: meta-llama/Llama-3.1-8B
[2026-07-22 17:30:01] [INFO] Downloading model from HuggingFace (this may take 15-30 minutes)...
```

Then the model downloads (~16GB). Grab coffee ☕.

**Expected output (after model download, starts training):**

```
[2026-07-22 17:45:30] [INFO] Base model loaded
[2026-07-22 17:45:30] [INFO] Applying LoRA configuration...
[2026-07-22 17:45:31] [INFO] LoRA applied successfully
[2026-07-22 17:45:31] [INFO] trainable params: 4,194,304 || all params: 8,031,810,560 || trainable%: 0.05
[2026-07-22 17:45:31] [INFO] Setting up training arguments...
[2026-07-22 17:45:31] [INFO] Initializing Trainer...
[2026-07-22 17:45:31] [INFO] Starting training...
[2026-07-22 17:45:31] [INFO]   Epochs: 3
[2026-07-22 17:45:31] [INFO]   Batch size: 4
[2026-07-22 17:45:31] [INFO]   Learning rate: 0.0002
[2026-07-22 17:45:31] [INFO] Tokenizing training data...
  100%|████████| 50866/50866 [05:30<00:00, 154.23it/s]
[2026-07-22 17:50:00] [INFO] Tokenized 50866/50866 pairs
```

**Expected output (training loop):**

```
Epoch 1/3
  50%|█████     | 6383/12716 [45:30<45:00, 0.234 samples/s]
    Loss: 0.452

Epoch 2/3
  25%|██        | 3200/12716 [22:15<67:00, 0.142 samples/s]
    Loss: 0.301

[Checkpoint saved at step 500, 1000, 1500, ...]
```

**Expected total runtime:**
- Model download: 15–30 min (first run only)
- Training: 2–4 hours (50K pairs, 3 epochs, batch_size=4 on T4)
- Validation: 5–10 min
- **Total first run: 2.5–4.5 hours**

### Resume from Checkpoint

If training is interrupted (Ctrl+C, power loss, etc.):

```bash
python phase_2_lora_finetuning.py
```

The script detects the last checkpoint and resumes. No data is lost; training continues from that step.

---

## Monitoring Training

### Watch GPU Usage (in separate terminal)

```bash
# Watch T4 GPU in real-time
nvidia-smi -l 1  # Refresh every 1 second

# Should show:
#   Process name: python
#   GPU Memory: ~14–15 GB (model + batch + optimizer)
#   GPU Util: 85–95% (active training)
```

### Watch Loss (in main terminal)

Training logs show loss every 100 steps:

```
[2026-07-22 17:XX:XX] Training Loss: 0.457  [batch 1000/12716]
[2026-07-22 17:XX:XX] Training Loss: 0.341  [batch 2000/12716]
[2026-07-22 17:XX:XX] Training Loss: 0.289  [batch 3000/12716]
```

Loss should generally decrease over time. If it plateaus or increases, training may be diverging—consider lowering learning rate.

---

## Outputs

After training completes:

```
data/
├── lora_training_data_combined.jsonl         (Phase 1 input)
├── llm_sample.csv                            (validation set)
├── lora_adapters/                            (LoRA weights)
│   ├── adapter_config.json
│   └── adapter_model.bin                     (~100 MB)
├── lora_checkpoints/                         (training checkpoints)
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-1500/                      (etc, keep 3 latest)
└── phase2_results/
    └── validation_results.json               (predictions + metrics)
```

### Validation Results (`phase2_results/validation_results.json`)

```json
{
  "accuracy": 0.765,
  "n_flagged": 42,
  "n_total": 200,
  "confusion_matrix": [[145, 13], [15, 27]],
  "predictions": [0, 1, 0, 1, ...],
  "true_labels": [0, 0, 0, 1, ...]
}
```

**Interpret:**
- `accuracy`: 76.5% (improvement over baseline 74.5%?)
- `n_flagged`: 42/200 patients flagged as HIGH RISK (vs. baseline 37/200)
- `confusion_matrix`: [TN, FP] / [FN, TP]

---

## Expected Results

### Baseline (llama3.1:8b, zero-shot, temp=0.0)
*From your LLM project:*
- Accuracy: 74.5%
- Flagged HIGH: 37/200
- Recall: 18.2% (4/22 deaths caught)
- Precision: 10.8%

### Expected Fine-tuned (llama3.1:8b + LoRA)
- Accuracy: 75–80% (slightly better discrimination)
- Flagged HIGH: 40–50/200 (should increase, more alert)
- Recall: 25–35% (should catch more deaths)
- Precision: 20–30% (better calibration)

### Comparison to Structured ML
*From your ML/DL project:*
- GBM on 17 features: AUC 0.757
- DNN on 17 features: AUC 0.754
- Embed + LogReg: AUC 0.743

Fine-tuned llama3.1:8b should approach or exceed these, especially on recall (catching deaths).

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GB
```

**Solution:**
1. Reduce batch size: Edit `BATCH_SIZE = 2` in script (line ~110)
2. Re-run training (resumes from last checkpoint)

### Issue: Model Download Stalled

**Error:**
```
Connection timeout downloading meta-llama/Llama-3.1-8B
```

**Solution:**
1. Check internet connection
2. HuggingFace may have rate limits; wait 10 minutes and retry
3. Alternatively, download model manually:
   ```bash
   huggingface-cli download meta-llama/Llama-3.1-8B
   ```

### Issue: Slow Training (< 10 samples/sec)

**Cause:** Normal for T4 with batch_size=4 (~0.2 sec/sample)

**Expected speed:**
- T4: ~5–10 samples/sec
- 50K samples / 8 samples/sec = ~1.7 hours per epoch
- 3 epochs = ~5 hours total (with validation)

If significantly slower, check GPU utilization (`nvidia-smi`). If <50%, something is blocking GPU (I/O, data loading).

### Issue: Training Loss Not Decreasing

**Cause:** Learning rate too low or data imbalance

**Solution:**
1. Check loss at epoch 1 vs. epoch 2 — should decrease
2. If stuck at same value, try higher learning rate: `LEARNING_RATE = 5e-4`
3. Imbalanced data (11.4% mortality) is normal; use weighted loss (in Phase 2 code v2)

---

## Next Steps (Phase 3)

Once Phase 2 completes:

1. **Verify adapter weights saved:**
   ```bash
   ls -lah data/lora_adapters/
   ```

2. **Inspect validation results:**
   ```bash
   python -c "import json; print(json.load(open('data/phase2_results/validation_results.json')))"
   ```

3. **Phase 3: RAG + Clinical Persona**
   - Load fine-tuned model + adapter
   - Implement RAG (retrieve similar cases from training set)
   - Apply your Pip-Boy personality architecture to clinical context
   - Inference on 200-patient sample
   - Compare all three versions

---

## Files

- `phase_2_lora_finetuning.py` — Main training script
- `phase_2_setup_check.py` — Pre-flight validation
- `PHASE_2_README.md` — This file

---

## Timeline for RISK Project

| Phase | Due | Status |
|-------|-----|--------|
| Phase 1 (data gen) | Jul 22 (TODAY) | ✅ Complete (3 sec!) |
| Phase 2 (LoRA) | Jul 23–24 | 🟡 In progress (2-6 hours) |
| Phase 3 (RAG) | Jul 24–25 | 🔲 Not started |
| Paper + Slides | Aug 10 | 🔲 Not started |
| Video | Aug 10 | 🔲 Not started |

You're on pace. Phase 2 tonight/tomorrow morning, Phase 3 tomorrow afternoon, then writing.
