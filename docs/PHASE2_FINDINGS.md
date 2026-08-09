# Phase 2 — Findings and Changes

Investigation of `risk.ipynb` Phase 2 (LoRA fine-tuning of Llama-3.1-8B on
50,866 mortality-prediction pairs).

**Date:** 2026-07-23
**Hardware:** RTX 2080 (8 GB, GPU 0) + Tesla T4 (15.9 GB, GPU 1)
**Environment:** `aihc-nlp` — torch 2.7.1+cu118, transformers 5.13.0, peft 0.19.1,
accelerate 1.14.0, bitsandbytes 0.49.2 (installed during this work)

Starting symptom: *"I can't access my T4, it is going to the 2080."* That turned
out to be one of three independent bugs, none of which was the actual cause of
the kernel crash.

---

## TL;DR

| | Before | After |
|---|---|---|
| GPU used | RTX 2080 (8 GB) | Tesla T4 (15.9 GB) |
| Model load | Kernel crash (native access violation) | Loads, 5.59 GB |
| Peak VRAM | n/a — never got there | 10.73 GB / 15.9 GB |
| Optimizer steps | 38,150 | 1,035 |
| Tokens processed | 78.1 M | ~1.6 M |
| Validation integrity | 200/200 patients also in training | Held out |
| Overfitting monitoring | None | 6 evals on a grouped split |
| Est. runtime (healthy card) | unknown | ~2 h |
| Est. runtime (current cooling) | — | **~9 h, thermally throttled** |

**Remaining blocker is hardware, not software:** the T4 throttles 4.4× within
100 seconds of sustained load. See [Thermal](#5-thermal-the-remaining-blocker).

---

## 1. GPU selection — the original question

### Root cause

Three compounding problems in the same kernel:

1. The Phase 2 preflight cell ran `torch.cuda.set_device(1)`, which
   **initializes the CUDA runtime**. From that moment `CUDA_VISIBLE_DEVICES` is
   ignored for the life of the process. The Phase 2 cell set it *after* that —
   dead code.
2. `torch.cuda.set_device()` only changes the *default allocation* device.
   `from_pretrained(..., device_map=...)` ignores it entirely.
3. `device_map="auto"` fills GPUs starting at `cuda:0` = the RTX 2080 (8 GB).

Consistent with the logs: `GPU: NVIDIA GeForce RTX 2080`, `GPU Memory: 8.6 GB`.

### Fix

New **first cell**, which must run before anything else in the kernel:

```python
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-5efc6340-9a1d-0768-7857-fd704b7433e2"  # Tesla T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert "torch" not in sys.modules   # fails loudly if run too late
```

Pinned by **UUID**, not index, so driver enumeration order can't silently
reassign it. The T4 becomes the only visible GPU and is therefore `cuda:0`, so
every pre-existing `...(0)` call in the notebook now refers to it.

Also: removed `torch.cuda.set_device(1)` from the preflight cell, and removed a
`pip install torch ... cu118` subprocess call that ran on every execution —
reinstalling torch into a live kernel cannot take effect until restart and can
silently swap the CUDA build.

> **Requires a kernel restart.** An already-running kernel has its CUDA state
> frozen and cannot be repointed.

---

## 2. The model did not fit

Llama-3.1-8B is 8.03 B parameters — **~16.1 GB in fp16**, against the T4's
15.9 GB. It could never have fit, before optimizer state, activations, or a
single batch.

The preflight check said `Model (fp16): 8.0 GB`, off by 2×, so it reported
"within safe limits" and passed.

### Fix — 4-bit QLoRA

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # T4 is Turing (sm_75): no bf16
)
```

plus `prepare_model_for_kbit_training()` and `device_map={"": 0}`.

**Measured:** weights 5.59 GB, trainable params 6,815,744 (0.085%).
Installed `bitsandbytes` 0.49.2 (was absent — `optim="paged_adamw_8bit"` would
also have failed).

---

## 3. The kernel crash was *not* OOM

The crash at `Loading weights: 58%` persisted on the T4 needing only 5.5 GB.
`faulthandler` gave the real story:

```
Windows fatal exception: access violation
  File "torch/storage.py", line 466 in __getitem__
  File "transformers/core_model_loading.py", line 1183 in _materialize_copy
  ...
  File "transformers/modeling_utils.py", line 4368 in from_pretrained
```

**transformers 5.13.0 hardcodes the safetensors `mmap` backend on every
non-Mac platform** (`modeling_utils.py:4489`):

```python
backend, device = ("pread", "mps") if is_mps else ("mmap", "cpu")
```

On this machine, mmap-backed reads of the 16 GB checkpoint fault with
`0xC0000005`. The process dies with no Python traceback — which in Jupyter
surfaces only as *"The Kernel crashed while executing code in the current
cell"*, indistinguishable from OOM.

Ruled out along the way: corrupt shards (all 4 verified tensor-by-tensor),
RAM exhaustion (crash occurs with 5+ GB free), bitsandbytes (a tiny 4-bit model
loads fine).

### Fix

Monkeypatch `safe_open` to use the `pread` backend, which reads shards with
ordinary file I/O instead of mapping them:

```python
import transformers.modeling_utils as _modeling_utils
_orig = _modeling_utils.safe_open
def _safe_open_pread(*a, **kw):
    kw["backend"] = "pread"
    return _orig(*a, **kw)
_modeling_utils.safe_open = _safe_open_pread
```

Remove this if a future transformers release fixes the mmap path.

---

## 4. Data integrity

Two problems found by inspecting the Phase 1 output directly.

### 4a. The validation set was entirely inside the training set — CRITICAL

**All 200/200** patient descriptions in `llm_sample.csv` appear verbatim in
`lora_training_data_combined.jsonl`.

As written, Phase 2 trained on those patients, then reported "validation
accuracy" on the same patients, and compared it against a zero-shot baseline
that had never seen them. That measures memorization, not generalization, and
the comparison is not like-for-like. **It would have invalidated the headline
result** (recall vs. the 18.2% baseline).

Fixed via `EXCLUDE_VALIDATION_PATIENTS = True`, which drops those 213 pairs
from training. Verified: 0 of the 200 remain in train or eval.

### 4b. Conflicting labels

**265 descriptions carry both `Yes` and `No` labels.** The description fields
(age, gender, insurance, LOS, diagnosis count, substance use) are coarse enough
that different admissions render identical text with different outcomes. This
is an irreducible noise floor — no amount of training removes it, and it caps
achievable accuracy.

### 4c. Structure

- 50,866 pairs over **49,377 distinct descriptions** (1,300 appear more than once)
- 8 distinct instruction templates
- Answer is **exactly 1 token** (`Yes` = 9642, `No` = 2822)
- Class balance: 45,076 `No` / 5,790 `Yes` — **11.4% positive**

---

## 5. Training efficiency

Measured token length across the whole dataset: **median 97, max 109**.

| Problem | Detail |
|---|---|
| `padding="max_length"` to 512 | Every sample padded 512 → **5.2× of compute spent on padding** |
| `labels = input_ids.copy()` | Loss over all ~97 prompt tokens; >99% of gradient taught boilerplate regurgitation |
| `pad_token == eos_token` | Combined with the above, trained the model to emit **~414 EOS tokens per sample** |
| 3 epochs | Excessive for a 1-token binary target from 8 fixed templates |
| 88.6% negative | Drives the model to answer `No` unconditionally — the exact failure mode of the 18.2%-recall baseline |

### Changes

1. **`MAX_SEQ_LENGTH` 512 → 128, no padding in `tokenize_pair`.**
   `DataCollatorForSeq2Seq` pads each batch to its own longest member.
   *Verified:* a batch of 8 collates to width **99, not 512**; zero truncation.

2. **Prompt tokens masked to `-100`.** Loss on the answer only.
   *Verified:* a batch of 8×99 = 792 positions has 776 masked, leaving exactly
   16 = 8 samples × 2 targets, decoding to `'No<|end_of_text|>'`.
   Prompt/answer boundary confirmed token-identical on 200 samples, so masking
   is exact.

3. **`NUM_EPOCHS` 3 → 1.** *This is a genuine tradeoff, not waste removal* —
   less training, not the same training faster. Watch the eval curve and raise
   it if loss is still falling at the end.

4. **`MAJORITY_CLASS_RATIO = 2.0`**, applied to the **training split only**,
   after the eval split is carved off. The eval set deliberately keeps natural
   prevalence — rebalancing it too would inflate eval precision and disagree
   with the final 200-patient validation.

   ```
               Yes    total   positive-rate
     train    5524   16572    33.3%
     eval      240    2000    12.0%   <- natural prevalence preserved
   ```

### Result

| | Original | Now |
|---|---|---|
| Training pairs | 50,866 | 16,572 |
| Sequence width | 512 (padded) | ~99 (dynamic) |
| Supervised tokens/sample | 512 | 2 |
| Optimizer steps | 38,150 | **1,035** |
| Tokens processed | 78.1 M | ~1.6 M |

---

## 6. Held-out eval split

Previously `EVAL_STEPS` was defined but never used — nothing monitored
overfitting, and the last checkpoint was kept blindly.

- **Grouped by patient description**, not random. 1,300 descriptions appear in
  multiple pairs; a naive split would put the same text on both sides.
  *Verified:* 48,653 train / 2,000 eval, **0 description overlap**, deterministic.
- Metrics on the single answer token: `accuracy`, `recall`, `precision`, `f1`,
  `n_flagged`, `n_positive`. **Recall is the metric that matters** — at an 11%
  base rate, always answering `No` scores 89% accuracy.
  *Verified* against a hand-built case (TP=3, FN=1, FP=2, TN=4 → recall 0.75,
  precision 0.60, accuracy 0.70).
- `preprocess_logits_for_metrics` reduces logits to argmax before accumulation.
  Without it Trainer would concatenate a ~100 GB float tensor across the eval set.
- `load_best_model_at_end` on `eval_f1`.

### Also fixed

- **`overwrite_output_dir` was removed in transformers 5.x.** Pre-existing;
  Phase 2 would have crashed at `TrainingArguments` regardless. All 23
  `TrainingArguments` and 7 `Trainer` kwargs audited against the installed
  signature — everything else valid.
- **`temperature=0.0` in `generate()`** is invalid; it is only read when
  `do_sample=True` and 0.0 is out of range. Replaced with `do_sample=False`.
- KV cache re-enabled for validation (disabled during training for checkpointing).

---

## 7. Thermal — the remaining blocker

Timed benchmark at the final config. **This is now the dominant cost.**

```
step   1     5.0 s
step   7     5.0 s
step  13     5.0 s
step  19     5.0 s     <- thermal slowdown engages
step  22    20.0 s
step  25    21.0 s
```

| | Cold | Heat-soaked |
|---|---|---|
| Step time | 4.67 s | 20.40 s (**4.4× slower**) |
| SM clock | 675 MHz | 300 MHz (of 1590 max) |
| Temp | 47 → 74 °C | 89 – 95 °C |

Collapse takes **~100 seconds of sustained load**, then never recovers. It is a
cliff, not a slope.

T4 thresholds: max operating **85 °C**, slowdown **93 °C**, shutdown **96 °C**.
The benchmark reached 95 °C and was killed manually.

### Projected full run (1,035 steps + 6 evals)

- **Current cooling: ~9 h** — throttles within two minutes and stays there.
  Likely to hit the 96 °C shutdown partway through.
- **With adequate airflow: ~2 h** (1.3 h training + ~0.7 h eval).

Treat 2 h as an *upper bound*: even the "cold" measurement was at 675 MHz
against a 1590 MHz maximum, so the card was already limited before the hard
throttle.

### Cause and remedy

The T4 is a 70 W **passively cooled** datacenter card — no fan, just a bare
heatsink shroud that expects forced front-to-back airflow from a server
chassis. In a desktop case it gets almost none.

A blower ducted onto the shroud (3D-printed T4 fan adapter with a 40–50 mm
blower, or an 80 mm fan aimed through it) **buys back more than every software
optimization in this document combined.**

Note: the card is *not* faulty at rest — it idles at 43–47 °C. It just sheds
heat very slowly under a passive heatsink, so post-load readings stay high for
a long time.

---

## 8. Correction to an earlier estimate

Eval overhead was initially estimated at 20–25%. **Measured, it is ~50%** —
about 7 min per pass over 2,000 pairs on a healthy card, ×6 ≈ 42 min against
1.3 h of training.

Options (not yet applied):

| Change | Effect | Cost |
|---|---|---|
| `EVAL_SPLIT_PAIRS = 1000` | Eval → ~21 min (~26% overhead) | Recall noise ±4.6 pp vs ±3.2 pp |
| `EVAL_STEPS = 300` | Eval → ~21 min | Only 3 evaluations |

---

## 9. Final configuration

```python
# Model / quantization
MODEL_NAME                  = "meta-llama/Llama-3.1-8B"
USE_4BIT                    = True
BNB_4BIT_QUANT_TYPE         = "nf4"
BNB_4BIT_COMPUTE_DTYPE      = torch.float16   # Turing: no bf16
GRADIENT_CHECKPOINTING      = True

# LoRA
LORA_RANK                   = 16
LORA_ALPHA                  = 32
LORA_DROPOUT                = 0.05
target_modules              = ["q_proj", "v_proj"]

# Training
LEARNING_RATE               = 2e-4
NUM_EPOCHS                  = 1
BATCH_SIZE                  = 8
GRAD_ACCUMULATION_STEPS     = 2      # effective batch 16
WARMUP_STEPS                = 100
MAX_SEQ_LENGTH              = 128    # ceiling only; batches are ~99 wide
MASK_PROMPT_LABELS          = True
MAJORITY_CLASS_RATIO        = 2.0    # training split only
SEED                        = 42

# Evaluation
EVAL_SPLIT_PAIRS            = 2000
EXCLUDE_VALIDATION_PATIENTS = True
EVAL_STEPS  = SAVE_STEPS    = 150    # 6 evals over 1,035 steps
```

### Memory measurements (T4, 15.9 GB)

| Config | Peak VRAM |
|---|---|
| batch 8 × accum 2, checkpointing **off** | **OOM** (>14.8 GB) |
| batch 4 × accum 4, checkpointing off | 13.25 GB (2.6 GB spare) |
| batch 8 × accum 2, checkpointing **on** | **10.73 GB** ← shipped |

Checkpointing is on deliberately: a mid-run OOM hours into a job costs far more
than the ~30% recompute, especially on a thermally limited card.

> **Coupling to watch:** `EVAL_STEPS` is sized against the actual step count.
> If you change `MAJORITY_CLASS_RATIO`, `NUM_EPOCHS`, or the batch settings, the
> step count moves and `EVAL_STEPS` needs re-tuning — at the old value of 500
> you would get only 2 evaluations.

---

## 10. Status

**Verified:**

- GPU pinning (T4 only, `cuda:0`, allocation confirmed)
- 4-bit load — 5.59 GB footprint, no crash
- Training runs — 40 steps, loss 6.82 → 0.50, gradients healthy
  (the `nan` grad_norm on step 1 is just the fp16 loss-scaler calibrating)
- `generate()` works
- Split/balance logic, no leakage, deterministic
- Metric math
- Eval loop on GPU — 10.73 GB peak
- All notebook cells compile

**Not verified:**

- A complete end-to-end Phase 2 run
- Any final model quality number

**Before starting the real run:**

1. Fix T4 airflow — this is worth more than everything else here.
2. Restart the kernel; run the GPU SELECTION cell first.
3. Consider the eval-cost tradeoff in §8.
4. Clear the stale Jupyter kernel holding 6.2 GB on the RTX 2080.

Backup of the original notebook: `risk.ipynb.bak`
### STEP AND EVAL NOTES
Step 450, val loss 0.203738, acc 0.8775, recall 0.216667, precision 0.477064, F1 0.297994, 109 flagged of 2,000, 240 positive. Plus: rank 16 / alpha 32 / lr 2e-4 / batch 4 × 2 accum / 1 epoch / ratio 2.0 / seed 42 / 4-bit nf4 / validation patients excluded / prompt labels masked.