# Can a low-rank adapter improve ICU mortality prediction?

Final project, AI in Healthcare. Fine-tunes **Llama-3.1-8B** with **QLoRA** to flag ICU
patients at high mortality risk, and compares it against the same untuned model
zero-shot on an identical held-out sample.

Everything here runs on a single **NVIDIA Tesla T4 (16 GB)**.

## Result

Held-out evaluation on 200 patients the adapter never saw during training:

| Model | Recall | Precision | Accuracy | F1 |
|---|---|---|---|---|
| **LoRA fine-tuned** | **0.455** | **0.556** | **0.900** | **0.500** |
| Zero-shot base | 0.182 | 0.108 | 0.745 | 0.136 |

Confusion matrix for the fine-tuned model (TN 170, FP 8, FN 12, TP 10). It flagged
18 of 200 patients as high risk against a true prevalence of 22.

Recall of 0.455 means the adapter still misses more than half of the patients who
died. The gain over zero-shot is large — precision improves roughly fivefold — but
the absolute numbers are well short of clinical usability, and the headline accuracy
of 0.900 is mostly a reflection of the 11% base rate rather than of skill.

## Repository layout

| Path | Role |
|---|---|
| `risk.ipynb` | **Canonical.** The end-to-end run that produced every number above. |
| `phase_1_lora_data_generation.py` | Builds instruction pairs from structured patient features. |
| `phase_2_setup_check.py` | Pre-flight validation: GPU, packages, data, VRAM estimate. |
| `contamination_probe.py` | Probes the tuned adapter for memorization of training text. |
| `phase_2_lora_finetuning.py` | **Superseded — see the warning below.** |
| `docs/figures/` | Confusion matrix, zero-shot vs. LoRA comparison, workflow diagram. |

### About `phase_2_lora_finetuning.py`

This file is an **earlier standalone draft of the Phase 2 pipeline and does not
reproduce the reported results.** It is kept for history. The authoritative Phase 2
implementation is the corresponding cell in `risk.ipynb`.

It differs from the canonical notebook in ways that matter:

| | `risk.ipynb` (canonical) | `phase_2_lora_finetuning.py` |
|---|---|---|
| Validation-patient exclusion | **yes** — drops the 213 leaking pairs | **absent** |
| 4-bit quantization (QLoRA) | NF4 + double quant | **absent** — loads fp16 |
| Class balancing | 2:1 | absent |
| Epochs | 1 | 3 |
| Gradient accumulation | 2 | 1 |
| Max sequence length | 128 | 512 |

The first row is the important one: run as written, the script trains on the same
200 patients it later evaluates on. The second row means it attempts to load ~16.1 GB
of fp16 weights onto a 16 GB card and will most likely exhaust memory first.

## Method

**Data.** Patient descriptions are generated from structured MIMIC-III-derived
features through a fully parameterized template — age, ethnicity, gender, insurance,
prior admissions, diagnosis count, ICU length of stay, and documented substance use.
No free-text notes are used. Phase 1 produced 50,866 instruction pairs, of which 5,790
(11.4%) are positive for mortality.

**Leakage control.** The 200-patient validation sample is held out at the source. Every
training pair whose input text matches one of those descriptions is dropped before the
train/eval split and before class balancing — 213 pairs in the reported run. The
remaining data splits 48,653 / 2,000 across 49,177 distinct patients with no description
spanning both sides. The training half is then balanced to 2:1 negative:positive,
yielding 16,572 pairs (5,524 positive, 11,048 negative).

This was not correct in an early version of the pipeline, where all 200 validation
descriptions appeared verbatim in training; `risk.ipynb` documents the fix and the
run log confirms the exclusion fired.

**Model.**

| Setting | Value |
|---|---|
| Base | `meta-llama/Llama-3.1-8B` |
| Quantization | 4-bit NF4, double quant, fp16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Trainable parameters | 6,815,744 (0.085% of 8.03B) |
| Epochs | 1 |
| Batch | 4 × 2 accumulation = 8 effective |
| Learning rate | 2e-4, 100 warmup steps |
| Max sequence length | 128 |
| Optimizer | `paged_adamw_8bit`, gradient checkpointing on |

fp16 rather than bf16 because the T4 is Turing (sm_75) and does not support bf16.
The run covers 2,071 optimizer steps over roughly 1.6M unpadded tokens; the log spans
2026-07-25 12:56 to 2026-07-26 11:36. Decoding is at temperature 0.0 throughout, for
both the fine-tuned adapter and the zero-shot baseline.

## Reproducing

Requires a CUDA GPU, and access to `meta-llama/Llama-3.1-8B`, which is gated on
Hugging Face.

```bash
cp .env.example .env      # then add your Hugging Face token
python phase_2_setup_check.py
```

Then run `risk.ipynb` top to bottom. The first cell pins the GPU by UUID and **must**
execute before anything imports `torch` — `CUDA_VISIBLE_DEVICES` is read once, when the
CUDA runtime initializes, and has no effect afterwards. Change the UUID to match your
own hardware. Phase 1 must complete before Phase 2, as Phase 2 consumes its output.

Phase 1 takes minutes on a modern CPU. Phase 2 took roughly a day on a T4.

## Data availability

**No patient data is included in this repository, and none will be.**

This project derives from MIMIC-III, which is distributed under the PhysioNet
Credentialed Health Data Use Agreement. That agreement permits sharing code and
aggregate results but prohibits redistributing the data or any individual-level
records derived from it. Accordingly, the following are excluded via `.gitignore`
and must be regenerated locally by a credentialed user:

- `data/` — features, generated descriptions, instruction pairs, predictions
- `checkpoint_storage/`, `data/lora_checkpoints/` — training checkpoints
- `data/lora_adapters/` — the trained adapter weights

The adapter weights are withheld deliberately, not merely for size. They are trained on
50,866 patient descriptions and could in principle carry memorized training records;
`contamination_probe.py` exists to test that. Access MIMIC-III through PhysioNet:
<https://physionet.org/content/mimiciii/>

All figures and metrics published here are aggregate.

## AI usage disclosure

Technical decisions, hardware troubleshooting, results interpretation, and final
judgments are the author's. AI assistance was used for code scaffolding, document
structure, slide generation, and narration, and for source discovery and citation
formatting — all sources were independently retrieved, read, and verified against the
publisher record. All prose, analysis, and argument are the author's own.

Full findings are reported in the accompanying paper and slide deck.
