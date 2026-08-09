# Can a low-rank adapter improve ICU mortality prediction?

Final project, AI in Healthcare. Fine-tunes **Llama-3.1-8B** with **QLoRA** to flag ICU
patients at high mortality risk, and compares it against the same untuned model
zero-shot on an identical held-out sample.

Everything here runs **locally** on a single **NVIDIA Tesla T4 (16 GB)** — no cloud
compute, on a 60 W power budget.

## On "high-risk"

The name refers to the assignment, not to patient risk stratification. The course
framing was to attempt something ambitious and account for it honestly — falling short
of a good number costs nothing, provided the attempt and the reporting are real.

That is why this README dwells on what broke. A recall of 0.455 is a weak classifier and
a genuine result at the same time: the substance is that an 8B model, quantized to 4 bits
on a passively cooled datacenter card wedged into a desktop, was made to beat its own
zero-shot baseline on every metric at once — and that a validation leak was caught before
it could invalidate the claim. The engineering account of the walls hit along the way is
in [`docs/process/CONSTRAINTS_NARRATIVE.md`](docs/process/CONSTRAINTS_NARRATIVE.md).

## Result

Held-out evaluation on 200 patients the adapter never saw during training:

| Model | Recall | Precision | Accuracy | F1 |
|---|---|---|---|---|
| **LoRA fine-tuned** | **0.455** | **0.556** | **0.900** | **0.500** |
| Zero-shot base | 0.182 | 0.108 | 0.745 | 0.136 |

Confusion matrix for the fine-tuned model (TN 170, FP 8, FN 12, TP 10). It flagged
18 of 200 patients as high risk against a true prevalence of 22.

Accuracy is close to uninformative here — answering "No" to all 200 already scores
89%. What moved is the composition: **10 deaths caught for 8 false alarms, against
the baseline's 4 caught for 33.** The adapter cut false alarms fourfold while catching
two and a half times as many deaths, so it did not buy its precision by retreating into
the majority class. Flagging 18 against a true count of 22 says it learned the scale of
the problem rather than its safest answer. Specificity rose from 81.5% to 95.5%.

The limits are real all the same. Recall of 0.455 still misses more than half the
patients who died, which is well short of clinical usability for a screening tool. And
22 positives is a small denominator: 10 of 22 carries a 95% interval of roughly
0.27–0.65, so the direction is clear but the point estimate is soft.

## Local compute, on a power budget

Nothing here ran in a cloud. The whole fine-tune executed on one Tesla T4 in a desktop —
a **70 W passively cooled datacenter card with no fan of its own**, power-capped to
**60 W** for the production run. Energy at the card came to roughly **1.4 kWh**: the
60 W cap across about 23 hours of wall clock, which is an upper bound, since the card
spent much of that throttled below the cap. Host system draw is not included.

Running inside that budget drove most of the technical decisions:

- **4-bit NF4 quantization was the price of admission, not an optimization.**
  Llama-3.1-8B needs ~16.1 GB in fp16 against the card's 15.9 GB — it does not fit before
  a single gradient is computed. Quantized, the weights occupy 5.59 GB. Peak usage
  measured 10.73 GB at micro-batch 8; the run as executed used micro-batch 4, since
  halving it was one of the thermal interventions below, so its true peak sat lower and
  was not separately measured.
- **Only 6.8M of 8.03B parameters were ever trained** (0.085%). Everything else stayed
  frozen and quantized.
- **Eliminating waste became a question of feasibility, not speed.** Padding every
  ~97-token sample out to 512 spent 5.2× of the compute on padding, and computing loss
  over the whole prompt taught the model to regurgitate its own instructions. Dynamic
  per-batch padding and masking the loss to the single answer token brought the run down
  to 1.6M tokens processed. On a thermally limited card, every wasted token is wasted
  heat.
- **The binding constraint was thermal, not computational.** The card throttles 4.4×
  within roughly 100 seconds of sustained load, and on an earlier attempt hit 97 °C and
  cut power to the whole machine — it still read above 69 °C twenty minutes later, with
  the computer switched off. A passive heatsink in a desktop has nowhere to put the heat
  even at zero load. What finally made the run possible was improvised: a 40 mm fan
  behind a **cardboard duct** over the leading edge of the fin stack, the case opened,
  and a floor fan aimed into it, plus the 60 W cap and a smaller batch. That held a
  stable 94 °C for the full run — roughly 2 °C under the hardware shutdown threshold.

This is what the report's closing argument rests on: the large energy cost of this
capability was spent upstream during pretraining and inherited here for free. Beating
that base model cost about a kilowatt-hour and $30 of hardware.

## Repository layout

| Path | Role |
|---|---|
| `risk.ipynb` | **Canonical.** The end-to-end run that produced every number above. |
| `phase_1_lora_data_generation.py` | Builds instruction pairs from structured patient features. |
| `phase_2_setup_check.py` | Pre-flight validation: GPU, packages, data, VRAM estimate. |
| `contamination_probe.py` | Probes the tuned adapter for memorization of training text. |
| `phase_2_lora_finetuning.py` | **Superseded — see the warning below.** |
| `docs/` | Report, slides, and engineering write-ups — see [Documentation](#documentation). |
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

## Documentation

| Document | What it is |
|---|---|
| [`docs/report/Phase2_Report.pdf`](docs/report/Phase2_Report.pdf) | The write-up, ACM `sigconf` format. Source in `Phase2_Report.tex`; compiles on Overleaf as-is against `docs/report/figures/`. |
| [`docs/Phase2_Findings_v2.pptx`](docs/Phase2_Findings_v2.pptx) | Presentation slides, with speaker notes in `Phase2_Presentation_Script.md`. |
| [`docs/PHASE2_FINDINGS.md`](docs/PHASE2_FINDINGS.md) | Engineering log for Phase 2 — the GPU-selection bug, the fp16 memory overrun, a `transformers` mmap crash, the leakage discovery, and the thermal ceiling. Written before the final run, so its projected step counts and batch sizes reflect a plan that was revised; the notebook and this README carry what actually ran. |
| [`docs/PHASE_1_README.md`](docs/PHASE_1_README.md), [`docs/PHASE_2_README.md`](docs/PHASE_2_README.md) | Per-phase pipeline documentation. |
| [`docs/process/CONSTRAINTS_NARRATIVE.md`](docs/process/CONSTRAINTS_NARRATIVE.md) | The engineering narrative the report draws on — why the constraints are the result. Source material rather than a deliverable, but it carries the argument. |
| [`docs/process/HANDOFF.md`](docs/process/HANDOFF.md) | A mid-project session note, kept for provenance. |

Two example patient descriptions in `PHASE_1_README.md` were redacted before publication; they were rendered from real MIMIC-III admissions and paired with their mortality outcomes. The description template that produced them is in [`phase_1_lora_data_generation.py`](phase_1_lora_data_generation.py).

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

Full findings are reported in [`docs/report/Phase2_Report.pdf`](docs/report/Phase2_Report.pdf).
