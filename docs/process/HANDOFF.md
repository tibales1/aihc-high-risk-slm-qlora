# HANDOFF — pick up here next session

**Written:** 2026-07-23 (evening)
**Deadline:** 2026-08-10 (~18 days, solo)
**One-line status:** Software is done and verified. The only blocker was heat. Cooler ordered — resume when it arrives.

---

## START HERE TOMORROW

The cooling part (see below) should arrive ~2026-07-24 (Amazon Prime). When it does:

1. **Install the cooler.** Noctua NF-A4x20 fan clips into the 40 mm adapter; adapter
   clips onto the T4 shroud so it blows *through* the fin tunnel (front-to-back).
   Power the fan from a **motherboard fan header** (or a Molex-to-fan adapter). The
   T4 has no fan header of its own.
2. **Restart the Jupyter kernel** and run the **GPU SELECTION cell (cell 0) FIRST.**
   An already-running kernel has frozen CUDA state and cannot be repointed to the T4.
3. **Ask me to run the monitored thermal burst** — a ~70 W worst-case load on the T4
   with a watchdog that hard-kills at **85 °C**, watching the ~100 s heat-soak window
   where it collapsed before. This confirms the cooling holds *before* committing to
   the full run. Nothing can shut the PC off — the watchdog stops well short of 96 °C.
4. **If it holds → kick off the full Phase 2 run.** ~2 h on a healthy T4. Checkpoints
   every 150 steps (resume-safe). Watch the first ~2 min for the heat-soak cliff.

If you want to start EARLIER (tonight) on the 2080-fan stopgap: the T4 idles at 56 °C
with the 2080's fans aimed at it. Ask me to run the same watchdog'd burst now — worst
case it stops at 85 °C and nothing crashes.

---

## What happened (2026-07-23)

- Started the QLoRA training run. The **T4 hit 97 °C in 3-4 minutes and the whole
  machine hard-shut-off**, needing 2 reboots to come back (card likely dropped off
  the PCIe bus).
- This is exactly the thermal cliff predicted in `PHASE2_FINDINGS.md` §7 — except it
  reached full thermal shutdown, not just throttle.
- **Not a software problem.** Every fix in `PHASE2_FINDINGS.md` is already in
  `risk.ipynb` and verified (GPU pinning, 4-bit QLoRA, mmap→pread crash fix, data
  leakage fix, masked-label training, held-out eval split). See that file for detail.

## Root cause & fix

- The T4 is a **70 W passively cooled datacenter card** — bare heatsink, no fan,
  expects forced front-to-back airflow it never gets in a desktop case.
- **Ordered (Amazon, Prime ~2026-07-24):**
  - **Noctua NF-A4x20 PWM** — 40×40×20 mm, high static pressure (the `x20` thickness
    matters; a thin fan stalls against the dense fins).
  - **2-pack 40 mm Tesla P4/T4 fan adapter** ("Low Profile 75 W PCIe Powered GPUs,
    fits 40 mm fans" — matches the T4: low-profile, single-slot, 70 W slot-powered).
- Interim stopgap in place: 2080's fans aimed at the T4 dropped idle **69 °C → 56 °C**
  (baseline was 43-47 °C, so still marginal — hence the pre-run burst test).

---

## Hardware facts (for the GPU-selection cell & monitoring)

| Thing | Value |
|---|---|
| T4 UUID (pinned in cell 0) | `GPU-5efc6340-9a1d-0768-7857-fd704b7433e2` |
| 2080 UUID | `GPU-201d5193-956c-ff92-7105-69e66741c839` |
| After cell 0, T4 is | the ONLY visible GPU = `cuda:0` |
| T4 power limit | fixed 70 W, floor 60 W (cap buys ≤14% heat — not a rescue) |
| T4 thermal thresholds | 85 °C max-op · 93 °C slowdown · **96 °C shutdown** |
| Watchdog kill temp for tests | **85 °C** |

---

## Notebook state (`risk.ipynb`, 5 cells)

- **Cell 0** — GPU SELECTION. Must run first. Pins T4 by UUID. ✅ in place.
- **Cell 4** — Phase 2 LoRA fine-tuning. All fixes applied. ✅
- Backup of pre-fix notebook: `risk.ipynb.bak`
- Final config reference: `PHASE2_FINDINGS.md` §9 (batch 8 × accum 2, grad-checkpointing
  on → 10.73 GB peak; 1 epoch; masked labels; EXCLUDE_VALIDATION_PATIENTS=True;
  EVAL_STEPS=150 → 6 evals over ~1,035 steps).

### Pre-run checklist (from findings §10)
- [ ] Cooler installed + fan spinning
- [ ] Kernel restarted, cell 0 run first
- [ ] Thermal burst passed (holds <85 °C through heat-soak)
- [ ] Clear any stale Jupyter kernel holding memory on the 2080
- [ ] (Optional) eval-cost tradeoff, findings §8 — drop EVAL_SPLIT_PAIRS to 1000 to
      cut eval overhead from ~50% to ~26% if runtime is tight

### Not yet verified
- A complete end-to-end Phase 2 run
- Any final model quality number

---

## Parallel work that does NOT need the T4 (can do anytime)

- **Version 3 — RAG clinical persona.** Embedding + retrieval on CPU/nomic. Plan and
  topic lore already drafted in `risk_project_plan.md`. Buildable now.
- **Related Work.** Find + read the 2-3 papers the ACM report needs (clinical LLM
  fine-tuning, RAG for clinical decision support, structured ML vs LLM on EHR).
- **Report scaffold.** ACM structure + known numbers already in hand (base model:
  74.5% acc / 18.2% recall; structured ML: GBM 0.757, DNN 0.754, Embed+LogReg 0.743).

---

## Project recap (three versions of llama3.1:8b on MIMIC-III mortality)

1. **Base** (zero-shot/CoT/ToT) — DONE, can't discriminate. The "before."
2. **LoRA fine-tuned** — the high-risk attempt. ← currently blocked on the run above.
3. **RAG + clinical persona** — not started, GPU-light.

Then compare all three against structured ML baselines. Rubric explicitly rewards
honest failure, so a completed run with any result is a win.
