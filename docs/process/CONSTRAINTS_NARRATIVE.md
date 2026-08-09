# Fine-Tuning at the Edge of the Hardware
### The constraint story behind the LoRA experiment

**Project:** Three Approaches to Clinical Mortality Prediction with llama3.1:8b
**Author-facing note:** This is report/slide source material — the engineering narrative
to draw from for Methods, Limitations, and the presentation. Technical detail lives in
`PHASE2_FINDINGS.md`; this file is the *story* those findings tell.

---

## Why the struggle is the result

The rubric is explicit: *"Even if you fail miserably and produce bad results, as long as
you have tried hard, you can still get a high score."* This project takes that seriously.
The interesting contribution was never going to be a clean AUC number that beats a
gradient-boosted baseline — a laptop can do that. The interesting contribution is what it
actually takes to fine-tune an 8-billion-parameter language model on clinical data using
**a 16 GB passively-cooled datacenter card wedged into a desktop, maxed out on every axis
at once.** Every wall I hit is a finding. This document is the honest record of those walls.

---

## The constraint stack

Everything below was operating at or beyond its limit simultaneously:

| Constraint | The limit | Why it bit |
|---|---|---|
| **VRAM** | T4 has 15.9 GB | Llama-3.1-8B is 16.1 GB in fp16 — it does not fit *before* a single gradient |
| **Compute arch** | T4 is Turing (sm_75) | No bf16; stuck on fp16 with its narrower dynamic range |
| **Thermals** | Passive 70 W card, no fan | Melts itself under sustained load in a desktop case |
| **Two-GPU routing** | 2080 (8 GB) is `cuda:0` | The framework kept sending the job to the *smaller* card |
| **Toolchain** | transformers 5.x on Windows | A hardcoded I/O backend crashed the kernel with no traceback |
| **Data** | Coarse feature-derived text | Validation leaked into training; identical inputs, opposite labels |
| **Time** | ~18 days, solo | No room to brute-force; every wrong turn is expensive |

The point of the project is not that these are unusual problems. It is that they are the
*normal* problems, and they all showed up together.

---

## The walls, in order

Each of these was a distinct problem with a distinct diagnosis. None was the one I expected.

### 1. The model doesn't fit — 8B into 16 GB
Llama-3.1-8B is **16.1 GB in fp16**, against the T4's 15.9 GB. It could never have loaded,
let alone trained. The naive preflight check reported "8.0 GB — within safe limits," off by
2×, and waved it through.
**Fix:** 4-bit QLoRA (NF4, double quantization). Weights drop to **5.59 GB**; only
**6.8 M of 8.0 B parameters (0.085%) are trainable.** The whole model plus optimizer plus a
batch now peaks at **10.73 GB** — with room to spare on the same card that couldn't hold the
weights alone.
**Lesson:** Quantization isn't an optimization here, it's the price of admission. Without it
there is no experiment.

### 2. The job kept running on the wrong GPU
The symptom was *"I can't access my T4, it keeps going to the 2080."* The cause was three
bugs stacked: a stray `set_device(1)` that initialized CUDA and froze `CUDA_VISIBLE_DEVICES`;
`device_map` ignoring the default device anyway; and `"auto"` filling from `cuda:0` — the
8 GB 2080.
**Fix:** Pin the T4 by **UUID** (not index — index reorders on driver whim) in the very first
cell, before torch is ever imported, with an assertion that fails loudly if run too late.
**Lesson:** On a multi-GPU box, *which card runs your job* is not automatic and not obvious.
You have to seize it explicitly, early, and by a name that can't be silently reassigned.

### 3. The kernel crash that wasn't out-of-memory
Loading stalled at 58% and the kernel died — no traceback, just *"the kernel crashed."* It
looked exactly like OOM, but it happened on the T4 needing only 5.5 GB, with GB to spare.
`faulthandler` caught the truth: a **Windows access violation (0xC0000005)** inside the
safetensors loader. transformers 5.x **hardcodes an mmap backend** on every non-Mac platform,
and mmap-backed reads of the 16 GB checkpoint fault on this machine.
**Fix:** Monkeypatch the loader to use the `pread` backend — ordinary file reads instead of
memory mapping.
**Lesson:** "The kernel crashed" is not a diagnosis. A whole day can hide between a symptom
that says *out of memory* and a cause that says *your library's default file I/O is broken on
your OS.* Ruling things out tensor-by-tensor is the job.

### 4. The data was quietly lying
Two problems, both invisible until I inspected the Phase 1 output directly:
- **All 200/200 validation patients were also in the training set.** As written, the
  experiment would have trained on the test patients, then reported "accuracy" on them, and
  compared it against a baseline that never saw them. That measures *memorization*, and it
  would have invalidated the headline result.
- **265 descriptions carry both `Yes` and `No` labels.** The features are coarse enough that
  different admissions with different outcomes render as identical text. This is an
  irreducible noise floor — a hard ceiling on achievable accuracy that no training removes.
**Lesson:** The most dangerous bug produces a *believable* number. Data leakage doesn't crash
anything; it just hands you a result you'd have been happy to report.

### 5. Training that mostly taught the model nothing
The original setup padded every ~97-token sample to 512 (**5.2× of compute spent on padding**),
computed loss over the whole prompt (teaching the model to regurgitate its own instructions),
and ran 3 epochs of an 88.6%-negative dataset — a recipe for a model that just answers "No"
to everything, the exact failure of the 18.2%-recall baseline it was meant to beat.
**Fix:** Dynamic padding (batches ~99 wide, not 512), **loss masked to the answer token only**,
class rebalancing on the *training split only*, 1 epoch. Optimizer steps fell from
**38,150 → 1,035**; tokens processed from **78.1 M → 1.6 M** — the same learning signal, an
order of magnitude less waste.
**Lesson:** On a thermally-limited card, *every wasted token is wasted heat.* Efficiency
stopped being about speed and became about whether the run can physically finish.

---

## The thermal wall — the one that actually stopped everything

All of the above was software, and all of it was solved. Then the real blocker showed up, and
it wasn't software at all.

I started the real run. **Within 3-4 minutes the T4 hit 97 °C and the entire machine cut
power** — a hard shutdown that took two reboots to recover from, the card having dropped off
the PCIe bus.

A timed benchmark tells the story exactly:

```
step  1     4.67 s      47 °C     675 MHz
step 19     5.0 s       ~74 °C            <- thermal slowdown engages
step 22    20.0 s
step 25    21.0 s       89-95 °C   300 MHz
```

- **Cold:** 4.67 s/step, 675 MHz. **Heat-soaked:** 20.40 s/step, 300 MHz — **4.4× slower.**
- Collapse takes **~100 seconds of sustained load, then never recovers.** It is a cliff, not
  a slope.
- T4 limits: 85 °C max-operating, 93 °C slowdown, **96 °C shutdown.** The run blew straight
  past all three.

The cause is physical and a little absurd: the T4 is a **70 W passively-cooled server card** —
a bare heatsink in a shroud, designed for a wall of chassis fans ramming air front-to-back
through the fins. In a desktop it gets almost none. It idles fine at 43-47 °C, then buries
itself the moment you ask it to work, because it sheds heat far slower than it makes it.

**The fix costs about $30 and undoes the single biggest bottleneck in the entire project:** a
40 mm high-static-pressure fan (Noctua NF-A4x20) in a printed adapter, ducted onto the shroud
to force air *through* the fin tunnel the way the card was designed for. A stopgap — the
neighboring 2080's fans aimed across it — already pulled idle from 69 °C back to 56 °C.

**Projected runtime, same software, same model:**
- Current passive cooling: **~9 h, throttled, likely to hit the shutdown partway through.**
- With adequate airflow: **~2 h.**

A $30 fan buys back more than every software optimization in this project combined.

**Lesson:** You can do everything right in software — quantize the model, pin the right GPU,
patch the crashing loader, clean the data, mask the labels, strip the waste — and still be
stopped cold by a heatsink with no air moving over it. The bottleneck is wherever the system
is actually maxed out, and it is not always in the code.

---

## What this experiment is really about

Read one way, this is a mortality-prediction study. Read honestly, it is a case study in
**applied machine learning under simultaneous, real constraints** — the version of the work
that doesn't appear in tutorials, where the model doesn't fit, the framework crashes, the data
lies, the wrong chip runs the job, and the right chip cooks itself.

Every workaround here is a small lesson in the gap between "fine-tuning an LLM" as a sentence
and as an afternoon. The clinical result, whatever it turns out to be, sits on top of that.
And per the rubric, the honest account of the walls — not a polished number — is the point.

---

## For the report and the slides

Pull directly from here:

- **Methods →** the constraint stack table + the QLoRA / GPU-pinning / data-integrity fixes.
- **Limitations →** the conflicting-label noise floor (§ wall 4), coarse features, single-center.
- **A results or discussion figure →** the thermal cliff (step-time vs temperature). It is the
  most vivid single image in the project: a 4.4× collapse in 100 seconds.
- **The presentation hook →** "I tried to fine-tune an 8-billion-parameter model on a card that
  couldn't hold it, on a chip that set my job to the wrong GPU, using a library that crashed my
  kernel — and then the card hit 97 °C and shut my whole computer off. Here's what each of those
  taught me." That opening earns attention honestly.

Candidate title for this section of the writeup:
**"Fine-Tuning at the Edge of the Hardware: What an 8B Model on a 16 GB Passive Card Actually Costs."**
