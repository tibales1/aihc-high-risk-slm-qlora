# Presentation Script — "Fine-Tuning a Small Language Model for In-Hospital Mortality Prediction"

**Deck:** `docs/Phase2_Findings_v2.pptx` (11 slides)
**Target length:** ~7 minutes (assignment allows 5–10). Spoken at ~140 words/min.
**Covers the required beats:** Introduction · Method · Results · Future Directions · (optional Demo).

**How to use this:** each block is one slide. The *[SLIDE n]* line tells you when to advance.
Read the body aloud; the *(cue)* notes are stage directions, not spoken. To hit a hard 5-minute
limit, cut slides 6 and 7 to one sentence each — the argument still stands.

---

### [SLIDE 1 — Title]  (~30 sec)

Hi, I'm Thomas Bales, and this is my high-risk project for AI in Healthcare. The question I set
out to answer is a simple one to state and a hard one to earn: can a *small*, open, general
language model be taught to beat *its own untuned self* at a real clinical prediction task —
in-hospital mortality — using a tiny adapter, on a single GPU, cheaply enough that a hospital IT
department could actually do it? The short answer is yes. The interesting part is everything I
had to get right to be allowed to say that.

### [SLIDE 2 — The result, up front]  (~40 sec)

Here's the headline, on 200 patients the model never saw during training. The fine-tuned model
caught two-and-a-half times as many deaths as the base model — recall went from eighteen percent
to forty-six. And it was about five times more precise when it did raise a flag. Accuracy rose
from seventy-four to ninety percent. The reason I lead with this: it improved on *both* axes at
once. It didn't catch more deaths by crying wolf on everyone, and it didn't get precise by going
quiet. In an imbalanced problem like this, doing both at the same time is the hard thing.

### [SLIDE 3 — Confusion matrix]  (~45 sec)

This is where those numbers come from — the held-out confusion matrix. Of twenty-two real deaths,
the model caught ten and missed twelve. Of a hundred seventy-eight survivors, it raised only eight
false alarms. Now, twelve missed deaths is not a solved problem — I want to be honest about that.
But look at the *scale*: the model flagged eighteen patients total, against twenty-two true deaths.
It learned roughly the right number of people to worry about. That's a calibrated, usable operating
point — not a model hiding in the majority class, and not one that panics and flags everybody.

### [SLIDE 4 — Baseline vs. fine-tuned]  (~30 sec)

Same comparison, side by side, across every metric. Orange is the untuned base model at zero-shot;
blue is the same model with a point-oh-eight-percent adapter on top. This is an apples-to-apples
test — identical base model, identical 200 patients, both run deterministically. The only thing
that changed is the adapter, and the adapter moved every bar in the right direction.

### [SLIDE 5 — Why you can trust these numbers]  (~60 sec)

Now the part that matters most, and the part I could only get right by reading the actual codebase
rather than trusting the pipeline. When I inspected the data directly, I found three things.

First, and critically: every single one of the 200 validation patients was *also* in the training
set. If I'd left that, I'd have trained on those patients and then "tested" on the same patients —
measuring memorization, not learning. That one bug would have invalidated the entire headline. I
removed those patients and verified zero overlap.

Second, two hundred sixty-five patient descriptions carry *contradictory* labels — identical text,
opposite outcomes — because the features are coarse. That's a noise floor no model can beat.

Third, I split the data by patient, not at random, and I rebalanced only the training side, leaving
the evaluation set at its natural death rate so the numbers stay honest. The point: the result on
the previous slides is a real measurement precisely *because* of this slide.

### [SLIDE 6 — Making it run]  (~45 sec)

Briefly, the engineering, because "it ran on one GPU" hides a lot. Three bugs stood between me and
a single training step. The model was silently loading onto the wrong GPU — an eight-gigabyte card
instead of the sixteen — so I pinned it by hardware ID. The model didn't fit in memory in full
precision, so I quantized it to four bits, which dropped it from sixteen gigabytes to under six.
And the kernel kept crashing on load — which looked like out-of-memory but was actually a
library bug in how the model file was read. None of the three was the problem it first appeared to
be, and that's exactly why looking at the real failure mattered.

### [SLIDE 7 — How it learned]  (~45 sec)

The training itself told a story. It started by *collapsing* — refusing to flag anyone, hiding in
the eighty-eight-percent base rate. Then class-balancing pulled it off that floor and it started
catching deaths, but it swung wildly between aggressive and cautious. As the learning rate decayed,
those swings damped down, and by the end the flag count settled near the true death count. No
overfitting — training and validation loss fell together the whole way.

### [SLIDE 8 — The tradeoff is a values decision]  (~45 sec)

And that oscillation isn't a flaw — it's the most important finding. Here are two training steps
right next to each other. Same model weights. At one, recall is fifty-one percent — catch half the
deaths, accept a lot of false alarms. At the very next, recall is fifteen — flag almost nobody, but
rarely cry wolf. Same model, opposite behavior. Which one you *deploy* isn't a technical setting.
It's a clinical-values decision — how many false alarms are worth one more life caught — and that
belongs to an accountable institution, not to a default buried in someone's code.

### [SLIDE 9 — What it cost]  (~40 sec)

What did all this cost? One passively-cooled datacenter GPU, in a desktop, running two degrees
short of a thermal shutdown, stabilized with a thirty-dollar floor fan and a cardboard air duct.
Four-bit quantization got the footprint under six gigabytes. The real point is the asymmetry: the
enormous energy of this capability was spent *upstream*, in pretraining, by someone else — and I
inherited it for free. Fine-tuning to beat that base cost almost nothing. Who authorized that
upstream expenditure, on behalf of everyone now depending on it, is itself a governance question.

### [SLIDE 10 — What was proven, and its limit]  (~45 sec)

So, what was proven: a small custom adapter, on a small open model, can beat that model on a
specialized clinical task — cheaply, deterministically, auditably — by *unlocking* knowledge
already frozen in the weights, not adding it. And the honest limit: the ceiling is the data, not
the model. Those contradictory labels come from coarse features. The single highest-value next step
is richer features — raw labs, vital signs, diagnosis codes — to tell apart patients the current
descriptions collapse together.

### [SLIDE 11 — For the record & future directions]  (~40 sec)

For reproducibility, everything's on the left — the base model, the adapter settings, one epoch,
temperature zero. That last one matters: deterministic inference means the same patient always gets
the same assessment, which is what makes the system auditable in the first place.

Going forward: richer features to lower that noise floor; reporting the full recall–precision
frontier instead of one point; and an ensemble check against my structured baselines to see whether
the text model misses *different* patients — if it does, they're complementary. If I did it again,
I'd save per-step results from the start and fix the GPU cooling *before* the run, not during it.

Thank you — I'm happy to take questions.

---

## Timing summary

| Slide | Topic | Approx. |
|---|---|---|
| 1 | Title / the question | 0:30 |
| 2 | Headline result | 0:40 |
| 3 | Confusion matrix | 0:45 |
| 4 | Baseline vs. fine-tuned | 0:30 |
| 5 | Data integrity (trust) | 1:00 |
| 6 | Systems / three bugs | 0:45 |
| 7 | Training arc | 0:45 |
| 8 | Values decision | 0:45 |
| 9 | Cost | 0:40 |
| 10 | Proven & limit | 0:45 |
| 11 | Record & future | 0:40 |
| — | **Total** | **~7:15** |

## Recording tips
- **PowerPoint** has built-in recording: *Slide Show → Record* — captures your voice per slide and
  exports to video (*File → Export → Create a Video*). This is the least-friction path.
- Do a quick mic check; record slide 5 first as a warm-up since it's the densest.
- Export at 1080p; label the file with your last name and first initial as the assignment requires.
