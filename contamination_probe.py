#!/usr/bin/env python3
"""
contamination_probe.py

Two probes, run against the SAME quantized base model used for LoRA training:

  PROBE 1 -- Verbatim memorization
      Seed the model with a near-empty prompt at greedy decoding and record
      what it emits unprompted. If it produces intact medical exam items,
      that text was memorized during pretraining.

  PROBE 2 -- Held-out contamination check
      Ask whether the specific MIMIC descriptions in the held-out split
      show signs of having been seen in pretraining, via (a) prefix
      continuation and (b) perplexity against shuffled controls.

Run on the base model only -- NOT the LoRA-adapted model. The question is
what the pretrained weights already contain.

Usage:
    python contamination_probe.py --heldout heldout_200.jsonl --out probe_results.json

The held-out file should be JSONL with one object per line containing a
"text" field holding the patient description as fed to the model.
"""

# GPU selection must happen before torch is imported -- CUDA reads these at
# first initialization and ignores later changes. This box has two GPUs
# (RTX 2080 + Tesla T4); the UUID pins us to the T4 regardless of enumeration
# order.
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    "GPU-5efc6340-9a1d-0768-7857-fd704b7433e2",  # Tesla T4
)

# Reduces allocator fragmentation, which matters here because training runs at
# ~10.7 GB of the T4's 15.9 GB and eval allocates a large logits tensor in bursts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

assert "torch" not in sys.modules, (
    "torch was imported before CUDA_VISIBLE_DEVICES was set -- the GPU pin is "
    "being ignored. Keep this block at the top of the file."
)

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL_ID = "meta-llama/Llama-3.1-8B"

# Minimal seeds. The point is that NOTHING here specifies medicine --
# any medical output is coming from the weights, not the prompt.
SEEDS = ["A", "A 25", "A 25-year-old", "Question:", "1.", "Answer:"]


def check_gpu():
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. Check that CUDA_VISIBLE_DEVICES matches a "
            "GPU UUID reported by `nvidia-smi -L`.")
    n = torch.cuda.device_count()
    assert n == 1, f"Expected 1 visible GPU, saw {n}"
    print(f"Visible GPU (cuda:0): {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def load_model(precision="auto"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"GPU: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
        if free < 7e9:
            log("WARNING: under 7 GB free -- another process likely holds "
                "the GPU. Shut down other kernels before continuing.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},              # pin to GPU, skip auto-planning
        low_cpu_mem_usage=True,          # stream shards, don't stage in RAM
        max_memory={0: "14GiB", "cpu": "8GiB"},
    )
    model.eval()
    return tok, model, "4-bit NF4, double quant, fp16 compute"


# ------------------------------------------------------------------ probe 1
def probe_memorization(tok, model, max_new=400):
    """Greedy generation from minimal seeds."""
    results = []
    for seed in SEEDS:
        ids = tok(seed, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=False,          # greedy == temperature 0.0
                num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        completion = text[len(seed):]
        results.append({
            "seed": seed,
            "completion": completion,
            "sha256": hashlib.sha256(completion.encode()).hexdigest()[:16],
            "looks_clinical": _clinical_markers(completion),
        })
        log(f"seed {seed!r}: {len(completion)} chars, "
            f"markers={results[-1]['looks_clinical']}")
    return results


def _clinical_markers(text):
    """Crude flag for medical-exam structure. Reported, not relied on."""
    t = text.lower()
    markers = ["mg/dl", "mm hg", "presents to the emergency",
               "physical examination", "most appropriate next step",
               "year-old", "laboratory results", "answer:"]
    return [m for m in markers if m in t]


# ------------------------------------------------------------------ probe 2
def probe_prefix_continuation(tok, model, texts, prefix_frac=0.4, n=50):
    """
    Feed the first prefix_frac of a held-out description and greedily
    continue. High token overlap with the true remainder is a memorization
    signal; low overlap is evidence the record was not in pretraining.
    """
    scores = []
    for text in texts[:n]:
        ids = tok(text, return_tensors="pt").input_ids[0]
        if len(ids) < 40:
            continue
        cut = int(len(ids) * prefix_frac)
        prefix, truth = ids[:cut], ids[cut:]
        with torch.no_grad():
            out = model.generate(
                prefix.unsqueeze(0).to(model.device),
                max_new_tokens=len(truth),
                do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[0][cut:]
        m = min(len(gen), len(truth))
        exact = (gen[:m].cpu() == truth[:m].cpu()).float().mean().item()
        scores.append(exact)
    return {
        "n_evaluated": len(scores),
        "mean_token_match": sum(scores) / len(scores) if scores else None,
        "max_token_match": max(scores) if scores else None,
        "note": ("Token match near chance indicates the record was not "
                 "memorized. Values approaching 1.0 indicate verbatim recall."),
    }


def perplexity(tok, model, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return torch.exp(loss).item()


def probe_perplexity_gap(tok, model, texts, n=50, seed=0):
    """
    Compare perplexity of real held-out descriptions against controls built
    by shuffling the sentence order within each description. Real text that
    was memorized should score markedly lower than its own shuffle. A small
    gap means the model finds these no more familiar than scrambled text.
    """
    rng = random.Random(seed)
    real, ctrl = [], []
    for text in texts[:n]:
        parts = [s for s in text.split(". ") if s.strip()]
        if len(parts) < 3:
            continue
        shuffled = parts[:]
        rng.shuffle(shuffled)
        real.append(perplexity(tok, model, text))
        ctrl.append(perplexity(tok, model, ". ".join(shuffled)))
    if not real:
        return {"error": "no usable descriptions"}
    mr, mc = sum(real) / len(real), sum(ctrl) / len(ctrl)
    return {
        "n": len(real),
        "mean_ppl_real": round(mr, 3),
        "mean_ppl_shuffled": round(mc, 3),
        "ratio": round(mr / mc, 4),
        "note": ("Ratio near 1.0 means real records are no more familiar "
                 "than scrambled ones -- evidence against memorization."),
    }


# ------------------------------------------------------------------ main
def load_texts(path, field=None):
    """
    Accept several shapes so you don't have to reformat first:
      - .jsonl : one JSON object per line
      - .json  : a list of objects, or a list of strings
      - .csv   : uses the named --field column
      - .txt   : one description per line
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    candidates = ["text", "description", "prompt", "input", "patient_text"]

    def pick(obj):
        if isinstance(obj, str):
            return obj
        if field and field in obj:
            return obj[field]
        for c in candidates:
            if c in obj:
                return obj[c]
        raise KeyError(
            f"no text field found; keys are {list(obj)[:10]}. "
            f"Re-run with --field <name>.")

    if ext == ".jsonl":
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(pick(json.loads(line)))
        return out
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next(v for v in data.values() if isinstance(v, list))
        return [pick(o) for o in data]
    if ext == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return [pick(r) for r in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=None,
                    help="held-out descriptions (.jsonl/.json/.csv/.txt). "
                         "Omit to run the memorization probe only.")
    ap.add_argument("--field", default=None,
                    help="name of the text column/key, if autodetect fails")
    ap.add_argument("--out", default="probe_results.json")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    check_gpu()

    texts = []
    if args.heldout:
        try:
            texts = load_texts(args.heldout, args.field)
            log(f"loaded {len(texts)} held-out descriptions")
        except FileNotFoundError:
            log(f"WARNING: {args.heldout} not found -- "
                f"running memorization probe only")
        except KeyError as e:
            log(f"WARNING: {e}")
    else:
        log("no --heldout given; running memorization probe only")

    tok, model, quant_desc = load_model()
    log(f"model loaded ({quant_desc})")

    results = {
        "model": MODEL_ID,
        "quantization": quant_desc,
        "decoding": "greedy (do_sample=False, num_beams=1)",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "probe_1_memorization": probe_memorization(tok, model),
    }
    if texts:
        results["probe_2_prefix_continuation"] = \
            probe_prefix_continuation(tok, model, texts, n=args.n)
        results["probe_2_perplexity_gap"] = \
            probe_perplexity_gap(tok, model, texts, n=args.n)
    else:
        results["probe_2_skipped"] = "no held-out data supplied"

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {args.out}")

    print("\n--- summary ---")
    for r in results["probe_1_memorization"]:
        print(f"seed {r['seed']!r:16} markers: {r['looks_clinical']}")
    if "probe_2_prefix_continuation" in results:
        print(json.dumps(results["probe_2_prefix_continuation"], indent=2))
        print(json.dumps(results["probe_2_perplexity_gap"], indent=2))


if __name__ == "__main__":
    main()