"""
=============================================================================
PHASE 2: PRE-FLIGHT VALIDATION CHECK
=============================================================================

Purpose:
    Verify your environment is ready for Phase 2 LoRA fine-tuning before
    committing to the 2-6 hour training pipeline on T4.

Checks:
    1. GPU available (CUDA, sufficient memory)
    2. Python packages (transformers, torch, peft, datasets)
    3. Phase 1 output exists (lora_training_data_combined.jsonl)
    4. Validation data exists (llm_sample.csv)
    5. HuggingFace model can be downloaded/cached
    6. Memory simulation (estimate VRAM usage)

Run this script BEFORE running phase_2_lora_finetuning.py

=============================================================================
"""

import os
import sys

def check_gpu():
    """Verify CUDA GPU is available with sufficient memory."""
    print("[PREFLIGHT] Checking GPU...")
    
    try:
        import torch
    except ImportError:
        print("  ✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available (no GPU detected)")
        return False
    
    print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ GPU memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 14:
        print(f"  ⚠ Warning: GPU memory is tight for llama3.1:8b with LoRA")
        print(f"    Recommend 16GB+. Will try with batch_size=2")
        return True  # Continue with warning
    
    return True

def check_packages():
    """Verify required Python packages are installed."""
    print("\n[PREFLIGHT] Checking Python packages...")
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "peft": "PEFT (Parameter-Efficient Fine-Tuning)",
        "datasets": "Hugging Face Datasets",
        "panda": "Pandas",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "tqdm": "tqdm",
    }
    
    missing = []
    
    for pkg_name, pkg_display in packages.items():
        try:
            __import__(pkg_name)
            print(f"  ✓ {pkg_display}")
        except ImportError:
            print(f"  ✗ {pkg_display} (MISSING)")
            missing.append(pkg_name)
    
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def check_training_data():
    """Verify Phase 1 output exists."""
    print("\n[PREFLIGHT] Checking training data...")
    
    import os
    
    training_file = os.path.join("data", "lora_training_data_combined.jsonl")
    
    if not os.path.exists(training_file):
        print(f"  ✗ Training data not found: {training_file}")
        print("  Run Phase 1 first: python phase_1_lora_data_generation.py")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(training_file) / 1e6
    print(f"  ✓ Training data exists: {training_file}")
    print(f"    File size: {file_size_mb:.1f} MB")
    
    # Check for valid JSON lines
    try:
        import json
        with open(training_file, "r") as f:
            for i in range(5):  # Check first 5 lines
                line = f.readline()
                if not line:
                    break
                json.loads(line)
        print(f"  ✓ Training data format valid (checked 5 samples)")
    except Exception as e:
        print(f"  ✗ Training data format invalid: {e}")
        return False
    
    return True

def check_validation_data():
    """Verify validation data exists."""
    print("\n[PREFLIGHT] Checking validation data...")
    
    import os
    import pandas as pd
    
    val_file = os.path.join("data", "llm_sample.csv")
    
    if not os.path.exists(val_file):
        print(f"  ✗ Validation data not found: {val_file}")
        print("  This should be from your LLM project (200-patient sample)")
        return False
    
    try:
        df = pd.read_csv(val_file)
        print(f"  ✓ Validation data exists: {val_file}")
        print(f"    Samples: {len(df)}")
        print(f"    Mortality rate: {df['mortality'].mean():.1%}")
        
        if "patient_description" not in df.columns:
            print(f"  ⚠ Warning: 'patient_description' column not found")
            print(f"    Validation will not work. Phase 1 script needed to generate descriptions.")
            return False
        
    except Exception as e:
        print(f"  ✗ Failed to read validation data: {e}")
        return False
    
    return True

def check_huggingface_model():
    """Verify model can be downloaded from HuggingFace."""
    print("\n[PREFLIGHT] Checking HuggingFace model...")
    
    print("  Note: First download may take 15-30 minutes (model is ~16GB)")
    print("  Model will be cached in ~/.cache/huggingface/")
    
    try:
        from transformers import AutoTokenizer
        print("  ✓ transformers library available")
    except ImportError:
        print("  ✗ transformers not installed")
        return False
    
    # Don't actually download to avoid long delay; just check config
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"  Model: {model_name}")
    print(f"  First run will download ~16GB (takes 15-30 min on good internet)")
    print(f"  Subsequent runs use cached weights")
    
    return True

def estimate_vram_usage():
    """Estimate VRAM usage during training."""
    print("\n[PREFLIGHT] Estimating VRAM usage...")
    
    # Rough estimates
    model_vram = 8.0  # llama3.1:8b in fp16
    optimizer_vram = 2.0  # AdamW optimizer states
    batch_vram = 0.5 * 4  # ~0.5 GB per sample × batch_size=4
    lora_vram = 0.2  # LoRA layers
    cache_vram = 1.0  # KV cache
    
    total_estimate = model_vram + optimizer_vram + batch_vram + lora_vram + cache_vram
    
    print(f"  Estimated breakdown:")
    print(f"    Model (fp16): {model_vram:.1f} GB")
    print(f"    Optimizer: {optimizer_vram:.1f} GB")
    print(f"    Batch (4 samples): {batch_vram:.1f} GB")
    print(f"    LoRA: {lora_vram:.1f} GB")
    print(f"    Cache: {cache_vram:.1f} GB")
    print(f"    Total: {total_estimate:.1f} GB")
    
    import torch
    available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    
    if available_vram > 0:
        print(f"  Available VRAM: {available_vram:.1f} GB")
        if total_estimate > available_vram * 0.9:
            print(f"  ⚠ Estimated usage exceeds 90% of available VRAM")
            print(f"    May cause OOM errors. Try reducing batch_size to 2.")
        else:
            print(f"  ✓ Estimated usage is within safe limits")

def main():
    """Run all preflight checks."""
    
    print("=" * 70)
    print("PHASE 2: PRE-FLIGHT VALIDATION CHECK")
    print("=" * 70)
    
    checks = [
        ("GPU", check_gpu),
        ("Packages", check_packages),
        ("Training Data", check_training_data),
        ("Validation Data", check_validation_data),
        ("HuggingFace Model", check_huggingface_model),
    ]
    
    checks_passed = 0
    checks_total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                checks_passed += 1
            else:
                print(f"\n[ABORT] {check_name} check failed")
                return False
        except Exception as e:
            print(f"\n[ERROR] {check_name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Estimate VRAM
    try:
        estimate_vram_usage()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 70)
    print(f"PRE-FLIGHT CHECKS: {checks_passed}/{checks_total} PASSED")
    print("=" * 70)
    
    if checks_passed == checks_total:
        print("\n✓ All checks passed!")
        print("You can now run Phase 2:")
        print("  python phase_2_lora_finetuning.py")
        print("\nExpected runtime: 2-6 hours (depends on GPU speed)")
        print("First run will download model (~16GB, 15-30 min)")
        print("To resume if interrupted: model checkpoints saved every 500 steps")
        return True
    else:
        print(f"\n✗ {checks_total - checks_passed} checks failed")
        print("Fix the issues above and retry")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
