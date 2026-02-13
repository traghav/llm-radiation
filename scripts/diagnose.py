"""Diagnostic script to investigate the sharp performance drop."""

import torch
import numpy as np
from llm_radiation.models.registry import load_model, snapshot_weights, restore_weights
from llm_radiation.bitflip.engine import BitFlipEngine


def check_nan_inf(model, label=""):
    """Count NaN and Inf values across all parameters."""
    total_params = 0
    nan_count = 0
    inf_count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            flat = param.data.float()
            total_params += flat.numel()
            nans = torch.isnan(flat).sum().item()
            infs = torch.isinf(flat).sum().item()
            nan_count += nans
            inf_count += infs
            if nans > 0 or infs > 0:
                print(f"  {label} {name}: {nans} NaN, {infs} Inf (of {flat.numel()})")
    return nan_count, inf_count, total_params


def check_restore(model, snapshot):
    """Verify restore_weights actually restores all parameters."""
    max_diff = 0.0
    mismatched = 0
    for name, param in model.named_parameters():
        if name in snapshot:
            restored = snapshot[name].to(device=param.device, dtype=param.dtype)
            diff = (param.data - restored).abs().max().item()
            if diff > 0:
                mismatched += 1
                max_diff = max(max_diff, diff)
                print(f"  MISMATCH {name}: max_diff={diff:.6e}")
    return mismatched, max_diff


def run_quick_generation(model, tokenizer, label=""):
    """Quick generation test to see if model produces garbage."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  [{label}] '{prompt}' -> '{text}'")

    # Check logits for NaN/Inf
    with torch.no_grad():
        logits = model(**inputs).logits
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print(f"  [{label}] Logits: NaN={has_nan}, Inf={has_inf}, "
              f"min={logits.min().item():.2f}, max={logits.max().item():.2f}")
    return has_nan, has_inf


print("=" * 70)
print("DIAGNOSTIC: Investigating sharp performance drop")
print("=" * 70)

# Load model
print("\n1. Loading TinyLlama...")
model, tokenizer = load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    device_map="auto",
)

# Snapshot
print("\n2. Snapshotting clean weights...")
snapshot = snapshot_weights(model)

# Baseline checks
print("\n3. Baseline NaN/Inf check...")
nan, inf, total = check_nan_inf(model, "baseline")
print(f"  TOTAL: {nan} NaN, {inf} Inf out of {total:,} params")

print("\n4. Baseline generation test...")
run_quick_generation(model, tokenizer, "baseline")

# Test restore mechanism
engine = BitFlipEngine(seed=42)
rates_to_test = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

for rate in rates_to_test:
    print(f"\n{'='*70}")
    print(f"TESTING flip_rate={rate:.0e}")
    print(f"{'='*70}")

    # Restore clean weights
    restore_weights(model, snapshot)

    # Verify restore worked
    mismatched, max_diff = check_restore(model, snapshot)
    if mismatched == 0:
        print("  Restore: OK (all params match snapshot)")
    else:
        print(f"  Restore: FAIL ({mismatched} params differ, max_diff={max_diff:.6e})")

    # Apply flips
    manifest = engine.apply_flips(model, flip_rate=rate, model_id="tinyllama", seed=42 + hash(rate) % (2**31))
    print(f"  Applied {manifest.num_flips:,} flips")

    # Check for NaN/Inf
    nan, inf, total = check_nan_inf(model, f"rate={rate:.0e}")
    print(f"  NaN/Inf summary: {nan:,} NaN, {inf:,} Inf out of {total:,} params")

    # Quick generation test
    has_nan, has_inf = run_quick_generation(model, tokenizer, f"rate={rate:.0e}")

print("\n\nDIAGNOSTIC COMPLETE")
