#!/usr/bin/env python3
"""Parity gates for the quant presets: every preset is compared with the bf16 PyTorch reference on the same DEV records.

A preset ships only if all gates pass:
  accuracy      no more than 0.5 points below the reference
  log loss      no more than +0.005 above the reference
  mean |dp|     at most 0.02 (mean over fields of the largest per-option probability difference)
  argmax        at least 98% of fields pick the same option as the reference

    python gates.py data/parity_gpu_bf16.json:BF16 data/parity_gpu_q8_0.json:Q8_0 ...
Prints a markdown table.
"""
import json
import sys

GATES = {"accuracy_delta_pt": -0.5, "log_loss_delta": 0.005, "mean_abs_dp": 0.02, "argmax_agreement": 0.98}


def check(result: dict) -> list[str]:
    failed = []
    if result["accuracy_delta_pt"] < GATES["accuracy_delta_pt"]:
        failed.append("accuracy")
    if result["log_loss_delta"] > GATES["log_loss_delta"]:
        failed.append("log loss")
    if result["mean_abs_dp"] > GATES["mean_abs_dp"]:
        failed.append("mean |dp|")
    if result["argmax_agreement"] < GATES["argmax_agreement"]:
        failed.append("argmax")
    if result.get("nonfinite_fields"):
        failed.append("nonfinite")
    return failed


def main() -> None:
    print("| preset | fields | accuracy (ref -> preset) | delta (pt) | log loss delta | mean abs dp | p99 abs dp | max abs dp "
          "| argmax agreement | gates |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for arg in sys.argv[1:]:
        path, _, name = arg.partition(":")
        r = json.load(open(path))
        failed = check(r)
        print(f"| {name or path} | {r['fields']:,} | {r['reference']['accuracy']:.4f} -> {r['test']['accuracy']:.4f} "
              f"| {r['accuracy_delta_pt']:+.2f} | {r['log_loss_delta']:+.4f} | {r['mean_abs_dp']:.4f} "
              f"| {r['p99_abs_dp']:.3f} | {r['max_abs_dp']:.3f} | {100 * r['argmax_agreement']:.2f}% "
              f"| {'pass' if not failed else 'FAIL: ' + ', '.join(failed)} |")


if __name__ == "__main__":
    main()
