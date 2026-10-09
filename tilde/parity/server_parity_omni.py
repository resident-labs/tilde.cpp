#!/usr/bin/env python3
"""DEV records (reference_omni.py encode output) through a running Tilde Omni server vs the PyTorch reference outputs:
choice / score fields, probabilities. Bodies are rebuilt from the DEV examples in the record's field order.

    TILDA_CODE=... python server_parity_omni.py --records omni_gpu_sub.jsonl --ref ref.jsonl --dev dev.jsonl \\
        --url http://127.0.0.1:18797 --n 400 [--max-tokens 15000] [--not-multiple-of 64]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18797")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=15000)
    ap.add_argument("--not-multiple-of", type=int, default=64)
    args = ap.parse_args()
    sys.path.insert(0, os.environ["TILDA_CODE"])
    from tilda.data import load_examples
    ex = load_examples(args.dev)["validation"]
    ref = {}
    for line in open(args.ref):
        r = json.loads(line)
        ref[r["id"]] = r["logits"]
    dps, agree, n_rec, acc_r, acc_t, n, errors = [], 0, 0, 0.0, 0.0, 0, 0
    for line in open(args.records):
        if n_rec >= args.n:
            break
        rec = json.loads(line)
        t = len(rec["input_ids"])
        if t > args.max_tokens or (args.not_multiple_of and t % args.not_multiple_of == 0) or rec["id"] not in ref:
            continue
        rows = [ex[i] for i in rec["rows"]]
        qs = {}
        for i, row in enumerate(rows):
            keys = list(row.option_keys or [chr(65 + j) for j in range(len(row.options))])
            qs[f"q{i}"] = {"type": row.kind, "instructions": row.question,
                           "criteria": list(row.options) if row.kind == "score" else dict(zip(keys, row.options))}
        body = json.dumps({"state": rows[0].state, "questions": qs}).encode()
        req = urllib.request.Request(args.url + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception as e:  # noqa: BLE001
            errors += 1
            print("ERR", rec["id"], str(e)[:200], file=sys.stderr)
            continue
        if resp.get("usage", {}).get("input_tokens") != t:
            errors += 1
            print("TOKENS", rec["id"], resp.get("usage"), t, file=sys.stderr)
            continue
        n_rec += 1
        for i, row in enumerate(rows):
            if row.kind not in ("choice", "score"):
                continue
            p = list(resp["answers"][f"q{i}"]["probabilities"].values())
            pr = softmax(ref[rec["id"]][i])
            if len(p) != len(pr):
                continue
            dps.append(max(abs(a - b) for a, b in zip(p, pr)))
            agree += int(p.index(max(p)) == pr.index(max(pr)))
            target = rec["questions"][i]["target"]
            acc_r += target[pr.index(max(pr))]
            acc_t += target[p.index(max(p))]
            n += 1
    dps.sort()
    print(json.dumps({"records": n_rec, "errors": errors, "fields": n, "mean_abs_dp": sum(dps) / max(1, n),
                      "p99_abs_dp": dps[int(0.99 * n)] if n else None, "max_abs_dp": dps[-1] if n else None,
                      "argmax_agreement": agree / max(1, n), "accuracy_ref": acc_r / max(1, n),
                      "accuracy_server": acc_t / max(1, n)}))


if __name__ == "__main__":
    main()
