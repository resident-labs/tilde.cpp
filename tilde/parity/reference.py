#!/usr/bin/env python3
"""Parity harness: encode DEV rows with the reference encoder, run the PyTorch reference, compare outputs.

Needs the tilda-train code (TILDA_CODE, a checkout of tilda-train) and its environment (torch, fla, transformers).
DEV rows are private: keep every file this writes outside the repository.

    # 1. records: the exact token ids and head spans the trainer's DEV evaluation uses
    python reference.py encode --dev DEV.jsonl --base BASE --checkpoint CKPT --out records.jsonl
    # 2. reference logits (GPU, batch 1, bf16 autocast: the serving computation), T divided out
    python reference.py run --records records.jsonl --base BASE --checkpoint CKPT --temperature 1.26 --out ref.jsonl
    # 3. tilde-eval -m model.gguf --records records.jsonl --out gguf.jsonl
    # 4. metrics
    python reference.py compare --records records.jsonl --ref ref.jsonl --test gguf.jsonl [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


def tilda_path() -> None:
    code = os.environ.get("TILDA_CODE")
    if not code:
        raise SystemExit("set TILDA_CODE to a tilda-train checkout")
    sys.path.insert(0, code)


def cmd_encode(args) -> None:
    tilda_path()
    from tilda.data import load_examples
    from tilda.serialization import load_tokenizer
    from tilda.training.joint.train import encode_training_records

    report = json.loads((args.checkpoint / "report.json").read_text())
    tokenizer = load_tokenizer(args.base, args.checkpoint)
    examples = load_examples(args.dev)["validation"]
    if args.max_rows:
        examples = examples[:args.max_rows]
    resolved = report.get("resolved_arguments") or report.get("args") or {}
    records, stats = encode_training_records(
        examples, tokenizer, int(resolved.get("max_request_tokens", 65536)),
        resolved.get("record_overflow_policy", "fail_whole_record"), resolved.get("field_order", "shuffle"),
        int(resolved.get("seed", 42)), None, False, report.get("input_layout", "state_first"))
    print(json.dumps({k: v for k, v in stats.items() if k != "split_states"}), file=sys.stderr)
    with open(args.out, "w") as out:
        for record in records:
            rows = list(record.row_indices) if record.row_indices else list(range(len(record.questions)))
            questions = []
            for question, example in zip(record.questions, record.examples, strict=True):
                questions.append({"type": int(question.question_type), "span": list(question.question_span),
                                  "options": [list(span) for span in question.option_spans],
                                  "target": list(example.target), "kind": example.kind})
            out.write(json.dumps({"id": record.record_id, "rows": rows, "input_ids": list(record.input_ids),
                                  "questions": questions}) + "\n")


def read_records(path: Path, limit: int | None = None, ids: set[str] | None = None):
    records = []
    with open(path) as stream:
        for line in stream:
            record = json.loads(line)
            if ids is not None and record["id"] not in ids:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def subset(records, every: int, max_tokens: int | None):
    """Deterministic subset: records whose id hash falls in 1/every, optionally capped by length."""
    out = []
    for record in records:
        if max_tokens and len(record["input_ids"]) > max_tokens:
            continue
        if every > 1 and int(hashlib.sha256(record["id"].encode()).hexdigest()[:8], 16) % every:
            continue
        out.append(record)
    return out


def cmd_subset(args) -> None:
    records = subset(read_records(args.records), args.every, args.max_tokens)
    with open(args.out, "w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    lengths = sorted(len(r["input_ids"]) for r in records)
    print(json.dumps({"records": len(records), "fields": sum(len(r["questions"]) for r in records),
                      "max_tokens": lengths[-1] if lengths else 0,
                      "median_tokens": lengths[len(lengths) // 2] if lengths else 0}))


def cmd_run(args) -> None:
    tilda_path()
    import torch
    from tilda import joint_serving
    from tilda.clef_joint import EncodedQuestion, EncodedRecord, forward_joint

    model, tokenizer, report = joint_serving.load_joint_checkpoint(
        str(args.base), str(args.checkpoint), device="cuda", verify=False, fast_head=True)
    pad = tokenizer.pad_token_id or 0
    records = read_records(args.records, args.limit)
    done = set()
    if args.out.exists():
        with open(args.out) as stream:
            done = {json.loads(line)["id"] for line in stream if line.strip()}
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with open(args.out, "a") as out:
        for index, record in enumerate(records):
            if record["id"] in done:
                continue
            questions = tuple(EncodedQuestion(f"field_{i + 1}", q["type"], tuple(q["span"]),
                                              tuple(tuple(span) for span in q["options"]),
                                              tuple(str(j) for j in range(len(q["options"]))))
                              for i, q in enumerate(record["questions"]))
            encoded = EncodedRecord(tuple(record["input_ids"]), questions, record["id"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), amp:
                logits = forward_joint(model, [encoded], pad)[0]
                logits = [(x.float() / args.temperature).cpu().tolist() for x in logits]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            out.write(json.dumps({"id": record["id"], "n_tokens": len(record["input_ids"]), "ms": ms,
                                  "logits": logits}) + "\n")
            if index % 200 == 0:
                out.flush()
                print(f"{index}/{len(records)}", file=sys.stderr, flush=True)


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def metrics_for(records, outputs):
    rows = correct = 0
    loss = 0.0
    for record in records:
        logits = outputs.get(record["id"])
        if logits is None:
            continue
        for question, row in zip(record["questions"], logits, strict=True):
            probs = softmax(row)
            target = question["target"]
            gold = max(range(len(target)), key=target.__getitem__)
            correct += int(max(range(len(probs)), key=probs.__getitem__) == gold)
            loss += -sum(t * math.log(max(p, 1e-300)) for t, p in zip(target, probs) if t > 0)
            rows += 1
    return {"fields": rows, "accuracy": correct / rows if rows else None, "log_loss": loss / rows if rows else None}


def load_outputs(path: Path):
    out = {}
    with open(path) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row["logits"]
    return out


def compare(records, ref, test):
    """Paired metrics on the records both outputs cover."""
    common = [r for r in records if r["id"] in ref and r["id"] in test]
    diffs, agree, fields, nonfinite = [], 0, 0, 0
    by_kind: dict[str, list] = {}
    worst = []
    for record in common:
        for question, a, b in zip(record["questions"], ref[record["id"]], test[record["id"]], strict=True):
            if len(a) != len(b):
                raise SystemExit(f"{record['id']}: option count differs")
            if not all(math.isfinite(x) for x in b):
                nonfinite += 1
                continue
            pa, pb = softmax(a), softmax(b)
            d = max(abs(x - y) for x, y in zip(pa, pb))
            diffs.append(d)
            same = max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__)
            agree += int(same)
            fields += 1
            by_kind.setdefault(question["kind"], []).append((d, same))
            worst.append((d, record["id"], len(record["input_ids"])))
    worst.sort(reverse=True)
    diffs.sort()
    result = {
        "records": len(common), "fields": fields, "nonfinite_fields": nonfinite,
        "reference": metrics_for(common, ref), "test": metrics_for(common, test),
        "max_abs_dp": diffs[-1] if diffs else None,
        "mean_abs_dp": sum(diffs) / len(diffs) if diffs else None,
        "p99_abs_dp": diffs[int(0.99 * (len(diffs) - 1))] if diffs else None,
        "argmax_agreement": agree / fields if fields else None,
        "by_kind": {k: {"fields": len(v), "mean_abs_dp": sum(d for d, _ in v) / len(v),
                        "max_abs_dp": max(d for d, _ in v), "argmax_agreement": sum(s for _, s in v) / len(v)}
                    for k, v in sorted(by_kind.items())},
        "worst": [{"max_abs_dp": d, "id": i, "n_tokens": n} for d, i, n in worst[:5]],
    }
    ra, ta = result["reference"]["accuracy"], result["test"]["accuracy"]
    rl, tl = result["reference"]["log_loss"], result["test"]["log_loss"]
    if ra is not None and ta is not None:
        result["accuracy_delta_pt"] = 100 * (ta - ra)
        result["log_loss_delta"] = tl - rl
    return result


def cmd_compare(args) -> None:
    records = read_records(args.records)
    result = compare(records, load_outputs(args.ref), load_outputs(args.test))
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode")
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-rows", type=int)
    p = sub.add_parser("subset")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--max-tokens", type=int)
    p = sub.add_parser("run")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--json", type=Path)
    args = ap.parse_args()
    {"encode": cmd_encode, "subset": cmd_subset, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
