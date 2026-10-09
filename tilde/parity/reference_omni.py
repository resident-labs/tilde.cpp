#!/usr/bin/env python3
"""Parity harness for Tilde Omni (tilde-4b-omni, a qwen35 joint checkpoint): the PyTorch reference on the same records.

Needs the tilda-train code (TILDA_CODE) and its environment (torch, fla, transformers). DEV rows are private: keep
every file this writes outside the repository.

    # 0. a text view of the checkpoint (the text serving path attaches no media towers): hard links + report.json
    python reference_omni.py textview --checkpoint final_folded --out text_view
    # 1. records: the trainer's encoder on the DEV rows, with the omni tokenizer; --ids keeps the records of another
    #    parity set (the Ling set tilde_parity_20261008/gpu_sub.jsonl: same states, re-encoded)
    python reference_omni.py encode --dev DEV.jsonl --base BASE --checkpoint text_view --ids gpu_sub.jsonl --out records.jsonl
    # 2. reference logits: GPU, batch 1, bf16 autocast, serve profile ampere24_bf16 (exact bf16), divided by the
    #    temperature the GGUF folds on top of the checkpoint's own (--temperature, total)
    python reference_omni.py run --records records.jsonl --base BASE --checkpoint text_view --temperature 1.40 --out ref.jsonl
    # 3. tilde-eval -m tilde-4b-omni.gguf --records records.jsonl --out gguf.jsonl
    # 4. python reference.py compare --records records.jsonl --ref ref.jsonl --test gguf.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import read_records, tilda_path  # noqa: E402

MEDIA_FILES = ("vision.safetensors", "audio.safetensors")


def cmd_textview(args) -> None:
    src, dst = args.checkpoint.resolve(), args.out
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir() or rel.name in MEDIA_FILES or rel.name in ("report.json", "CHECKPOINT_READY.json"):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dst / rel).exists():
            try:
                os.link(path, dst / rel)
            except OSError:  # another filesystem
                import shutil
                shutil.copy2(path, dst / rel)
    report = json.loads((src / "report.json").read_text())
    report["files"] = {k: v for k, v in report["files"].items() if k not in MEDIA_FILES}
    for key in ("vision_spec", "audio_spec"):
        report.pop(key, None)
    report["text_view_of"] = str(src)
    (dst / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"text_view": str(dst), "files": sorted(report["files"])}))


def checkpoint_temperature(checkpoint: Path) -> float:
    report = json.loads((checkpoint / "report.json").read_text())
    return float((report.get("temperature_fold") or {}).get("temperature") or 1.0)


def cmd_encode(args) -> None:
    tilda_path()
    from tilda.data import load_examples
    from tilda.serialization import load_tokenizer
    from tilda.training.joint.train import encode_training_records

    report = json.loads((args.checkpoint / "report.json").read_text())
    tokenizer = load_tokenizer(args.base, args.checkpoint)
    examples = load_examples(args.dev)["validation"]
    resolved = report.get("resolved_arguments") or report.get("args") or {}
    records, stats = encode_training_records(
        examples, tokenizer, int(resolved.get("max_request_tokens", 65536)),
        resolved.get("record_overflow_policy", "fail_whole_record"), resolved.get("field_order", "shuffle"),
        int(resolved.get("seed", 42)), None, False, report.get("input_layout", "state_first"))
    print(json.dumps({k: v for k, v in stats.items() if k != "split_states"}), file=sys.stderr)
    wanted = None
    if args.ids:
        wanted = {}
        for record in read_records(args.ids):
            wanted[record["id"]] = record.get("rows")
    kept = mismatched = 0
    with open(args.out, "w") as out:
        for record in records:
            if wanted is not None and record.record_id not in wanted:
                continue
            rows = list(record.row_indices) if record.row_indices else list(range(len(record.questions)))
            if wanted is not None and wanted[record.record_id] is not None and wanted[record.record_id] != rows:
                mismatched += 1
            questions = []
            for question, example in zip(record.questions, record.examples, strict=True):
                questions.append({"type": int(question.question_type), "span": list(question.question_span),
                                  "options": [list(span) for span in question.option_spans],
                                  "target": list(example.target), "kind": example.kind})
            out.write(json.dumps({"id": record.record_id, "rows": rows, "input_ids": list(record.input_ids),
                                  "questions": questions}) + "\n")
            kept += 1
    print(json.dumps({"records": kept, "wanted": None if wanted is None else len(wanted),
                      "rows_differ_from_ids_file": mismatched}))


def cmd_run(args) -> None:
    tilda_path()
    import torch
    from tilda import joint_serving
    from tilda.clef_joint import EncodedQuestion, EncodedRecord, forward_joint

    model, tokenizer, report = joint_serving.load_joint_checkpoint(
        str(args.base), str(args.checkpoint), device="cuda", verify=False, fast_head=True,
        serve_profile=args.serve_profile)
    model.eval()
    pad = tokenizer.pad_token_id or 0
    extra = args.temperature / checkpoint_temperature(args.checkpoint)
    print(json.dumps({"attention_mode": getattr(model, "attention_mode", None), "extra_temperature": extra}),
          file=sys.stderr)
    records = read_records(args.records, args.limit)
    if args.max_tokens:
        records = [r for r in records if len(r["input_ids"]) <= args.max_tokens]
    if args.min_tokens:
        records = [r for r in records if len(r["input_ids"]) > args.min_tokens]
    done = set()
    if args.out.exists():
        with open(args.out) as stream:
            done = {json.loads(line)["id"] for line in stream if line.strip()}
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
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_joint(model, [encoded], pad)[0]
                logits = [(x.float() / extra).cpu().tolist() for x in logits]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            out.write(json.dumps({"id": record["id"], "n_tokens": len(record["input_ids"]), "ms": ms,
                                  "logits": logits}) + "\n")
            if index % 200 == 0:
                out.flush()
                print(f"{index}/{len(records)}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("textview")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("encode")
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ids", type=Path, help="keep only the records (ids) of this records file")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True, help="total serving temperature (as the GGUF)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, help="only records up to this many tokens")
    p.add_argument("--min-tokens", type=int, help="only records over this many tokens")
    p.add_argument("--serve-profile", default="ampere24_bf16",
                   help="qwen35 serve profile: ampere24_bf16 (exact bf16, bf16 load) or off (the eager fp32-master build)")
    args = ap.parse_args()
    {"textview": cmd_textview, "encode": cmd_encode, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
