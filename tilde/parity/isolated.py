#!/usr/bin/env python3
"""Isolated questions vs joint decisions, on the PyTorch reference (measurement only; tilde.cpp has no isolated mode).

Three readings of a multi-question request:
  (a) joint: today's behavior, one pass; the MLA layers are bidirectional over the whole request, so the state and
      every question see each other, and the head sees every field (reference.py `run`).
  (b) isolated: the state is read once and never sees a question; question i sees the state and itself only; the KDA
      and convolution states of question i start from the end of the state, not from earlier questions; the head of
      question i sees its own fields and the state. Each question keeps the positions it has in a one-question
      request. Computed here per question on the one-question request [preamble + state + schema header + field i + end]
      with the MLA split at the end of the schema header: rows before it attend to those keys only, rows after it to
      every key. KDA and the convolutions are causal, so the prefix never sees the question and question i starts
      from the state's final recurrent state; this is exactly a state pass shared by all questions.
  (c) independent: one one-question request per question (the state sees its question).

    python isolated.py prepare --records gpu_sub.jsonl --dev DEV.jsonl --base BASE --checkpoint CKPT --out singles.jsonl
    python isolated.py run --singles singles.jsonl --mode isolated|independent --base B --checkpoint C --temperature T \
        --out out.jsonl [--skip-single-question-records]
    python isolated.py latency --records bench_records.jsonl --base B --checkpoint C --out latency.json
    python isolated.py compare --records gpu_sub.jsonl --joint ref.jsonl --isolated iso.jsonl --independent ind.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import read_records, tilda_path  # noqa: E402
from isolated_questions import prefix_end as _prefix_end  # noqa: E402

def prefix_end(input_ids, tokenizer) -> int:
    return _prefix_end(input_ids, tokenizer)


def cmd_prepare(args) -> None:
    tilda_path()
    from tilda.clef_joint import encode_joint_record
    from tilda.data import load_examples
    from tilda.serialization import load_tokenizer

    tokenizer = load_tokenizer(args.base, args.checkpoint)
    examples = load_examples(args.dev)["validation"]
    records = read_records(args.records)
    checked = n_out = 0
    with open(args.out, "w") as out:
        for record in records:
            rows = record["rows"]
            exs = [examples[r] for r in rows]
            if checked < args.check:
                joint = encode_joint_record(exs, tokenizer, 65536, tuple(rows))
                if list(joint.input_ids) != record["input_ids"]:
                    raise SystemExit(f"{record['id']}: re-encoding the record's rows does not reproduce it")
                checked += 1
            surgery = None
            if checked <= args.check:
                from isolated_questions import split_record
                surgery = split_record(encode_joint_record(exs, tokenizer, 65536, tuple(rows)), tokenizer)
            for i, (row, ex) in enumerate(zip(rows, exs, strict=True)):
                single = encode_joint_record([ex], tokenizer, 65536, (row,))
                if surgery is not None and (list(surgery[i][0].input_ids) != list(single.input_ids)
                                            or surgery[i][0].questions[0].option_spans != single.questions[0].option_spans
                                            or surgery[i][0].questions[0].question_span != single.questions[0].question_span):
                    raise SystemExit(f"{record['id']} field {i}: split_record differs from the one-question encoding")
                q = single.questions[0]
                ids = list(single.input_ids)
                out.write(json.dumps({
                    "id": f"{record['id']}#{i}", "parent": record["id"], "index": i,
                    "n_questions": len(rows), "input_ids": ids, "prefix": prefix_end(ids, tokenizer),
                    "questions": [{"type": int(q.question_type), "span": list(q.question_span),
                                   "options": [list(s) for s in q.option_spans],
                                   "target": list(ex.target), "kind": ex.kind}]}) + "\n")
                n_out += 1
    print(json.dumps({"records": len(records), "singles": n_out, "checked_reencodes": checked}))


def install_isolation() -> None:
    from isolated_questions import _install
    _install()


def load_model(args):
    tilda_path()
    from tilda import joint_serving
    model, tokenizer, _ = joint_serving.load_joint_checkpoint(str(args.base), str(args.checkpoint), device="cuda",
                                                              verify=False, fast_head=True)
    return model, tokenizer


def encoded(record):
    from tilda.clef_joint import EncodedQuestion, EncodedRecord
    questions = tuple(EncodedQuestion(f"field_{i + 1}", q["type"], tuple(q["span"]),
                                      tuple(tuple(s) for s in q["options"]),
                                      tuple(str(j) for j in range(len(q["options"]))))
                      for i, q in enumerate(record["questions"]))
    return EncodedRecord(tuple(record["input_ids"]), questions, record["id"])


def cmd_run(args) -> None:
    import torch
    model, tokenizer = load_model(args)
    from tilda.clef_joint import forward_joint
    pad = tokenizer.pad_token_id or 0
    from isolated_questions import mla_split
    singles = read_records(args.singles)
    if args.skip_single_question_records:
        singles = [s for s in singles if s["n_questions"] > 1]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in open(args.out) if line.strip()}
    with open(args.out, "a") as out:
        for n, s in enumerate(singles):
            if s["id"] in done:
                continue
            torch.cuda.synchronize()
            t = time.perf_counter()
            split = mla_split(s["prefix"]) if args.mode == "isolated" else contextlib.nullcontext()
            with split, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_joint(model, [encoded(s)], pad)[0]
                logits = [(x.float() / args.temperature).cpu().tolist() for x in logits]
            torch.cuda.synchronize()
            out.write(json.dumps({"id": s["id"], "ms": (time.perf_counter() - t) * 1000, "logits": logits}) + "\n")
            if n % 500 == 0:
                out.flush()
                print(f"{args.mode} {n}/{len(singles)}", file=sys.stderr, flush=True)


def cmd_latency(args) -> None:
    """Per bench record: (a) the joint pass; (b) as a cached implementation would run it, measured as its parts: one
    pass over the prefix (preamble + state + schema header) and one batched pass over the N question suffixes (field
    block + end marker each; their attention to the prefix keys is not included: N x ~100 queries over the prefix, a
    few ms at 32k); (b, uncached) and (c): N one-question passes (with and without the isolation split)."""
    import torch
    model, tokenizer = load_model(args)
    from tilda.clef_joint import forward_joint
    pad = tokenizer.pad_token_id or 0
    install_isolation()
    backbone = model.backbone
    from tilda.ling import attention

    def timed(fn, repeat=3):
        best = None
        for _ in range(repeat + 1):  # first is warm-up
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            best = ms if best is None or ms < best else best
        return best

    def run_backbone(ids_2d, valid):
        attention.set_key_padding(model.base_backbone(), valid)
        backbone(input_ids=ids_2d, attention_mask=None, use_cache=False, return_dict=True)

    results = []
    for record in read_records(args.records):
        ids = record["input_ids"]
        p = prefix_end(ids, tokenizer)
        starts = []
        for q in record["questions"]:
            # each field block starts at the newline before "FIELD i"; approximate it as the end of the previous field
            starts.append(q["span"][0])
        n = len(record["questions"])
        # field blocks: from the end of the previous field's last option (or the prefix) to the end of this field's
        # last option, plus the end marker for the last one
        bounds, prev = [], p
        for q in record["questions"]:
            end = max(s[1] for s in q["options"]) + 1
            bounds.append((prev, end))
            prev = end
        suffixes = [ids[a:b] + [ids[-1]] for a, b in bounds]
        dev = "cuda"
        joint_ms = timed(lambda: forward_joint(model, [encoded(record)], pad))
        prefix_t = torch.tensor([ids[:p]], device=dev)
        prefix_ms = timed(lambda: run_backbone(prefix_t, torch.ones_like(prefix_t, dtype=torch.bool)))
        width = max(len(s) for s in suffixes)
        batch = torch.full((n, width), pad, device=dev, dtype=torch.long)
        valid = torch.zeros((n, width), device=dev, dtype=torch.bool)
        for i, s in enumerate(suffixes):
            batch[i, :len(s)] = torch.tensor(s, device=dev)
            valid[i, :len(s)] = True
        suffix_ms = timed(lambda: run_backbone(batch, valid))
        results.append({"id": record["id"], "tokens": len(ids), "questions": n, "prefix_tokens": p,
                        "suffix_tokens": sum(len(s) for s in suffixes), "joint_ms": joint_ms,
                        "isolated_prefix_ms": prefix_ms, "isolated_suffixes_ms": suffix_ms,
                        "isolated_est_ms": prefix_ms + suffix_ms})
        print(json.dumps(results[-1]), file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def fitted_log_loss(items):
    """Log loss at the best single temperature factor (golden-section over log T), and that factor."""
    def ll(scale):
        total = 0.0
        for logits, target in items:
            p = softmax([x / scale for x in logits])
            total += -sum(t * math.log(max(q, 1e-300)) for t, q in zip(target, p) if t > 0)
        return total / len(items)
    lo, hi = math.log(0.25), math.log(4.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = ll(math.exp(a)), ll(math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = ll(math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = ll(math.exp(b))
    best = (a + b) / 2
    return ll(math.exp(best)), math.exp(best), ll(1.0)


def cmd_compare(args) -> None:
    records = read_records(args.records)
    joint = {json.loads(l)["id"]: json.loads(l)["logits"] for l in open(args.joint) if l.strip()}

    def per_question(path, fallback_single):
        out = {}
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                parent, index = row["id"].rsplit("#", 1)
                out[(parent, int(index))] = row["logits"][0]
        for r in records:  # one-question records: (c) is (a) exactly
            if fallback_single and len(r["questions"]) == 1 and (r["id"], 0) not in out and r["id"] in joint:
                out[(r["id"], 0)] = joint[r["id"]][0]
        return out

    modes = {"joint (a)": {(r["id"], i): joint[r["id"]][i] for r in records if r["id"] in joint
                           for i in range(len(r["questions"]))}}
    if args.isolated:
        modes["isolated (b)"] = per_question(args.isolated, False)
    if args.independent:
        modes["independent (c)"] = per_question(args.independent, True)

    fields = [(r["id"], i, q, len(r["questions"])) for r in records for i, q in enumerate(r["questions"])]
    common = [f for f in fields if all((f[0], f[1]) in m for m in modes.values())]

    def bucket(n):
        return "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"

    def summarize(sel):
        res = {}
        for name, m in modes.items():
            correct, dps, agree, items = 0, [], 0, []
            for rid, i, q, _ in sel:
                logits = m[(rid, i)]
                p = softmax(logits)
                pa = softmax(modes["joint (a)"][(rid, i)])
                gold = max(range(len(q["target"])), key=q["target"].__getitem__)
                arg = max(range(len(p)), key=p.__getitem__)
                correct += int(arg == gold)
                agree += int(arg == max(range(len(pa)), key=pa.__getitem__))
                dps.append(max(abs(x - y) for x, y in zip(p, pa)))
                items.append((logits, q["target"]))
            fitted, scale, raw = fitted_log_loss(items)
            res[name] = {"fields": len(sel), "accuracy": correct / len(sel), "log_loss": raw,
                         "fitted_log_loss": fitted, "fitted_scale": scale,
                         "mean_abs_dp_vs_a": sum(dps) / len(dps), "max_abs_dp_vs_a": max(dps),
                         "argmax_agreement_vs_a": agree / len(sel)}
        return res

    report = {"fields": len(common), "records": len({f[0] for f in common}), "all": summarize(common),
              "by_questions_per_request": {}, "by_qtype": {}}
    groups = defaultdict(list)
    for f in common:
        groups[bucket(f[3])].append(f)
    for k in ["1", "2", "3-4", "5-8", "9+"]:
        if groups.get(k):
            report["by_questions_per_request"][k] = summarize(groups[k])
    kinds = defaultdict(list)
    for f in common:
        kinds[f[2]["kind"]].append(f)
    for k, sel in sorted(kinds.items()):
        report["by_qtype"][k] = summarize(sel)
    text = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    # markdown
    def table(title, res_map):
        lines = [f"### {title}", "", "| slice | mode | fields | accuracy | log loss | fitted log loss (scale) | "
                 "mean \\|dp\\| vs a | max \\|dp\\| vs a | argmax = a |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for slice_name, res in res_map.items():
            for name, m in res.items():
                lines.append(f"| {slice_name} | {name} | {m['fields']:,} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | "
                             f"{m['fitted_log_loss']:.4f} ({m['fitted_scale']:.2f}) | {m['mean_abs_dp_vs_a']:.4f} | "
                             f"{m['max_abs_dp_vs_a']:.3f} | {m['argmax_agreement_vs_a']:.2%} |")
        return "\n".join(lines)
    print(table("All fields", {"all": report["all"]}))
    print()
    print(table("By questions per request", report["by_questions_per_request"]))
    print()
    print(table("By question type", report["by_qtype"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check", type=int, default=200, help="records whose joint re-encoding is checked")
    p = sub.add_parser("run")
    p.add_argument("--singles", type=Path, required=True)
    p.add_argument("--mode", choices=("isolated", "independent"), required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-single-question-records", action="store_true")
    p = sub.add_parser("latency")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--joint", type=Path, required=True)
    p.add_argument("--isolated", type=Path)
    p.add_argument("--independent", type=Path)
    p.add_argument("--json", type=Path)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "latency": cmd_latency, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
