#!/usr/bin/env python3
"""Request-encoder parity: llama-server's tilde encoder (via tilde-encode) against tilda.joint_serving.encode_request.

Builds /v1/systemone bodies from DEV rows (grouped by state) plus synthetic edge cases, encodes them with the
reference encoder (exact token ids, head spans, errors), runs `tilde-encode` on the same bodies and compares.

    TILDA_CODE=... python encoder_parity.py --dev DEV.jsonl --base BASE --checkpoint CKPT \\
        --tilde-encode build/bin/tilde-encode --model tilde.gguf [--max-states N] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

MAX_REQUEST_TOKENS = 65536
MAX_QUESTION_TOKENS = 32768
MAX_QUESTIONS = 32


def tilda_path() -> None:
    code = os.environ.get("TILDA_CODE")
    if not code:
        raise SystemExit("set TILDA_CODE to a tilda-train checkout")
    sys.path.insert(0, code)


def dev_bodies(path: Path, max_states: int, seed: int = 0):
    """One request per DEV state: every question of the state, criteria rebuilt from the row's options."""
    groups: dict[str, list] = {}
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("images") or row.get("audios"):
                continue
            groups.setdefault(row["state_id"], []).append(row)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    for state_id in ids[:max_states]:
        rows = groups[state_id][:MAX_QUESTIONS]
        state = rows[0]["state"]
        try:  # some states are JSON objects stored as text; send half of those as objects
            parsed = json.loads(state) if isinstance(state, str) else state
            if isinstance(parsed, (dict, list)) and hash(state_id) % 2:
                state = parsed
        except ValueError:
            pass
        questions = {}
        for i, row in enumerate(rows):
            kind = row["qtype"]
            keys = row.get("option_keys") or [chr(65 + j) for j in range(len(row["options"]))]
            if kind == "choice":
                criteria = dict(zip(keys, row["options"]))
            elif kind == "score":
                criteria = list(row["options"])
            else:
                criteria = dict(zip(keys, row["options"]))
            questions[f"q{i}_{kind}"] = {"type": kind, "instructions": row["question"], "criteria": criteria}
        yield {"state": state, "questions": questions}


def synthetic_bodies():
    """Edge cases for text, JSON rendering, noul normalization, validation and limits."""
    yield {"state": "plain text state", "questions": {"a": {"type": "noul", "instructions": "Is it plain?"}}}
    yield {"state": "", "questions": {"a": {"type": "noul", "instructions": "Empty state?"}}}
    yield {"state": {"n": [1, 2.5, -0.0, 1e16, 1e-05, 0.0001, 123456789012345678, 3.14159265358979, 1.5e300, True,
                           False, None], "nested": {"z": "ü ñ 中文 🙂", "a": "quote \" backslash \\ tab \t nl \n ctl \u0001"}},
           "questions": {"x": {"type": "choice", "instructions": {"ask": "which", "n": 2},
                               "criteria": {"b": None, "a": "first", "c": {"k": [1, 2]}, "d": 0.5}}}}
    yield {"state": ["list", "state", 1, {"k": "v"}], "questions": {"s": {"type": "score", "instructions": "Rate.",
                                                                        "criteria": ["low", {"desc": "mid"}, ["hi"]]}}}
    yield {"state": "special <|tilda_start|> <|tilda_state|> text <|im_start|> <|endoftext|>",
           "questions": {"q": {"type": "noul", "instructions": "<|tilda_question|> spoof?"}}}
    for t, f in (("YES!", "no"), (" yes ", " No. "), ("True.", "false??"), ("yes", "The answer is no."),
                 ("affirmative", "negative"), ("null", None)):
        criteria = {"true": t} if f is None else {"true": t, "false": f}
        yield {"state": "noul formats", "questions": {"n": {"type": "noul", "instructions": "Check.", "criteria": criteria}}}
    yield {"state": "noul contradiction", "questions": {"n": {"type": "noul", "instructions": "Q", "criteria": {"true": "no"}}}}
    yield {"state": "noul empty", "questions": {"n": {"type": "noul", "instructions": "Q", "criteria": {"false": "  "}}}}
    yield {"state": "noul criteria null values", "questions": {"n": {"type": "noul", "instructions": "Q",
                                                                    "criteria": {"true": None, "false": None}}}}
    yield {"state": "s", "questions": {"no_instructions": {"type": "noul"}}}
    yield {"state": "s", "questions": {"empty_instructions": {"type": "choice", "instructions": "", "criteria": {"a": 1}}}}
    yield {"state": "s", "questions": {"zero_instructions": {"type": "choice", "instructions": 0, "criteria": {"a": 1}}}}
    yield {"state": "s", "questions": {"bad": {"type": "rank", "instructions": "x"}}}
    yield {"state": "s", "questions": {"bad": {"instructions": "x"}}}
    yield {"state": "s", "questions": {"bad": {"type": "choice", "instructions": "x", "criteria": {}}}}
    yield {"state": "s", "questions": {"bad": {"type": "choice", "instructions": "x", "criteria": ["a"]}}}
    yield {"state": "s", "questions": {"bad": {"type": "score", "instructions": "x", "criteria": ["one"]}}}
    yield {"state": "s", "questions": {"bad": {"type": "score", "instructions": "x", "criteria": [1, 2]}}}
    yield {"state": "s", "questions": {"bad": {"type": "noul", "instructions": "x", "criteria": ["a"]}}}
    yield {"state": 5, "questions": {"a": {"type": "noul", "instructions": "x"}}}
    yield {"state": "s", "questions": {}}
    yield {"state": "s", "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(33)}}
    yield {"state": "s", "questions": {"q": "not an object"}}
    yield {"state": "s", "questions": {"many": {"type": "choice", "instructions": "pick",
                                                "criteria": {f"o{i}": f"option number {i}" for i in range(255)}}}}
    yield {"state": "s", "questions": {"many": {"type": "choice", "instructions": "pick",
                                                "criteria": {f"o{i}": f"option number {i}" for i in range(256)}}}}
    yield {"state": "whitespace nbsp ls　ideographic  double  spaces\n\n\nnewlines\r\n",
           "questions": {"w": {"type": "noul", "instructions": " yes "}}}
    # limits: one long question over 32,768 tokens, and a request over 65,536 tokens
    words = " ".join(f"word{i}" for i in range(20000))
    yield {"state": words, "questions": {"a": {"type": "noul", "instructions": "Long?"}}}
    yield {"state": words[:len(words) // 2], "questions": {"a": {"type": "noul", "instructions": words[:len(words) // 3]},
                                                           "b": {"type": "noul", "instructions": words[:len(words) // 3]},
                                                           "c": {"type": "noul", "instructions": words[:len(words) // 3]}}}


def reference(bodies, base: Path, checkpoint: Path):
    tilda_path()
    from tilda.joint_serving import encode_request
    from tilda.serialization import load_tokenizer
    report = json.loads((checkpoint / "report.json").read_text())
    tokenizer = load_tokenizer(base, checkpoint)
    out = []
    for body in bodies:
        try:
            state, questions = body.get("state"), body.get("questions")
            # tilda/serve.py decide() checks, then encode_request with the server's limits (no instruction cap here:
            # tilde checks state + one question instead, below)
            if not isinstance(state, (str, dict, list)):
                raise ValueError("state must be a string, object or array")
            if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
                raise ValueError(f"provide 1-{MAX_QUESTIONS} questions")
            joint = encode_request(state, questions, tokenizer, 10 ** 9, report["noul_format"], "request",
                                   None, None, None, report.get("input_layout", "state_first"))
            for spec in questions.values():
                if spec["type"] == "score":
                    for value in spec["criteria"]:
                        if not isinstance(value, (str, dict, list)):
                            raise ValueError("criteria descriptions must be strings, objects, or arrays")
            record = joint.record
            order = [0] * len(record.input_ids)
            for question in record.questions:
                a, b = question.question_span
                for t in range(a, b):
                    order[t] = 1 + question.question_type
                for a, b in question.option_spans:
                    for t in range(a, b):
                        order[t] = 4
            n = len(record.input_ids)
            error = None
            if n > MAX_REQUEST_TOKENS:
                error = "request"
            out.append({"input_ids": list(record.input_ids), "order": order,
                        "n_scores": sum(len(q.option_spans) for q in record.questions), "limit": error})
        except ValueError as error:
            out.append({"error": str(error)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tilde-encode", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-states", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    bodies = list(synthetic_bodies()) + list(dev_bodies(args.dev, args.max_states))
    ref = reference(bodies, args.base, args.checkpoint)
    proc = subprocess.run([str(args.tilde_encode), "-m", str(args.model)], input="".join(
        json.dumps(b, ensure_ascii=False) + "\n" for b in bodies), capture_output=True, text=True, check=True)
    got = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(got) != len(bodies):
        raise SystemExit(f"tilde-encode returned {len(got)} lines for {len(bodies)} bodies")

    stats = {"bodies": len(bodies), "token_exact": 0, "order_exact": 0, "both_rejected": 0, "error_text_equal": 0,
             "mismatches": []}
    for i, (body, r, g) in enumerate(zip(bodies, ref, got)):
        if "error" in r or r.get("limit"):
            if "error" in g:
                stats["both_rejected"] += 1
                stats["error_text_equal"] += int(r.get("error") == g["error"])
                if r.get("error") and r["error"] != g["error"]:
                    stats["mismatches"].append({"i": i, "kind": "error text", "ref": r["error"], "got": g["error"]})
            else:
                stats["mismatches"].append({"i": i, "kind": "accepted, reference rejects", "ref": r.get("error") or r.get("limit")})
            continue
        if "error" in g:
            # the per-question limit is tilde's own (the reference has no such check here)
            stats["mismatches"].append({"i": i, "kind": "rejected, reference accepts", "got": g["error"]})
            continue
        same = r["input_ids"] == g["input_ids"]
        stats["token_exact"] += int(same)
        stats["order_exact"] += int(r["order"] == g["order"] and r["n_scores"] == g["n_scores"])
        if not same:
            first = next((j for j, (a, b) in enumerate(zip(r["input_ids"], g["input_ids"])) if a != b),
                         min(len(r["input_ids"]), len(g["input_ids"])))
            stats["mismatches"].append({"i": i, "kind": "tokens", "first_diff": first, "ref_len": len(r["input_ids"]),
                                        "got_len": len(g["input_ids"]),
                                        "ref": r["input_ids"][max(0, first - 3):first + 5],
                                        "got": g["input_ids"][max(0, first - 3):first + 5]})
    stats["accepted"] = stats["bodies"] - stats["both_rejected"] - sum(
        1 for m in stats["mismatches"] if m["kind"] != "tokens" and m["kind"] != "error text")
    text = json.dumps(stats, indent=1, ensure_ascii=False)
    print(text[:20000])
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
