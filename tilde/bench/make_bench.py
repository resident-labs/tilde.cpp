#!/usr/bin/env python3
"""Benchmark requests at exact token lengths, from any long public text (e.g. public-domain reports).

Writes /v1/systemone bodies (for the server benchmark) and pre-encoded records (for tilde-eval --profile). Lengths are
total request tokens as llama-server's tilde encoder counts them (tilde-encode), hit by bisection on the state text.

    python make_bench.py --text docs.txt --tilde-encode BIN --model M.gguf --out-bodies b.jsonl --out-records r.jsonl \\
        [--lengths 1024,4096,16384,32768,65536] [--questions 1,8] [--no-limits]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

QUESTION_TEMPLATES = [
    ("noul", "Does the document describe a decision that was already taken?", None),
    ("choice", "Which kind of document is this?", {"report": "a committee or agency report", "bill": "the text of a bill",
                                                     "letter": "a letter", "other": "something else"}),
    ("score", "How technical is the language?", ["plain", "mixed", "technical", "highly technical"]),
    ("noul", "Is money mentioned anywhere?", None),
    ("choice", "Which branch of government is the main actor?", {"legislative": None, "executive": None,
                                                                  "judicial": None, "none": "no branch is the actor"}),
    ("noul", "Does the text mention a specific date?", None),
    ("score", "How long would an expert need to review it?", ["minutes", "an hour", "a day", "a week"]),
    ("noul", "Is there a list of recommendations?", None),
]


def encode(binary, model, bodies, no_limits):
    cmd = [str(binary), "-m", str(model)] + (["--no-limits"] if no_limits else [])
    out = subprocess.run(cmd, input="".join(json.dumps(b) + "\n" for b in bodies), capture_output=True, text=True,
                         check=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def body_for(text, chars, n_questions, pad_questions):
    questions = {}
    for i in range(n_questions):
        kind, instructions, criteria = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        if pad_questions:
            # long instructions: each question carries a slice of the text (a 64k request needs them, since the
            # state + one question is at most 32k)
            start = (i * 7919) % max(1, len(text) - pad_questions)
            instructions = instructions + " Context: " + text[start:start + pad_questions]
        q = {"type": kind, "instructions": instructions}
        if criteria is not None:
            q["criteria"] = criteria
        questions[f"q{i + 1}"] = q
    return {"state": {"document": text[:chars]}, "questions": questions}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", type=Path, required=True, help="text file, or a .jsonl with a 'text' field per line")
    ap.add_argument("--tilde-encode", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lengths", default="1024,4096,16384,32768,65536")
    ap.add_argument("--questions", default="1,8")
    ap.add_argument("--no-limits", action="store_true", help="also make requests the server rejects (engine runs)")
    ap.add_argument("--out-bodies", type=Path, required=True)
    ap.add_argument("--out-records", type=Path, required=True)
    args = ap.parse_args()

    if args.text.suffix == ".jsonl":
        text = "\n\n".join(json.loads(line)["text"] for line in args.text.read_text().splitlines() if line.strip())
    else:
        text = args.text.read_text()
    bodies, records = [], []
    for n_q in [int(x) for x in args.questions.split(",")]:
        for target in [int(x) for x in args.lengths.split(",")]:
            # a request over 32k needs long questions: the state + one question is capped at 32,768 tokens
            pad = 0 if (target <= 32768 or args.no_limits or n_q == 1) else int((target - 16384) / n_q * 3.2)
            lo, hi = 0, len(text)
            best = None
            for _ in range(40):
                mid = (lo + hi) // 2
                enc = encode(args.tilde_encode, args.model, [body_for(text, mid, n_q, pad)], args.no_limits)[0]
                if "error" in enc:
                    hi = mid
                    continue
                n = len(enc["input_ids"])
                if n <= target:
                    best = (mid, enc)
                    lo = mid + 1
                else:
                    hi = mid
                if lo >= hi:
                    break
            if best is None:
                print(f"{n_q} questions, {target} tokens: not possible within the limits")
                continue
            chars, enc = best
            body = body_for(text, chars, n_q, pad)
            body["request_id"] = f"bench-{n_q}q-{target}"
            bodies.append(body)
            order, questions = enc["order"], []
            i = 0
            while i < len(order):
                j = i
                while j < len(order) and order[j] == order[i]:
                    j += 1
                if 1 <= order[i] <= 3:
                    questions.append({"type": order[i] - 1, "span": [i, j], "options": []})
                elif order[i] == 4:
                    questions[-1]["options"].append([i, j])
                i = j
            records.append({"id": body["request_id"], "input_ids": enc["input_ids"], "questions": questions})
            print(f"{n_q} questions, target {target}: {len(enc['input_ids'])} tokens")
    with open(args.out_bodies, "w") as f:
        for b in bodies:
            f.write(json.dumps(b) + "\n")
    with open(args.out_records, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
