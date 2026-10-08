#!/usr/bin/env python3
"""Conformance check of a running /v1/systemone server (tilde-server, or tilda/serve.py for comparison).

- the expectation set of tilda-train release/check_requests.json (choice keys and yes/no answers);
- the response shape of tilda/serve.py: model, answers (type; choice: choice/probabilities/confidence; noul:
  noul/probabilities{true,false}/confidence; score: score/legend/probabilities/confidence), usage.input_tokens,
  latency_ms, request_id echo;
- rejections (400, never truncation): too many questions, bad types, bad criteria, a request over the per-request
  limit, a state + question over the per-question limit.

    python check_server.py --url http://127.0.0.1:8080 --requests check_requests.json [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request


def post(url: str, body: dict, timeout: float = 600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read())
        except ValueError:
            payload = {}
        return error.code, payload, (time.perf_counter() - start) * 1000


def check_shape(body: dict, resp: dict) -> list[str]:
    problems = []
    for key in ("model", "answers", "usage"):
        if key not in resp:
            problems.append(f"missing {key}")
    if "request_id" in body and resp.get("request_id") != body["request_id"]:
        problems.append("request_id not echoed")
    if not isinstance(resp.get("usage", {}).get("input_tokens"), int):
        problems.append("usage.input_tokens")
    answers = resp.get("answers", {})
    if list(answers) != list(body["questions"]):
        problems.append("answers not in question order")
    for qid, spec in body["questions"].items():
        a = answers.get(qid, {})
        kind = spec["type"]
        if a.get("type") != kind:
            problems.append(f"{qid}: type")
            continue
        if kind == "noul":
            p = a.get("noul")
            if not isinstance(p, float) or not 0 <= p <= 1:
                problems.append(f"{qid}: noul")
            elif abs(a["probabilities"]["true"] - p) > 1e-9 or abs(a["probabilities"]["false"] - (1 - p)) > 1e-9:
                problems.append(f"{qid}: noul probabilities")
            elif abs(a["confidence"] - abs(2 * p - 1)) > 1e-9:
                problems.append(f"{qid}: noul confidence")
        else:
            probs = a.get("probabilities", {})
            keys = list(spec["criteria"]) if kind == "choice" else [str(i) for i in range(len(spec["criteria"]))]
            if list(probs) != keys or abs(sum(probs.values()) - 1) > 1e-5:
                problems.append(f"{qid}: probabilities")
                continue
            n = len(keys)
            conf = max(0.0, min(1.0, (n * max(probs.values()) - 1) / (n - 1))) if n > 1 else 1.0
            if abs(a.get("confidence", -1) - conf) > 1e-6:
                problems.append(f"{qid}: confidence")
            if kind == "choice" and a.get("choice") != max(keys, key=lambda k: probs[k]):
                problems.append(f"{qid}: choice is not the argmax")
            if kind == "score":
                expected = sum(i * probs[str(i)] for i in range(n))
                if abs(a.get("score", -1) - expected) > 1e-6:
                    problems.append(f"{qid}: score")
                if list(a.get("legend", {})) != keys:
                    problems.append(f"{qid}: legend")
    return problems


def rejection_cases() -> list[tuple[str, dict]]:
    words = " ".join(["alpha"] * 40000)        # ~40k tokens: state + one question over 32,768
    state = " ".join(["beta"] * 8000)          # ~8k tokens
    question = " ".join(["gamma"] * 23000)     # ~23k tokens: each state + question fits, the request does not
    return [
        ("33 questions", {"state": "s", "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(33)}}),
        ("no questions", {"state": "s", "questions": {}}),
        ("bad state", {"state": 5, "questions": {"a": {"type": "noul", "instructions": "x"}}}),
        ("bad type", {"state": "s", "questions": {"a": {"type": "rank", "instructions": "x"}}}),
        ("256 options", {"state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                                         "criteria": {f"o{i}": None for i in range(256)}}}}),
        ("1 score level", {"state": "s", "questions": {"a": {"type": "score", "instructions": "x", "criteria": ["a"]}}}),
        ("state + question over 32,768", {"state": words, "questions": {"a": {"type": "noul", "instructions": "x"}}}),
        ("request over 65,536", {"state": state, "questions": {
            f"q{i}": {"type": "noul", "instructions": question} for i in range(3)}}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--requests", required=True, help="release/check_requests.json")
    ap.add_argument("--json")
    args = ap.parse_args()

    data = json.load(open(args.requests))
    items = data["requests"] if "requests" in data else data
    report = {"expectations": [], "shape_problems": [], "rejections": []}
    met = total = 0
    for item in items:
        body = dict(item["body"], request_id=item["id"])
        status, resp, ms = post(args.url, body)
        if status != 200:
            report["shape_problems"].append({"id": item["id"], "status": status, "error": resp})
            continue
        problems = check_shape(body, resp)
        if problems:
            report["shape_problems"].append({"id": item["id"], "problems": problems})
        for qid, want in item.get("expect", {}).items():
            a = resp["answers"][qid]
            if a["type"] == "choice":
                got = a.get("choice")
            elif a["type"] == "noul":
                got = a["noul"] > 0.5
            else:
                got = int(max(a["probabilities"], key=a["probabilities"].get))
            ok = got == want
            met += ok
            total += 1
            report["expectations"].append({"id": item["id"], "question": qid, "want": want, "got": got, "ok": ok,
                                           "p": a.get("noul", a.get("probabilities", {}).get(str(want)))})
    for case, body in (("not JSON", None), ("JSON array", [])):
        req = urllib.request.Request(args.url + "/v1/systemone", data=b"{oops" if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                status, payload = resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as error:
            status, payload = error.code, json.loads(error.read() or b"{}")
        report["rejections"].append({"case": case, "status": status, "error": payload.get("error"), "ok": status == 400})
    for name, body in rejection_cases():
        status, resp, ms = post(args.url, body)
        report["rejections"].append({"case": name, "status": status, "error": resp.get("error"), "ok": status == 400})
    report["summary"] = {"expectations_met": met, "expectations": total,
                         "shape_ok": not report["shape_problems"],
                         "rejections_ok": all(r["ok"] for r in report["rejections"])}
    text = json.dumps(report, indent=1)
    print(json.dumps(report["summary"]))
    for r in report["rejections"]:
        print(f"  {r['case']}: {r['status']} {str(r['error'])[:160]}")
    for p in report["shape_problems"]:
        print("  shape:", p)
    for e in report["expectations"]:
        if not e["ok"]:
            print("  missed:", e)
    if args.json:
        open(args.json, "w").write(text)


if __name__ == "__main__":
    main()
