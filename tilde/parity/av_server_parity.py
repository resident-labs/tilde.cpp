#!/usr/bin/env python3
"""Media requests (e.g. omni_av2_both: 2 fps frame pairs + 1 s audio slices, ordered by time, stream layout) through a
running Tilde Omni server, compared with the PyTorch reference outputs of reference_omni_media.py run (same records).

The request is rebuilt from the record's media (declared-size frame files with their scale, audio clip + slice) and
its rows (questions, in the record's field order); media times are the record's block order.

    python av_server_parity.py --records ref_omni_av2_both.jsonl --rows EVAL_SUBSETS_AV/omni_av2_both.jsonl \\
        --url http://127.0.0.1:18797 --out server_av.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

RATE = 13.0  # audio tokens per second


def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def body_for(rec: dict, rows: dict) -> dict:
    first = rows[rec["questions"][0]["uid"]]
    images, audio = [], []
    for t, m in enumerate(rec["media"]):
        if m["kind"] == "image":
            item = {"label": "video", "time": float(t), "stored_scale": m["scale"]}
            if len(m["paths"]) == 2:
                item["frames"] = [b64(p) for p in m["paths"]]
            else:
                item["data"] = b64(m["paths"][0])
            images.append(item)
        else:
            a, b = m["frames"]
            item = {"time": float(t), "start": a / RATE, "end": b / RATE, "label": "soundtrack"}
            first_use = next((j for j, x in enumerate(audio) if x.get("_path") == m["path"]), None)
            if first_use is None:
                item.update({"data": b64(m["path"]), "_path": m["path"]})
            else:
                item["clip"] = first_use
            audio.append(item)
    # labels: the rows' own (first image / audio of the first row)
    for item, src in zip(images, [x for x in first.get("images", [])]):
        item["label"] = src.get("label", item["label"])
    if first.get("audios"):
        for item in audio:
            item["label"] = first["audios"][0].get("label", "audio")
    for item in audio:
        item.pop("_path", None)
    questions = {}
    for i, q in enumerate(rec["questions"]):
        row = rows[q["uid"]]
        kind = row["qtype"]
        if kind == "point":
            questions[f"q{i}"] = {"type": "point", "instructions": row["question"], "image": int(row["point"]["image"])}
            continue
        keys = row.get("option_keys") or [chr(65 + j) for j in range(len(row["options"]))]
        questions[f"q{i}"] = {"type": kind, "instructions": row["question"],
                              "criteria": list(row["options"]) if kind == "score" else dict(zip(keys, row["options"]))}
    return {"request_id": rec["id"], "state": first["state"], "images": images, "audio": audio,
            "media_order": first.get("media_order", "row"), "media_layout": first.get("media_layout", "blocks"),
            "questions": questions, "budget_cells": 100000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18797")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--media-order", choices=("row", "time"), help="override the rows' media_order (row: images first)")
    args = ap.parse_args()
    rows = {}
    for line in open(args.rows):
        r = json.loads(line)
        rows[r["uid"]] = r
    dps, agree, fields, acc_r, acc_s, errors, tok_diff, n = [], 0, 0, 0.0, 0.0, 0, 0, 0
    pts = {"n": 0, "same_token": 0, "ref_hit": 0, "srv_hit": 0}
    with open(args.out, "w") as out:
        for line in open(args.records):
            if args.limit and n >= args.limit:
                break
            rec = json.loads(line)
            n += 1
            body = body_for(rec, rows)
            if args.media_order:
                body["media_order"] = args.media_order
            req = urllib.request.Request(args.url + "/v1/systemone", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
            except urllib.error.HTTPError as e:
                errors += 1
                out.write(json.dumps({"id": rec["id"], "error": e.read().decode()[:300]}) + "\n")
                continue
            tokens = resp.get("usage", {}).get("input_tokens")
            tok_diff += int(tokens != rec["n_tokens"])
            out.write(json.dumps({"id": rec["id"], "input_tokens": tokens, "n_tokens": rec["n_tokens"],
                                  "answers": resp["answers"]}) + "\n")
            if rec.get("ref") is None:
                continue
            ref_points = iter(rec["ref"]["points"])
            for i, q in enumerate(rec["questions"]):
                ans = resp["answers"][f"q{i}"]
                if q["kind"] == "point":
                    rp = next(ref_points)
                    pts["n"] += 1
                    pts["same_token"] += int((ans.get("none") and rp["none"]) or
                                             (not ans.get("none") and not rp["none"] and
                                              abs(ans["stored_xy"][0] - rp["click"][0]) < 1e-3 and
                                              abs(ans["stored_xy"][1] - rp["click"][1]) < 1e-3))
                    continue
                pr = softmax(rec["ref"]["logits"][i])
                row = rows[q["uid"]]
                if q["kind"] == "noul":
                    keys = [str(k).lower() for k in (row.get("option_keys") or ["no", "yes"])]
                    yes = next((j for j, k in enumerate(keys) if k in ("yes", "true")), 1)
                    p = [0.0] * len(pr)
                    p[yes] = ans["noul"]
                    p[1 - yes] = 1 - ans["noul"]
                else:
                    p = list(ans["probabilities"].values())
                if len(p) != len(pr):
                    continue
                dps.append(max(abs(a - b) for a, b in zip(p, pr)))
                agree += int(p.index(max(p)) == pr.index(max(pr)))
                acc_r += q["target"][pr.index(max(pr))]
                acc_s += q["target"][p.index(max(p))]
                fields += 1
    dps.sort()
    print(json.dumps({"records": n, "errors": errors, "token_count_differs": tok_diff, "fields": fields,
                      "accuracy_ref": acc_r / max(1, fields), "accuracy_server": acc_s / max(1, fields),
                      "delta_pt": 100 * (acc_s - acc_r) / max(1, fields),
                      "mean_abs_dp": sum(dps) / max(1, fields), "p99_abs_dp": dps[int(0.99 * fields)] if fields else None,
                      "argmax_agreement": agree / max(1, fields), "points": pts}, indent=1))


if __name__ == "__main__":
    main()
