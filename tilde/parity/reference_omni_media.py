#!/usr/bin/env python3
"""Media parity for Tilde Omni: the PyTorch reference (TLive towers online, the checkpoint's bridges and point head)
on the omni checkpoint-eval subsets, written as records tilde-eval --media can run, with the reference outputs.

Needs the tilda-train code (TILDA_CODE) and its environment, the subsets (tilda.qwen35.train.checkpoint_evals select,
seed 20261007: vision, pointing_dev, audio from tilda-mega-v1 DEV; screenspot_v2 from eval_pack_v1) and their media.

    # 1. online encoder dirs (layout index: grids, blank-merging maps, audio frame counts; CPU)
    python reference_omni_media.py prepare --base BASE --subsets EVAL_SUBSETS --sets vision pointing_dev audio \\
        --media-root MEGA --out enc_mix
    python reference_omni_media.py prepare --base BASE --subsets EVAL_SUBSETS --sets screenspot_v2 \\
        --media-root EVAL_PACK --out enc_epack
    # 2. records + reference outputs (GPU), one file per set
    python reference_omni_media.py run --base BASE --checkpoint final_folded --temperature 1.40 --subsets EVAL_SUBSETS \\
        --set vision --enc enc_mix --media-root MEGA --out vision.jsonl
    # 3. tilde-eval -m tilde-4b-omni.gguf --mmproj mmproj.gguf --records vision.jsonl --out vision_gguf.jsonl
    # 4. python reference_omni_media.py compare --records vision.jsonl --test vision_gguf.jsonl

Record: id, set, input_ids (the tilda-vl placeholder ids in place: image start / end, image token, point query / none;
audio blocks use the image start / end ids too), questions [{type 0-2 | 3 (point), span, options, target, kind,
image (points)}] in sequence order, points [{question, box, targets}], media [{kind, start, tokens, path(s), scale |
audio start / end}] in sequence order; ref: logits per question (point questions: [none option]) divided by the extra
temperature, and per point field the point head's token logits plus sub-patch / location / size at its best token.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import tilda_path  # noqa: E402
from reference_omni import checkpoint_temperature  # noqa: E402

DROP_ARGS = {"--lm-validation-fixture", "--checkpoint-evals", "--data", "--media-workers", "--media-host-gb",
             "--media-vram-gb", "--first-save-rows", "--save-every-rows", "--keep-resume"}


def train_argv(report: dict, base: Path, vision: Path, audio: Path, media_root: Path) -> list[str]:
    """The run's arguments (report args) with this box's paths."""
    a = report["args"]
    argv = ["--recipe", a["recipe"], "--backbone", str(base), "--base-identity", "configs/base_qwen35_tlive4b.json",
            "--attention-mode", a["attention_mode"], "--point-head", a["point_head"], "--omni", "--no-audio-acif",
            "--vision-rope", a["vision_rope"], "--seed", str(a["seed"]), "--online-encoders",
            "--vision-features", str(vision), "--audio-features", str(audio), "--media-root", str(media_root)]
    return argv


def cmd_prepare(args) -> None:
    tilda_path()
    from tilda.qwen35.model.online import prepare
    data = [args.subsets / f"{name}.jsonl" for name in args.sets]
    prepare(args.base, data, args.media_root, args.out, args.workers)


def _decl_pixels(root: Path, entry: dict, fixed: Path) -> list[str]:
    """Paths of the image as the reference reads it (decode_image: the declared size); a file stored at another size is
    resized by the reference's rule and saved under `fixed`."""
    from PIL import Image
    import numpy as np
    from tilda.vision.common import TOKEN_PX
    from tilda.vision.extract import to_declared
    out = []
    width, height = entry["cols"] * TOKEN_PX, entry["rows"] * TOKEN_PX
    for p in entry.get("frames") or [entry["path"]]:
        path = root / p
        with Image.open(path) as im:
            if im.size == (width, height) and im.mode == "RGB" and path.suffix.lower() == ".png":
                out.append(str(path.resolve()))
                continue
            pixels = np.array(to_declared(np.asarray(im.convert("RGB")), width, height))
        target = fixed / (path.stem + f"_{width}x{height}.png")
        if not target.exists():
            fixed.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(target)
        out.append(str(target.resolve()))
    return out


def cmd_run(args) -> None:
    for key in ("base", "checkpoint", "subsets", "enc", "media_root", "out"):
        setattr(args, key, getattr(args, key).resolve())  # before the chdir below
    tilda_path()
    os.chdir(os.environ["TILDA_CODE"])  # recipe / base identity paths are relative to the code
    import torch
    from tilda import clef_joint
    from tilda.agents.omni_t4 import build_t4
    from tilda.audio.common import is_audio
    from tilda.qwen35.model.online import use_stores
    from tilda.training.joint import train as T
    from tilda.vision.common import point_image, point_targets
    from tilda.vision.point import point_decode
    sys.path.insert(0, os.environ["TILDA_CODE"])
    from scripts.evaluation.evaluate_vision_sets import load_set, run_tokenizer

    report = json.loads((args.checkpoint / "report.json").read_text())
    enc = args.enc.resolve()
    _, targs = T.parse_arguments(["--data", os.devnull, "--output", os.devnull,
                                  *train_argv(report, args.base.resolve(), enc / "vision", enc / "audio",
                                              args.media_root.resolve())])
    tokenizer, marker_start = run_tokenizer(targs)
    extra = args.temperature / checkpoint_temperature(args.checkpoint, args.unfolded_head)
    if args.records_only:  # the records alone (CPU): stores without a model
        from types import SimpleNamespace
        from tilda.qwen35.model.online import open_audio_store, open_vision_store
        model = SimpleNamespace(vision=SimpleNamespace(store=open_vision_store(enc / "vision", args.media_root.resolve())),
                                audio=SimpleNamespace(store=open_audio_store(enc / "audio", args.media_root.resolve())))
    else:
        model, how = build_t4(targs, tokenizer, marker_start, args.checkpoint.resolve(), args.serve_profile, verify=False)
        model.eval()
        print(json.dumps({"loaded_by": how[:200], "extra_temperature": extra}), file=sys.stderr, flush=True)
    vision_ids = clef_joint.vision_token_ids(tokenizer)
    pad = tokenizer.pad_token_id or 0
    fixed = args.out.parent / "media_fixed"

    import contextlib
    stores = contextlib.nullcontext() if args.records_only else use_stores(
        model, vision_dir=enc / "vision", audio_dir=enc / "audio", media_root=args.media_root.resolve())
    with stores:
        vstore = model.vision.store
        astore = model.audio.store
        astore = getattr(astore, "_store", astore)
        rows, records = load_set(args.subsets / f"{args.set}.jsonl", tokenizer, vstore, targs)
        rate = float(astore.meta.get("frame_rate_hz", 13.0))
        with open(args.out, "w") as out:
            for index, record in enumerate(records[:args.limit] if args.limit else records):
                if not record.points and not record.questions:
                    continue
                media_items = record.examples[0].images if record.examples else record.points[0].example.images
                media, image_index = [], 0
                for item, (start, count) in zip(media_items, record.image_spans, strict=True):
                    if is_audio(item):
                        entry = astore.index[item.id]
                        a, b = item.frame_span(rate, int(entry["frames"]))
                        media.append({"kind": "audio", "start": start, "tokens": count,
                                      "path": str((astore.root / entry["path"]).resolve()),
                                      "frames": [a, b], "clip_frames": int(entry["frames"])})
                    else:
                        entry = vstore.index[item.id]
                        media.append({"kind": "image", "image": image_index, "start": start, "tokens": count,
                                      "paths": _decl_pixels(vstore.root, entry, fixed), "scale": item.scale,
                                      "rows": entry["rows"], "cols": entry["cols"]})
                        image_index += 1
                questions = []
                for q, ex in zip(record.questions, record.examples, strict=True):
                    questions.append({"type": int(q.question_type), "span": list(q.question_span),
                                      "options": [list(s) for s in q.option_spans], "target": list(ex.target),
                                      "kind": ex.kind})
                points = []
                for p in record.points:
                    image = point_image(p.example)
                    spec = p.example.point
                    tokens = None
                    layout_of = [m for m in media if m["kind"] == "image"][p.image_index]
                    if spec.box is not None:
                        tok = vstore.token_of_cell(image.id, targs.vision_merge)
                        tokens = point_targets(spec, image, tok)["tokens"]
                    questions.append({"type": 3, "span": list(p.question_span), "options": [[p.none_position, p.none_position + 1]],
                                      "target": [1.0], "kind": "point", "image": p.image_index,
                                      "query": p.query_position})
                    points.append({"question": len(questions) - 1, "box": None if spec.box is None else list(spec.box),
                                   "targets": tokens, "image": p.image_index, "image_start": layout_of["start"]})
                order = sorted(range(len(questions)), key=lambda i: questions[i]["span"][0])
                remap = {old: new for new, old in enumerate(order)}
                questions = [questions[i] for i in order]
                for p in points:
                    p["question"] = remap[p["question"]]
                ids = list(record.input_ids)
                if args.records_only:
                    out.write(json.dumps({"id": record.record_id, "set": args.set, "n_tokens": len(ids), "input_ids": ids,
                                          "questions": questions, "points": points, "media": media,
                                          "vision_ids": vision_ids, "ref": None}) + "\n")
                    continue
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, outs = clef_joint.forward_joint_outputs(model, [record], pad)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                clef = [(x.float() / extra).cpu().tolist() for x in logits[0]]
                ref_points = []
                for out_p in outs[0]:
                    click = point_decode(out_p)
                    best = click["token"]
                    ref_points.append({"token_logits": out_p["token_logits"].float().cpu().tolist(), "best": best,
                                       "sub_logits": out_p["sub_logits"][best].float().cpu().tolist(),
                                       "location": out_p["location_logits"][best].float().cpu().flatten().tolist(),
                                       "size": out_p["size"][best].float().cpu().tolist(),
                                       "click": [click["x"], click["y"]], "none": click["none"],
                                       "p_none": click["p_none"]})
                # logits in sequence order: Clef questions keep theirs, point questions get an empty slot
                n_clef = len(record.questions)
                ref_logits = [None] * len(questions)
                for old, values in enumerate(clef):
                    ref_logits[remap[old]] = values
                rec = {"id": record.record_id, "set": args.set, "n_tokens": len(ids), "input_ids": ids,
                       "questions": questions, "points": points, "media": media, "vision_ids": vision_ids,
                       "ref": {"ms": ms, "logits": ref_logits, "points": ref_points}}
                assert n_clef + len(points) == len(questions)
                out.write(json.dumps(rec) + "\n")
                if index % 100 == 0:
                    out.flush()
                    print(f"{index}/{len(records)}", file=sys.stderr, flush=True)


# ---- compare ----------------------------------------------------------------------------------------------------------

def _softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def cmd_compare(args) -> None:
    recs = [json.loads(line) for line in open(args.records)]
    test = {}
    for line in open(args.test):
        r = json.loads(line)
        test[r["id"]] = r
    fields = agree = 0
    diffs, ref_ll, test_ll, ref_ok, test_ok = [], 0.0, 0.0, 0, 0
    pts = {"n": 0, "click_agree": 0, "ref_hit": 0, "test_hit": 0, "pos": 0, "dp_none": [], "token_dp": []}
    for rec in recs:
        t = test.get(rec["id"])
        if t is None:
            continue
        for qi, q in enumerate(rec["questions"]):
            if q["type"] == 3:
                continue
            a, b = rec["ref"]["logits"][qi], t["logits"][qi]
            pa, pb = _softmax(a), _softmax(b)
            gold = max(range(len(q["target"])), key=q["target"].__getitem__)
            diffs.append(max(abs(x - y) for x, y in zip(pa, pb)))
            agree += int(max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__))
            ref_ok += int(max(range(len(pa)), key=pa.__getitem__) == gold)
            test_ok += int(max(range(len(pb)), key=pb.__getitem__) == gold)
            ref_ll += -sum(tt * math.log(max(p, 1e-300)) for tt, p in zip(q["target"], pa) if tt > 0)
            test_ll += -sum(tt * math.log(max(p, 1e-300)) for tt, p in zip(q["target"], pb) if tt > 0)
            fields += 1
        for pi, p in enumerate(rec["points"]):
            r, s = rec["ref"]["points"][pi], t["points"][pi]
            pts["n"] += 1
            pr, ps = _softmax(r["token_logits"]), _softmax(s["token_logits"])
            pts["dp_none"].append(abs(pr[0] - ps[0]))
            pts["token_dp"].append(max(abs(x - y) for x, y in zip(pr, ps)))
            same = (r["none"] == s["none"]) and (r["none"] or (abs(r["click"][0] - s["click"][0]) < 1e-3
                                                               and abs(r["click"][1] - s["click"][1]) < 1e-3))
            pts["click_agree"] += int(same)
            if p["box"] is None:
                pts["ref_hit"] += int(r["none"])
                pts["test_hit"] += int(s["none"])
            else:
                x0, y0, x1, y1 = p["box"]
                pts["pos"] += 1
                pts["ref_hit"] += int(not r["none"] and x0 <= r["click"][0] <= x1 and y0 <= r["click"][1] <= y1)
                pts["test_hit"] += int(not s["none"] and x0 <= s["click"][0] <= x1 and y0 <= s["click"][1] <= y1)
    diffs.sort()
    out = {"records": len([r for r in recs if r["id"] in test]), "fields": fields}
    if fields:
        out.update({"reference": {"accuracy": ref_ok / fields, "log_loss": ref_ll / fields},
                    "test": {"accuracy": test_ok / fields, "log_loss": test_ll / fields},
                    "accuracy_delta_pt": 100 * (test_ok - ref_ok) / fields, "log_loss_delta": (test_ll - ref_ll) / fields,
                    "mean_abs_dp": sum(diffs) / fields, "p99_abs_dp": diffs[int(0.99 * (fields - 1))],
                    "max_abs_dp": diffs[-1], "argmax_agreement": agree / fields})
    if pts["n"]:
        n = pts["n"]
        out["points"] = {"fields": n, "on_screen": pts["pos"], "reference_hit": pts["ref_hit"] / n,
                         "test_hit": pts["test_hit"] / n, "hit_delta_pt": 100 * (pts["test_hit"] - pts["ref_hit"]) / n,
                         "same_click": pts["click_agree"] / n, "mean_abs_dp_none": sum(pts["dp_none"]) / n,
                         "mean_max_abs_dp_token": sum(pts["token_dp"]) / n}
    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--subsets", type=Path, required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("run")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--unfolded-head", type=Path, help="joint_head.safetensors of the unfolded checkpoint (see reference_omni)")
    p.add_argument("--subsets", type=Path, required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--enc", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--serve-profile", default="ampere24_bf16")
    p.add_argument("--records-only", action="store_true", help="write the records without running the reference (CPU)")
    p = sub.add_parser("compare")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--json", type=Path)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
