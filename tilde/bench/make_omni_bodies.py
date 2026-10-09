#!/usr/bin/env python3
"""/v1/systemone bodies for a Tilde Omni server benchmark (bench_server.py --bodies): text requests of about N tokens
(the state is prose cut to ~3.6 characters per token; the server's usage.input_tokens has the exact count), one
image + point request, one audio + choice request.

    python make_omni_bodies.py --text a.md b.md --image screen.png --audio clip.flac --out bodies.jsonl \\
        [--lengths 1024,4096,16384,32768]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

QUESTIONS = {
    "kind": {"type": "choice", "instructions": "Which kind of document is this?",
             "criteria": {"manual": "technical documentation", "story": "fiction", "news": "a news report",
                          "other": "something else"}},
    "money": {"type": "noul", "instructions": "Is money mentioned anywhere?"},
}


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", type=Path, nargs="+", required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--lengths", default="1024,4096,16384,32768")
    ap.add_argument("--chars-per-token", type=float, default=3.6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prose = "\n\n".join(p.read_text(errors="replace") for p in args.text)
    rows = []
    for n in (int(x) for x in args.lengths.split(",")):
        need = int(n * args.chars_per_token)
        text = (prose * (need // max(1, len(prose)) + 1))[:need]
        rows.append({"name": f"text_{n}", "body": {"state": text, "questions": QUESTIONS}})
    if args.image:
        rows.append({"name": "image_point", "body": {
            "state": "A screenshot of a desktop application.",
            "images": [{"data": data_url(args.image, "image/png"), "label": "screenshot"}],
            "questions": {"click": {"type": "point", "instructions": "Click the close button.", "image": 0},
                          "dark": {"type": "noul", "instructions": "Is the screen in dark mode?"}}}})
    if args.audio:
        rows.append({"name": "audio_choice", "body": {
            "state": "A voice command for a media player.",
            "audio": [{"data": data_url(args.audio, "audio/flac"), "label": "audio"}],
            "questions": {"command": {"type": "choice", "instructions": "Which command was spoken?",
                                      "criteria": {"play": None, "pause": None, "next": None, "volume": None,
                                                   "other": "something else"}}}}})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"request_id": r["name"], **r["body"]}) + "\n")
    print(f"{len(rows)} bodies -> {args.out}")


if __name__ == "__main__":
    main()
