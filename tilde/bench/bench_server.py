#!/usr/bin/env python3
"""Latency/memory benchmark of a /v1/systemone server with real HTTP requests.

Starts the server command, waits until it answers, then for every body in --bodies (make_bench.py output) sends it
once (cold: the first request of that shape) and --repeat more times (warm: the median). While each request runs it
samples the GPU memory in use (nvidia-smi, whole device) and the server's resident memory (VmRSS of the process
tree). Prints a markdown table and writes JSON lines.

    python bench_server.py --cmd "tilde-server -m M.gguf --port 8091" --url http://127.0.0.1:8091 \\
        --bodies bench_bodies.jsonl --out results.jsonl [--repeat 3] [--label "Q8_0 3090 Ti"]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def post(url, body, timeout=1200.0):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
            return resp.status, payload, (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode()[:300]}, (time.perf_counter() - start) * 1000


def tree_rss_kb(pid):
    """VmRSS of pid and its children (kB)."""
    total = 0
    pids = [pid]
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        pids += [int(p) for p in out]
    except OSError:
        pass
    for p in pids:
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return total


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return int(out[0]) if out else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


class Sampler:
    def __init__(self, pid, period=0.05):
        self.pid, self.period = pid, period
        self.gpu = self.rss = 0
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_flag.is_set():
            g = gpu_used_mib()
            if g is not None:
                self.gpu = max(self.gpu, g)
            self.rss = max(self.rss, tree_rss_kb(self.pid))
            time.sleep(self.period)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_flag.set()
        self.thread.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd", help="server command (omit to use a running server)")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--ready-timeout", type=float, default=900)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    bodies = [json.loads(line) for line in open(args.bodies) if line.strip()]
    proc = None
    gpu_before = gpu_used_mib()  # whatever else holds GPU memory (desktop, a VRAM ballast): subtracted from the peaks
    t_start = time.time()
    if args.cmd:
        log = open(args.log or os.devnull, "w")
        proc = subprocess.Popen(shlex.split(args.cmd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    probe = {"state": "ready?", "questions": {"q": {"type": "noul", "instructions": "Is this a probe?"}}}
    while True:
        try:
            status, _, _ = post(args.url, probe, timeout=30)
            if status == 200:
                break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}")
        if time.time() - t_start > args.ready_timeout:
            raise SystemExit("server did not become ready")
        time.sleep(1)
    ready_s = time.time() - t_start
    idle_gpu = gpu_used_mib()
    pid = proc.pid if proc else None
    print(f"{args.label}: ready in {ready_s:.1f} s, GPU in use {idle_gpu} MiB (before start {gpu_before}), RSS {tree_rss_kb(pid) / 1024 if pid else 0:.0f} MiB")

    rows = []
    with open(args.out, "a") as out:
        for body in bodies:
            with Sampler(pid or os.getpid()) as s:
                status, resp, cold_ms = post(args.url, body)
            if status != 200:
                row = {"label": args.label, "id": body.get("request_id"), "status": status, "error": resp.get("error")}
                print(row)
                out.write(json.dumps(row) + "\n")
                continue
            warm, server_ms = [], []
            with Sampler(pid or os.getpid()) as s2:
                for _ in range(args.repeat):
                    st, r2, ms = post(args.url, body)
                    warm.append(ms)
                    server_ms.append(r2.get("latency_ms"))
            n = resp["usage"]["input_tokens"]
            w = statistics.median(warm) if warm else None
            row = {"label": args.label, "id": body.get("request_id"), "questions": len(body["questions"]),
                   "input_tokens": n, "cold_ms": round(cold_ms, 1), "warm_ms": round(w, 1) if w else None,
                   "warm_server_ms": statistics.median([x for x in server_ms if x is not None]) if server_ms else None,
                   "tokens_per_s": round(n / (w / 1000)) if w else None,
                   "peak_gpu_mib": max(s.gpu, s2.gpu),
                   "peak_gpu_server_mib": max(s.gpu, s2.gpu) - gpu_before if gpu_before is not None else None,
                   "peak_rss_mib": round(max(s.rss, s2.rss) / 1024),
                   "ready_s": round(ready_s, 1), "idle_gpu_mib": idle_gpu}
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"  {row['id']}: {n} tokens, cold {row['cold_ms']} ms, warm {row['warm_ms']} ms, "
                  f"{row['tokens_per_s']} tok/s, GPU {row['peak_gpu_server_mib']} MiB, RSS {row['peak_rss_mib']} MiB", flush=True)
    if proc is not None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
