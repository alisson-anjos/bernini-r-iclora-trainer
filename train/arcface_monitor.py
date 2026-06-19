"""Sidecar arcface monitor for a running head-swap IC-LoRA training.

The in-training insightface init was flaky, so compute the metric out-of-process
from the saved validation mp4s. Polls the run's validation/ dir, computes the
mean ArcFace per step as new previews appear, appends to validation_metrics.jsonl
and logs an arcface curve to its own wandb run (x = training step).

    .venv/bin/python train/arcface_monitor.py --run_dir /data/training/bernini/<run>
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import imageio.v3 as iio

sys.path.insert(0, "/data/bernini-src")
from train import validate as V

REF_DIR = "/data/datasets/head_swap_unified/reference"


def steps_on_disk(vdir):
    by_step = {}
    for mp4 in glob.glob(os.path.join(vdir, "step*.mp4")):
        m = re.match(r"step(\d+)_(.+)\.mp4", os.path.basename(mp4))
        if m:
            by_step.setdefault(int(m.group(1)), []).append((m.group(2), mp4))
    return by_step


def mean_arcface(clips):
    sims = []
    for vid, mp4 in clips:
        ref = os.path.join(REF_DIR, f"{vid}.png")
        try:
            frames = iio.imread(mp4).astype(np.float32) / 255.0
        except Exception:
            continue
        s = V.arcface_similarity(ref, frames)
        if s is not None:
            sims.append(s)
    return (float(np.mean(sims)), len(sims)) if sims else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--project", default="bernini-headswap")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    vdir = os.path.join(args.run_dir, "validation")
    metrics_path = os.path.join(vdir, "validation_metrics.jsonl")
    run_name = os.path.basename(args.run_dir.rstrip("/")) + "_arcface"

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb
            wb.init(project=args.project, name=run_name)
            wb.define_metric("step")
            wb.define_metric("arcface_mean", step_metric="step")
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            wb = None

    results = {}   # step -> (mean, n, clip_count)  (recompute while clip_count grows)
    logged = {}    # step -> clip_count already pushed to wandb
    idle = 0
    while True:
        by_step = steps_on_disk(vdir)
        latest = max(by_step, default=-1)
        changed = False
        for step, clips in by_step.items():
            prev = results.get(step)
            # (re)compute if unseen or more clips have landed since last time;
            # a step is only "settled" once a newer batch exists or 3+ clips.
            if prev is None or prev[2] < len(clips):
                mean, n = mean_arcface(clips)
                if mean is not None:
                    results[step] = (mean, n, len(clips))
                    changed = True
        if changed:
            with open(metrics_path, "w") as f:  # idempotent rewrite, no dupes
                for s in sorted(results):
                    m, n, _ = results[s]
                    f.write(json.dumps({"step": s, "arcface_mean": round(m, 4), "n": n}) + "\n")
        if wb is not None:
            for step in sorted(results):
                m, n, cc = results[step]
                settled = step < latest or cc >= 3
                if settled and logged.get(step, -1) < cc:
                    wb.log({"step": step, "arcface_mean": m})
                    logged[step] = cc
                    print(f"[arcface] step {step}: mean={m:.3f} (n={n})", flush=True)
        idle = idle + 1 if not changed else 0
        if max(results, default=0) >= args.max_steps or idle > 30:
            print("[arcface] done.")
            break
        time.sleep(args.poll)

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
