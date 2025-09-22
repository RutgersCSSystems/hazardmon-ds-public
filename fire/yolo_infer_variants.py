#!/usr/bin/env python3
"""
yolo_infer_variants.py

Run YOLO inference across image variants produced by make_variants_* and record per-image latency.
Reads manifest.csv so output includes dataset_label and mode for grouped analysis.

Usage examples:
  # CPU, run all variants listed in manifest.csv
  python3 yolo_infer_variants.py \
    --manifest confuse_data/variants_min/manifest.csv \
    --out confuse_data/variants_min/timings.csv \
    --model yolov8n.pt --device cpu --warmup 5

  # GPU 0, only specific modes, limit each mode to 300 images
  python3 yolo_infer_variants.py \
    --manifest confuse_data/variants_min/manifest.csv \
    --out confuse_data/variants_min/timings_subset.csv \
    --model yolov8n.pt --device 0 --modes q30,q90,small224,large2048 \
    --per_mode_limit 300 --shuffle --seed 42 --warmup 10
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
import psutil

# Optional deps
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def measure_sys():
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_mb": proc.memory_info().rss / (1024*1024),
    }

def load_manifest(path: Path, modes_filter: List[str] | None, per_mode_limit: int | None,
                  shuffle: bool, seed: int):
    df = pd.read_csv(path)
    # Expect at least: variant_path, dataset_label, mode
    cols_needed = {"variant_path", "dataset_label", "mode"}
    if not cols_needed.issubset(set(df.columns)):
        raise SystemExit(f"Manifest missing required columns: {cols_needed}")
    if modes_filter:
        df = df[df["mode"].isin(modes_filter)]
    # basic existence filter
    df = df[df["variant_path"].apply(lambda p: Path(p).suffix.lower() in IMG_EXTS and Path(p).exists())]
    # Optional per-mode downsample
    if per_mode_limit and per_mode_limit > 0:
        rng = random.Random(seed)
        parts = []
        for mode, group in df.groupby("mode"):
            items = group.sample(n=min(per_mode_limit, len(group)), random_state=rng.randint(0, 1_000_000))
            parts.append(items)
        df = pd.concat(parts, ignore_index=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

def init_model(model_path: str, device: str):
    if not ULTRALYTICS_AVAILABLE:
        print("[warn] ultralytics not available; falling back to image decode timing only.")
        return None
    model = YOLO(model_path)
    # Optional: force device string like "cpu" or "0"
    return model

def run_infer(model, img_path: str, device: str):
    """Return elapsed seconds. If no model, do a minimal decode to measure something."""
    t0 = time.perf_counter()
    if model is not None:
        _ = model(img_path, device=device)
        if TORCH_AVAILABLE and device != "cpu":
            # ensure GPU kernels finished
            torch.cuda.synchronize()
    else:
        # Fallback: decode only
        from PIL import Image
        with Image.open(img_path) as im:
            im.load()
    return time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to variants manifest.csv")
    ap.add_argument("--out", required=True, help="Output timings CSV")
    ap.add_argument("--model", required=False, default=None, help="Ultralytics model file (e.g., yolov8n.pt)")
    ap.add_argument("--device", default="cpu", help="cpu or CUDA id (e.g., 0, 1)")
    ap.add_argument("--modes", default="", help="Comma-separated subset of modes to include")
    ap.add_argument("--per_mode_limit", type=int, default=0, help="Cap examples per mode (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle order before running")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup inferences to stabilize timings")
    args = ap.parse_args()

    random.seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    modes_filter = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes else None
    df = load_manifest(manifest_path, modes_filter, args.per_mode_limit, args.shuffle, args.seed)
    if df.empty:
        raise SystemExit("No images to process after filtering.")

    # Initialize model if provided
    model = None
    if args.model:
        model = init_model(args.model, args.device)

    # Warmup (use a few random images)
    warm_imgs = df["variant_path"].head(min(args.warmup, len(df))).tolist()
    for p in warm_imgs:
        try:
            _ = run_infer(model, p, args.device)
        except Exception:
            pass

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow([
        "image", "dataset_label", "mode",
        "latency_s", "cpu_percent", "rss_mb", "timestamp"
    ])

    # Main loop
    n = len(df)
    print(f"[info] Running inference on {n} images...")
    for i, row in df.iterrows():
        img = row["variant_path"]
        try:
            dt = run_infer(model, img, args.device)
            sysm = measure_sys()
            writer.writerow([
                img, row["dataset_label"], row["mode"],
                f"{dt:.6f}", f"{sysm['cpu_percent']:.2f}", f"{sysm['rss_mb']:.2f}", f"{time.time():.6f}"
            ])
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"  {i+1}/{n} done")
        except Exception as e:
            # record failure with NaN latency
            writer.writerow([
                img, row["dataset_label"], row["mode"],
                "", "", "", f"{time.time():.6f}"
            ])
    fout.close()
    print(f"[done] Wrote timings to {out_path}")

if __name__ == "__main__":
    main()

