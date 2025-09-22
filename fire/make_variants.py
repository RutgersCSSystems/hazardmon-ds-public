#!/usr/bin/env python3
"""
make_variants_min_safe_parallel.py

Parallel variant generator (process pool).
- Keeps short hashed filenames to avoid "File name too long".
- Same modes as before: q30,q50,q70,q90, small224, large2048, copy, truncateXX.

Usage:
  python3 make_variants_min_safe_parallel.py \
    --dataset confuse_data/pyrodataset/train:pyro \
    --dataset confuse_data/openfire_images/train:openfire \
    --out confuse_data/variants_min \
    --modes q30,q90,small224,large2048,copy \
    --seed 42 --limit 0 --workers 8
"""

import argparse
import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import random
import shutil
import os
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# NOTE: Pillow is imported INSIDE workers to avoid heavy state in parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_dataset_arg(s: str):
    if ":" in s:
        p, label = s.split(":", 1)
        return Path(p), label
    p = Path(s); return p, p.name

def gather_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def hash8(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:8]

def unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{stem}_{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1

def dst_name(prefix: str, mode: str, src: Path, ext: str):
    h = hash8(src)
    pre = prefix if prefix else "ds"
    return f"{pre}_{h}_{mode}{ext}"

# --- Worker-side logic ---
def worker_process_one(task: Tuple[str, str, str, str, str]) -> Optional[Dict]:
    """
    Runs in a subprocess.
    task tuple: (src_path, label, mode, out_root, prefix)
    Returns manifest row dict or None if skipped/failed.
    """
    src_path, label, mode, out_root, prefix = task
    src = Path(src_path)
    out_root = Path(out_root)

    try:
        from PIL import Image  # import inside worker
    except Exception:
        return None

    def save_jpeg(img: "Image.Image", dst: Path, quality=90):
        img.save(dst, format="JPEG", quality=quality, optimize=True, progressive=False)

    def resize_keep_aspect(img: "Image.Image", target_short=None, target_long=None):
        w, h = img.size
        if target_short:
            if w <= h:
                new_w = target_short
                new_h = max(1, int(h * (target_short / max(1, w))))
            else:
                new_h = target_short
                new_w = max(1, int(w * (target_short / max(1, h))))
        elif target_long:
            if w >= h:
                new_w = target_long
                new_h = max(1, int(h * (target_long / max(1, w))))
            else:
                new_h = target_long
                new_w = max(1, int(w * (target_long / max(1, h))))
        else:
            return img
        return img.resize((new_w, new_h), Image.BILINEAR)

    def write_truncated_copy(src: Path, dst: Path, fraction: float):
        data = src.read_bytes()
        n = max(1, int(len(data) * fraction))
        dst.write_bytes(data[:n])

    try:
        out_dir = out_root / label / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        # Non-decoding modes first
        if mode.startswith("truncate"):
            pct_str = mode.replace("truncate", "")
            frac = float(pct_str) / 100.0
            name = dst_name(prefix, mode, src, src.suffix.lower())
            dst = unique_path(out_dir / name)
            write_truncated_copy(src, dst, frac)
            return {
                "variant_path": str(dst),
                "original_path": str(src),
                "dataset_label": label,
                "mode": mode,
                "params_json": json.dumps({"op":"truncate","fraction":frac}, sort_keys=True),
                "seed": None,  # filled by parent later
                "timestamp_iso": now_iso(),
            }

        if mode == "copy":
            name = dst_name(prefix, mode, src, src.suffix.lower())
            dst = unique_path(out_dir / name)
            shutil.copy2(src, dst)
            return {
                "variant_path": str(dst),
                "original_path": str(src),
                "dataset_label": label,
                "mode": mode,
                "params_json": json.dumps({"op":"copy"}, sort_keys=True),
                "seed": None,
                "timestamp_iso": now_iso(),
            }

        # Decode once for transforms
        try:
            with Image.open(src) as im:
                im.load()
                im = im.convert("RGB")

                if mode in {"q30","q50","q70","q90"}:
                    q = int(mode[1:])
                    name = dst_name(prefix, mode, src, ".jpg")
                    dst = unique_path(out_dir / name)
                    save_jpeg(im, dst, quality=q)
                    params = {"op":"jpeg_reencode","quality":q}

                elif mode == "small224":
                    im2 = resize_keep_aspect(im, target_short=224)
                    name = dst_name(prefix, mode, src, ".jpg")
                    dst = unique_path(out_dir / name)
                    save_jpeg(im2, dst, quality=90)
                    params = {"op":"resize","target_short":224}

                elif mode == "large2048":
                    im2 = resize_keep_aspect(im, target_long=2048)
                    name = dst_name(prefix, mode, src, ".jpg")
                    dst = unique_path(out_dir / name)
                    save_jpeg(im2, dst, quality=90)
                    params = {"op":"resize","target_long":2048}

                else:
                    return None

            return {
                "variant_path": str(dst),
                "original_path": str(src),
                "dataset_label": label,
                "mode": mode,
                "params_json": json.dumps(params, sort_keys=True),
                "seed": None,
                "timestamp_iso": now_iso(),
            }

        except Exception:
            return None

    except Exception:
        return None

def build_tasks(datasets: List[str], modes: List[str], seed: int, limit: int, prefix: str):
    random.seed(seed)
    tasks = []
    for entry in datasets:
        src_root, label = parse_dataset_arg(entry)
        imgs = gather_images(src_root)
        if not imgs:
            print(f"[warn] No images under {src_root}, skipping.")
            continue
        imgs.sort(); random.shuffle(imgs)
        if limit > 0 and len(imgs) > limit:
            imgs = imgs[:limit]
        for src in imgs:
            for mode in modes:
                tasks.append((str(src), label, mode, "", prefix))  # out_root filled later
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="path[:label], repeatable")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--modes", required=True, help="Comma-separated: q30,q50,q70,q90,small224,large2048,copy,truncateXX")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Max images per dataset (0=all)")
    ap.add_argument("--prefix", default="", help="Optional filename prefix (defaults to dataset label)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                    help="Number of worker processes (default: CPU count)")
    args = ap.parse_args()

    out_root = Path(args.out); ensure_dir(out_root)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    allowed = {"q30","q50","q70","q90","small224","large2048","copy"}
    for m in modes:
        if m.startswith("truncate"):
            pct = m.replace("truncate","")
            if not pct.isdigit() or not (1 <= int(pct) <= 99):
                raise SystemExit(f"Invalid truncate mode: {m} (use truncate10..truncate90)")
        elif m not in allowed:
            raise SystemExit(f"Unknown/disabled mode: {m}")

    # Build tasks
    tasks = build_tasks(args.dataset, modes, args.seed, args.limit, args.prefix or "")  # out_root to be filled below
    # Fill out_root into each task tuple (avoid closing over Path objects pre-fork on some platforms)
    tasks = [(src, label, mode, str(out_root), (args.prefix or label)) for (src, label, mode, _, _) in tasks]

    manifest_rows = []
    total = len(tasks)
    if total == 0:
        print("[info] Nothing to do.")
        return

    print(f"[info] Processing {total} tasks with {args.workers} workers...")
    completed = 0

    # Use a process pool
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker_process_one, t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                row["seed"] = args.seed  # set seed in parent
                manifest_rows.append(row)
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"  progress: {completed}/{total}")

    # Write manifest
    mf = out_root / "manifest.csv"
    ensure_dir(mf.parent)
    with mf.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant_path","original_path","dataset_label","mode",
            "params_json","seed","timestamp_iso"
        ])
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"[done] Created {len(manifest_rows)} variants. Manifest: {mf}")

if __name__ == "__main__":
    main()

