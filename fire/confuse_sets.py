#!/usr/bin/env python3
"""
confuse_sets.py

Randomly merges images from src_extra_dir into src_base_dir producing merged_dir.
Produces a manifest CSV with provenance and randomized order for reproducibility.

Usage:
  python3 confuse_sets.py \
    --base confuse_data/pyrodataset/train \
    --extra confuse_data/openfire_images/train \
    --out confuse_data/merged_train \
    --fraction 0.30 \
    --seed 42 \
    --copy
"""

import argparse
import random
import shutil
import os
from pathlib import Path
import csv
import uuid

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def gather_images(dirpath):
    p = Path(dirpath)
    imgs = [f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in IMG_EXTS]
    return imgs

def safe_copy(src: Path, dst_dir: Path, prefix=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = src.name
    if prefix:
        name = f"{prefix}_{name}"
    dst = dst_dir / name
    # avoid overwrite: append uuid if exists
    if dst.exists():
        dst = dst.with_name(dst.stem + "_" + uuid.uuid4().hex + dst.suffix)
    shutil.copy2(src, dst)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to base dataset dir (e.g., pyrodataset/train)')
    ap.add_argument('--extra', required=True, help='Path to extra images dir (e.g., openfire_images/train)')
    ap.add_argument('--out', required=True, help='Output merged directory')
    ap.add_argument('--fraction', type=float, default=0.5, help='Fraction of extra images to inject (0-1) relative to base count')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    ap.add_argument('--copy', action='store_true', help='Copy files (default); omit to move extra images')
    ap.add_argument('--shuffle', action='store_true', help='Shuffle final ordering of copied files')
    args = ap.parse_args()

    random.seed(args.seed)

    base_imgs = gather_images(args.base)
    extra_imgs = gather_images(args.extra)

    if len(base_imgs) == 0:
        raise SystemExit(f"No images found in base dir: {args.base}")
    if len(extra_imgs) == 0:
        raise SystemExit(f"No images found in extra dir: {args.extra}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy base images into out_dir preserving subdirs
    manifest = []
    for p in base_imgs:
        rel = p.relative_to(Path(args.base))
        dst_subdir = out_dir / rel.parent
        dst = safe_copy(p, dst_subdir, prefix='base')
        manifest.append((str(dst), 'base', str(p)))

    # decide how many extra images to inject
    num_to_inject = int(len(base_imgs) * args.fraction)
    num_to_inject = min(num_to_inject, len(extra_imgs))
    chosen_extra = random.sample(extra_imgs, num_to_inject)

    for p in chosen_extra:
        # place extras distributed across same relative structure as base or a flat dir; here we mirror structure if possible
        try:
            rel = p.relative_to(Path(args.extra))
        except Exception:
            rel = Path(p.name)
        dst_subdir = out_dir / rel.parent
        if args.copy:
            dst = safe_copy(p, dst_subdir, prefix='extra')
        else:
            dst_subdir.mkdir(parents=True, exist_ok=True)
            name = f"extra_{p.name}"
            dst = dst_subdir / name
            if dst.exists():
                dst = dst.with_name(dst.stem + "_" + uuid.uuid4().hex + dst.suffix)
            shutil.move(str(p), str(dst))
        manifest.append((str(dst), 'extra', str(p)))

    if args.shuffle:
        random.shuffle(manifest)

    # write manifest CSV
    manifest_path = out_dir / 'manifest.csv'
    with manifest_path.open('w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['merged_path', 'origin', 'original_path'])
        for row in manifest:
            writer.writerow(row)

    print(f"Merged {len(base_imgs)} base images and {len(chosen_extra)} extra images into {out_dir}")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
