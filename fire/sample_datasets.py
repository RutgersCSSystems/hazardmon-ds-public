#!/usr/bin/env python3
"""
sample_from_datasets.py

Randomly sample K images from each dataset directory and copy them into a single output directory,
prefixing each filename with the dataset name to avoid collisions.

Usage examples:
  # sample 10 images from two datasets
  python3 sample_from_datasets.py --datasets pyrodataset/train openfire_images/train \
    --out confuse_data/sampled --k 10 --seed 42 --copy

  # sample K images from multiple named datasets (give a label for each dataset)
  python3 sample_from_datasets.py --dataset pyrodataset/train:pyro \
    --dataset openfire_images/train:openfire \
    --out confuse_data/sampled --k 5 --seed 123 --copy
"""
import argparse
import random
from pathlib import Path
import shutil
import csv
import sys

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def gather_images(d: Path):
    if not d.exists():
        return []
    return [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]

def parse_dataset_arg(s: str):
    """
    Accept either:
      path
    or
      path:label
    Return (Path(path), label)
    """
    if ':' in s:
        path_str, label = s.split(':', 1)
        return Path(path_str), label
    else:
        p = Path(s)
        # label default: folder name
        return p, p.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', help='List of dataset paths (legacy). Each entry may be path or path:label')
    ap.add_argument('--dataset', action='append', help='Use multiple --dataset path[:label] arguments (preferred)')
    ap.add_argument('--out', required=True, help='Output directory where sampled images will be copied')
    ap.add_argument('--k', type=int, default=10, help='Number of images to sample from each dataset (default 10)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    ap.add_argument('--copy', action='store_true', help='Copy files (default behavior). If omitted, files are copied anyway to be safe.')
    args = ap.parse_args()

    # build dataset list from either --dataset or --datasets
    raw = []
    if args.dataset:
        raw.extend(args.dataset)
    if args.datasets:
        raw.extend(args.datasets)
    if not raw:
        print("No datasets specified. Use --dataset path[:label] or --datasets ...", file=sys.stderr)
        sys.exit(2)

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_sampled = 0

    for entry in raw:
        src_path, label = parse_dataset_arg(entry)
        imgs = gather_images(src_path)
        if not imgs:
            print(f"Warning: no images found in {src_path} (skipping)", file=sys.stderr)
            continue

        k = min(args.k, len(imgs))
        chosen = random.sample(imgs, k)
        for src in chosen:
            # create destination filename with prefix label
            dst_name = f"{label}_{src.name}"
            dst = out_dir / dst_name
            # avoid accidental overwrite by appending counter if needed
            counter = 1
            while dst.exists():
                dst = out_dir / f"{label}_{dst.stem}_{counter}{dst.suffix}"
                counter += 1
            shutil.copy2(src, dst)
            manifest.append((str(dst), label, str(src)))
            total_sampled += 1

    # write manifest
    manifest_path = out_dir / 'manifest.csv'
    with manifest_path.open('w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['sampled_path', 'dataset_label', 'original_path'])
        for row in manifest:
            writer.writerow(row)

    print(f"Sampled {total_sampled} images into {out_dir} (manifest: {manifest_path})")

if __name__ == '__main__':
    main()
