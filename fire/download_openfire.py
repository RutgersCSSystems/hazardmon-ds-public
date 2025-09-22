#!/usr/bin/env python3
"""
Download the PyroNear OpenFire wildfire image classification dataset from Hugging Face.

This script fetches the `pyronear/openfire` dataset, reads each split (train/validation),
and downloads every image URL listed under `image_url` into split-specific directories.

Usage:
    python download_openfire.py --output_dir ./openfire_images

Options:
    --output_dir  Directory to save downloaded images (default: ./openfire_images)
    --timeout     HTTP timeout in seconds for image download (default: 10)
    --max_workers Number of parallel download threads (default: 8)
"""
import os
import argparse
import threading
from queue import Queue
import requests
from datasets import load_dataset


def worker(queue, output_dir, timeout):
    while True:
        item = queue.get()
        if item is None:
            break
        url, split = item
        try:
            # derive filename from URL (strip query params)
            fname = os.path.basename(url.split('?')[0])
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            out_path = os.path.join(split_dir, fname)
            if os.path.exists(out_path):
                print(f"Already downloaded: {out_path}")
            else:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                print(f"Downloaded: {out_path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        finally:
            queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Download OpenFire dataset images.")
    parser.add_argument(
        "--output_dir", default="./openfire_images", help="Directory to save images"
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="HTTP timeout (seconds)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of download threads"
    )
    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset pyronear/openfire...")
    ds = load_dataset("pyronear/openfire")

    # Prepare a queue of (url, split)
    queue = Queue()
    for split in ds:
        print(f"Queueing {len(ds[split])} images for split '{split}'...")
        for ex in ds[split]:
            url = ex.get("image_url")
            if url:
                queue.put((url, split))

    # Start worker threads
    threads = []
    for _ in range(args.max_workers):
        t = threading.Thread(target=worker, args=(queue, args.output_dir, args.timeout))
        t.daemon = True
        t.start()
        threads.append(t)

    # Wait for all downloads to finish
    queue.join()
    # Stop workers
    for _ in threads:
        queue.put(None)
    for t in threads:
        t.join()

    print("All downloads completed.")

if __name__ == "__main__":
    main()

