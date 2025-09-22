#!/usr/bin/env python3
import os
import argparse
import threading
import multiprocessing
import logging
import subprocess
import sys
import shutil
#import dfilter


import torch
from datasets import load_dataset
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.nn.modules.block import Bottleneck

torch.serialization.add_safe_globals([Bottleneck])


def export_yolo_dataset(dataset, split, output_dir):
    """
    Export a Hugging Face wildfire dataset split to YOLO-style folder structure.
    """
    img_out = os.path.join(output_dir, 'images', split)
    lbl_out = os.path.join(output_dir, 'labels', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for ex in dataset:
        img = ex['image']  # PIL image
        name = ex['image_name']
        # save image
        img_path = os.path.join(img_out, name)
        img.save(img_path)
        # save labels (already YOLO formatted)
        label_txt = ex['annotations']
        label_name = os.path.splitext(name)[0] + '.txt'
        with open(os.path.join(lbl_out, label_name), 'w') as f:
            f.write(label_txt)


def download_and_prepare(repo_id, output_dir):
    """
    Download the HF dataset and export to local YOLO format.
    """
    print(f"Downloading dataset {repo_id}...")
    ds = load_dataset(repo_id)
    print("Exporting train split...")
    export_yolo_dataset(ds['train'], 'train', output_dir)
    print("Exporting validation split...")
    export_yolo_dataset(ds['val'],   'val',   output_dir)
    print("Dataset ready at", output_dir)


def get_thread_logger(thread_local, file_name):
    """Create or retrieve a thread-specific logger."""
    if not hasattr(thread_local, 'logger'):
        thread_local.logger = logging.getLogger(f"Thread-{threading.get_ident()}")
        thread_local.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(file_name)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        thread_local.logger.addHandler(fh)
    return thread_local.logger


def count_items_in_dir(folder):
    try:
        res = subprocess.run(['ls','-1',folder], capture_output=True, text=True, check=True)
        return len(res.stdout.splitlines())
    except:
        return -1


def process_safe_predict(batch_size, image_path, device_index, output_file):
    # determine device
    if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
        device_str = str(device_index)
    else:
        device_str = 'cpu'

    # safe globals
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, C3, SPPF
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.conv import ConvTranspose
    from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, SiLU
    add_safe_globals([DetectionModel,Conv,C2f,C3,SPPF,Detect,ConvTranspose,Module,Sequential,Conv2d,BatchNorm2d,ReLU,SiLU])

    # load model
    model = YOLO('runs/train/wildfire-detect/weights/best.pt')
    # logger
    thread_local = threading.local()
    logger = get_thread_logger(thread_local, output_file)
    logger.info(f"Model loaded on {device_str}")
    # find fire class
    class_names = model.names
    fire_idx = next((i for i,n in class_names.items() if 'fire' in n.lower()), 0)
    logger.info(f"Using fire class index: {fire_idx}")

    results = model.predict(image_path, imgsz=480, stream=True, batch=batch_size, device=device_str)
    fire_list=[]; total=0; cnt=0
    for i,res in enumerate(results):
        path = res.path; sp=res.speed
        pre,inf,post = sp['preprocess'],sp['inference'],sp['postprocess']
        total += inf; cnt +=1
        cls = res.boxes.cls.cpu().numpy().astype(int)
        has_fire = (cls==fire_idx).any()
        logger.info(f"Image {i}: {path}")
        logger.info(f"  preprocess: {pre:.2f} ms, inference: {inf:.2f} ms, post: {post:.2f} ms")
        if has_fire:
            logger.info(f"  🔥 FIRE detected: {int((cls==fire_idx).sum())} boxes")
            fire_list.append(path)
        else:
            logger.info("  — no fire")
    # save summary
    with open(output_file.replace('.txt','_fire.txt'),'w') as f:
        for p in fire_list: f.write(p+'\n')
    if cnt: logger.info(f"Avg inf time: {total/cnt:.2f} ms")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo', default='pyronear/pyro-sdis')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--num_threads',type=int,default=1)
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--gpu_device', type=int,default=0)
    parser.add_argument('--log_folder', default='./inference')
    args=parser.parse_args()

    # download dataset
    #download_and_prepare(args.repo, args.data_dir)
    # apply dfilter
    from dfilter import dfilter
    dfilter(args.data_dir)

    # inference
    os.makedirs(args.log_folder,exist_ok=True)
    multiprocessing.set_start_method('spawn')
    procs=[]
    for i in range(args.num_threads):
        lf = os.path.join(args.log_folder,f'part_{i+1}.txt')
        img_folder=os.path.join(args.data_dir,'images','val')
        p=multiprocessing.Process(target=process_safe_predict,args=(args.batch_size,img_folder,args.gpu_device,lf))
        p.start(); procs.append(p)
    for p in procs: p.join()

if __name__=='__main__': main()
