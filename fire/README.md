# 🔥 Dataset Preparation and YOLO Timing Experiments

This repository provides scripts to **download datasets**, **generate
variants**, and **run YOLO inference timing experiments**. The aim is to
evaluate model robustness under dataset transformations and measure inference
performance.

---

## 📂 Repository Structure
```
.
├── scripts/
│ ├── setvars.sh # Environment variable setup
│ ├── download_openfire.py # Downloads the OpenFire dataset
│ ├── pyrodataset.py # Prepares dataset with augmentations
│ ├── yolo-ds.sh # Runs YOLO inference
│ ├── yolo_infer_timing.py # Measures inference time, logs to timings.csv
│ ├── make_variants.sh # Generates dataset variants (confusions, distortions, etc.)
│ ├── plot_timings.py # Plots graphs from timings.csv
│
├── confuse_data/ # Generated dataset variants
│ ├── variants_min/ # Minimal variants subset
│ ├── timings.csv # Timing results from inference
│ └── timings_subset.csv # Smaller subset of timings
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo/scripts
```

### 2. Set Up Environment

Source the environment variables:

```
source setvars.sh
```

Install dependencies:
```
pip install -r requirements.txt
```

### 3. Download Dataset
Run the download script in background and wait for completion:

```
mkdir generate_data
cd generate_data

python3 ../download_openfire.py &> download_openfire.out &
pid1=$!

python3 ../pyrodataset.py &> pyrodataset.out &
pid2=$!

wait $pid1 $pid2
```

### 4. Run YOLO Inference
Run YOLO inference across dataset variants:

```
cd ..

python3 confuse_sets.py     --base generate_data/pyrodataset/train     --extra generate_data/openfire_images/train     --out generate_data/merged_train     --fraction 0.30     --seed 42     --copy

python3 make_variants.py \
--dataset generate_data/pyrodataset/train:pyro \
--dataset generate_data/openfire_images/train:openfire \
--out generate_data/variants_min \
--modes q30,q90,small224,large2048,copy \
--seed 42 --limit 0 --workers 8

python3 yolo_infer_variants.py     --manifest generate_data/variants_min/manifest.csv     --out generate_data/variants_min/timings_subset.csv     --model yolov8n.pt --device 0 --modes q30,q90,small224,large2048     --per_mode_limit 300 --shuffle --seed 42 --warmup 10
```

### 5. Plot the graphs

```
   python3 plot_timings_variants.py \
  --csv generate_data/variants_min/timings_subset.csv \
  --out generate_data/variants_min/plots_subset \
  --group-cols dataset_label,mode
```
