#source scripts/setvars.sh
mkdir generate_data 
cd generate_data

cd ..
 python3 confuse_sets.py     --base generate_data/pyrodataset/train     --extra generate_data/openfire_images/train     --out generate_data/merged_train     --fraction 0.30     --seed 42     --copy

  python3 make_variants.py \
    --dataset generate_data/pyrodataset/train:pyro \
    --dataset generate_data/openfire_images/train:openfire \
    --out generate_data/variants_min \
    --modes q30,q90,small224,large2048,copy \
    --seed 42 --limit 0 --workers 8

   python3 yolo_infer_variants.py     --manifest generate_data/variants_min/manifest.csv     --out generate_data/variants_min/timings_subset.csv     --model yolov8n.pt --device 0 --modes q30,q90,small224,large2048     --per_mode_limit 300 --shuffle --seed 42 --warmup 10

   python3 plot_timings_variants.py \
  --csv generate_data/variants_min/timings_subset.csv \
  --out generate_data/variants_min/plots_subset \
  --group-cols dataset_label,mode
