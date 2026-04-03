# Experiment Plan — Pupil Segmentation

## Overview

Four batches of experiments. Batch 1 is fully independent (run in parallel).
Batch 2 depends on Batch 1 results. Batch 3 is augmentation ablation. Batch 4 is
post-hoc analysis (no training).

**Data directory:** set `DATA_DIR` to your OpenEDS root (containing `train/`, `validation/`, `test/`).

---

## Batch 1: Independent runs (6 experiments)

All can run simultaneously if resources allow.

### Loss function ablation (4 runs)

```bash
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss ce       --save_dir checkpoints/ritnet_ce       --num_epochs 50 --no_wandb
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss dice     --save_dir checkpoints/ritnet_dice     --num_epochs 50 --no_wandb
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss ce_dice  --save_dir checkpoints/ritnet_ce_dice  --num_epochs 50 --no_wandb
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss compound --save_dir checkpoints/ritnet_compound --num_epochs 50 --no_wandb
```

### Architecture baseline (1 run)

```bash
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model unet --loss ce --save_dir checkpoints/unet_ce --num_epochs 50 --no_wandb
```

### Preprocessing ablation (1 run)

```bash
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss ce --save_dir checkpoints/ritnet_no_preproc --num_epochs 50 --no_wandb --no_preprocessing
```

### Or use the ablation runner

```bash
uv run python src/pupil_segmentation/run_ablations.py --data_dir $DATA_DIR --num_epochs 50
```

### After each run: benchmark

```bash
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_ce/best_model.pth       --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_ce.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_dice/best_model.pth     --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_dice.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_ce_dice/best_model.pth  --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_ce_dice.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_compound/best_model.pth --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_compound.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/unet_ce/best_model.pth         --data_dir $DATA_DIR --model unet   --output_json results/unet_ce.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_no_preproc/best_model.pth --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_no_preproc.json
```

### What each run produces

```
checkpoints/<name>/
├── best_model.pth     # Best checkpoint (by val IoU)
├── metrics.csv        # Per-epoch: train_loss, val_loss, val_iou, val_dice, lr
└── checkpoint_epoch_*.pth  # Periodic snapshots

results/<name>.json    # Full benchmark: IoU, Dice, ellipse errors, inference time
```

---

## Batch 1 analysis: pick the best

Compare results JSONs. The key metric is **pupil axis error** (not just IoU).
If best loss is not CE, proceed to Batch 2. If it is CE, skip to Batch 3.

---

## Batch 2: Conditional runs (0-2 experiments)

Only needed if best loss from Batch 1 is not CE.

```bash
# Architecture comparison with best loss
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model unet --loss {best} --save_dir checkpoints/unet_{best} --num_epochs 50 --no_wandb

# Preprocessing with best loss
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss {best} --save_dir checkpoints/ritnet_{best}_no_preproc --num_epochs 50 --no_wandb --no_preprocessing
```

Benchmark as above.

---

## Batch 3: Augmentation ablation (2 experiments)

Uses the best loss from Batch 1 (update "compound" below to the actual best).

### Standard augmentation (flip, rotate, brightness, contrast, noise)

```bash
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss compound --augmentation standard --save_dir checkpoints/ritnet_compound_aug_std --num_epochs 50 --no_wandb
```

### Domain adaptation (standard + sclera brightening + resolution downsample)

```bash
uv run python src/pupil_segmentation/train.py --data_dir $DATA_DIR --model ritnet --loss compound --augmentation domain --save_dir checkpoints/ritnet_compound_aug_domain --num_epochs 50 --no_wandb
```

### Benchmark

```bash
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_compound_aug_std/best_model.pth    --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_compound_aug_std.json
uv run python src/pupil_segmentation/benchmark.py --checkpoint checkpoints/ritnet_compound_aug_domain/best_model.pth --data_dir $DATA_DIR --model ritnet --output_json results/ritnet_compound_aug_domain.json
```

### Or use the ablation runner

```bash
uv run python src/pupil_segmentation/run_ablations.py --data_dir $DATA_DIR --experiment ritnet_compound_aug_std
uv run python src/pupil_segmentation/run_ablations.py --data_dir $DATA_DIR --experiment ritnet_compound_aug_domain
```

---

## Batch 4: Post-hoc analysis (no training)

### PLR robustness experiments

Can run on any machine (synthetic data only):

```bash
uv run python -m src.pupil_segmentation.evaluation.robustness --output_dir results/robustness --n_trials 100
```

### Generate figures

```bash
# PLR fitted curve explainer (no dependencies)
uv run python -m src.pupil_segmentation.evaluation.plots --plot plr --figures_dir figures

# Training curves (needs checkpoints/ with metrics.csv files)
uv run python -m src.pupil_segmentation.evaluation.plots --plot training --csv_dir checkpoints --figures_dir figures

# Robustness plots (needs results/robustness/ from above)
uv run python -m src.pupil_segmentation.evaluation.plots --plot noise --robustness_dir results/robustness --figures_dir figures
uv run python -m src.pupil_segmentation.evaluation.plots --plot blinks --robustness_dir results/robustness --figures_dir figures
```

### Headset validation

Run best model on collected headset images:

```bash
uv run python src/pupil_segmentation/predict.py \
    --checkpoint checkpoints/{best}/best_model.pth \
    --model ritnet \
    --input {headset_image_dir} \
    --output results/headset_validation
```

---

## What goes in the report

### Main body tables

| Table | Source | Key columns |
|-------|--------|-------------|
| Architecture comparison | `results/ritnet_{best}.json`, `results/unet_{best}.json` | Params, Size, Pupil IoU, Iris IoU, Axis err, FPS |
| Loss ablation | `results/ritnet_*.json` (4 losses) | Loss, Pupil IoU, Iris IoU, Axis err |
| Preprocessing ablation | `results/ritnet_{best}.json`, `results/ritnet_*_no_preproc.json` | Preprocessing, Pupil IoU, Axis err |
| Augmentation ablation | `results/ritnet_*_aug_*.json` | Augmentation, Pupil IoU, Axis err |

### Main body figures

| Figure | Source | Description |
|--------|--------|-------------|
| Pipeline diagram | Manual (TikZ/draw.io) | Image → preprocess → segment → ellipse → PLR |
| PLR fitted curve | `figures/plr_fitted_curve.pdf` | Synthetic data + fitted model + annotations |
| Noise robustness | `figures/robustness_noise.pdf` | Parameter recovery vs noise σ |
| Blink robustness | `figures/robustness_blinks.pdf` | Parameter recovery vs blink duration |
| Headset validation | `results/headset_validation/` | Segmentation overlays on real images |

### Appendix

| Figure/Table | Source | Description |
|--------------|--------|-------------|
| Training curves | `figures/training_curves.pdf` | Loss and IoU vs epoch for all runs |
| Full per-class metrics | `results/*.json` | Background, sclera, iris, pupil IoU + Dice |
| Combined noise+blink grid | `results/robustness/combined.json` | Full degradation grid |
