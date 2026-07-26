# Running On A 2 x 2080Ti Server

The recommended strategy is two independent Python processes:

```text
GPU 0 -> shard 0
GPU 1 -> shard 1
```

This is simpler and more robust than `DataParallel` for offline pseudo-label
generation. Each process loads one SAM3 image model and writes a disjoint subset
of pseudo labels.

## 1. Copy Required Files

Put these on the server:

```text
/data/code/sam3-main/sam3-main
/data/code/sam3_remote_wsss
/data/remote_dataset/remote/Postdam
/data/checkpoints/sam3.pt
```

Then edit `configs/potsdam_sam3_2080ti.json` if your paths differ.

## 2. Environment

Use the SAM3 environment. For 2080Ti, start with smaller tiles because each card
has 11 GB memory:

```text
tile_size = 768
tile_overlap = 192
```

If OOM happens, reduce `tile_size` to `512` and `tile_overlap` to `128`.

Install:

```bash
cd /data/code/sam3-main/sam3-main
pip install -e .

cd /data/code/sam3_remote_wsss
pip install -e .
```

## 3. Build Image-Level Labels

If you are simulating image-level labels from Potsdam masks:

```bash
python -m sam3_remote_wsss.build_image_level_labels \
  --config configs/potsdam_sam3_2080ti.json \
  --output data/potsdam_image_level_labels.csv
```

If you already have real image-level labels, prepare the same CSV format:

```csv
image_id,impervious_surface,building,low_vegetation,tree,car
top_potsdam_2_10,1,1,1,1,1
```

## 4. Smoke Test One Image

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config configs/potsdam_sam3_2080ti.json \
  --labels-csv data/potsdam_image_level_labels.csv \
  --output-dir runs/smoke \
  --limit 1
```

Check:

```text
runs/smoke/overlays
runs/smoke/pseudo_labels
```

## 5. Run Both GPUs

```bash
bash scripts/run_two_2080ti.sh \
  configs/potsdam_sam3_2080ti.json \
  data/potsdam_image_level_labels.csv \
  runs/potsdam_sam3_2080ti
```

Logs:

```text
runs/potsdam_sam3_2080ti/logs/gpu0.log
runs/potsdam_sam3_2080ti/logs/gpu1.log
```

The script uses `--skip-existing`, so it can be rerun after interruption.

## 6. Evaluate Pseudo Labels

```bash
python -m sam3_remote_wsss.evaluate_pseudo_labels \
  --config configs/potsdam_sam3_2080ti.json \
  --pseudo-label-dir runs/potsdam_sam3_2080ti/pseudo_labels \
  --output runs/potsdam_sam3_2080ti/pseudo_metrics.json
```

Inspect `foreground_miou` and class IoU before training the student model.

## 7. Train The SegFormer-Head Student

Single GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.train_student \
  --config configs/potsdam_sam3_2080ti.json \
  --pseudo-label-dir runs/potsdam_sam3_2080ti/pseudo_labels \
  --output-dir runs/student_smoke \
  --epochs 1 \
  --batch-size 2 \
  --limit 2 \
  --amp
```

Two visible GPUs with simple DataParallel:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.train_student \
  --config configs/potsdam_sam3_2080ti.json \
  --pseudo-label-dir runs/potsdam_sam3_2080ti/pseudo_labels \
  --output-dir runs/student_segformer_resnet50 \
  --epochs 20 \
  --batch-size 8 \
  --crop-size 512 \
  --output-stride 16 \
  --head segformer \
  --segformer-embed-dim 256 \
  --samples-per-image 16 \
  --cat-max-ratio 0.75 \
  --min-component-area 16 \
  --ignore-boundary-width 1 \
  --num-workers 4 \
  --data-parallel \
  --amp
```

If training OOMs on 2080Ti, lower `--batch-size` first, then lower
`--crop-size` to `384`, then use `--output-stride 32`.

## Notes

- Use the image SAM3 model only. Do not use SAM3.1 video/multiplex on 2080Ti for
  this WSSS pseudo-labeling experiment.
- 2080Ti does not support BF16. If you add custom autocast later, use FP16, not
  BF16.
- If all prompts are enabled for all positive classes, generation will be slow.
  First run one image and inspect overlays before launching the full dataset.
- To use RemoteCLIP prompt selection, install `open-clip-torch`, set
  `remoteclip.enabled=true`, and point `remoteclip.checkpoint_path` to the
  RemoteCLIP checkpoint. This ranks prompts per tile and only sends the top-k
  prompts to SAM3.
