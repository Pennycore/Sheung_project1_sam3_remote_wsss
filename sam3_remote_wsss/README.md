# SAM3 Remote WSSS

This project is a lightweight prototype for using SAM3 to generate semantic
pseudo labels from image-level labels on remote-sensing datasets such as
Potsdam.

The design is intentionally separate from the original SAM3 repository. SAM3 is
used as a backend mask generator, while this project owns the WSSS logic:

```text
image-level labels
-> class-aware text prompts
-> SAM3 instance masks on overlapping tiles
-> score filtering and mask fusion
-> semantic pseudo-label PNGs
-> SegFormer-head student segmentation model
```

## Dataset Assumption

The default config targets your local Potsdam dataset:

```text
C:\Users\28457\Desktop\remote_dataset\remote\Postdam
  4_Ortho_RGBIR
    top_potsdam_2_10_RGBIR.tif
    ...
  5_Labels_all
    top_potsdam_2_10_label.tif
    ...
```

Potsdam images are 6000 x 6000 RGBIR TIFFs. The pseudo-label generator only
uses image-level labels at inference time. The full pixel labels can be used to
derive weak image-level labels and later evaluate pseudo-label quality.

## Why Tiles

SAM3 resizes images internally, so running it on a full 6000 x 6000 scene may
hide small remote-sensing objects. This project cuts each image into overlapping
tiles, runs prompts on each tile, and stitches the masks back into original
image coordinates.

## Install

Use the same Python environment as SAM3. Install SAM3 first, then install this
project in editable mode:

```powershell
cd C:\Users\28457\Desktop\CODE\sam3-main\sam3-main
pip install -e .

cd C:\Users\28457\Documents\Codex\2026-06-16\c-users-28457-desktop-code-sam3\outputs\sam3_remote_wsss
pip install -e .
```

Recommended extra packages:

```powershell
pip install tifffile pillow numpy tqdm
```

SAM3 itself requires a CUDA-capable environment and checkpoint access. Student
training additionally needs PyTorch and TorchVision, which are usually already
present in a SAM3 environment.

## Step 1: Build Image-Level Labels

For early experiments, we can derive weak image-level labels from Potsdam's
pixel labels. This simulates image-level supervision and produces a CSV.

```powershell
python -m sam3_remote_wsss.build_image_level_labels ^
  --config configs/potsdam_sam3_only.json ^
  --output data/potsdam_image_level_labels.csv
```

CSV format:

```csv
image_id,impervious_surface,building,low_vegetation,tree,car
top_potsdam_2_10,1,1,1,1,1
```

`clutter/background` is treated as background, not a prompted foreground class.

## Step 2: Generate SAM3 Pseudo Labels

```powershell
python -m sam3_remote_wsss.generate_pseudo_labels ^
  --config configs/potsdam_sam3_only.json ^
  --labels-csv data/potsdam_image_level_labels.csv ^
  --output-dir runs/potsdam_sam3_only ^
  --limit 2
```

For a 2 x 2080Ti Linux server, see
[`docs/server_2080ti.md`](docs/server_2080ti.md). The short version is to run
two independent shards:

```bash
bash scripts/run_two_2080ti.sh \
  configs/potsdam_sam3_2080ti.json \
  data/potsdam_image_level_labels.csv \
  runs/potsdam_sam3_2080ti
```

Important config fields:

- `sam3_repo`: path to the original SAM3 repository.
- `checkpoint_path`: optional local SAM3 checkpoint. If null, SAM3 may try to
  download from HuggingFace.
- `prompting.style`: prompt generation strategy. `remoteclip_b2c` expands class
  names into RemoteCLIP-like remote-sensing captions.
- `tile_size`: default `1024`.
- `tile_overlap`: default `256`.
- `score_threshold`: minimum SAM3 score to keep a mask.
- `ignore_index`: pixels with unresolved conflicts are written as `255`.

## Output Layout

```text
runs/potsdam_sam3_only
  pseudo_labels
    top_potsdam_2_10.png
  overlays
    top_potsdam_2_10.jpg
  metadata
    top_potsdam_2_10.json
```

Pseudo-label IDs:

```text
0   background
1   impervious_surface
2   building
3   low_vegetation
4   tree
5   car
255 ignore
```

## Current Model Idea

The first implementation is SAM3-only:

```text
positive image-level class
-> multiple remote-sensing text prompts
-> SAM3 instance masks
-> class-specific mask fusion
```

Prompt generation now follows a RemoteCLIP-inspired B2C style. RemoteCLIP uses
Box-to-Caption to turn object categories, locations, and counts into natural
captions. In this WSSS setting we do not have boxes at prompt-generation time,
so the project approximates that idea with class-level remote-sensing templates:

```text
building
buildings in satellite imagery
there is a building in the center of the remote sensing image
there is a building away from the center of the aerial image
there are several buildings in the remote sensing image
there are many buildings in the aerial image
a lot of buildings can be seen in the satellite image
overhead view of buildings
```

You can control this in config:

```json
"prompting": {
  "style": "remoteclip_b2c",
  "include_manual_prompts": true,
  "max_prompts_per_class": 8
}
```

This deliberately avoids a classification network at first. If SAM3-only masks
are noisy, the next extension should add one of these optional modules:

- RemoteCLIP tile/prompt filtering.
- CAM-generated boxes or points as SAM3 geometric prompts.
- Mask verification using a small class-specific classifier.

## Optional RemoteCLIP Prompt Selection

RemoteCLIP does not generate new text prompts by itself. In this project it can
rank a hand-written prompt bank for each tile and keep only the most relevant
prompts before calling SAM3:

```json
"remoteclip": {
  "enabled": true,
  "model_name": "ViT-B-32",
  "checkpoint_path": "/data/checkpoints/RemoteCLIP-ViT-B-32.pt",
  "device": "cuda",
  "top_k_per_class": 2,
  "min_score": null
}
```

The resulting flow is:

```text
class prompt bank
-> RemoteCLIP tile-text similarity
-> top-k prompts per class
-> SAM3 masks
```

## Step 3: Evaluate Pseudo Labels

Because Potsdam includes pixel labels, you can measure whether the SAM3 pseudo
labels are usable before training a student model:

```powershell
python -m sam3_remote_wsss.evaluate_pseudo_labels ^
  --config configs/potsdam_sam3_only.json ^
  --pseudo-label-dir runs/potsdam_sam3_only/pseudo_labels ^
  --output runs/potsdam_sam3_only/pseudo_metrics.json
```

The most useful number here is `foreground_miou`. If it is extremely low, fix
prompting, tiling, score thresholds, or RemoteCLIP filtering before training.

## Step 4: Train A SegFormer-Head Student

The project now includes a student segmentation model:

```text
RGB tile
-> TorchVision ResNet multi-scale feature extractor
-> SegFormer MLP decoder head
-> semantic logits
-> ToCo-style balanced foreground/background CE loss
```

The SegFormer head is adapted from
`C:\Users\28457\Desktop\CODE\SegFormer-master\SegFormer-master\mmseg\models\decode_heads\segformer_head.py`
without depending on mmcv/mmseg. The older ToCo `LargeFOV` head remains
available with `--head large_fov` for ablation.

Training-time preprocessing follows common MMSeg/SegFormer and WSSS practice:
random scale crop, category-ratio crop filtering, horizontal/vertical flips,
90-degree rotation, photometric distortion, blur, small pseudo-label component
removal, and uncertain boundary pixels set to `255` ignore.

Train on SAM3 pseudo labels:

```powershell
python -m sam3_remote_wsss.train_student ^
  --config configs/potsdam_sam3_only.json ^
  --pseudo-label-dir runs/potsdam_sam3_only/pseudo_labels ^
  --output-dir runs/student_segformer_resnet50 ^
  --epochs 20 ^
  --batch-size 4 ^
  --crop-size 512 ^
  --output-stride 16 ^
  --head segformer ^
  --segformer-embed-dim 256 ^
  --samples-per-image 16 ^
  --cat-max-ratio 0.75 ^
  --min-component-area 16 ^
  --ignore-boundary-width 1 ^
  --amp
```

For a quick smoke test:

```powershell
python -m sam3_remote_wsss.train_student ^
  --config configs/potsdam_sam3_only.json ^
  --pseudo-label-dir runs/potsdam_sam3_only/pseudo_labels ^
  --output-dir runs/student_smoke ^
  --epochs 1 ^
  --batch-size 2 ^
  --limit 1
```

Student outputs:

```text
runs/student_segformer_resnet50
  train_log.jsonl
  checkpoints
    last.pt
```
