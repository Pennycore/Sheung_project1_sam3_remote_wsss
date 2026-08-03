# SAM3 Remote WSSS

This project is a lightweight prototype for using SAM3 to generate semantic
pseudo labels from image-level labels on remote-sensing datasets such as
Potsdam.

## Project Handoff And Reproduction

The project now keeps its long-term context in Chinese documents so a new
machine or a new Codex task can continue without the original chat history:

- [`docs/handoff.md`](docs/handoff.md): canonical current handoff, measured
  results, CAM/SAM3 status, server paths, and next commands.
- [`docs/current_status.md`](docs/current_status.md): earlier implementation
  snapshot retained for historical context.
- [`docs/environment_setup.md`](docs/environment_setup.md): reproducible Linux
  environment setup, including the 2080Ti BF16 compatibility check.
- [`docs/experiment_log.md`](docs/experiment_log.md): results that have already
  been run and validated.
- [`docs/runbook.md`](docs/runbook.md): daily smoke, evaluation, two-GPU pseudo
  labeling, and student training commands.

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

The config expects a Potsdam dataset root with this layout:

```text
/path/to/Postdam
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

The recommended workflow first writes explicit 512 x 512 patches and derives a
separate image-level label row for each patch. This avoids applying every class
from a 6000 x 6000 parent image to every small SAM3 tile.

## Why Tiles

SAM3 resizes images internally, so running it on a full 6000 x 6000 scene may
hide small remote-sensing objects. This project cuts each image into overlapping
tiles, runs prompts on each tile, and stitches the masks back into original
image coordinates.

## Install

Use the same Python environment as SAM3. From this repository root, install
SAM3 first and then the WSSS package:

```bash
python -m pip install -e sam3-main/sam3-main
python -m pip install -e sam3_remote_wsss
```

Recommended extra packages:

```powershell
pip install tifffile pillow numpy tqdm
```

SAM3 itself requires a CUDA-capable environment and checkpoint access. Student
training additionally needs PyTorch and TorchVision, which are usually already
present in a SAM3 environment. See `docs/environment_setup.md` for the tested
2080Ti dependency pins and the NumPy/OpenCV compatibility fix.

## Step 0: Prepare An Explicit Patch Dataset

For Potsdam, use the pixel GT once to simulate patch-level image tags and keep
the copied patch GT isolated for evaluation. The SAM3 generator and student do
not read the pixel GT during training. The recommended full-data command also
applies a parent-tile split before any overlapping patches are used:

```bash
python -m sam3_remote_wsss.prepare_potsdam_patches \
  --config configs/potsdam_server_prompt4.json \
  --parent-split configs/potsdam_parent_split_17_6_14.json \
  --output-root /home/undergr/remote_dataset/Postdam_patches_512_full \
  --patch-size 512 \
  --patch-overlap 128 \
  --min-class-pixels 16 \
  --class-min-pixels car=4
```

The split contains 17 train, 6 validation, 14 held-out test, and one excluded
parent tile (`top_potsdam_7_10`). Outputs include the all-patch CSV plus
`image_level_labels_train.csv`, `image_level_labels_val.csv`,
`image_level_labels_test.csv`, `parent_split.json`, `patches.csv`, patch
RGBIR/GT TIFFs, and `potsdam_patches_config.json`. Each `patches.csv` row records
its parent ID and split. The generated config points to the patch dataset and
sets SAM3 tiling to one tile per patch.

This yields 9,472 patches: 4,352 train, 1,536 validation, and 3,584 test. The
split validator rejects duplicate, unknown, or unassigned parent IDs so
overlapping patches cannot leak between sets.

If a source RGB label has noisy colors from an earlier lossy conversion, repair
it to the nearest configured Potsdam palette without overwriting the source:

```bash
python -m sam3_remote_wsss.repair_palette_label \
  --config configs/potsdam_server_prompt4.json \
  --input /path/to/noisy_label.tif \
  --output /path/to/repaired_label.tif \
  --max-distance 80
```

The repair is written only when every pixel is within the distance guard. A
JSON sidecar records maximum distance and per-class pixel counts.

## Step 1: Build Full-Image Labels (Legacy Baseline)

For comparison with the earlier full-image baseline, derive one weak label row
per 6000 x 6000 parent image. Do not use this CSV for the recommended patch
workflow because nearly every parent image contains all foreground classes.

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

`clutter/background` is class ID `0`, not a prompted foreground class. Unknown
GT colors remain `255` ignore.

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
  /path/to/Postdam_patches_512/potsdam_patches_config.json \
  /path/to/Postdam_patches_512/image_level_labels.csv \
  runs/potsdam_sam3_2080ti
```

Only patch IDs present in `--labels-csv` are processed. With two shards, the
generator writes `summary_shard0.json` and `summary_shard1.json`; retries with
`--skip-existing` preserve prior summary entries and complete missing outputs.

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
- `uncovered_label`: label used where no accepted SAM3 mask exists. The
  recommended value is `255`, so uncertain pixels are ignored during student
  training. Set it to `0` only to reproduce the earlier hard-background
  baseline.
- `background_prompting`: optional PromptBG settings. Accepted background masks
  fill only uncovered pixels; similarly scored foreground/background overlaps
  become `255` instead of forcing either class.

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

Pseudo-label evaluation reports two complementary protocols. `miou` and
`foreground_miou` are strict: uncovered pixels count as missed ground-truth
pixels. `labeled_miou` and `labeled_foreground_miou` measure correctness only
where the pseudo label did not abstain. Always report `labeled_coverage`
alongside the labeled metrics to prevent low-coverage pseudo labels from
appearing artificially strong.

PromptBG can be enabled with a separate conservative threshold:

```json
"background_prompting": {
  "enabled": true,
  "prompts": [
    "clutter and miscellaneous objects in aerial imagery",
    "unclassified background regions in a remote sensing image",
    "boundary clutter in overhead imagery",
    "miscellaneous non-target areas in satellite imagery"
  ],
  "score_threshold": 0.6,
  "min_mask_area": 16,
  "max_mask_area_ratio": 0.5,
  "conflict_margin": 0.03
}
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

This deliberately avoids a classification network in the SAM3-only baseline.
The implemented CAM/SAM3 extension adds a multi-label classifier when denser
foreground coverage and reliable background exclusion are required:

- RemoteCLIP tile/prompt filtering.
- CAM low-response background seeds and high-response foreground completion.
- CAM/SAM3 conflict rejection with `255` ignore labels.

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

## Step 4: Train CAM And Fuse It With SAM3

The CAM classifier predicts only the five foreground classes from image-level
labels. Background is inferred where every positive foreground CAM is weak;
it is never trained as an image-level class.

Train the classifier on the explicit patch dataset:

```bash
FULL_ROOT=/home/undergr/remote_dataset/Postdam_patches_512_full

CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.train_cam \
  --config "$FULL_ROOT/potsdam_patches_config.json" \
  --labels-csv "$FULL_ROOT/image_level_labels_train.csv" \
  --val-labels-csv "$FULL_ROOT/image_level_labels_val.csv" \
  --output-dir runs/cam_resnet50_full \
  --epochs 20 \
  --batch-size 8 \
  --image-size 512 \
  --backbone resnet50 \
  --output-stride 16 \
  --pretrained-backbone \
  --num-workers 4 \
  --data-parallel \
  --amp
```

Validation is deterministic and parent-disjoint. The trainer refuses shared
parent tiles, logs validation loss plus micro/macro/per-class F1, and selects
`checkpoints/best.pt` by validation macro-F1 (validation loss breaks ties).
Training F1 is diagnostic only. The trainer refuses to reuse an output directory
containing `train_log.jsonl`, `best.pt`, or `last.pt`, because resume training is
not implemented. Use a new output directory for each run. The destructive
`--overwrite-output` flag is available only for an intentional replacement.
When mirroring console output, use `tee -a` so the shell does not truncate an
existing console log before the trainer can perform this check.

Generate normalized multi-scale CAMs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_cams \
  --config "$FULL_ROOT/potsdam_patches_config.json" \
  --labels-csv "$FULL_ROOT/image_level_labels_train.csv" \
  --checkpoint runs/cam_resnet50_full/checkpoints/best.pt \
  --output-dir runs/cam_resnet50_full/cams_train \
  --scales 0.5,1.0,1.5 \
  --visualize-limit 10 \
  --amp
```

Fuse CAMs with the `Ignore255` SAM3 pseudo labels:

```bash
python -m sam3_remote_wsss.fuse_cam_sam \
  --config "$FULL_ROOT/potsdam_patches_config.json" \
  --labels-csv "$FULL_ROOT/image_level_labels_train.csv" \
  --sam-pseudo-label-dir runs/sam3_prompt4_full_train/pseudo_labels \
  --cam-dir runs/cam_resnet50_full/cams_train \
  --output-dir runs/cam_sam_background_only_full_train \
  --background-threshold 0.0 \
  --background-only \
  --require-complete \
  --skip-existing
```

The recommended fusion policy is conservative: SAM3 owns all foreground labels,
while CAMs add background only where every active foreground channel is exactly
zero. Other uncovered pixels remain `255`:

```text
SAM3 foreground                         -> keep the SAM3 class
no SAM3 mask + every CAM == 0           -> background 0
all other uncovered pixels              -> ignore 255
```

`--require-complete` verifies that every image-level training row has both a
SAM3 PNG and CAM NPZ before writing any fused output. On retries,
`--skip-existing` reprocesses incomplete output triplets and rebuilds
`summary.json` from all per-image metadata rather than only the current retry.
Each output directory also stores `fusion_run.json` with input and setting
fingerprints. Reusing an output directory after changing the labels, SAM3
PNGs, CAM NPZs, thresholds, or fusion mode is rejected to prevent stale masks.

Omit `--background-only` and set the foreground/support thresholds to reproduce
the experimental full hybrid mode. On the current one-parent-tile smoke data,
CAM foreground completion did not improve foreground mIoU, so it is retained as
an ablation rather than the recommended pseudo-label source.

Evaluate `runs/cam_sam_fused/pseudo_labels` with the same
`evaluate_pseudo_labels` command used for the SAM3-only ablations. CAM training
and fusion consume only RGB imagery and image-level CSV labels; pixel GT stays
isolated for offline evaluation.

Evaluation JSON also reports `input_pseudo_labels`, `evaluated_images`, and
explicit skip reasons/examples. Use `--require-all` for a strict run that fails
when any PNG has no matching item, label, or valid Potsdam GT pixels.

## Step 5: Train A SegFormer-Head Student

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
  --pseudo-label-dir runs/cam_sam_fused/pseudo_labels ^
  --cam-checkpoint runs/cam_resnet50/checkpoints/best.pt ^
  --output-dir runs/student_segformer_resnet50 ^
  --epochs 20 ^
  --batch-size 4 ^
  --crop-size 512 ^
  --output-stride 16 ^
  --head segformer ^
  --segformer-embed-dim 256 ^
  --samples-per-image 1 ^
  --cat-max-ratio 0.75 ^
  --min-component-area 16 ^
  --ignore-boundary-width 1 ^
  --amp
```

For the SAM3-only student ablation, point `--pseudo-label-dir` to the SAM3
output and omit `--cam-checkpoint`.

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
    best.pt
    last.pt
```

When `--val-labels-csv` is supplied, validation uses the corresponding pixel
GT only for evaluation. Training still reads pseudo labels exclusively. The
trainer rejects train/validation patches from the same parent tile and selects
`best.pt` by validation mIoU. Existing logs and checkpoints are protected;
choose a new output directory or explicitly pass `--overwrite-output`.

### Fully Supervised Upper Bound

Use the same train/validation split, model, augmentation, and evaluator with
pixel GT to measure the WSSS model against a matched fully supervised upper
bound. `--train-labels-csv` is mutually exclusive with `--pseudo-label-dir`.
The default `--loss auto` selects ordinary cross-entropy for this mode.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.train_student \
  --config "$FULL_ROOT/potsdam_patches_config.json" \
  --train-labels-csv "$FULL_ROOT/image_level_labels_train.csv" \
  --val-labels-csv "$FULL_ROOT/image_level_labels_val.csv" \
  --output-dir runs/student_segformer_fully_supervised_full_v1 \
  --epochs 20 \
  --batch-size 8 \
  --val-batch-size 8 \
  --crop-size 512 \
  --samples-per-image 1 \
  --backbone resnet50 \
  --head segformer \
  --segformer-embed-dim 256 \
  --output-stride 16 \
  --pretrained-backbone \
  --num-workers 4 \
  --data-parallel \
  --amp
```

Do not pass the CAM checkpoint to the primary fully supervised upper bound.
After validation selects `best.pt`, evaluate it on the locked test split using
the same stitched command in Step 6.

For pseudo-label experiments, ToCo loss normally gives the separately averaged
background and foreground terms equal weight. Use
`--background-loss-weight` and `--foreground-loss-weight` for a predeclared
validation-only sweep when sparse or noisy background seeds should contribute
less. The defaults are both `1.0`, which exactly preserves prior experiments.

## Step 6: Evaluate And Stitch The Student

Use the parent-disjoint test CSV only after model selection is complete:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.evaluate_student \
  --config "$FULL_ROOT/potsdam_patches_config.json" \
  --labels-csv "$FULL_ROOT/image_level_labels_test.csv" \
  --checkpoint runs/student_segformer_background_only_full_v1/checkpoints/best.pt \
  --output-dir runs/student_segformer_background_only_full_v1/test_best \
  --batch-size 8 \
  --image-size 512 \
  --num-workers 4 \
  --data-parallel \
  --amp
```

The evaluator reports both overlapping patch metrics and stitched parent-tile
metrics. It resolves overlap by preferring predictions farther from patch
boundaries, so each original Potsdam pixel contributes exactly once to the
final confusion matrix. Use `stitched_metrics` as the final result.
