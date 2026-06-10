# Conference Paper Segmentation Pipeline

This repository is a standalone subset of `C:\Dev\Digitech` prepared for the conference paper workflow. It contains only the Atlas vertebra segmentation model, evaluation scripts, inference, and mask-to-keypoint extraction.

## What is included

- Patch-based multiclass vertebra segmentation training
- Fold-based evaluation
- Batch inference with postprocessing
- Mask-to-keypoint extraction for downstream metric computation
- Shared project paths and output folders for a clean standalone workflow

## Repository layout

```text
.
|-- Src/
|   |-- Atlas/
|   |   |-- data/                     # Atlas datasets
|   |   |-- training/                 # Training entrypoints + YAML config
|   |   |-- evaluation/               # Fold evaluation
|   |   `-- inference/                # Segmentation inference + keypoint extraction
|   |-- Models/                       # Shared train/test loops
|   |-- Utils/                        # Utilities
|   `-- project_paths.py              # Centralized paths
|-- Tests/                            # Small local smoke-test assets
|-- data/                             # Local datasets, not tracked
|-- artifacts/checkpoints/            # Local checkpoints, not tracked
|-- outputs/                          # Generated outputs, not tracked
|-- requirements.txt
`-- README.md
```

Legacy flat entrypoints under `Src/Atlas/` are kept as compatibility wrappers and forward to the newer structured modules.

## Installation

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Expected data layout

The standalone pipeline expects the same dataset layout as the parent project:

```text
data/
|-- atlas_vertebra/
|   |-- datasets-PNG/
|   `-- datasets-NPY/
`-- folds/
    `-- atlas_vertebra/
        |-- train_atlas_vertebra_fold_0.csv
        |-- val_atlas_vertebra_fold_0.csv
        `-- ...
```

Paths are centralized in `Src/project_paths.py`.

## Training

Default training config:

`Src/Atlas/training/configs/atlas_multiclass_patch.yaml`

Run training:

```powershell
python Src/Atlas/training/atlas_model_multiclass_patch.py `
  --config Src/Atlas/training/configs/atlas_multiclass_patch.yaml
```

Checkpoints are saved to `artifacts/checkpoints/segmentation/` and TensorBoard logs to `outputs/runs/`.

## Evaluation

Run fold evaluation:

```powershell
python Src/Atlas/evaluation/atlas_test_multiclass_patch.py
```

CSV summaries are written to `outputs/results/Atlas/multiclass/`.

## Batch inference

Run inference on a directory of PNG images:

```powershell
python Src/Atlas/inference/inference.py `
  --model_path artifacts/checkpoints/segmentation/Atlas-heqv-multi-patch-10000-fold-0-final.pth `
  --input_dir data/atlas_vertebra/datasets-PNG `
  --output_dir outputs/predictions `
  --sw_batch_size 4 `
  --overlap 0.25
```

Outputs include:

- `*_image.png`
- `*_maskhat.png`
- `*_blended_hat.png`

## Single-image smoke test

If you keep a local sample image in `Tests/`, you can run:

```powershell
python Src/Atlas/single_inference.py `
  --model_path artifacts/checkpoints/segmentation/Atlas-heqv-multi-patch-10000-fold-0-final.pth `
  --image_path Tests/0001035.png `
  --output_dir outputs/predictions\smoke_test `
  --no_blend
```

## Keypoint extraction

Convert color segmentation masks to LabelMe-style point annotations:

```powershell
python Src/Atlas/inference/extraction.py `
  --in outputs/predictions `
  --out_dir outputs/keypoints `
  --pattern *_maskhat.png `
  --verbose
```

The legacy alias still works:

```powershell
python Src/Atlas/keypoint_extraction.py `
  --in outputs/predictions `
  --out_dir outputs/keypoints `
  --pattern *_maskhat.png `
  --verbose
```

## Notes

- This repo intentionally focuses on the conference-paper segmentation pipeline, not the full `Digitech` analysis stack.
- If you want to change output locations, do it in `Src/project_paths.py`.
- `requirements.txt` contains CPU-safe defaults. For CUDA, install the PyTorch build matching your local CUDA version.
