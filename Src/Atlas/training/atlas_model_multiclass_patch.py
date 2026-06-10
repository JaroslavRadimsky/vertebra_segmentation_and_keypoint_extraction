import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import DataLoader

from Src.Atlas.data.atlas_dataset_patch import AtlasDataset
from Src.Atlas.training.common import build_model
from Src.Atlas.training.config_loader import load_training_config
from Src.Models.training_new import load_model, save_model, training_loop
from Src.Utils.data_utils import get_split_files
from Src.Utils.image_utils import HistogramEqualizationd
from Src.Utils.replicability import get_generator, make_worker_seed_fn, set_seed
from Src.project_paths import ATLAS_DATA_DIR, ATLAS_FOLDS_DIR, SEGMENTATION_CHECKPOINTS_DIR


DEFAULT_CONFIG = {
    "run": {
        "name": "Atlas-heqv-multi-patch-100",
        "train": True,
        "load_train_model": None,
        "save_epoch": 10,
        "min_val_dice": 0.65,
        "seed": 42,
        "max_epochs": 200,
    },
    "data": {
        "dataset_name": "atlas_vertebra",
        "max_files": 100,
        "start_fold": 4,
        "binary": False,
        "heqv": True,
        "train_batch_size": 2,
        "test_batch_size": 2,
        "num_workers": 0,
        "patches_per_image": 20,
    },
    "model": {
        "num_classes": 7,
        "in_channels": 3,
        "channels": [16, 32, 64, 128, 256],
        "strides": [2, 2, 2, 2],
        "num_res_units": 2,
    },
    "optimizer": {"lr": 1e-2},
    "scheduler": {
        "reduce_on_plateau": False,
        "patience": 5,
        "ratio": 0.8,
        "t0": 5,
        "t_mult": 2,
        "eta_min": 1e-8,
    },
    "transforms": {
        "image_size": [512, 512],
        "rand_flip_prob": 0.5,
        "rand_rotate90_prob": 0.5,
        "rand_rotate90_max_k": 3,
        "rand_zoom_min": 0.9,
        "rand_zoom_max": 1.1,
        "rand_zoom_prob": 0.3,
        "rand_gaussian_noise_prob": 0.2,
    },
    "validation": {
        "sw_roi_size": [512, 512],
        "sw_batch_size": 4,
        "sw_overlap": 0.25,
    },
    "loss": {
        "to_onehot_y": False,
        "softmax": True,
        "lambda_dice": 1.0,
        "lambda_ce": 1.0,
    },
}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "atlas_multiclass_patch.yaml"


def build_runtime_config(raw_config):
    image_size = tuple(raw_config["transforms"]["image_size"])
    runtime_config = {
        "NAME": raw_config["run"]["name"],
        "NUM_CLASSES": raw_config["model"]["num_classes"],
        "BINARY": raw_config["data"]["binary"],
        "MAX_EPOCHS": raw_config["run"]["max_epochs"],
        "LR": raw_config["optimizer"]["lr"],
        "TRAIN_BATCH_SIZE": raw_config["data"]["train_batch_size"],
        "TEST_BATCH_SIZE": raw_config["data"]["test_batch_size"],
        "TRAIN": raw_config["run"]["train"],
        "LOAD_TRAIN_MODEL": raw_config["run"]["load_train_model"],
        "SAVE_EPOCH": raw_config["run"]["save_epoch"],
        "MIN_VAL_DICE": raw_config["run"]["min_val_dice"],
        "SEED": raw_config["run"]["seed"],
        "HEQV": raw_config["data"]["heqv"],
        "MAX_FILES": raw_config["data"]["max_files"],
        "START_FOLD": raw_config["data"]["start_fold"],
        "NUM_WORKERS": raw_config["data"]["num_workers"],
        "PATCHES_PER_IMAGE": raw_config["data"]["patches_per_image"],
        "DATASET_NAME": raw_config["data"]["dataset_name"],
        "IMAGE_SIZE": image_size,
        "MODEL_IN_CHANNELS": raw_config["model"]["in_channels"],
        "MODEL_CHANNELS": tuple(raw_config["model"]["channels"]),
        "MODEL_STRIDES": tuple(raw_config["model"]["strides"]),
        "MODEL_NUM_RES_UNITS": raw_config["model"]["num_res_units"],
        "RAND_FLIP_PROB": raw_config["transforms"]["rand_flip_prob"],
        "RAND_ROTATE90_PROB": raw_config["transforms"]["rand_rotate90_prob"],
        "RAND_ROTATE90_MAX_K": raw_config["transforms"]["rand_rotate90_max_k"],
        "RAND_ZOOM_MIN": raw_config["transforms"]["rand_zoom_min"],
        "RAND_ZOOM_MAX": raw_config["transforms"]["rand_zoom_max"],
        "RAND_ZOOM_PROB": raw_config["transforms"]["rand_zoom_prob"],
        "RAND_GAUSSIAN_NOISE_PROB": raw_config["transforms"]["rand_gaussian_noise_prob"],
        "SW_ROI_SIZE": tuple(raw_config["validation"]["sw_roi_size"]),
        "SW_BATCH_SIZE": raw_config["validation"]["sw_batch_size"],
        "SW_OVERLAP": raw_config["validation"]["sw_overlap"],
        "LOSS_TO_ONEHOT_Y": raw_config["loss"]["to_onehot_y"],
        "LOSS_SOFTMAX": raw_config["loss"]["softmax"],
        "LOSS_LAMBDA_DICE": raw_config["loss"]["lambda_dice"],
        "LOSS_LAMBDA_CE": raw_config["loss"]["lambda_ce"],
    }
    lrconfig = {
        "LRReduceOnPlato": raw_config["scheduler"]["reduce_on_plateau"],
        "LR_PATIENCE": raw_config["scheduler"]["patience"],
        "LR_RATIO": raw_config["scheduler"]["ratio"],
        "T0": raw_config["scheduler"]["t0"],
        "T_MULT": raw_config["scheduler"]["t_mult"],
        "ETA_MIN": raw_config["scheduler"]["eta_min"],
    }
    return runtime_config, lrconfig


def get_train_transformation(config):
    transform_steps = []
    if config["HEQV"]:
        transform_steps.append(HistogramEqualizationd(keys=["image"]))
    transform_steps.extend(
        [
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
            RandSpatialCropd(keys=["image", "label"], roi_size=config["IMAGE_SIZE"], random_size=False),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=config["RAND_FLIP_PROB"]),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=config["RAND_FLIP_PROB"]),
            RandRotate90d(
                keys=["image", "label"],
                prob=config["RAND_ROTATE90_PROB"],
                max_k=config["RAND_ROTATE90_MAX_K"],
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=config["RAND_ZOOM_MIN"],
                max_zoom=config["RAND_ZOOM_MAX"],
                prob=config["RAND_ZOOM_PROB"],
            ),
            RandGaussianNoised(keys=["image"], prob=config["RAND_GAUSSIAN_NOISE_PROB"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    transforms = Compose(transform_steps)
    transforms.set_random_state(config["SEED"])
    return transforms


def get_test_transformation(config):
    transform_steps = []
    if config["HEQV"]:
        transform_steps.append(HistogramEqualizationd(keys=["image"]))
    transform_steps.extend(
        [
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return Compose(transform_steps)


def main():
    raw_config, config_path = load_training_config(
        default_config=DEFAULT_CONFIG,
        default_config_path=DEFAULT_CONFIG_PATH,
        description="Train the patch-based multiclass Atlas segmentation model.",
    )
    config, lrconfig = build_runtime_config(raw_config)

    print(f"Using config: {config_path}")
    set_seed(config["SEED"])
    worker_init_fn = make_worker_seed_fn(config["SEED"])
    seeded_generator = get_generator(config["SEED"])

    print("Atlas patch multiclass segmentation model")
    images_path = ATLAS_DATA_DIR / "datasets-PNG"
    labels_path = ATLAS_DATA_DIR / "datasets-NPY"
    folds = get_split_files(
        images_path=images_path,
        labels_path=labels_path,
        folds_path=ATLAS_FOLDS_DIR,
        dataset_name=config["DATASET_NAME"],
        max_files=config["MAX_FILES"],
    )

    for fold_id, fold in enumerate(folds[config["START_FOLD"] :], start=config["START_FOLD"]):
        print(f"Fold {fold_id}")
        print(f"Train images: {len(fold['train']['image_name'])}")
        print(f"Train labels: {len(fold['train']['label_name'])}")
        print(f"Val images: {len(fold['val']['image_name'])}")
        print(f"Val labels: {len(fold['val']['label_name'])}")

        train_dataset = AtlasDataset(
            images=fold["train"]["image_path"],
            masks=fold["train"]["label_path"],
            classes=config["NUM_CLASSES"],
            image_transform=get_train_transformation(config),
            binary=config["BINARY"],
            patches_per_image=config["PATCHES_PER_IMAGE"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["TRAIN_BATCH_SIZE"],
            shuffle=True,
            num_workers=config["NUM_WORKERS"],
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            generator=seeded_generator,
        )

        test_dataset = AtlasDataset(
            images=fold["val"]["image_path"],
            masks=fold["val"]["label_path"],
            classes=config["NUM_CLASSES"],
            image_transform=get_test_transformation(config),
            binary=config["BINARY"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["TEST_BATCH_SIZE"],
            shuffle=False,
            num_workers=config["NUM_WORKERS"],
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            generator=seeded_generator,
        )

        x, y = next(iter(train_loader))
        print("image:", x.shape, x.min().item(), x.mean().item(), x.max().item())
        print("label:", y.shape, y.min().item(), y.max().item())
        print("label sum per-pixel (mean):", y.sum(dim=1).float().mean().item())
        print("label class occupancy:", y.sum(dim=(0, 2, 3)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config, device)
        loss_function = DiceCELoss(
            to_onehot_y=config["LOSS_TO_ONEHOT_Y"],
            softmax=config["LOSS_SOFTMAX"],
            lambda_dice=config["LOSS_LAMBDA_DICE"],
            lambda_ce=config["LOSS_LAMBDA_CE"],
        )
        optimizer = torch.optim.Adam(model.parameters(), config["LR"])

        if not config["TRAIN"]:
            continue

        epoch = 0
        if config["LOAD_TRAIN_MODEL"] is not None:
            model, optimizer, epoch = load_model(
                model,
                optimizer,
                filepath=SEGMENTATION_CHECKPOINTS_DIR / f"{config['LOAD_TRAIN_MODEL']}-fold-{fold_id}.pth",
                device=device,
            )

        training_loop(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=test_loader,
            config=config,
            lrconfig=lrconfig,
            fold_id=fold_id,
            start_epoch=epoch,
        )

        save_model(
            model,
            optimizer,
            epoch=config["MAX_EPOCHS"],
            filepath=SEGMENTATION_CHECKPOINTS_DIR / f"{config['NAME']}-fold-{fold_id}-final.pth",
        )


if __name__ == "__main__":
    main()
