from __future__ import annotations

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    ToTensord,
)

from Src.Utils.image_utils import HistogramEqualizationd


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
        "LOSS_INCLUDE_BACKGROUND": raw_config["loss"].get("include_background", False),
        "LOSS_TO_ONEHOT_Y": raw_config["loss"]["to_onehot_y"],
        "LOSS_SOFTMAX": raw_config["loss"]["softmax"],
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
    transforms = Compose(
        [
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
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
    transforms.set_random_state(config["SEED"])
    return transforms


def get_train_transformation_with_heqv(config):
    transforms = Compose(
        [
            HistogramEqualizationd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
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
    transforms.set_random_state(config["SEED"])
    return transforms


def get_test_transformation(config):
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_test_transformation_with_heqv(config):
    return Compose(
        [
            HistogramEqualizationd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["IMAGE_SIZE"]),
            ToTensord(keys=["image", "label"]),
        ]
    )


def build_model(config, device):
    return UNet(
        spatial_dims=2,
        in_channels=config["MODEL_IN_CHANNELS"],
        out_channels=config["NUM_CLASSES"],
        channels=config["MODEL_CHANNELS"],
        strides=config["MODEL_STRIDES"],
        num_res_units=config["MODEL_NUM_RES_UNITS"],
    ).to(device)
