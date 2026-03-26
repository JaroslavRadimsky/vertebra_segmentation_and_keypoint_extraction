#!/usr/bin/env python3
"""
single_inference.py


Příklad použití:

python .\Src\Atlas\single_inference.py --model_path .\Src\Models\Atlas-heqv-multi-patch-10000-fold-0-final.pth --image_path .\Tests\0001035.png --output_dir .\Tests\ --no_blend
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image
import cv2

import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd, ToTensord

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Src.Utils.image_utils import HistogramEqualizationd

from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.measure import label, regionprops


# ==========================
# 1) Konfigurace
# ==========================

NUM_CLASSES = 7
IN_CHANNELS = 3
ROI_SIZE = (512, 512)
CONF_THRESH_DEFAULT = 0.5

def bgr_to_rgb(c):
    b, g, r = c
    return (r, g, b)

VIVID_COLORS_MULTI = {
    1: (0, 255, 255),
    2: (255, 0, 255),
    3: (255, 255, 0),
    4: (0, 128, 255),
    5: (255, 0, 0),
    6: (0, 0, 255),
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: bgr_to_rgb(VIVID_COLORS_MULTI[1]),
    2: bgr_to_rgb(VIVID_COLORS_MULTI[2]),
    3: bgr_to_rgb(VIVID_COLORS_MULTI[3]),
    4: bgr_to_rgb(VIVID_COLORS_MULTI[4]),
    5: bgr_to_rgb(VIVID_COLORS_MULTI[5]),
    6: bgr_to_rgb(VIVID_COLORS_MULTI[6]),
}

def mask_to_color_bgr(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, NUM_CLASSES):
        colored[mask == c] = VIVID_COLORS_MULTI[c]
    return colored


def blend_like_testing_visu(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    colored_mask = mask_to_color_bgr(mask)
    overlay = image_bgr.copy()
    overlay[mask > 0] = colored_mask[mask > 0]
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return image_bgr, colored_mask, blended


# ==========================
# 2) Model + transformace
# ==========================

def build_model(num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS) -> torch.nn.Module:
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def load_trained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Neznámý formát checkpointu (čekal jsem dict).")

    print("Checkpoint keys sample:", list(state_dict.keys())[:5])
    print("Num keys:", len(state_dict))

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    w = next(iter(model.parameters())).detach().float().cpu()
    print(
        f"First param stats after load: "
        f"min={w.min().item():.6f} mean={w.mean().item():.6f} "
        f"max={w.max().item():.6f} std={w.std().item():.6f}"
    )
    return model


def get_inference_transform_sw(roi_size=ROI_SIZE):
    return Compose([
        HistogramEqualizationd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "image_raw"], spatial_size=roi_size),
        ToTensord(keys=["image", "image_raw"]),
    ])


# ==========================
# 3) Postprocessing
# ==========================

def clean_class(mask_cls: np.ndarray, min_size: int = 500) -> np.ndarray:
    if not mask_cls.any():
        return mask_cls

    labeled = label(mask_cls)
    regions = regionprops(labeled)

    for r in regions:
        if r.area < min_size:
            labeled[labeled == r.label] = 0

    labeled = label(labeled > 0)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return np.zeros_like(mask_cls, dtype=bool)

    areas = [r.area for r in regions]
    largest_label = regions[int(np.argmax(areas))].label
    mask_main = (labeled == largest_label)

    struct = np.ones((3, 3), dtype=bool)
    mask_main = binary_opening(mask_main, structure=struct)
    mask_main = binary_closing(mask_main, structure=struct)
    mask_main = binary_fill_holes(mask_main)

    return mask_main


def postprocess_mask(pred_mask: np.ndarray, num_classes: int = NUM_CLASSES, min_size: int = 500) -> np.ndarray:
    h, w = pred_mask.shape
    cleaned = np.zeros((h, w), dtype=np.uint8)

    for cls in range(1, num_classes):
        cls_mask = (pred_mask == cls)
        cls_clean = clean_class(cls_mask, min_size=min_size)
        cleaned[cls_clean] = cls

    return cleaned


def relabel_by_vertical_position(mask: np.ndarray, class_ids=(1, 2, 3, 4, 5, 6)) -> np.ndarray:
    centroids = []

    for cls in class_ids:
        m = (mask == cls)
        if not m.any():
            continue
        lab = label(m)
        regs = regionprops(lab)
        if not regs:
            continue

        areas = [r.area for r in regs]
        r_main = regs[int(np.argmax(areas))]
        y, _x = r_main.centroid
        centroids.append((cls, y))

    centroids_sorted = sorted(centroids, key=lambda t: t[1])

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for new_label, (old_label, _) in enumerate(centroids_sorted, start=1):
        new_mask[mask == old_label] = new_label

    return new_mask


# ==========================
# 4) Inference pro jeden obrázek
# ==========================

@torch.no_grad()
def run_inference_on_image_sw(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    transform,
    roi_size=ROI_SIZE,
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    conf_thresh: float = CONF_THRESH_DEFAULT,
    apply_relabel: bool = True,
    min_size: int = 500,
) -> tuple[np.ndarray, Image.Image]:
    img_pil = Image.open(img_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)
    orig_h, orig_w = img_np_hw3.shape[:2]

    img_np = np.transpose(img_np_hw3, (2, 0, 1))

    data = {"image": img_np, "image_raw": img_np.copy()}
    data = transform(data)

    img_tensor = data["image"].unsqueeze(0).to(device)
    raw_tensor = data["image_raw"]

    logits = sliding_window_inference(
        inputs=img_tensor,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
        device=device,
        sw_device=device,
    )

    probs = torch.softmax(logits, dim=1)
    max_prob, pred = probs.max(dim=1)

    pred_mask = pred.squeeze(0).cpu().numpy()
    max_prob_np = max_prob.squeeze(0).cpu().numpy()

    pred_mask[max_prob_np < conf_thresh] = 0

    cleaned_mask = postprocess_mask(pred_mask, num_classes=NUM_CLASSES, min_size=min_size)
    if apply_relabel:
        cleaned_mask = relabel_by_vertical_position(cleaned_mask)

    cleaned_mask = cleaned_mask[:orig_h, :orig_w].astype(np.uint8)

    raw_img_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
    raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)
    raw_img_np = raw_img_np[:orig_h, :orig_w, :]
    overlay_base = Image.fromarray(raw_img_np, mode="RGB")

    return cleaned_mask, overlay_base


# ==========================
# 5) CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Atlas UNet: sliding window inference + postprocessing pro jeden PNG soubor."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Cesta k uloženému .pth modelu.")
    parser.add_argument("--image_path", type=str, required=True, help="Cesta k jednomu vstupnímu PNG obrázku.")
    parser.add_argument("--output_dir", type=str, required=True, help="Složka pro uložení výstupů.")

    parser.add_argument("--no_relabel", action="store_true", help="Nevykonávat anatomické přerelabelování 1..6.")
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=CONF_THRESH_DEFAULT,
        help=f"Prahová hodnota jistoty softmax (default {CONF_THRESH_DEFAULT}).",
    )
    parser.add_argument("--min_size", type=int, default=500, help="Min. velikost komponenty v pixelech (default 500).")
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Kolik oken najednou na GPU (default 4).")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap oken (default 0.5).")
    
    parser.add_argument("--no_blend", action="store_true", help="Neukládat blended vizualizaci.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Vstupní obrázek neexistuje: {args.image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Použité zařízení: {device}")
    print(f"Načítám model z: {args.model_path}")
    print(f"Vstupní obrázek: {args.image_path}")

    model = load_trained_model(args.model_path, device)
    transform = get_inference_transform_sw(roi_size=ROI_SIZE)

    final_mask, overlay_base = run_inference_on_image_sw(
        model=model,
        img_path=args.image_path,
        device=device,
        transform=transform,
        roi_size=ROI_SIZE,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        conf_thresh=args.conf_thresh,
        apply_relabel=not args.no_relabel,
        min_size=args.min_size,
    )

    img_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Nelze načíst obrázek přes cv2: {args.image_path}")

    img_bgr = img_bgr[: final_mask.shape[0], : final_mask.shape[1], :]

    image_bgr, maskhat_bgr, blended_hat_bgr = blend_like_testing_visu(
        img_bgr, final_mask, alpha=0.5
    )

    fname = os.path.basename(args.image_path)
    stem = os.path.splitext(fname)[0]

    out_maskhat = os.path.join(args.output_dir, f"{stem}_maskhat.png")
    out_blended = os.path.join(args.output_dir, f"{stem}_blended_hat.png")

    cv2.imwrite(out_maskhat, maskhat_bgr)
    if not args.no_blend:
        cv2.imwrite(out_blended, blended_hat_bgr)


    print(f"→ uloženo: {out_maskhat}")
    print(f"→ uloženo: {out_blended}")
    print("Hotovo.")


if __name__ == "__main__":
    main()
