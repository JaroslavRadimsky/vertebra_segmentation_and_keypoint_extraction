import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd, ToTensord
from monai.inferers import sliding_window_inference

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Src.Atlas.inference.inference import (
    build_model,
    load_trained_model,
    get_inference_transform_sw,
    postprocess_mask,
    mask_to_color_bgr,
)

import os



@torch.no_grad()
def infer_single_image(model_path, img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path, device)
    transform = get_inference_transform_sw()

    # 1) Načti a připrav obrázek
    img_pil = Image.open(img_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)
    orig_h, orig_w = img_np_hw3.shape[:2]
    img_np = np.transpose(img_np_hw3, (2, 0, 1))  # (3, H, W)

    data = {"image": img_np, "image_raw": img_np.copy()}
    data = transform(data)
    img_tensor = data["image"].unsqueeze(0).to(device)

    # 2) Inference (sliding window)
    logits = sliding_window_inference(
        inputs=img_tensor,
        roi_size=(512, 512),
        sw_batch_size=4,
        predictor=model,
        overlap=0.25,
        mode="gaussian",
        device=device,
        sw_device=device,
    )
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    # 3) Ulož výstupní masku před postprocessingem
    raw_mask_color = mask_to_color_bgr(pred)
    cv2.imwrite(f"{output_dir}/mask_raw.png", raw_mask_color)

    # 4) Postprocessing
    cleaned_mask = postprocess_mask(pred, num_classes=7, min_size=500)
    cleaned_mask_color = mask_to_color_bgr(cleaned_mask)
    cv2.imwrite(f"{output_dir}/mask_postprocessed.png", cleaned_mask_color)

    print("Hotovo: masky uloženy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Cesta k modelu (.pth)")
    parser.add_argument("--img_path", required=True, help="Cesta k PNG obrázku")
    parser.add_argument("--output_dir", required=True, help="Cesta k výstupní složce")
    args = parser.parse_args()

    infer_single_image(args.model_path, args.img_path, args.output_dir)
