#!/usr/bin/env python3
"""
infer_one_sw_anim_poststeps.py

Jedno-snímková inference Atlas UNet s věrnou akumulací jako MONAI
(sliding window + gaussian blending) +:

1) Animace sliding window průjezdu (GIF/MP4) s postupně se skládající maskou
2) Uložení obrázků pro každý krok postprocessingu, aby bylo vidět čištění masky

Poznámky:
- Implementuje "gaussian" blending: sum_probs += probs * w, sum_w += w, probs = sum_probs / sum_w
- Animace ukazuje:
  - původní obraz
  - průběžný overlay segmentace
  - aktuální okno (červený obdélník)
- Postprocess step images:
  01_raw_pred_argmax.png
  02_after_conf_thresh.png
  03_post_cls1_remove_small.png / 04_post_cls1_keep_largest.png / 05_post_cls1_open_close.png / 06_post_cls1_fill_holes.png
  ... pro každou třídu
  XX_after_postprocess_all_classes.png
  YY_after_relabel_vertical.png (pokud zapnuto)

Použití:
  python Src/Atlas/inference/infer_one_sw_anim_poststeps.py \
  --model_path artifacts/checkpoints/segmentation/Atlas-heqv-multi-100-fold-4-final.pth \
    --image_path ./input_pngs/snim.png \
    --output_dir ./out_one \
    --roi 512 512 \
    --overlap 0.25 \
    --conf_thresh 0.5 \
    --min_size 500 \
    --fps 12 \
    --stride_frames 2 \
    --max_frames 400 \
    --save_mp4

Závislosti:
  pip install monai torch numpy pillow scipy scikit-image imageio imageio-ffmpeg
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd, ToTensord

from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.measure import label, regionprops

import imageio.v2 as imageio


# --------------------------
# Import projektu (HistogramEqualizationd)
# --------------------------
# Pokud máš projektovou strukturu jako ve tvém skriptu:
#   python -m Src....
# tak je to OK. Tady podporujeme i přímé spuštění.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from Src.Utils.image_utils import HistogramEqualizationd
except Exception as e:
    raise ImportError(
        "Nepodařilo se importovat Src.Utils.image_utils.HistogramEqualizationd.\n"
        "Uprav sys.path nebo spusť skript z rootu projektu.\n"
        f"Chyba: {e}"
    )


# ==========================
# Konfigurace
# ==========================
NUM_CLASSES = 7
IN_CHANNELS = 3

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}


# ==========================
# Model + transformace
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

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_inference_transform(roi_size):
    return Compose([
        HistogramEqualizationd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "image_raw"], spatial_size=roi_size),
        ToTensord(keys=["image", "image_raw"]),
    ])


# ==========================
# Vizualizace
# ==========================
def mask_to_color(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color_img[mask == cls] = rgb
    return Image.fromarray(color_img, mode="RGB")


def overlay_mask(base_rgb: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
    cm = mask_to_color(mask)
    return Image.blend(base_rgb, cm, alpha=alpha)


def draw_window(pil_img: Image.Image, x1, y1, x2, y2, width=4) -> Image.Image:
    out = pil_img.copy()
    dr = ImageDraw.Draw(out)
    dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=width)
    return out


def save_img(pil_img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_img.save(path)


# ==========================
# Sliding window (věrná gaussian akumulace)
# ==========================
def gaussian_weight_2d(roi_h: int, roi_w: int, sigma_scale: float = 0.125) -> np.ndarray:
    """
    2D gaussian weight window (peak=1.0).
    sigma ~ sigma_scale * roi_dim (podobně jako MONAI gaussian blending).
    """
    ys = np.arange(roi_h, dtype=np.float32)
    xs = np.arange(roi_w, dtype=np.float32)
    cy = (roi_h - 1) / 2.0
    cx = (roi_w - 1) / 2.0

    sy = max(1e-6, roi_h * sigma_scale)
    sx = max(1e-6, roi_w * sigma_scale)

    gy = np.exp(-0.5 * ((ys - cy) / sy) ** 2)
    gx = np.exp(-0.5 * ((xs - cx) / sx) ** 2)
    w = np.outer(gy, gx)
    w = w / (w.max() + 1e-8)
    return w.astype(np.float32)


def generate_positions(H, W, roi_h, roi_w, overlap: float):
    sh = max(1, int(round(roi_h * (1.0 - overlap))))
    sw = max(1, int(round(roi_w * (1.0 - overlap))))

    ys = list(range(0, max(1, H - roi_h + 1), sh))
    xs = list(range(0, max(1, W - roi_w + 1), sw))

    # "na doraz"
    if ys[-1] != H - roi_h:
        ys.append(max(0, H - roi_h))
    if xs[-1] != W - roi_w:
        xs.append(max(0, W - roi_w))

    for y in ys:
        for x in xs:
            yield y, x


@torch.no_grad()
def sliding_window_gaussian_with_animation(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,      # (1,3,H,W) padded
    base_pil: Image.Image,         # (Horig,Worig)
    orig_h: int,
    orig_w: int,
    roi_size=(512, 512),
    overlap: float = 0.25,
    conf_thresh: float = 0.5,
    device: torch.device = torch.device("cpu"),
    save_gif_path: str = None,
    save_mp4_path: str = None,
    fps: int = 12,
    stride_frames: int = 2,
    max_frames: int = 400,
    alpha_overlay: float = 0.4,
):
    _, _, H, W = img_tensor.shape
    roi_h, roi_w = roi_size
    if roi_h > H or roi_w > W:
        raise ValueError(f"ROI {roi_size} je větší než vstup {(H,W)} po paddingu. Zkontroluj SpatialPad/roi.")

    # akumulace v float32 na CPU (můžeš přepnout na GPU, ale pro animaci to typicky stačí na CPU)
    sum_probs = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    sum_w = np.zeros((H, W), dtype=np.float32)

    w2d = gaussian_weight_2d(roi_h, roi_w, sigma_scale=0.125)  # podobné MONAI gaussian
    w2d_b = w2d[None, :, :]  # (1,roi_h,roi_w)

    frames = []
    frame_count = 0
    patch_count = 0

    # predikce průběžná (bez conf thresh zatím) jen pro animaci (můžeš chtít i s thresh)
    for y, x in generate_positions(H, W, roi_h, roi_w, overlap):
        patch_count += 1
        patch = img_tensor[:, :, y:y+roi_h, x:x+roi_w].to(device)   # (1,3,roi_h,roi_w)

        logits = model(patch)                                       # (1,7,roi_h,roi_w)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().float().cpu().numpy()  # (7,roi_h,roi_w)

        # gaussian blending
        sum_probs[:, y:y+roi_h, x:x+roi_w] += probs * w2d_b
        sum_w[y:y+roi_h, x:x+roi_w] += w2d

        # frame?
        if (save_gif_path or save_mp4_path) and (patch_count % max(1, stride_frames) == 0):
            # safe division
            denom = (sum_w + 1e-8)
            probs_full = sum_probs / denom[None, :, :]    # (7,H,W)
            pred_full = np.argmax(probs_full, axis=0).astype(np.uint8)

            # pro animaci můžeš rovnou aplikovat conf thresh na max prob
            max_prob = np.max(probs_full, axis=0)
            pred_vis = pred_full.copy()
            pred_vis[max_prob < conf_thresh] = 0

            pred_vis = pred_vis[:orig_h, :orig_w]
            vis = overlay_mask(base_pil, pred_vis, alpha=alpha_overlay)
            vis = draw_window(vis, x, y, x+roi_w, y+roi_h, width=4)

            frames.append(np.array(vis))
            frame_count += 1
            if frame_count >= max_frames:
                break

    # final full probs
    denom = (sum_w + 1e-8)
    probs_full = sum_probs / denom[None, :, :]
    pred_full = np.argmax(probs_full, axis=0).astype(np.uint8)
    max_prob = np.max(probs_full, axis=0).astype(np.float32)

    # ulož animaci
    if (save_gif_path or save_mp4_path) and len(frames) > 0:
        if save_gif_path:
            imageio.mimsave(save_gif_path, frames, fps=fps)
        if save_mp4_path:
            imageio.mimsave(save_mp4_path, frames, fps=fps)  # vyžaduje imageio-ffmpeg

    # oříznout na orig
    pred_full = pred_full[:orig_h, :orig_w]
    max_prob = max_prob[:orig_h, :orig_w]

    return pred_full, max_prob, probs_full  # probs_full je stále (7,Hpad,Wpad) na CPU


# ==========================
# Postprocessing (krokované ukládání)
# ==========================
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


def _save_step(base_pil: Image.Image, mask: np.ndarray, out_path: str, alpha=0.4):
    ov = overlay_mask(base_pil, mask, alpha=alpha)
    save_img(ov, out_path)


def clean_class_with_steps(
    base_pil: Image.Image,
    pred_mask: np.ndarray,
    cls: int,
    out_dir_steps: str,
    min_size: int = 500,
    alpha: float = 0.4,
):
    """
    Vrací vyčištěnou bool masku pro třídu cls + ukládá mezikroky.
    """
    cls_mask = (pred_mask == cls)
    if not cls_mask.any():
        # uložit "prázdný" krok pro konzistenci (volitelné)
        return cls_mask

    # A) remove small components
    lab = label(cls_mask)
    regs = regionprops(lab)
    lab_a = lab.copy()
    for r in regs:
        if r.area < min_size:
            lab_a[lab_a == r.label] = 0
    mask_a = lab_a > 0
    _save_step(base_pil, (mask_a.astype(np.uint8) * cls), os.path.join(out_dir_steps, f"cls{cls:02d}_03_remove_small.png"), alpha)

    # B) keep largest
    lab_b = label(mask_a)
    regs_b = regionprops(lab_b)
    if len(regs_b) == 0:
        mask_b = np.zeros_like(cls_mask, dtype=bool)
        _save_step(base_pil, (mask_b.astype(np.uint8) * cls), os.path.join(out_dir_steps, f"cls{cls:02d}_04_keep_largest.png"), alpha)
        return mask_b

    areas = [r.area for r in regs_b]
    largest_label = regs_b[int(np.argmax(areas))].label
    mask_b = (lab_b == largest_label)
    _save_step(base_pil, (mask_b.astype(np.uint8) * cls), os.path.join(out_dir_steps, f"cls{cls:02d}_04_keep_largest.png"), alpha)

    # C) opening/closing
    struct = np.ones((3, 3), dtype=bool)
    mask_c = binary_opening(mask_b, structure=struct)
    mask_c = binary_closing(mask_c, structure=struct)
    _save_step(base_pil, (mask_c.astype(np.uint8) * cls), os.path.join(out_dir_steps, f"cls{cls:02d}_05_open_close.png"), alpha)

    # D) fill holes
    mask_d = binary_fill_holes(mask_c)
    _save_step(base_pil, (mask_d.astype(np.uint8) * cls), os.path.join(out_dir_steps, f"cls{cls:02d}_06_fill_holes.png"), alpha)

    return mask_d.astype(bool)


def postprocess_with_steps(
    base_pil: Image.Image,
    pred_mask: np.ndarray,
    out_dir_steps: str,
    min_size: int = 500,
    apply_relabel: bool = True,
    alpha: float = 0.4,
):
    """
    pred_mask: (H,W) uint8 0..6 (už po conf thresh)
    Ukládá průběžné výsledky.
    """
    os.makedirs(out_dir_steps, exist_ok=True)

    # 00: raw input pred (už argmax)
    _save_step(base_pil, pred_mask, os.path.join(out_dir_steps, "00_pred_argmax.png"), alpha)

    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)

    # pro každou třídu uložíme i "start" masku (před čištěním)
    for cls in range(1, NUM_CLASSES):
        cls_start = (pred_mask == cls).astype(np.uint8) * cls
        _save_step(base_pil, cls_start, os.path.join(out_dir_steps, f"cls{cls:02d}_02_before_clean.png"), alpha)

        cls_clean_bool = clean_class_with_steps(
            base_pil=base_pil,
            pred_mask=pred_mask,
            cls=cls,
            out_dir_steps=out_dir_steps,
            min_size=min_size,
            alpha=alpha,
        )
        cleaned[cls_clean_bool] = cls

    _save_step(base_pil, cleaned, os.path.join(out_dir_steps, "10_after_postprocess_all_classes.png"), alpha)

    if apply_relabel:
        rel = relabel_by_vertical_position(cleaned)
        _save_step(base_pil, rel, os.path.join(out_dir_steps, "11_after_relabel_vertical.png"), alpha)
        return rel

    return cleaned


# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Jedno-snímková SW inference s animací a krokovým postprocess.")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--roi", type=int, nargs=2, default=[512, 512], help="ROI (h w)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf_thresh", type=float, default=0.5)
    ap.add_argument("--min_size", type=int, default=500)
    ap.add_argument("--no_relabel", action="store_true")

    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride_frames", type=int, default=2, help="uloží každý N-tý patch jako frame")
    ap.add_argument("--max_frames", type=int, default=400)

    ap.add_argument("--save_gif", action="store_true", help="uložit GIF animaci")
    ap.add_argument("--save_mp4", action="store_true", help="uložit MP4 animaci (vyžaduje imageio-ffmpeg)")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_anim_dir = os.path.join(args.output_dir, "anim")
    out_steps_dir = os.path.join(args.output_dir, "post_steps")
    os.makedirs(out_anim_dir, exist_ok=True)
    os.makedirs(out_steps_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    roi_h, roi_w = args.roi
    roi_size = (roi_h, roi_w)

    model = load_trained_model(args.model_path, device)
    transform = get_inference_transform(roi_size=roi_size)

    # load image
    img_pil = Image.open(args.image_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)
    orig_h, orig_w = img_np_hw3.shape[:2]
    img_np_chw = np.transpose(img_np_hw3, (2, 0, 1))  # (3,H,W)

    data = {"image": img_np_chw, "image_raw": img_np_chw.copy()}
    data = transform(data)

    img_tensor = data["image"].unsqueeze(0).to(device)   # (1,3,Hpad,Wpad)
    raw_tensor = data["image_raw"]                        # (3,Hpad,Wpad)

    raw_img_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
    raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)
    raw_img_np = raw_img_np[:orig_h, :orig_w, :]
    base_pil = Image.fromarray(raw_img_np, mode="RGB")

    # save base
    save_img(base_pil, os.path.join(args.output_dir, "base_image.png"))

    gif_path = None
    mp4_path = None
    stem = os.path.splitext(os.path.basename(args.image_path))[0]
    if args.save_gif:
        gif_path = os.path.join(out_anim_dir, f"{stem}_sw_gaussian.gif")
    if args.save_mp4:
        mp4_path = os.path.join(out_anim_dir, f"{stem}_sw_gaussian.mp4")

    # SW gaussian + animation
    print("Running sliding window gaussian blending + animation...")
    pred_argmax, max_prob, _probs_full = sliding_window_gaussian_with_animation(
        model=model,
        img_tensor=img_tensor,
        base_pil=base_pil,
        orig_h=orig_h,
        orig_w=orig_w,
        roi_size=roi_size,
        overlap=args.overlap,
        conf_thresh=args.conf_thresh,
        device=device,
        save_gif_path=gif_path,
        save_mp4_path=mp4_path,
        fps=args.fps,
        stride_frames=args.stride_frames,
        max_frames=args.max_frames,
        alpha_overlay=0.4,
    )

    # 01: raw pred argmax overlay
    _save_step(base_pil, pred_argmax, os.path.join(args.output_dir, "01_raw_pred_argmax_overlay.png"), alpha=0.4)
    mask_to_color(pred_argmax).save(os.path.join(args.output_dir, "01_raw_pred_argmax_color.png"))

    # 02: apply conf threshold
    pred_thresh = pred_argmax.copy()
    pred_thresh[max_prob < float(args.conf_thresh)] = 0

    _save_step(base_pil, pred_thresh, os.path.join(args.output_dir, "02_after_conf_thresh_overlay.png"), alpha=0.4)
    mask_to_color(pred_thresh).save(os.path.join(args.output_dir, "02_after_conf_thresh_color.png"))

    # Postprocess with steps (per class + final + relabel)
    print("Running postprocess with step images...")
    final_mask = postprocess_with_steps(
        base_pil=base_pil,
        pred_mask=pred_thresh.astype(np.uint8),
        out_dir_steps=out_steps_dir,
        min_size=int(args.min_size),
        apply_relabel=(not args.no_relabel),
        alpha=0.4,
    )

    # Save final outputs
    mask_to_color(final_mask).save(os.path.join(args.output_dir, "final_mask_color.png"))
    final_overlay = overlay_mask(base_pil, final_mask, alpha=0.4)
    save_img(final_overlay, os.path.join(args.output_dir, "final_overlay.png"))

    print("Done.")
    if gif_path:
        print(f"GIF:  {gif_path}")
    if mp4_path:
        print(f"MP4:  {mp4_path}")
    print(f"Post steps dir: {out_steps_dir}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
