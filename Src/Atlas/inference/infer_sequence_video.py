#!/usr/bin/env python3
"""
infer_sequence_sw_video.py

Sekvenční inference přes více snímků:
- pro každý snímek udělá sliding-window gaussian blending a ULOŽÍ animované framy
  (overlay segmentace + červené aktuální okno) stejně jako v původním skriptu
- všechny framy poskládá do jednoho MP4/GIF videa, takže "video běží" přes snímky

Použití:
python Src/Atlas/inference/infer_sequence_video.py --model_path artifacts/checkpoints/segmentation/Atlas-heqv-multi-100-fold-4-final.pth --input_dir ./input_pngs --pattern "*.png" --output_dir outputs/sequences --num_frames 120 --roi 512 512 --overlap 0.25 --conf_thresh 0.5 --fps 12 --stride_frames 2 --max_frames_per_image 200 --save_mp4

Závislosti:
  pip install monai torch numpy pillow scipy scikit-image imageio imageio-ffmpeg
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

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


# ==========================
# Sliding window (věrná gaussian akumulace) + FRAMES RETURN
# ==========================
def gaussian_weight_2d(roi_h: int, roi_w: int, sigma_scale: float = 0.125) -> np.ndarray:
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

    if ys[-1] != H - roi_h:
        ys.append(max(0, H - roi_h))
    if xs[-1] != W - roi_w:
        xs.append(max(0, W - roi_w))

    for y in ys:
        for x in xs:
            yield y, x


@torch.no_grad()
def sliding_window_gaussian_with_frames(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,      # (1,3,Hpad,Wpad)
    base_pil: Image.Image,         # (Horig,Worig) PIL
    orig_h: int,
    orig_w: int,
    roi_size=(512, 512),
    overlap: float = 0.25,
    conf_thresh: float = 0.5,
    device: torch.device = torch.device("cpu"),
    stride_frames: int = 2,
    max_frames: int = 400,
    alpha_overlay: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Vrátí:
      pred_argmax (Horig,Worig)
      max_prob    (Horig,Worig)
      frames      list[np.ndarray(H,W,3)] – animace SW skládání + obdélník okna
    """
    _, _, H, W = img_tensor.shape
    roi_h, roi_w = roi_size
    if roi_h > H or roi_w > W:
        raise ValueError(f"ROI {roi_size} je větší než vstup {(H,W)} po paddingu.")

    sum_probs = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    sum_w = np.zeros((H, W), dtype=np.float32)

    w2d = gaussian_weight_2d(roi_h, roi_w, sigma_scale=0.125)
    w2d_b = w2d[None, :, :]

    frames: List[np.ndarray] = []
    patch_count = 0

    for y, x in generate_positions(H, W, roi_h, roi_w, overlap):
        patch_count += 1
        patch = img_tensor[:, :, y:y+roi_h, x:x+roi_w].to(device)

        logits = model(patch)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().float().cpu().numpy()

        sum_probs[:, y:y+roi_h, x:x+roi_w] += probs * w2d_b
        sum_w[y:y+roi_h, x:x+roi_w] += w2d

        if patch_count % max(1, stride_frames) == 0:
            denom = (sum_w + 1e-8)
            probs_full = sum_probs / denom[None, :, :]
            pred_full = np.argmax(probs_full, axis=0).astype(np.uint8)

            max_p = np.max(probs_full, axis=0)
            pred_vis = pred_full.copy()
            pred_vis[max_p < conf_thresh] = 0

            pred_vis = pred_vis[:orig_h, :orig_w]
            vis = overlay_mask(base_pil, pred_vis, alpha=alpha_overlay)

            # okno kreslíme v koordinátech paddingu, ale overlay je oříznutý na orig
            # -> okno může částečně "trčet" mimo, tak ho ořízneme do rozsahu orig
            x1 = max(0, min(x, orig_w-1))
            y1 = max(0, min(y, orig_h-1))
            x2 = max(0, min(x + roi_w, orig_w-1))
            y2 = max(0, min(y + roi_h, orig_h-1))

            vis = draw_window(vis, x1, y1, x2, y2, width=4)
            frames.append(np.array(vis))

            if len(frames) >= max_frames:
                break

    denom = (sum_w + 1e-8)
    probs_full = sum_probs / denom[None, :, :]
    pred_full = np.argmax(probs_full, axis=0).astype(np.uint8)
    max_prob = np.max(probs_full, axis=0).astype(np.float32)

    pred_full = pred_full[:orig_h, :orig_w]
    max_prob = max_prob[:orig_h, :orig_w]
    return pred_full, max_prob, frames


# ==========================
# Utils: snímky
# ==========================
def list_frames(input_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def ensure_same_size_np(frame_rgb: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    # target_wh je (W,H)
    pil = Image.fromarray(frame_rgb)
    if pil.size == target_wh:
        return frame_rgb
    pil = pil.resize(target_wh, resample=Image.BILINEAR)
    return np.array(pil)


# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser(description="SW animace pro více snímků -> jedno video.")
    ap.add_argument("--model_path", type=str, required=True)

    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.png")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--num_frames", type=int, default=0, help="Kolik vstupních snímků zpracovat (0 = všechny).")
    ap.add_argument("--start_index", type=int, default=0)

    ap.add_argument("--roi", type=int, nargs=2, default=[512, 512], help="ROI (h w)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf_thresh", type=float, default=0.5)

    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride_frames", type=int, default=2, help="uloží každý N-tý patch jako frame")
    ap.add_argument("--max_frames_per_image", type=int, default=200, help="max počet animovaných framů na 1 vstupní snímek")
    ap.add_argument("--hold_last", type=int, default=0, help="kolik framů zopakovat na konci každého snímku (pauza)")

    ap.add_argument("--save_gif", action="store_true")
    ap.add_argument("--save_mp4", action="store_true")

    ap.add_argument("--alpha_overlay", type=float, default=0.4)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_video_dir = os.path.join(args.output_dir, "video")
    os.makedirs(out_video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    roi_h, roi_w = args.roi
    roi_size = (roi_h, roi_w)

    model = load_trained_model(args.model_path, device)
    transform = get_inference_transform(roi_size=roi_size)

    all_paths = list_frames(args.input_dir, args.pattern)
    if not all_paths:
        raise FileNotFoundError(f"Nenašel jsem nic v {args.input_dir} pro pattern {args.pattern}")

    paths = all_paths[args.start_index:]
    if args.num_frames and args.num_frames > 0:
        paths = paths[:args.num_frames]

    print(f"Input images: {len(paths)} (start_index={args.start_index}, num_frames={args.num_frames})")

    gif_path = os.path.join(out_video_dir, "sequence_sw_overlay.gif") if args.save_gif else None
    mp4_path = os.path.join(out_video_dir, "sequence_sw_overlay.mp4") if args.save_mp4 else None

    writer_mp4 = imageio.get_writer(mp4_path, fps=args.fps) if mp4_path else None
    frames_gif: List[np.ndarray] = []

    target_wh: Optional[Tuple[int, int]] = None  # (W,H)

    total_written = 0

    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        print(f"[{i+1}/{len(paths)}] {stem}")

        img_pil = Image.open(p).convert("RGB")
        img_np_hw3 = np.array(img_pil, dtype=np.float32)
        orig_h, orig_w = img_np_hw3.shape[:2]
        img_np_chw = np.transpose(img_np_hw3, (2, 0, 1))

        data = {"image": img_np_chw, "image_raw": img_np_chw.copy()}
        data = transform(data)

        img_tensor = data["image"].unsqueeze(0).to(device)
        raw_tensor = data["image_raw"]

        raw_img_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
        raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)
        raw_img_np = raw_img_np[:orig_h, :orig_w, :]
        base_pil = Image.fromarray(raw_img_np, mode="RGB")

        # SW + animované framy
        _pred_argmax, _max_prob, sw_frames = sliding_window_gaussian_with_frames(
            model=model,
            img_tensor=img_tensor,
            base_pil=base_pil,
            orig_h=orig_h,
            orig_w=orig_w,
            roi_size=roi_size,
            overlap=args.overlap,
            conf_thresh=args.conf_thresh,
            device=device,
            stride_frames=args.stride_frames,
            max_frames=args.max_frames_per_image,
            alpha_overlay=args.alpha_overlay,
        )

        if not sw_frames:
            continue

        # sjednotit velikost videa podle prvního frame
        if target_wh is None:
            first = sw_frames[0]
            target_wh = (first.shape[1], first.shape[0])  # (W,H)

        # zapsat framy
        last_frame = None
        for fr in sw_frames:
            fr2 = ensure_same_size_np(fr, target_wh)
            last_frame = fr2
            if writer_mp4:
                writer_mp4.append_data(fr2)
            if gif_path:
                frames_gif.append(fr2)
            total_written += 1

        # pauza na konci snímku (volitelně)
        if args.hold_last > 0 and last_frame is not None:
            for _ in range(args.hold_last):
                if writer_mp4:
                    writer_mp4.append_data(last_frame)
                if gif_path:
                    frames_gif.append(last_frame)
                total_written += 1

    if writer_mp4:
        writer_mp4.close()
        print(f"MP4 saved: {mp4_path}")

    if gif_path and frames_gif:
        imageio.mimsave(gif_path, frames_gif, fps=args.fps)
        print(f"GIF saved: {gif_path}")

    print(f"Done. Total frames written: {total_written}")


if __name__ == "__main__":
    main()
