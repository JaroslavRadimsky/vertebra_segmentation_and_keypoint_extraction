import json
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

Point = Tuple[float, float]

# ---------- I/O ----------
def load_points(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = {}
    for shp in data.get("shapes", []):
        label = shp.get("label", "")
        coords = shp.get("points", [])
        if not coords:
            continue
        x, y = coords[0]
        pts[label] = (float(x), float(y))
    return pts, data.get("imagePath", None)

def resolve_image_path(json_path: Path, image_arg: Optional[Path], img_dir: Optional[Path], image_path_in_json: Optional[str]) -> Path:
    if image_arg is not None:
        return image_arg
    if image_path_in_json:
        ip = Path(image_path_in_json)
        if img_dir:
            cand = img_dir / ip.name
            if cand.exists():
                return cand
        cand = json_path.parent / ip
        if cand.exists():
            return cand
    stem = json_path.stem
    candidates = []
    if img_dir:
        candidates += [img_dir / f"{stem}.png", img_dir / f"{stem}.jpg", img_dir / f"{stem}.jpeg"]
    candidates += [json_path.parent / f"{stem}.png", json_path.parent / f"{stem}.jpg", json_path.parent / f"{stem}.jpeg"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Unable to locate image file. Provide --image or --img_dir, or ensure JSON has a valid imagePath.")

# ---------- geometry helpers ----------
def angle_deg_between_lines(p1: Point, p2: Point, q1: Point, q2: Point) -> float:
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)
    v2 = np.array([q2[0]-q1[0], q2[1]-q1[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    det = float(v1[0]*v2[1] - v1[1]*v2[0])
    ang = math.degrees(math.atan2(det, dot))
    return ang

def angle_wrt_horizontal_deg(p1: Point, p2: Point) -> float:
    v = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)
    return math.degrees(math.atan2(v[1], v[0]))

def draw_infinite_line(img, p, q, color, thickness):
    h, w = img.shape[:2]
    if p == q:
        return
    x1, y1 = p; x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    t_vals = []
    def add_t(x=None, y=None):
        if x is not None and dx != 0:
            t_vals.append((x - x1) / dx)
        if y is not None and dy != 0:
            t_vals.append((y - y1) / dy)
    add_t(x=0); add_t(x=w-1); add_t(y=0); add_t(y=h-1)
    pts = []
    for t in t_vals:
        X = x1 + t*dx; Y = y1 + t*dy
        if 0 <= X <= w-1 and 0 <= Y <= h-1:
            pts.append((int(round(X)), int(round(Y))))
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[-1], color, thickness, lineType=cv2.LINE_AA)

def midpoint(a, b):
    return (int(round((a[0]+b[0])/2)), int(round((a[1]+b[1])/2)))

def norm360(a):
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def line_from_pts(A, B):
    A = (float(A[0]), float(A[1])); B = (float(B[0]), float(B[1]))
    # Ax + By + C = 0
    return (A[1]-B[1], B[0]-A[0], A[0]*B[1]-B[0]*A[1])

def intersect(L1, L2):
    A1, B1, C1 = L1; A2, B2, C2 = L2
    det = A1*B2 - A2*B1
    if abs(det) < 1e-6:
        return None
    x = (B1*C2 - B2*C1) / det
    y = (C1*A2 - C2*A1) / det
    return (int(round(x)), int(round(y)))

def draw_ray(img, origin, angle_deg, length, color, thickness):
    theta = math.radians(angle_deg)
    end = (int(origin[0] + length*math.cos(theta)), int(origin[1] + length*math.sin(theta)))
    cv2.line(img, origin, end, color, thickness, lineType=cv2.LINE_AA)
    return end

def draw_angle_arc(img, center, ang_a_deg, ang_b_deg, radius, color, thickness):
    A = norm360(ang_a_deg)
    B = norm360(ang_b_deg)
    d = (B - A) % 360.0
    if d <= 180.0:
        start, end = A, B
    else:
        start, end = B, A
    cv2.ellipse(img, center, (radius, radius), 0, start, end, color, thickness, lineType=cv2.LINE_AA)

def signed_distance_point_to_line(A, B, P):
    Ax, Ay = A; Bx, By = B; Px, Py = P
    vx, vy = Bx - Ax, By - Ay
    wx, wy = Px - Ax, Py - Ay
    cross = vx*wy - vy*wx
    norm  = math.hypot(vx, vy)
    if norm == 0:
        return 0.0
    return cross / norm

def px_tolerance_from_scale(mm_per_px: float | None, px_per_mm: float | None, default_px: float = 2.0) -> float:
    if mm_per_px is not None and mm_per_px > 0:
        return 2.0 / mm_per_px
    if px_per_mm is not None and px_per_mm > 0:
        return 2.0 * px_per_mm
    return default_px

def posterior_point_for_vertebra(pts: Dict[str, Tuple[float,float]], v: str) -> Tuple[float,float] | None:
    keys = [f"{v} top right", f"{v} bottom right"]
    have = [k for k in keys if k in pts]
    if len(have) == 2:
        a, b = pts[have[0]], pts[have[1]]
        return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
    if have:
        return pts[have[0]]
    return None

def get_endplate(pts: Dict[str, Point], vertebra: str, plate: str):
    """
    plate: 'top' nebo 'bottom'
    vrací (left_point, right_point)
    """
    lk = f"{vertebra} {plate} left"
    rk = f"{vertebra} {plate} right"
    if lk not in pts or rk not in pts:
        return None
    return pts[lk], pts[rk]

# ---------- metrics ----------
def compute_metrics(pts: Dict[str, Tuple[float,float]]) -> Dict[str, float]:
    required = [
        "C2 bottom left", "C2 bottom right", "C2 centroid",
        "C7 bottom left", "C7 bottom right", "C7 top left"
    ]
    for k in required:
        if k not in pts:
            missing = ", ".join([r for r in required if r not in pts])
            raise KeyError(f"Missing required points: {missing}")

    c2_inf = (pts["C2 bottom left"], pts["C2 bottom right"])
    c7_inf = (pts["C7 bottom left"], pts["C7 bottom right"])

    slope_c2 = -angle_wrt_horizontal_deg(*c2_inf)
    cobb_c2_c7 = -angle_deg_between_lines(*c2_inf, *c7_inf)

    x_c2c = pts["C2 centroid"][0]
    x_c7_tl = pts["C7 top left"][0]
    sva_signed = x_c2c - x_c7_tl
    sva_abs = abs(sva_signed)

    metrics = {
        "Cobb_C2_C7_deg": cobb_c2_c7,
        "Slope_C2_deg": slope_c2,
        "SVA_C2_C7_px": sva_signed,
        "SVA_C2_C7_abs_px": sva_abs,
    }

    # segmentální úhly mezi sousedními obratli
    vertebrae = ["C2", "C3", "C4", "C5", "C6", "C7"]
    for i in range(len(vertebrae) - 1):
        upper = vertebrae[i]
        lower = vertebrae[i + 1]

        upper_inf = get_endplate(pts, upper, "bottom")
        lower_sup = get_endplate(pts, lower, "top")

        if upper_inf is None or lower_sup is None:
            continue

        seg_angle = -angle_deg_between_lines(
            upper_inf[0], upper_inf[1],
            lower_sup[0], lower_sup[1]
        )
        metrics[f"Segmental_{upper}_{lower}_deg"] = seg_angle

    return metrics

def toyama_classify(pts: Dict[str, Tuple[float,float]],
                    px_tol: float,
                    flip_side: bool=False) -> tuple[str, dict]:
    required = ["C2 bottom right", "C7 top right"]
    for k in required:
        if k not in pts:
            return "rovný tvar", {"reason": f"missing {k}", "counts": {}, "per_level": {}}

    A = pts["C2 bottom right"]
    B = pts["C7 top right"]

    vertebrae = ["C3","C4","C5","C6"]
    cats = {}
    counts = {"on":0, "pos":0, "neg":0}

    for v in vertebrae:
        P = posterior_point_for_vertebra(pts, v)
        if P is None:
            continue
        d = signed_distance_point_to_line(A, B, P)
        if abs(d) <= px_tol:
            cats[v] = "on"
        else:
            cats[v] = "pos" if d > 0 else "neg"
        counts[cats[v]] += 1

    if not cats:
        return "rovný tvar", {"reason":"no posterior points", "counts": counts, "per_level": cats}

    if counts["on"] == len(cats):
        return "rovný tvar", {"counts": counts, "per_level": cats}

    if counts["pos"] > 0 and counts["neg"] == 0:
        label = "kyfoza" if flip_side else "lordoza"
        return label, {"counts": counts, "per_level": cats}
    if counts["neg"] > 0 and counts["pos"] == 0:
        label = "lordoza" if flip_side else "kyfoza"
        return label, {"counts": counts, "per_level": cats}

    return "esovitý tvar", {"counts": counts, "per_level": cats}

# ---------- common canvas ----------
def base_canvas(image_path: Path, pts: Dict[str, Tuple[float,float]], draw_points=True):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    r = max(2, int(round((h + w) / 1000)))
    thick = max(1, int(round((h + w) / 1200)))
    font_scale = max(0.40, min(0.9, (h + w) / 3500))

    if draw_points:
        for (x, y) in pts.values():
            p = (int(round(x)), int(round(y)))
            cv2.circle(vis, p, r, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return vis, h, w, r, thick, font_scale

# ---------- COBB ----------
def draw_cobb(image_path: Path, pts: Dict[str, Tuple[float,float]], metrics: Dict[str, float], out_path: Path, draw_points=True):
    vis, h, w, r, thick, font_scale = base_canvas(image_path, pts, draw_points)
    c2l = tuple(map(int, map(round, pts["C2 bottom left"])))
    c2r = tuple(map(int, map(round, pts["C2 bottom right"])))
    c7l = tuple(map(int, map(round, pts["C7 bottom left"])))
    c7r = tuple(map(int, map(round, pts["C7 bottom right"])))

    draw_infinite_line(vis, c2l, c2r, (255, 0, 0), thick)
    draw_infinite_line(vis, c7l, c7r, (0, 255, 255), thick)

    v_c2 = np.array([c2r[0]-c2l[0], c2r[1]-c2l[1]], dtype=float)
    v_c7 = np.array([c7r[0]-c7l[0], c7r[1]-c7l[1]], dtype=float)
    v_c2_perp = np.array([-v_c2[1], v_c2[0]], dtype=float)
    v_c7_perp = np.array([-v_c7[1], v_c7[0]], dtype=float)

    mid_c2 = midpoint(c2l, c2r)
    mid_c7 = midpoint(c7l, c7r)
    offset_x = int(0.15 * w)
    mid_c2s = (mid_c2[0] + offset_x, mid_c2[1])
    mid_c7s = (mid_c7[0] + offset_x, mid_c7[1])

    p2_c2s = (int(mid_c2s[0] + v_c2_perp[0]), int(mid_c2s[1] + v_c2_perp[1]))
    p2_c7s = (int(mid_c7s[0] + v_c7_perp[0]), int(mid_c7s[1] + v_c7_perp[1]))

    draw_infinite_line(vis, mid_c2s, p2_c2s, (255, 150, 0), max(1, thick-1))
    draw_infinite_line(vis, mid_c7s, p2_c7s, (0, 200, 255), max(1, thick-1))

    Lp1 = line_from_pts(mid_c2s, p2_c2s)
    Lp2 = line_from_pts(mid_c7s, p2_c7s)
    P = intersect(Lp1, Lp2)
    if P is None:
        P = (int(0.6*w), int(0.2*h))

    ang1 = math.degrees(math.atan2(v_c2_perp[1], v_c2_perp[0]))
    ang2 = math.degrees(math.atan2(v_c7_perp[1], v_c7_perp[0]))

    L = max(60, int(0.05*(h+w)))
    draw_ray(vis, P, ang1, L, (255, 150, 0), max(1, thick-1))
    draw_ray(vis, P, ang2, L, (0, 200, 255), max(1, thick-1))
    rad = max(30, int(0.035*(h+w)))
    draw_angle_arc(vis, P, ang1, ang2, rad, (0, 255, 0), max(1, thick-1))

    txt = f"Cobb C2-C7: {abs(metrics['Cobb_C2_C7_deg']):.3f} deg"
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------- SVA ----------
def draw_sva(image_path: Path, pts: Dict[str, Tuple[float,float]], metrics: Dict[str, float], out_path: Path, draw_points=True):
    vis, h, w, r, thick, font_scale = base_canvas(image_path, pts, draw_points)
    c2c = tuple(map(int, map(round, pts["C2 centroid"])))
    c7_tl = tuple(map(int, map(round, pts["C7 top left"])))

    cv2.line(vis, (c2c[0], 0), (c2c[0], h-1), (0, 165, 255), max(1, thick-1), lineType=cv2.LINE_AA)
    cv2.line(vis, (min(c2c[0], c7_tl[0]), c7_tl[1]), (max(c2c[0], c7_tl[0]), c7_tl[1]), (0, 165, 255), max(1, thick-1), lineType=cv2.LINE_AA)

    txt = f"SVA C2-C7: {metrics['SVA_C2_C7_px']:.1f} px"
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------- C2 slope ----------
def smallest_signed_delta_deg(a_deg: float, b_deg: float) -> float:
    return (b_deg - a_deg + 540.0) % 360.0 - 180.0

def draw_small_arc_from_horizontal(img, center, delta_deg, radius, color, thickness):
    if delta_deg >= 0:
        start, end = 0.0, float(delta_deg)
    else:
        start, end = 360.0 + float(delta_deg), 360.0
    cv2.ellipse(img, center, (radius, radius), 0, start, end, color, thickness, lineType=cv2.LINE_AA)

def draw_c2_slope(image_path: Path,
                  pts: Dict[str, Tuple[float,float]],
                  metrics: Dict[str, float],
                  out_path: Path,
                  draw_points=True,
                  arc_radius_px: int | None = None,
                  arc_offset_px: int = 0):
    vis, h, w, r, thick, font_scale = base_canvas(image_path, pts, draw_points)
    c2l = tuple(map(int, map(round, pts["C2 bottom left"])))
    c2r = tuple(map(int, map(round, pts["C2 bottom right"])))

    draw_infinite_line(vis, c2l, c2r, (255, 0, 0), max(1, thick))

    mid_c2 = midpoint(c2l, c2r)
    cv2.line(vis, (0, mid_c2[1]), (w-1, mid_c2[1]), (200, 200, 200), max(1, thick-1), lineType=cv2.LINE_AA)

    ang_c2 = math.degrees(math.atan2(c2r[1]-c2l[1], c2r[0]-c2l[0]))
    delta  = smallest_signed_delta_deg(0.0, ang_c2)

    L   = max(200, int(0.05*(h+w)))
    rad = arc_radius_px if arc_radius_px is not None else max(200, int(0.035*(h+w)))

    bisector_deg = delta / 2.0
    cx, cy = mid_c2
    if arc_offset_px:
        cx = int(round(cx + arc_offset_px * math.cos(math.radians(bisector_deg))))
        cy = int(round(cy + arc_offset_px * math.sin(math.radians(bisector_deg))))
    arc_center = (cx, cy)

    draw_ray(vis, arc_center, 0.0,  L, (200, 200, 200), max(1, thick-1))
    draw_ray(vis, arc_center, ang_c2, L, (255, 0, 0),   max(1, thick-1))
    draw_small_arc_from_horizontal(vis, arc_center, delta, rad, (0, 255, 0), max(1, thick-1))

    slope_deg = metrics["Slope_C2_deg"]
    txt = f"C2 slope: {slope_deg:.3f} deg"
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------- segmentální úhly ----------
def draw_segmental_angle(image_path: Path,
                         pts: Dict[str, Tuple[float,float]],
                         metrics: Dict[str, float],
                         upper: str,
                         lower: str,
                         out_path: Path,
                         draw_points: bool = True):
    vis, h, w, r, thick, font_scale = base_canvas(image_path, pts, draw_points)

    upper_inf = get_endplate(pts, upper, "bottom")
    lower_sup = get_endplate(pts, lower, "top")
    if upper_inf is None or lower_sup is None:
        raise KeyError(f"Missing points for segment {upper}/{lower}")

    u_l = tuple(map(int, map(round, upper_inf[0])))
    u_r = tuple(map(int, map(round, upper_inf[1])))
    l_l = tuple(map(int, map(round, lower_sup[0])))
    l_r = tuple(map(int, map(round, lower_sup[1])))

    # dvě ploténky
    draw_infinite_line(vis, u_l, u_r, (255, 0, 0), thick)
    draw_infinite_line(vis, l_l, l_r, (0, 255, 255), thick)

    # kolmice pro vykreslení úhlu
    v1 = np.array([u_r[0]-u_l[0], u_r[1]-u_l[1]], dtype=float)
    v2 = np.array([l_r[0]-l_l[0], l_r[1]-l_l[1]], dtype=float)
    v1_perp = np.array([-v1[1], v1[0]], dtype=float)
    v2_perp = np.array([-v2[1], v2[0]], dtype=float)

    mid1 = midpoint(u_l, u_r)
    mid2 = midpoint(l_l, l_r)

    offset_x = int(0.12 * w)
    mid1s = (mid1[0] + offset_x, mid1[1])
    mid2s = (mid2[0] + offset_x, mid2[1])

    p1b = (int(mid1s[0] + v1_perp[0]), int(mid1s[1] + v1_perp[1]))
    p2b = (int(mid2s[0] + v2_perp[0]), int(mid2s[1] + v2_perp[1]))

    draw_infinite_line(vis, mid1s, p1b, (255, 150, 0), max(1, thick-1))
    draw_infinite_line(vis, mid2s, p2b, (0, 200, 255), max(1, thick-1))

    L1 = line_from_pts(mid1s, p1b)
    L2 = line_from_pts(mid2s, p2b)
    P = intersect(L1, L2)
    if P is None:
        P = (int(0.65*w), int(0.25*h))

    ang1 = math.degrees(math.atan2(v1_perp[1], v1_perp[0]))
    ang2 = math.degrees(math.atan2(v2_perp[1], v2_perp[0]))

    L = max(60, int(0.05*(h+w)))
    draw_ray(vis, P, ang1, L, (255, 150, 0), max(1, thick-1))
    draw_ray(vis, P, ang2, L, (0, 200, 255), max(1, thick-1))
    rad = max(30, int(0.035*(h+w)))
    draw_angle_arc(vis, P, ang1, ang2, rad, (0, 255, 0), max(1, thick-1))

    key = f"Segmental_{upper}_{lower}_deg"
    txt = f"{upper}/{lower}: {metrics[key]:.3f} deg"
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------- Toyama ----------
# def draw_toyama(image_path: Path,
#                 pts: Dict[str, Tuple[float,float]],
#                 label: str,
#                 details: dict,
#                 out_path: Path,
#                 draw_points: bool=True):
#     vis, h, w, r, thick, font_scale = base_canvas(image_path, pts, draw_points)

#     if "C2 bottom right" in pts and "C7 top right" in pts:
#         A = tuple(map(int, map(round, pts["C2 bottom right"])))
#         B = tuple(map(int, map(round, pts["C7 top right"])))
#         cv2.line(vis, A, B, (0, 255, 0), max(1, thick), lineType=cv2.LINE_AA)

#     color_map = {"on": (200,200,200), "pos": (0,140,255), "neg": (255,80,80)}
#     for v in ["C3","C4","C5","C6"]:
#         P = posterior_point_for_vertebra(pts, v)
#         if P is None:
#             continue
#         cat = details.get("per_level", {}).get(v, "on")
#         C = color_map.get(cat, (200,200,200))
#         p = (int(round(P[0])), int(round(P[1])))
#         cv2.circle(vis, p, max(3, r+1), C, -1, lineType=cv2.LINE_AA)
#         cv2.putText(vis, v, (p[0]+4, p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, font_scale*0.9), C, 1, cv2.LINE_AA)

#     cv2.putText(vis, f"Toyama: {label}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), 2, cv2.LINE_AA)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), vis)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Draw each metric into a separate image, including segmental angles.")
    ap.add_argument("--json", required=True, type=Path, help="Path to LabelMe JSON file")
    ap.add_argument("--image", type=Path, default=None, help="Path to the X-ray image (PNG/JPG). If omitted, resolves automatically.")
    ap.add_argument("--img_dir", type=Path, default=None, help="Directory containing images; used with JSON's imagePath or json stem.")
    ap.add_argument("--out_dir", type=Path, default=Path("./out"), help="Output directory")
    ap.add_argument("--no_points", action="store_true", help="Do not draw point markers")
    ap.add_argument("--mm_per_px", type=float, default=None, help="mm na 1 px (pro 2mm toleranci Toyama).")
    ap.add_argument("--px_per_mm", type=float, default=None, help="px na 1 mm (alternativa k --mm_per_px).")
    ap.add_argument("--toyama_flip_side", action="store_true", help="Prohoď mapování stran (lordoza/kyfoza), pokud projekce vede k opačné interpretaci.")
    args = ap.parse_args()

    pts, image_path_in_json = load_points(args.json)
    image_path = resolve_image_path(args.json, args.image, args.img_dir, image_path_in_json)
    metrics = compute_metrics(pts)

    print(f"\n📄 {args.json.name}")
    for k, v in metrics.items():
        unit = "deg" if "deg" in k else "px"
        print(f"{k:25s} : {v: .3f} {unit}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem

    # 1) COBB
    cobb_path = out_dir / f"{stem}_cobb.png"
    draw_cobb(image_path, pts, metrics, cobb_path, draw_points=(not args.no_points))
    print(f"💾 COBB      -> {cobb_path}")

    # 2) SVA
    sva_path = out_dir / f"{stem}_sva.png"
    draw_sva(image_path, pts, metrics, sva_path, draw_points=(not args.no_points))
    print(f"💾 SVA       -> {sva_path}")

    # 3) C2 slope
    slope_path = out_dir / f"{stem}_c2slope.png"
    draw_c2_slope(image_path, pts, metrics, slope_path, draw_points=(not args.no_points))
    print(f"💾 C2 slope  -> {slope_path}")

    # 4+) segmentální úhly - každý zvlášť
    vertebrae = ["C2", "C3", "C4", "C5", "C6", "C7"]
    for i in range(len(vertebrae) - 1):
        upper = vertebrae[i]
        lower = vertebrae[i + 1]
        key = f"Segmental_{upper}_{lower}_deg"
        if key not in metrics:
            print(f"⚠️  Přeskakuji {upper}/{lower} - chybí body.")
            continue
        seg_path = out_dir / f"{stem}_segmental_{upper}_{lower}.png"
        draw_segmental_angle(image_path, pts, metrics, upper, lower, seg_path, draw_points=(not args.no_points))
        print(f"💾 {upper}/{lower} -> {seg_path}")

    # px_tol = px_tolerance_from_scale(args.mm_per_px, args.px_per_mm, default_px=2.0)
    # if args.mm_per_px is None and args.px_per_mm is None:
    #     print("⚠️  Nebylo zadáno měřítko; pro Toyamu používám toleranci 2 px.")

    # toyama_label, toyama_details = toyama_classify(pts, px_tol, flip_side=args.toyama_flip_side)
    # print(f"Toyama: {toyama_label}  (tol=±{px_tol:.2f} px, per-level={toyama_details.get('per_level',{})})")

    # toyama_path = out_dir / f"{stem}_toyama.png"
    # draw_toyama(image_path, pts, toyama_label, toyama_details, toyama_path, draw_points=(not args.no_points))
    # print(f"💾 Toyama    -> {toyama_path}")

if __name__ == "__main__":
    main()