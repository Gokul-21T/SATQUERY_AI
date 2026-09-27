import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Dict, Any, List


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB NumPy array."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)


def get_quadrant_description(mask: np.ndarray) -> str:
    """
    Given a binary mask, identify the dominant spatial quadrant
    (e.g., 'northern', 'southern', 'south-western', 'central').
    """
    h, w = mask.shape[:2]
    total_active = np.sum(mask > 0)
    if total_active == 0:
        return "none detected"

    # Divide into 3x3 grid for finer spatial resolution
    h_third, w_third = h // 3, w // 3
    grid_scores = {}
    row_names = ["northern", "central", "southern"]
    col_names = ["western", "central", "eastern"]

    for r_idx in range(3):
        for c_idx in range(3):
            submask = mask[r_idx * h_third : (r_idx + 1) * h_third, c_idx * w_third : (c_idx + 1) * w_third]
            score = np.sum(submask > 0)
            name = f"{row_names[r_idx]}-{col_names[c_idx]}" if row_names[r_idx] != "central" or col_names[c_idx] != "central" else "central"
            grid_scores[name] = score

    best_region = max(grid_scores, key=grid_scores.get)
    if grid_scores[best_region] / total_active > 0.4:
        return f"{best_region} region"
    
    # Check vertical bias
    top = np.sum(mask[:h//2, :])
    bottom = np.sum(mask[h//2:, :])
    left = np.sum(mask[:, :w//2])
    right = np.sum(mask[:, w//2:])

    v_dir = "northern" if top > 1.3 * bottom else ("southern" if bottom > 1.3 * top else "")
    h_dir = "western" if left > 1.3 * right else ("eastern" if right > 1.3 * left else "")

    if v_dir and h_dir:
        return f"{v_dir}-{h_dir} region"
    elif v_dir:
        return f"{v_dir} region"
    elif h_dir:
        return f"{h_dir} region"
    return "distributed across the region"


def extract_vegetation_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes a remote sensing vegetation index (Visible-Atmospherically Resistant / Excess Green).
    Returns (binary_mask, coverage_percentage).
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # Remote sensing visible vegetation index: 2*G - R - B
    exg = 2.0 * g - r - b
    # Also ratio: (G - R) / (G + R + 1e-5)
    gr_ratio = (g - r) / (g + r + 1e-5)

    veg_mask = ((exg > 15) & (gr_ratio > 0.05) & (g > 45)).astype(np.uint8) * 255
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)

    percentage = float(np.sum(veg_mask > 0) / (rgb.shape[0] * rgb.shape[1]) * 100.0)
    return veg_mask, round(percentage, 2)


def extract_water_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes water body mask using Blue-to-Red ratio & HSV darkness thresholding.
    Returns (binary_mask, coverage_percentage).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)

    # Water criteria: Blue reflectance higher than Red, low-to-moderate brightness, hue in blue/cyan range (85-135)
    blue_ratio = (b - r) / (b + r + 1e-5)
    hsv_water = (h >= 85) & (h <= 140) & (s >= 35) & (v <= 220)
    spectral_water = (b > r * 1.15) & (blue_ratio > 0.08) & (r < 110)

    water_mask = ((hsv_water | spectral_water) & (g < 160)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

    percentage = float(np.sum(water_mask > 0) / (rgb.shape[0] * rgb.shape[1]) * 100.0)
    return water_mask, round(percentage, 2)


def extract_built_up_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extracts built-up/urban regions using edge density, high grayscale intensity, and low color saturation.
    Returns (binary_mask, coverage_percentage).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]

    # Buildings have low color saturation (grays, whites, concrete) and high brightness or distinct edges
    edges = cv2.Canny(gray, 60, 150)
    edge_density = cv2.boxFilter(edges.astype(np.float32), -1, (15, 15))

    built_mask = ((s < 50) & (gray > 130) & (edge_density > 10)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    built_mask = cv2.morphologyEx(built_mask, cv2.MORPH_CLOSE, kernel)

    percentage = float(np.sum(built_mask > 0) / (rgb.shape[0] * rgb.shape[1]) * 100.0)
    return built_mask, round(percentage, 2)


def extract_road_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extracts linear transport network and roads using line detection and morphological filters.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=25, maxLineGap=10)

    road_mask = np.zeros_like(gray, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(road_mask, (x1, y1), (x2, y2), 255, 3)

    percentage = float(np.sum(road_mask > 0) / (rgb.shape[0] * rgb.shape[1]) * 100.0)
    return road_mask, round(percentage, 2)


def find_prominent_regions(mask: np.ndarray, max_regions: int = 5) -> List[Dict[str, Any]]:
    """
    Finds bounding boxes and metadata of the top largest connected components in a mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]

    regions = []
    h, w = mask.shape[:2]
    for i, cnt in enumerate(sorted_cnts):
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        submask = np.zeros_like(mask)
        cv2.drawContours(submask, [cnt], -1, 255, -1)
        quadrant = get_quadrant_description(submask)
        regions.append({
            "id": i + 1,
            "xmin": int(x),
            "ymin": int(y),
            "xmax": int(x + bw),
            "ymax": int(y + bh),
            "area_pixels": int(area),
            "area_pct": round(float(area / (h * w) * 100.0), 2),
            "quadrant": quadrant
        })
    return regions


def compute_land_cover_breakdown(rgb: np.ndarray) -> Dict[str, float]:
    """
    Analyzes full scene land cover breakdown.
    """
    veg_mask, veg_pct = extract_vegetation_mask(rgb)
    water_mask, water_pct = extract_water_mask(rgb)
    built_mask, built_pct = extract_built_up_mask(rgb)

    # Remaining is bare soil / unclassified
    other_pct = max(0.0, round(100.0 - (veg_pct + water_pct + built_pct), 2))

    return {
        "vegetation": veg_pct,
        "water": water_pct,
        "built_up": built_pct,
        "bare_soil_other": other_pct
    }
