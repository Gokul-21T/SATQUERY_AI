import os
import uuid
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_image_to_outputs(img_arr: np.ndarray, prefix: str = "overlay") -> str:
    """Save a NumPy RGB array to the outputs folder and return relative URL."""
    fname = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    fpath = os.path.join(OUTPUTS_DIR, fname)
    if img_arr.dtype != np.uint8:
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    Image.fromarray(img_arr).save(fpath, "PNG")
    return f"/outputs/{fname}"


def create_grounding_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_class: str,
    regions: List[Dict[str, Any]],
    confidence: float = 0.92
) -> str:
    """
    Creates a visual grounding overlay:
    - Translucent colored highlight over segmented pixels
    - Glowing contour outlines
    - Bounding boxes with label badges
    """
    h, w = rgb.shape[:2]
    overlay = rgb.copy()

    # Color scheme per feature
    color_map = {
        "vegetation": (40, 220, 90),     # Bright Emerald Green
        "water": (30, 144, 255),          # Dodger Blue / Cyan
        "built_up": (255, 120, 40),       # High-viz Orange
        "buildings": (255, 120, 40),      # High-viz Orange
        "roads": (255, 215, 0),           # Bright Yellow
        "default": (0, 230, 200)          # Aqua Teal
    }
    color = color_map.get(target_class.lower(), color_map["default"])

    # 1. Semi-transparent mask blending
    alpha = 0.42
    color_layer = np.full_like(rgb, color, dtype=np.uint8)
    mask_indices = mask > 0

    blended = cv2.addWeighted(rgb, 1.0 - alpha, color_layer, alpha, 0)
    overlay[mask_indices] = blended[mask_indices]

    # 2. Draw crisp contour outlines
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

    # 3. Draw bounding boxes and HUD labels for prominent regions
    for r in regions:
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        
        # Outer box
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
        
        # Corner brackets for mission-control aesthetic
        b_len = min(15, (xmax - xmin) // 4, (ymax - ymin) // 4)
        for (cx, cy, dx, dy) in [
            (xmin, ymin, 1, 1), (xmax, ymin, -1, 1),
            (xmin, ymax, 1, -1), (xmax, ymax, -1, -1)
        ]:
            cv2.line(overlay, (cx, cy), (cx + dx * b_len, cy), (255, 255, 255), 3)
            cv2.line(overlay, (cx, cy), (cx, cy + dy * b_len), (255, 255, 255), 3)

        # Label tag
        label_text = f"{target_class.upper()} #{r['id']} ({r.get('area_pct', 0)}%)"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tw, th = label_size
        
        tag_ymin = max(0, ymin - th - 8)
        cv2.rectangle(overlay, (xmin, tag_ymin), (xmin + tw + 10, tag_ymin + th + 6), (15, 23, 42), -1)
        cv2.rectangle(overlay, (xmin, tag_ymin), (xmin + tw + 10, tag_ymin + th + 6), color, 1)
        cv2.putText(overlay, label_text, (xmin + 5, tag_ymin + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return save_image_to_outputs(overlay, prefix=f"grounding_{target_class}")


def create_change_detection_overlay(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    change_mask: np.ndarray,
    change_details: Dict[str, Any]
) -> Dict[str, str]:
    """
    Creates change detection visualizations:
    1. Single overlaid post-change image with highlighted red/amber change zones
    2. Side-by-side composite: [BEFORE T1] | [AFTER T2] | [CHANGE HEATMAP]
    """
    # Ensure equal dimensions
    h, w = rgb_t2.shape[:2]
    if rgb_t1.shape[:2] != (h, w):
        rgb_t1 = cv2.resize(rgb_t1, (w, h))
    if change_mask.shape[:2] != (h, w):
        change_mask = cv2.resize(change_mask, (w, h))

    # Single overlay on T2
    t2_overlay = rgb_t2.copy()
    
    # Red-amber heat coloring for change
    change_indices = change_mask > 0
    change_color = (255, 69, 0) # Red-orange
    color_layer = np.full_like(rgb_t2, change_color, dtype=np.uint8)

    blended = cv2.addWeighted(rgb_t2, 0.45, color_layer, 0.55, 0)
    t2_overlay[change_indices] = blended[change_indices]
    
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(t2_overlay, contours, -1, (255, 230, 0), 2, cv2.LINE_AA)

    # Highlight major change bounding box
    major_regions = change_details.get("regions", [])
    for r in major_regions[:3]:
        cv2.rectangle(t2_overlay, (r["xmin"], r["ymin"]), (r["xmax"], r["ymax"]), (255, 255, 255), 2, cv2.LINE_AA)
        tag = f"CHANGE ({r.get('area_pct', 0)}%)"
        cv2.rectangle(t2_overlay, (r["xmin"], r["ymin"] - 18), (r["xmin"] + 110, r["ymin"]), (20, 20, 20), -1)
        cv2.putText(t2_overlay, tag, (r["xmin"] + 4, r["ymin"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Create Side-by-Side 3-Panel Composite: [T1 BEFORE] | [T2 AFTER] | [CHANGE MAP]
    # Heatmap of change
    heatmap = cv2.applyColorMap(change_mask, cv2.COLORMAP_JET)
    heatmap[change_mask == 0] = [15, 23, 42] # Dark background

    # Annotate panels with headers
    def add_header(img, title, subtitle=""):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.putText(res, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(res, subtitle, (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        return res

    p1 = add_header(rgb_t1, "1. BASELINE (T1)", "Pre-event")
    p2 = add_header(rgb_t2, "2. CURRENT (T2)", "Post-event")
    p3 = add_header(t2_overlay, "3. CHANGE OVERLAY", f"{change_details.get('changed_pct', 0)}% Detected")

    # Combine into a horizontal panorama
    composite = np.concatenate([p1, p2, p3], axis=1)

    t1_url = save_image_to_outputs(rgb_t1, prefix="t1_before")
    t2_url = save_image_to_outputs(rgb_t2, prefix="t2_after")
    overlay_url = save_image_to_outputs(t2_overlay, prefix="change_overlay")
    composite_url = save_image_to_outputs(composite, prefix="change_composite")

    return {
        "overlay_url": overlay_url,
        "composite_url": composite_url,
        "before_url": t1_url,
        "after_url": t2_url
    }


def create_fusion_overlay(
    optical_rgb: np.ndarray,
    sar_img: np.ndarray,
    fusion_details: Dict[str, Any]
) -> Dict[str, str]:
    """
    Creates multi-sensor Optical + SAR fusion visualizations:
    - False-color composite combining Optical spectral channels with SAR roughness
    - Cross-sensor feature highlights
    """
    h, w = optical_rgb.shape[:2]
    if sar_img.shape[:2] != (h, w):
        sar_img = cv2.resize(sar_img, (w, h))

    sar_gray = cv2.cvtColor(sar_img, cv2.COLOR_RGB2GRAY) if len(sar_img.shape) == 3 else sar_img

    # Fusion composite:
    # Red Channel: SAR backscatter intensity (highlights metallic, urban, rough structures)
    # Green Channel: Optical Green (highlights vegetation)
    # Blue Channel: Optical Blue (highlights water and shadow)
    opt_g = optical_rgb[:, :, 1]
    opt_b = optical_rgb[:, :, 2]
    fused_rgb = np.stack([sar_gray, opt_g, opt_b], axis=-1)

    # Highlight cross-sensor confirmed built-up areas
    built_mask = fusion_details.get("built_mask")
    if built_mask is not None:
        if built_mask.shape[:2] != (h, w):
            built_mask = cv2.resize(built_mask, (w, h))
        contours, _ = cv2.findContours(built_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fused_rgb, contours, -1, (255, 235, 59), 2, cv2.LINE_AA)

    # Side-by-side composite: [Optical] | [SAR Radar] | [Fused Multi-Sensor]
    def add_header(img, title, subtitle=""):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.putText(res, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(res, subtitle, (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        return res

    p1 = add_header(optical_rgb, "OPTICAL SENSOR", "Multispectral RGB")
    p2 = add_header(sar_img, "SAR SENSOR", "Radar Backscatter")
    p3 = add_header(fused_rgb, "OPTICAL+SAR FUSION", "Multi-Modal Joint View")

    composite = np.concatenate([p1, p2, p3], axis=1)

    opt_url = save_image_to_outputs(optical_rgb, prefix="fusion_opt")
    sar_url = save_image_to_outputs(sar_img, prefix="fusion_sar")
    fused_url = save_image_to_outputs(fused_rgb, prefix="fusion_fused")
    composite_url = save_image_to_outputs(composite, prefix="fusion_composite")

    return {
        "overlay_url": composite_url,
        "fused_url": fused_url,
        "before_url": opt_url,
        "after_url": sar_url
    }
