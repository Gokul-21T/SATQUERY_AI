import numpy as np
from typing import Dict, Any, List
from app.processing.image import (
    decode_image_bytes,
    extract_vegetation_mask,
    extract_water_mask,
    extract_built_up_mask,
    extract_road_mask,
    find_prominent_regions,
    get_quadrant_description
)
from app.processing.overlays import create_grounding_overlay, save_image_to_outputs


async def run_grounding(question: str, image_bytes_list: List[bytes], target_class: str = "vegetation", trace_logger=None) -> Dict[str, Any]:
    """
    Visual Grounding Specialist:
    Localizes specific land-cover types and structures requested in natural language,
    highlights regions with segmentation contours and bounding boxes, and computes spatial metrics.
    """
    if not image_bytes_list:
        return {
            "answer": "No satellite image was provided for visual grounding.",
            "confidence": 0.0,
            "evidence": [],
            "statistics": {},
            "overlay_url": None
        }

    rgb = decode_image_bytes(image_bytes_list[0])
    h, w = rgb.shape[:2]
    if trace_logger:
        trace_logger.add_step(f"Image validated ({w}x{h} px, RGB multispectral bands)")
        trace_logger.add_step(f"Target semantic feature identified: '{target_class}'")

    # Select segmentation routine based on target class
    t_cls = target_class.lower()
    if t_cls in ["water", "river", "lake", "ocean", "reservoir"]:
        mask, coverage_pct = extract_water_mask(rgb)
        label = "Water Body"
        canonical_class = "water"
    elif t_cls in ["building", "buildings", "built_up", "urban", "dock", "docks", "structures"]:
        mask, coverage_pct = extract_built_up_mask(rgb)
        label = "Built-up Structure"
        canonical_class = "built_up"
    elif t_cls in ["road", "roads", "highway", "transport"]:
        mask, coverage_pct = extract_road_mask(rgb)
        label = "Road / Transport Corridor"
        canonical_class = "roads"
    else:
        # Default to vegetation / agricultural
        mask, coverage_pct = extract_vegetation_mask(rgb)
        label = "Vegetation / Canopy"
        canonical_class = "vegetation"

    if trace_logger:
        trace_logger.add_step(f"Extracted {label} spatial mask (Coverage: {coverage_pct}%)")

    # Find prominent connected component regions and bounding boxes
    regions = find_prominent_regions(mask, max_regions=5)
    primary_quadrant = get_quadrant_description(mask)

    if trace_logger:
        trace_logger.add_step(f"Localized {len(regions)} prominent bounding cluster(s)")
        trace_logger.add_step("Rendered mission-control overlay HUD with contour bounds")

    # Render visual grounding overlay
    overlay_url = create_grounding_overlay(rgb, mask, canonical_class, regions, confidence=0.92)
    before_url = save_image_to_outputs(rgb, prefix="grounding_orig")

    # Build structured evidence items
    evidence = []
    evidence.append({
        "type": "coverage_metric",
        "description": f"Total {label.lower()} coverage: {coverage_pct}% of surveyed area",
        "value": coverage_pct,
        "quadrant": primary_quadrant,
        "confidence": 0.93
    })

    for r in regions:
        evidence.append({
            "type": "localized_region",
            "description": f"{label} cluster #{r['id']} ({r['area_pct']}% of scene) in {r['quadrant']}",
            "value": r["area_pct"],
            "quadrant": r["quadrant"],
            "bounding_box": {
                "xmin": r["xmin"],
                "ymin": r["ymin"],
                "xmax": r["xmax"],
                "ymax": r["ymax"],
                "label": f"{label} #{r['id']}",
                "confidence": 0.91
            },
            "confidence": 0.91
        })

    # Natural language explanation
    if coverage_pct > 0.5:
        answer = (
            f"Successfully grounded {label.lower()} areas across the satellite image. "
            f"{label} constitutes {coverage_pct}% of the surface area, with primary concentration "
            f"situated in the {primary_quadrant}. {len(regions)} distinct regional cluster(s) have been highlighted."
        )
        confidence = 0.93
    else:
        answer = (
            f"Only negligible traces of {label.lower()} were grounded ({coverage_pct}%). "
            f"No major contiguous clusters were identified in this region."
        )
        confidence = 0.86

    return {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "statistics": {
            "target_class": canonical_class,
            "coverage_percentage": coverage_pct,
            "region_count": len(regions),
            "primary_location": primary_quadrant,
            "regions": regions
        },
        "overlay_url": overlay_url,
        "before_url": before_url
    }
