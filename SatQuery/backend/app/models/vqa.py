import numpy as np
from typing import Dict, Any, List
from app.processing.image import (
    decode_image_bytes,
    compute_land_cover_breakdown,
    extract_vegetation_mask,
    extract_water_mask,
    extract_built_up_mask,
    get_quadrant_description
)
from app.processing.overlays import save_image_to_outputs


async def run_vqa(question: str, image_bytes_list: List[bytes], trace_logger=None) -> Dict[str, Any]:
    """
    VQA Specialist:
    Answers natural language inquiries regarding land cover, dominant features,
    water presence, vegetation, and spatial composition.
    """
    if not image_bytes_list:
        return {
            "answer": "No satellite image was provided for VQA analysis.",
            "confidence": 0.0,
            "evidence": [],
            "statistics": {}
        }

    rgb = decode_image_bytes(image_bytes_list[0])
    h, w = rgb.shape[:2]
    if trace_logger:
        trace_logger.add_step(f"Image validated ({w}x{h} px, RGB multispectral bands)")

    # Compute land cover composition
    stats = compute_land_cover_breakdown(rgb)
    veg_pct = stats["vegetation"]
    water_pct = stats["water"]
    built_pct = stats["built_up"]
    other_pct = stats["bare_soil_other"]

    # Dominant land cover
    classes = [("Vegetation / Forest", veg_pct), ("Water Bodies", water_pct), ("Built-up / Urban", built_pct), ("Bare Soil / Other", other_pct)]
    dominant_name, dominant_pct = max(classes, key=lambda x: x[1])

    q_lower = question.lower()
    evidence = []
    
    # Analyze spatial location for dominant feature
    veg_mask, _ = extract_vegetation_mask(rgb)
    veg_loc = get_quadrant_description(veg_mask)

    water_mask, _ = extract_water_mask(rgb)
    water_loc = get_quadrant_description(water_mask)

    built_mask, _ = extract_built_up_mask(rgb)
    built_loc = get_quadrant_description(built_mask)

    if trace_logger:
        trace_logger.add_step("Computed spectral indices (NDVI/NDWI/NDBI approximations)")
        trace_logger.add_step("Extracted spatial land-cover distribution")

    # Formulate contextual answers based on user query
    if "dominant" in q_lower or "major" in q_lower or "main" in q_lower:
        answer = f"The dominant land cover in this satellite image is {dominant_name}, covering approximately {dominant_pct}% of the surveyed surface area."
        confidence = 0.94
        evidence.append({
            "type": "spectral_distribution",
            "description": f"Dominant class: {dominant_name} ({dominant_pct}%)",
            "value": dominant_pct,
            "quadrant": veg_loc if "vegetation" in dominant_name.lower() else (water_loc if "water" in dominant_name.lower() else built_loc),
            "confidence": 0.94
        })

    elif "water" in q_lower:
        if water_pct > 1.5:
            answer = f"Yes, significant water bodies are detected covering {water_pct}% of the scene, predominantly situated in the {water_loc}."
            confidence = 0.93
        else:
            answer = f"No extensive open water bodies were detected (less than 1% coverage). The scene is primarily comprised of {dominant_name} ({dominant_pct}%)."
            confidence = 0.89
        evidence.append({
            "type": "water_detection",
            "description": f"Water coverage: {water_pct}%, concentrated in the {water_loc}",
            "value": water_pct,
            "quadrant": water_loc,
            "confidence": confidence
        })

    elif "vegetation" in q_lower or "forest" in q_lower or "green" in q_lower or "crop" in q_lower:
        if veg_pct > 5.0:
            answer = f"Yes, healthy vegetation and agricultural land cover approximately {veg_pct}% of the area, concentrated in the {veg_loc}."
            confidence = 0.95
        else:
            answer = f"Vegetation coverage is minimal ({veg_pct}%). The dominant cover is {dominant_name}."
            confidence = 0.88
        evidence.append({
            "type": "vegetation_density",
            "description": f"Vegetation coverage: {veg_pct}%, localized in {veg_loc}",
            "value": veg_pct,
            "quadrant": veg_loc,
            "confidence": confidence
        })

    elif "building" in q_lower or "urban" in q_lower or "built-up" in q_lower or "structure" in q_lower or "port" in q_lower:
        if built_pct > 2.0:
            answer = f"Built-up structures and developed areas comprise {built_pct}% of the image, concentrated primarily in the {built_loc}."
            confidence = 0.91
        else:
            answer = f"Few or no major built-up structures are detected ({built_pct}% coverage). The region appears largely undeveloped."
            confidence = 0.87
        evidence.append({
            "type": "urban_footprint",
            "description": f"Built-up area: {built_pct}%, concentrated in {built_loc}",
            "value": built_pct,
            "quadrant": built_loc,
            "confidence": confidence
        })

    else:
        # General scene description
        answer = (
            f"This satellite scene contains a diverse mosaic of land cover: "
            f"Vegetation covers {veg_pct}% (mostly {veg_loc}), "
            f"Water bodies occupy {water_pct}% ({water_loc}), and "
            f"Built-up structures account for {built_pct}% ({built_loc}). "
            f"The overall dominant class is {dominant_name}."
        )
        confidence = 0.91
        evidence.append({
            "type": "scene_overview",
            "description": f"Full land cover breakdown: Veg {veg_pct}%, Water {water_pct}%, Built {built_pct}%",
            "value": stats,
            "quadrant": "Multi-region",
            "confidence": confidence
        })

    primary_url = save_image_to_outputs(rgb, prefix="vqa_input")

    return {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "statistics": {
            "dominant_class": dominant_name,
            "dominant_percentage": dominant_pct,
            "vegetation_percentage": veg_pct,
            "water_percentage": water_pct,
            "built_up_percentage": built_pct,
            "bare_soil_percentage": other_pct
        },
        "overlay_url": primary_url,
        "before_url": primary_url
    }
