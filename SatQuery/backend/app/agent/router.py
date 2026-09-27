import os
import re
from typing import Dict, Any, Tuple

TASK_KEYWORDS = {
    "CHANGE_DETECTION": [
        "change", "changed", "difference", "before and after", "before", "after",
        "increased", "decreased", "growth", "loss", "over time", "between these",
        "between two", "temporal", "development over", "what happened to"
    ],
    "FUSION": [
        "sar", "optical", "radar", "risat", "cartosat", "sentinel-1", "sentinel-2",
        "both sensors", "multi-sensor", "multisensor", "multi sensor", "fusion",
        "combine", "cross-sensor", "complementary", "backscatter"
    ],
    "GROUNDING": [
        "where is", "where are", "where", "locate", "location", "show me", "highlight",
        "find", "find the", "bound", "bounding box", "detect where", "mark the",
        "pinpoint", "segment", "isolate", "which part", "draw"
    ],
    "VQA": [
        "what is", "what are", "is there", "are there", "does this", "how much",
        "percentage of", "dominant", "describe", "identify", "tell me about",
        "explain", "overview", "what type", "count", "any water", "any vegetation"
    ]
}

TARGET_CLASSES = {
    "water": ["water", "river", "lake", "reservoir", "ocean", "sea", "canal", "pond"],
    "vegetation": ["vegetation", "forest", "tree", "trees", "crop", "crops", "agriculture", "agricultural", "green", "greenery", "plant"],
    "built_up": ["building", "buildings", "built-up", "urban", "house", "houses", "structure", "structures", "settlement", "city", "industrial", "dock", "docks"],
    "roads": ["road", "roads", "highway", "street", "streets", "transport", "runway", "track"]
}


def extract_target_class(query: str) -> str:
    """Identify target geographic class from query."""
    q = query.lower()
    for class_name, keywords in TARGET_CLASSES.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q):
                return class_name
    return "vegetation" # Sensible remote-sensing default


def route_query(question: str, image_count: int, task_override: str = None) -> Tuple[str, float, str]:
    """
    Intelligent Query Router for SatQuery AI.
    
    Returns:
        (task_name: str, confidence: float, target_class: str)
    """
    if task_override and task_override.upper() in ["VQA", "GROUNDING", "CHANGE_DETECTION", "FUSION", "GENERAL"]:
        target_cls = extract_target_class(question)
        return task_override.upper(), 0.98, target_cls

    q = re.sub(r"[^a-z0-9+ \-]", " ", question.lower())
    scores = {task: 0 for task in TASK_KEYWORDS}

    for task, keywords in TASK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                # exact word boundary gives stronger score
                if re.search(r'\b' + re.escape(keyword) + r'\b', q):
                    scores[task] += 2
                else:
                    scores[task] += 1

    target_cls = extract_target_class(question)

    # Multi-sensor fusion intent has high specificity
    if scores["FUSION"] >= 2 or ("optical" in q and "sar" in q):
        return "FUSION", 0.95, target_cls

    # Dual images + change keywords -> CHANGE_DETECTION
    if image_count >= 2:
        if scores["FUSION"] > 0:
            return "FUSION", 0.92, target_cls
        if scores["CHANGE_DETECTION"] > 0 or "change" in q or "differ" in q:
            return "CHANGE_DETECTION", 0.96, target_cls
        # If 2 images are passed without change keywords, check if fusion or change detection fits best
        if any(w in q for w in ["combine", "sensor", "joint"]):
            return "FUSION", 0.90, target_cls
        return "CHANGE_DETECTION", 0.88, target_cls

    # Single image routing
    if scores["GROUNDING"] > 0:
        return "GROUNDING", 0.92, target_cls

    if scores["CHANGE_DETECTION"] > 0:
        # Prompt user for second image or run change analysis against baseline
        return "CHANGE_DETECTION", 0.85, target_cls

    if scores["VQA"] > 0:
        return "VQA", 0.91, target_cls

    # Fallback to VQA for general inquiries
    return "VQA", 0.82, target_cls
