from typing import Dict, Any, List


def build_response(
    question: str,
    task: str,
    images_count: int,
    specialist_result: Dict[str, Any],
    trace_steps: List[str] = None
) -> Dict[str, Any]:
    """
    Constructs the final explainable SatQuery AI API response.
    """
    trace = trace_steps or []
    if "Evidence generated" not in " ".join(trace):
        trace.append(f"Evidence generated ({len(specialist_result.get('evidence', []))} items)")
    if "Explainable answer synthesized" not in " ".join(trace):
        trace.append("Final answer and visualization synthesized")

    return {
        "task": task,
        "answer": specialist_result.get("answer", "Analysis complete."),
        "confidence": specialist_result.get("confidence", 0.90),
        "evidence": specialist_result.get("evidence", []),
        "statistics": specialist_result.get("statistics", {}),
        "overlay_url": specialist_result.get("overlay_url"),
        "before_url": specialist_result.get("before_url"),
        "after_url": specialist_result.get("after_url"),
        "trace": trace,
        "model_used": "SatQuery-RS-Intelligence"
    }
