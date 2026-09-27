import os
import json
from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.agent.router import route_query
from app.agent.planner import ExecutionTrace
from app.models.vqa import run_vqa
from app.models.grounding import run_grounding
from app.models.change_detection import run_change_detection
from app.models.fusion import run_fusion
from app.processing.evidence import build_response

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
SAMPLES_JSON_PATH = os.path.join(DATA_DIR, "sample_scenarios.json")


@router.get("/samples")
def get_sample_scenarios():
    """Returns curated demo scenarios for SIH 26167 presentation."""
    if os.path.exists(SAMPLES_JSON_PATH):
        try:
            with open(SAMPLES_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load sample scenarios: {e}")
    return []


@router.post("/query")
async def query_endpoint(
    query: Optional[str] = Form(None),
    question: Optional[str] = Form(None), # Backward compatibility
    task: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[])
):
    """
    Main SatQuery AI multimodal analysis endpoint.
    Accepts natural-language query and satellite image(s),
    intelligently routes to specialist engine, and returns explainable results.
    """
    user_query = (query or question or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    trace = ExecutionTrace(user_query)

    # Read image contents into memory
    image_bytes_list = []
    for img in images:
        if img.filename:
            content = await img.read()
            if len(content) > 0:
                image_bytes_list.append(content)

    trace.add_step(f"Received {len(image_bytes_list)} satellite image(s)")

    # 1. Autonomous Intent Routing
    detected_task, router_conf, target_class = route_query(user_query, len(image_bytes_list), task_override=task)
    trace.add_step(f"Intent classified as {detected_task} (Confidence: {round(router_conf * 100)}%)")

    # 2. Specialist Execution
    if detected_task == "VQA":
        result = await run_vqa(user_query, image_bytes_list, trace_logger=trace)

    elif detected_task == "GROUNDING":
        result = await run_grounding(user_query, image_bytes_list, target_class=target_class, trace_logger=trace)

    elif detected_task == "CHANGE_DETECTION":
        result = await run_change_detection(user_query, image_bytes_list, trace_logger=trace)

    elif detected_task == "FUSION":
        result = await run_fusion(user_query, image_bytes_list, trace_logger=trace)

    else:
        # Fallback to VQA
        result = await run_vqa(user_query, image_bytes_list, trace_logger=trace)
        detected_task = "VQA"

    # 3. Formulate explainable response
    return build_response(user_query, detected_task, len(image_bytes_list), result, trace.to_list())


@router.post("/analyze")
async def analyze_alias(
    question: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[])
):
    """Backward-compatible alias for /api/query."""
    return await query_endpoint(query=query, question=question, task=task, images=images)
