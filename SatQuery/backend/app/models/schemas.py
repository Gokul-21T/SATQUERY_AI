from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: Optional[str] = None
    confidence: Optional[float] = None


class EvidenceItem(BaseModel):
    type: str = Field(..., description="Evidence category, e.g. region, metric, spectral, temporal")
    description: str = Field(..., description="Human-readable explanation of this evidence piece")
    value: Optional[Union[float, int, str, Dict[str, Any]]] = None
    quadrant: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: Optional[float] = None


class QueryResponse(BaseModel):
    task: str = Field(..., description="Selected specialist task: VQA, GROUNDING, CHANGE_DETECTION, FUSION")
    answer: str = Field(..., description="Synthesized natural-language answer")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Structured supporting evidence")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative analysis statistics")
    overlay_url: Optional[str] = Field(None, description="URL of generated visualization overlay")
    before_url: Optional[str] = Field(None, description="URL of pre-processed primary image")
    after_url: Optional[str] = Field(None, description="URL of secondary/temporal image if provided")
    trace: List[str] = Field(default_factory=list, description="Step-by-step explainability trace")
    model_used: Optional[str] = Field("SatQuery-RS-Core", description="Active model or fallback engine")


class SampleScenario(BaseModel):
    id: str
    task: str
    title: str
    description: str
    image_count: int
    primary_image: str
    secondary_image: Optional[str] = None
    default_query: str
    suggested_queries: List[str] = Field(default_factory=list)
    benchmark: str
    sensor: str
