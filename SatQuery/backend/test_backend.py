import os
import sys
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200, res.text
    data = res.json()
    print("[TEST PASSED] /api/health:", data)

def test_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200, res.text
    samples = res.json()
    assert len(samples) >= 4, f"Expected at least 4 samples, got {len(samples)}"
    print(f"[TEST PASSED] /api/samples: {len(samples)} scenarios loaded")

def test_vqa():
    sample_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc1_vqa_agricultural_lake.png")
    with open(sample_img_path, "rb") as f:
        res = client.post(
            "/api/query",
            data={"query": "What is the dominant land cover in this image?"},
            files=[("images", ("sample.png", f.read(), "image/png"))]
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["task"] == "VQA", f"Expected VQA, got {data['task']}"
    assert "dominant" in data["answer"].lower() or "vegetation" in data["answer"].lower()
    assert len(data["trace"]) > 0
    print("[TEST PASSED] VQA Query:", data["answer"][:70], "...")

def test_grounding():
    sample_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc2_grounding_reservoir_fields.png")
    with open(sample_img_path, "rb") as f:
        res = client.post(
            "/api/query",
            data={"query": "Show me the water bodies."},
            files=[("images", ("sample.png", f.read(), "image/png"))]
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["task"] == "GROUNDING", f"Expected GROUNDING, got {data['task']}"
    assert data["overlay_url"] is not None
    assert len(data["evidence"]) > 0
    print("[TEST PASSED] Grounding Query:", data["answer"][:70], "...", "Overlay:", data["overlay_url"])

def test_change_detection():
    t1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc3_change_t1_before.png")
    t2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc3_change_t2_after.png")
    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
        res = client.post(
            "/api/query",
            data={"query": "What changed between these two images?"},
            files=[
                ("images", ("t1.png", f1.read(), "image/png")),
                ("images", ("t2.png", f2.read(), "image/png"))
            ]
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["task"] == "CHANGE_DETECTION", f"Expected CHANGE_DETECTION, got {data['task']}"
    assert data["statistics"]["total_changed_percentage"] > 0
    print("[TEST PASSED] Change Detection Query:", data["answer"][:70], "...", "Changed %:", data["statistics"]["total_changed_percentage"])

def test_fusion():
    opt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc4_fusion_cartosat_optical.png")
    sar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sc4_fusion_risat_sar.png")
    with open(opt_path, "rb") as f1, open(sar_path, "rb") as f2:
        res = client.post(
            "/api/query",
            data={"query": "Combine optical and SAR imagery to identify built-up areas."},
            files=[
                ("images", ("opt.png", f1.read(), "image/png")),
                ("images", ("sar.png", f2.read(), "image/png"))
            ]
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["task"] == "FUSION", f"Expected FUSION, got {data['task']}"
    assert data["statistics"]["fused_built_up_percentage"] > 0
    print("[TEST PASSED] Fusion Query:", data["answer"][:70], "...")

if __name__ == "__main__":
    test_health()
    test_samples()
    test_vqa()
    test_grounding()
    test_change_detection()
    test_fusion()
    print("\nALL BACKEND TESTS PASSED SUCCESSFULLY!")
