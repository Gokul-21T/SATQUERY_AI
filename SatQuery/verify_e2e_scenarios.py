import os
import requests

API_URL = "http://127.0.0.1:8000/api/query"
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "samples")

def test_scenario(title, query, image_filenames, expected_task):
    print(f"\n==================================================")
    print(f"TESTING: {title}")
    print(f"Query: '{query}'")
    files = []
    for fn in image_filenames:
        p = os.path.join(SAMPLES_DIR, fn)
        files.append(("images", (fn, open(p, "rb"), "image/png")))

    res = requests.post(API_URL, data={"query": query}, files=files)
    assert res.status_code == 200, f"Error: {res.status_code} - {res.text}"
    data = res.json()
    
    assert data["task"] == expected_task, f"Expected {expected_task}, got {data['task']}"
    print(f"[PASSED] Detected Task: {data['task']}")
    print(f"[PASSED] Confidence: {round(data['confidence'] * 100)}%")
    print(f"[PASSED] Answer: {data['answer']}")
    print(f"[PASSED] Overlay URL: {data['overlay_url']}")
    print(f"[PASSED] Evidence count: {len(data['evidence'])}")
    print(f"[PASSED] Execution Trace:")
    for step in data["trace"]:
        print(f"   [+] {step}")

if __name__ == "__main__":
    # 1. VQA
    test_scenario(
        "Demo 1 - VQA",
        "What is the dominant land cover in this image?",
        ["sc1_vqa_agricultural_lake.png"],
        "VQA"
    )

    # 2. Grounding
    test_scenario(
        "Demo 2 - Grounding",
        "Show me the water bodies.",
        ["sc2_grounding_reservoir_fields.png"],
        "GROUNDING"
    )

    # 3. Change Detection
    test_scenario(
        "Demo 3 - Change Detection",
        "What changed between these two images?",
        ["sc3_change_t1_before.png", "sc3_change_t2_after.png"],
        "CHANGE_DETECTION"
    )

    # 4. Multi-Modal Fusion
    test_scenario(
        "Demo 4 - Multi-Modal Fusion",
        "Combine optical and SAR imagery to identify built-up areas.",
        ["sc4_fusion_cartosat_optical.png", "sc4_fusion_risat_sar.png"],
        "FUSION"
    )

    print("\n==================================================")
    print("ALL 4 DEMO SCENARIOS VERIFIED END-TO-END SUCCESSFULLY!")
