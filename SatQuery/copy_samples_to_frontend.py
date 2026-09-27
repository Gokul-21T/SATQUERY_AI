import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_SAMPLES = os.path.join(BASE_DIR, "backend", "data", "samples")
SRC_CATALOG = os.path.join(BASE_DIR, "backend", "data", "sample_scenarios.json")

DEST_PUBLIC_SAMPLES = os.path.join(BASE_DIR, "frontend", "public", "sample_data")
os.makedirs(DEST_PUBLIC_SAMPLES, exist_ok=True)

if os.path.exists(SRC_SAMPLES):
    for f in os.listdir(SRC_SAMPLES):
        s_path = os.path.join(SRC_SAMPLES, f)
        d_path = os.path.join(DEST_PUBLIC_SAMPLES, f)
        if os.path.isfile(s_path):
            shutil.copy2(s_path, d_path)
    print(f"Copied sample images to {DEST_PUBLIC_SAMPLES}")

if os.path.exists(SRC_CATALOG):
    shutil.copy2(SRC_CATALOG, os.path.join(DEST_PUBLIC_SAMPLES, "sample_scenarios.json"))
    print("Copied sample_scenarios.json to frontend/public/sample_data/")
