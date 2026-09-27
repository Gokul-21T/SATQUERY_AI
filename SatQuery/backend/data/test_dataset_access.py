import urllib.request
import json
import os

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def check_url(name, url):
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as res:
            size = res.headers.get("Content-Length", "unknown")
            print(f"[OK] {name}: status={res.status}, size={size}")
            return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

# 1. CDVQA
print("--- CDVQA ---")
check_url("CDVQA Val Questions", "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Val_questions.json")
check_url("CDVQA Val Answers", "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Val_answers.json")
check_url("CDVQA Val Images", "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Val_images.json")

# 2. VRSBench
print("\n--- VRSBench ---")
check_url("VRSBench GitHub sample", "https://raw.githubusercontent.com/lx709/VRSBench/main/fig_example.png")
check_url("VRSBench rs_instruction", "https://raw.githubusercontent.com/lx709/VRSBench/main/rs_instruction.txt")
check_url("VRSBench HuggingFace API", "https://huggingface.co/api/datasets/xiang709/VRSBench")

# 3. BigEarthNet.txt
print("\n--- BigEarthNet.txt ---")
check_url("BigEarthNet.txt HuggingFace API", "https://huggingface.co/api/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")
check_url("BigEarthNet official website", "https://txt.bigearth.net/")

# 4. RSVQA
print("\n--- RSVQA ---")
check_url("RSVQA official website", "https://rsvqa.sylvainlobry.com/")
check_url("RSVQA GitHub TorchRS", "https://raw.githubusercontent.com/isaaccorley/torchrs/main/torchrs/datasets/rsvqa.py")
