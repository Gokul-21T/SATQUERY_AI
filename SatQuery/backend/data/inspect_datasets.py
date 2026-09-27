import urllib.request
import json

headers = {"User-Agent": "Mozilla/5.0"}

# Check BigEarthNet.txt HuggingFace files
req = urllib.request.Request("https://huggingface.co/api/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", headers=headers)
with urllib.request.urlopen(req) as res:
    ben_info = json.loads(res.read().decode())
    print("BigEarthNet.txt files in HF:")
    for sibling in ben_info.get("siblings", [])[:15]:
        print("  -", sibling.get("rfilename"))

# Check VRSBench HuggingFace files
req = urllib.request.Request("https://huggingface.co/api/datasets/xiang709/VRSBench", headers=headers)
with urllib.request.urlopen(req) as res:
    vrs_info = json.loads(res.read().decode())
    print("\nVRSBench files in HF:")
    for sibling in vrs_info.get("siblings", [])[:15]:
        print("  -", sibling.get("rfilename"))

# Inspect a snippet of CDVQA Val_images and Val_questions
req = urllib.request.Request("https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Val_images.json", headers=headers)
with urllib.request.urlopen(req) as res:
    cdvqa_images = json.loads(res.read().decode())
    print(f"\nCDVQA Val_images count: {len(cdvqa_images)}")
    print("Sample CDVQA image entry:", list(cdvqa_images.items())[:2] if isinstance(cdvqa_images, dict) else cdvqa_images[:2])

req = urllib.request.Request("https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Val_questions.json", headers=headers)
with urllib.request.urlopen(req) as res:
    cdvqa_q = json.loads(res.read().decode())
    print(f"\nCDVQA Val_questions count: {len(cdvqa_q)}")
    print("Sample CDVQA question:", list(cdvqa_q.items())[:2] if isinstance(cdvqa_q, dict) else cdvqa_q[:2])
