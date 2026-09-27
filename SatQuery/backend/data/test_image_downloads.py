import urllib.request
import json
import os

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

test_urls = [
    ("VRSBench_example", "https://raw.githubusercontent.com/lx709/VRSBench/main/fig_example.png"),
    ("Sentinel2_Agri", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Sentinel-2_captures_agricultural_fields_in_Aragon%2C_Spain.jpg/640px-Sentinel-2_captures_agricultural_fields_in_Aragon%2C_Spain.jpg"),
    ("Sentinel1_SAR", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sentinel-1_sees_Namibia.jpg/640px-Sentinel-1_sees_Namibia.jpg"),
    ("Change_T1_Before", "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Lake_Mead_July_2000.jpg/640px-Lake_Mead_July_2000.jpg"),
    ("Change_T2_After", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Lake_Mead_July_2021.jpg/640px-Lake_Mead_July_2021.jpg"),
]

for name, url in test_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as res:
            data = res.read()
            print(f"[OK] {name}: {len(data)} bytes downloaded")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
