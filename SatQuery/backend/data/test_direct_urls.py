import urllib.request
import json
import os

headers = {"User-Agent": "SatQueryBot/1.0 (contact: satquery@example.com) Mozilla/5.0"}

test_urls = [
    ("Sentinel2_Direct", "https://upload.wikimedia.org/wikipedia/commons/c/cf/Sentinel-2_captures_agricultural_fields_in_Aragon%2C_Spain.jpg"),
    ("Sentinel1_SAR_Direct", "https://upload.wikimedia.org/wikipedia/commons/e/e0/Sentinel-1_sees_Namibia.jpg"),
    ("Lake_Mead_T1", "https://upload.wikimedia.org/wikipedia/commons/5/52/Lake_Mead_July_2000.jpg"),
    ("Lake_Mead_T2", "https://upload.wikimedia.org/wikipedia/commons/1/14/Lake_Mead_July_2021.jpg"),
    ("Dubai_T1_2000", "https://upload.wikimedia.org/wikipedia/commons/6/69/Dubai_2000-2010.jpg"),
    ("VRSBench_raw", "https://raw.githubusercontent.com/lx709/VRSBench/main/fig_example.png")
]

for name, url in test_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as res:
            data = res.read()
            print(f"[OK] {name}: {len(data)} bytes downloaded")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
