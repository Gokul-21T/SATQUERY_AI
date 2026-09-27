import urllib.request
import os
from PIL import Image
import numpy as np

headers = {"User-Agent": "Mozilla/5.0"}

def download_and_inspect(url, out_name):
    print(f"Downloading {url}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as res:
        data = res.read()
    with open(out_name, "wb") as f:
        f.write(data)
    im = Image.open(out_name)
    print(f"[OK] {out_name}: format={im.format}, size={im.size}, mode={im.mode}")
    return im

# 1. Real agricultural remote sensing photo
url_agri = "https://raw.githubusercontent.com/microsoft/torchgeo/main/tests/data/ucmerced/UCMerced_LandUse/Images/agricultural/agricultural00.tif"
download_and_inspect(url_agri, "real_agri00.tif")

# 2. Real LandCover.ai orthophoto
url_lc = "https://raw.githubusercontent.com/microsoft/torchgeo/main/tests/data/landcoverai/images/M-33-20-D-c-4-2.tif"
download_and_inspect(url_lc, "real_landcover.tif")
