import urllib.request
from PIL import Image

headers = {"User-Agent": "Mozilla/5.0"}
base = "https://raw.githubusercontent.com/microsoft/torchgeo/main/tests/data/oscd/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images/val0"

for sub in ["imgs_1_rect", "imgs_2_rect"]:
    for band in ["B02", "B03", "B04"]:
        url = f"{base}/{sub}/{band}.tif"
        out = f"oscd_{sub}_{band}.tif"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as res:
            with open(out, "wb") as f:
                f.write(res.read())
        im = Image.open(out)
        print(f"[OK] {out}: size={im.size}, mode={im.mode}")
