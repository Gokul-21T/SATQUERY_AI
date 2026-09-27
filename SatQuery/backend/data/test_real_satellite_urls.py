import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

urls = [
    ('NASA_Dubai_T1', 'https://eoimages.gsfc.nasa.gov/images/imagerecords/4000/4950/dubai_etm_2000287_lrg.jpg'),
    ('NASA_Dubai_T2', 'https://eoimages.gsfc.nasa.gov/images/imagerecords/4000/4950/dubai_oli_2010279_lrg.jpg'),
    ('NASA_LasVegas_T1', 'https://eoimages.gsfc.nasa.gov/images/imagerecords/4000/4636/lasvegas_etm_1984296_lrg.jpg'),
    ('NASA_LasVegas_T2', 'https://eoimages.gsfc.nasa.gov/images/imagerecords/4000/4636/lasvegas_oli_2007297_lrg.jpg'),
    ('ESA_Sentinel2_TorchRS', 'https://raw.githubusercontent.com/isaaccorley/torchrs/main/tests/assets/sentinel2.png'),
    ('VRSBench_Port', 'https://raw.githubusercontent.com/lx709/VRSBench/main/fig_example.png')
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

for name, url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as res:
            head = res.read(1024*50)
            cl = res.headers.get('Content-Length', 'unknown')
            print(f"[OK] {name}: downloaded chunk ({len(head)} bytes), total={cl}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
