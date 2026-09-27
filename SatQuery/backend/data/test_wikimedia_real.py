import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Test compliant Wikimedia User-Agent
headers = {
    'User-Agent': 'SatQueryAI-Bot/1.0 (https://github.com/satquery-demo; contact@satquery.org) Python/3.10 urllib'
}

test_urls = [
    ("Sentinel2_Fields_Spain", "https://upload.wikimedia.org/wikipedia/commons/c/cf/Sentinel-2_captures_agricultural_fields_in_Aragon%2C_Spain.jpg"),
    ("Sentinel1_Radar_Namibia", "https://upload.wikimedia.org/wikipedia/commons/e/e0/Sentinel-1_sees_Namibia.jpg"),
    ("Lake_Mead_T1_2000", "https://upload.wikimedia.org/wikipedia/commons/5/52/Lake_Mead_July_2000.jpg"),
    ("Lake_Mead_T2_2021", "https://upload.wikimedia.org/wikipedia/commons/1/14/Lake_Mead_July_2021.jpg"),
    ("Dubai_Satellite_2000", "https://upload.wikimedia.org/wikipedia/commons/6/69/Dubai_2000-2010.jpg")
]

for name, url in test_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as res:
            data = res.read(1024*100)
            print(f"[OK] {name}: downloaded chunk ({len(data)} bytes), total={res.headers.get('Content-Length')}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
