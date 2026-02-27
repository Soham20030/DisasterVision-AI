import urllib.request
import time
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

pairs = [
    (
        'https://upload.wikimedia.org/wikipedia/commons/3/30/Antakya_before_2023_earthquake_%28satellite%29.jpg',
        r'Test Images\turkey_pre_antakya.jpg'
    ),
    (
        'https://upload.wikimedia.org/wikipedia/commons/9/9e/Antakya_after_2023_earthquake_%28satellite%29.jpg',
        r'Test Images\turkey_post_antakya.jpg'
    ),
]

for i, (url, path) in enumerate(pairs):
    if i > 0:
        time.sleep(3)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(path, 'wb') as f:
                f.write(data)
            print(f"OK: {path} ({len(data):,} bytes)")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} FAIL: {e}")
            time.sleep(5)
