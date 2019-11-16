import requests
from requests.adapters import HTTPAdapter
from pathlib import Path

def download(url, dest, cache=True):
    if cache and Path(dest).exists():
        print("File cache exists, skipping download")
        return
    print(f"Downloading {dest} ...")
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=10))
    r = s.get(url, allow_redirects=True)
    with open(dest, 'wb') as f:
        f.write(r.content)
