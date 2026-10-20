from pathlib import Path

import requests

url = (
    "https://query1.finance.yahoo.com/v7/finance/download/HG=F?period1=0&period2=9999999999&interval=1d&events=history"
)
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/vnd.ms-excel,application/xml;q=0.9,*/*;q=0.8",
}

resp = requests.get(url, headers=headers, timeout=30)
print("status:", resp.status_code)
print("content-type:", resp.headers.get("content-type"))
print("len:", len(resp.content))
if resp.status_code == 200 and len(resp.content) > 100:
    p = Path("data/raw/yahoo")
    p.mkdir(parents=True, exist_ok=True)
    fn = p / "yahoo_HG=F_live.csv"
    fn.write_bytes(resp.content)
    print("Saved to", fn)
else:
    print("Failed to fetch live Yahoo CSV; status", resp.status_code)
