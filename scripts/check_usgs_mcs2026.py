import re
from urllib.parse import urljoin

import requests

base = "https://pubs.usgs.gov/periodicals/mcs2026/"
print("Fetching", base)
try:
    r = requests.get(base, timeout=20)
    print("status", r.status_code)
    if r.status_code == 200:
        links = set(re.findall(r'href=["\']([^"\']+)["\']', r.text))
        structured = [
            link
            for link in links
            if link.lower().endswith((".csv", ".xlsx", ".xls"))
        ]
        print("Found structured links:", structured)
        for link in structured:
            url = urljoin(base, link)
            print("Testing", url)
            try:
                rr = requests.head(url, timeout=10)
                print(" ->", rr.status_code, rr.headers.get("content-type"))
            except Exception as e:
                print(" -> head error", e)
    else:
        print("Failed to fetch index")
except Exception as e:
    print("error", e)
