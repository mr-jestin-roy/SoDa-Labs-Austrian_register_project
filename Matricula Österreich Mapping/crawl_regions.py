"""
Crawl Matricula Online (data.matricula-online.eu) Austria region pages.

Level 1: /en/oesterreich/                       -> 20 diocese/church-district regions
Level 2: /en/oesterreich/{region_slug}/          -> parish list per region

Output:
  output/diocese_state_mapping.csv   one row per region (diocese/church-district)
  output/parishes.csv                one row per parish, joined to region metadata
  output/parishes.json                same data, nested by region
  raw_html/{region_slug}.html         cached raw HTML per region page
"""
import concurrent.futures
import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://data.matricula-online.eu"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw_html"
OUT_DIR = ROOT / "output"

# Level-1 regions, hand-classified from /en/oesterreich/ (see conversation).
# state = modern Bundesland this diocese/church-district maps to (political mapping,
# not always 1:1 -- e.g. "wien" fond covers Vienna + eastern Lower Austria).
REGIONS = [
    {"slug": "alt-ev", "name": "Altösterreich (evangelisch)", "state": "cross-region", "denomination": "Evangelical", "count": 10, "reichsgau_1942": "N/A (all parishes outside 1942 Austria/Ostmark -- Bukovina, Transylvania, Trieste/Gorizia/Pola, Teschen)"},
    {"slug": "burgenland-ab-hb", "name": "Burgenland: Ev. Kirchen A.B. und H.B.", "state": "Burgenland", "denomination": "Evangelical", "count": 29, "reichsgau_1942": "Niederdonau, Steiermark"},
    {"slug": "burgenland", "name": "Burgenland: Rk. Pfarren", "state": "Burgenland", "denomination": "Catholic", "count": 2, "reichsgau_1942": "Niederdonau, Steiermark"},
    {"slug": "kaernten-evAB", "name": "Kärnten: Ev. Kirche A.B", "state": "Kärnten", "denomination": "Evangelical", "count": 31, "reichsgau_1942": "Kärnten"},
    {"slug": "gurk", "name": "Kärnten: Rk. Diözese Gurk", "state": "Kärnten", "denomination": "Catholic", "count": 328, "reichsgau_1942": "Kärnten"},
    {"slug": "metropolis", "name": "Metropolis von Austria, griech.-orth. Kirche", "state": "cross-region", "denomination": "Greek Orthodox", "count": 2, "reichsgau_1942": "Wien"},
    {"slug": "niederoesterreich-ev-kirche-AB", "name": "Niederösterreich: Ev. Kirche A.B.", "state": "Niederösterreich", "denomination": "Evangelical", "count": 17, "reichsgau_1942": "Niederdonau"},
    {"slug": "noela", "name": "Niederösterreich: NÖ Landesarchiv", "state": "Niederösterreich", "denomination": "State archive", "count": 2, "reichsgau_1942": "Niederdonau"},
    {"slug": "st-poelten", "name": "Niederösterreich (Westen): Rk. Diözese St. Pölten", "state": "Niederösterreich", "denomination": "Catholic", "count": 410, "reichsgau_1942": "Niederdonau"},
    {"slug": "oberoesterreich-ev-kirche-AB", "name": "Oberösterreich: Ev. Kirche A.B", "state": "Oberösterreich", "denomination": "Evangelical", "count": 9, "reichsgau_1942": "Oberdonau"},
    {"slug": "oberoesterreich", "name": "Oberösterreich: Rk. Diözese Linz", "state": "Oberösterreich", "denomination": "Catholic", "count": 427, "reichsgau_1942": "Oberdonau"},
    {"slug": "salzburg", "name": "Salzburg: Rk. Erzdiözese Salzburg", "state": "Salzburg", "denomination": "Catholic", "count": 137, "reichsgau_1942": "Salzburg"},
    {"slug": "steiermark-ev-kirche-AB", "name": "Steiermark: Ev. Kirche A.B", "state": "Steiermark", "denomination": "Evangelical", "count": 25, "reichsgau_1942": "Steiermark"},
    {"slug": "graz-seckau", "name": "Steiermark: Rk. Diözese Graz-Seckau", "state": "Steiermark", "denomination": "Catholic", "count": 358, "reichsgau_1942": "Steiermark"},
    {"slug": "tirol", "name": "Tirol: Matriken Tirol Online", "state": "Tirol", "denomination": "State archive", "count": 353, "reichsgau_1942": "Tirol-Vorarlberg"},
    {"slug": "vorarlberg", "name": "Vorarlberg: Matriken (Land Vorarlberg)", "state": "Vorarlberg", "denomination": "State archive", "count": 127, "reichsgau_1942": "Tirol-Vorarlberg"},
    {"slug": "daw", "name": "Wien, Diözesanarchiv", "state": "Wien", "denomination": "Catholic", "count": 1, "reichsgau_1942": "Wien"},
    {"slug": "wien-evang-dioezese-AB", "name": "Wien: Ev. Kirche A.B.", "state": "Wien", "denomination": "Evangelical", "count": 16, "reichsgau_1942": "Wien"},
    {"slug": "wien-evang-dioezese-HB", "name": "Wien: Ev. Kirche H.B.", "state": "Wien", "denomination": "Evangelical", "count": 4, "reichsgau_1942": "Wien"},
    {"slug": "wien", "name": "Wien/Niederösterreich (Osten): Rk. Erzdiözese Wien", "state": "Wien/Niederösterreich (Ost)", "denomination": "Catholic", "count": 567, "reichsgau_1942": "Wien, Niederdonau"},
]


def fetch_region(region):
    slug = region["slug"]
    url = f"{BASE}/en/oesterreich/{slug}/"
    cache_path = RAW_DIR / f"{slug}.html"
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        last_err = None
        html = None
        for attempt in range(5):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                html = resp.text
                break
            except requests.exceptions.HTTPError as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        if html is None:
            raise last_err
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(0.5)  # be polite

    soup = BeautifulSoup(html, "html.parser")
    parishes = []
    for a in soup.select("div.list-group a.list-group-item"):
        href = a.get("href", "")
        m = re.match(rf"^/en/oesterreich/{re.escape(slug)}/([^/]+)/$", href)
        if not m:
            continue
        badge = a.find("span", class_="badge")
        name = a.get_text(strip=True)
        if badge:
            name = name.replace(badge.get_text(strip=True), "", 1).strip()
        parishes.append(
            {
                "parish_name": name,
                "parish_slug": m.group(1),
                "parish_url": BASE + href,
            }
        )
    return slug, parishes


def main():
    RAW_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_region, r): r for r in REGIONS}
        for fut in concurrent.futures.as_completed(futures):
            region = futures[fut]
            slug, parishes = fut.result()
            results[slug] = parishes
            expected = region["count"]
            status = "OK" if len(parishes) == expected else f"MISMATCH (expected {expected})"
            print(f"{slug:35s} {len(parishes):4d} parishes  [{status}]")

    # diocese/state mapping CSV
    with (OUT_DIR / "diocese_state_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region_slug", "region_name", "state", "denomination", "reichsgau_1942", "parish_count_listed", "parish_count_scraped", "region_url"])
        for r in REGIONS:
            w.writerow([r["slug"], r["name"], r["state"], r["denomination"], r["reichsgau_1942"], r["count"], len(results.get(r["slug"], [])), f"{BASE}/en/oesterreich/{r['slug']}/"])

    # flat parish CSV
    with (OUT_DIR / "parishes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region_slug", "region_name", "state", "denomination", "reichsgau_1942", "parish_name", "parish_slug", "parish_url"])
        for r in REGIONS:
            for p in results.get(r["slug"], []):
                w.writerow([r["slug"], r["name"], r["state"], r["denomination"], r["reichsgau_1942"], p["parish_name"], p["parish_slug"], p["parish_url"]])

    # nested JSON
    nested = []
    for r in REGIONS:
        entry = dict(r)
        entry["parishes"] = results.get(r["slug"], [])
        nested.append(entry)
    (OUT_DIR / "parishes.json").write_text(json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(v) for v in results.values())
    print(f"\nTotal parishes scraped: {total}")


if __name__ == "__main__":
    main()
