"""Optional data collection helper.

Scrapes headlines/summaries from news homepages into scraped_articles.csv.
The output is UNLABELED: add a Topic column before using it to train ML_Model.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = [
    "https://www.bbc.com/news",
    "https://www.nytimes.com/",
    "https://www.reuters.com/",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsClassifier/1.0)"}
OUTPUT = Path(__file__).resolve().parent / "scraped_articles.csv"


def scrape_url(url: str) -> list[list[str]]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[skip] {url}: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []
    for article in soup.find_all("article"):
        headline_tag = article.find(["h1", "h2", "h3"])
        summary_tag = article.find("p")
        time_tag = article.find("time")
        if headline_tag is None or summary_tag is None:
            continue

        headline = headline_tag.get_text(" ", strip=True)
        summary = summary_tag.get_text(" ", strip=True)
        date = time_tag.get("datetime", "") if time_tag else ""
        if not headline or not summary:
            continue
        rows.append([headline, summary, date, url])
    return rows


def main() -> None:
    rows: list[list[str]] = []
    for url in URLS:
        rows.extend(scrape_url(url))

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Headline", "Summary", "Date", "Source URL"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} articles to {OUTPUT}")


if __name__ == "__main__":
    main()
