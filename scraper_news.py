import requests
from bs4 import BeautifulSoup
import uuid
from datetime import datetime
import json
import os
from urllib.parse import urljoin

# базовые разделы
BASE_VARIETY_SOURCES = [
    "https://variety.com/v/film/news/",
    "https://variety.com/v/tv/news/",
    "https://variety.com/v/digital/news/",
]

DATA_DIR = "data"

# сколько страниц на каждый раздел попробуем обойти
PAGES_PER_SECTION = 5  # можно потом увеличить/уменьшить


def fetch_html(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    resp.raise_for_status()
    return resp.text


def parse_variety_listing(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            full = urljoin(base_url, href)
        else:
            full = href
        # фильтруем только реальные статьи с годом
        if "variety.com" in full and "/202" in full:
            links.add(full.split("?")[0])
    return list(links)


def parse_variety_article(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    headline = h1.get_text(strip=True) if h1 else ""

    date_str = ""
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        date_str = time_tag["datetime"][:10]

    article_div = soup.find("div", class_="article__content") or soup
    body_parts = []
    for p in article_div.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt:
            body_parts.append(txt)
    body = "\n".join(body_parts)

    return headline, date_str, body


def main(target_count=300):
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    out_path = os.path.join(DATA_DIR, f"articles_{today_str}.jsonl")

    collected = 0
    seen_urls = set()

    with open(out_path, "w", encoding="utf-8") as f:
        for base_src in BASE_VARIETY_SOURCES:
            for page in range(1, PAGES_PER_SECTION + 1):
                if page == 1:
                    src = base_src
                else:
                    # у variety странички так
                    src = f"{base_src}page/{page}/"

                print(f"\nScraping listing: {src}")
                try:
                    listing_html = fetch_html(src)
                except Exception as e:
                    print("  Error fetching listing:", e)
                    continue

                article_links = parse_variety_listing(listing_html, src)
                print(f"  Found {len(article_links)} candidate links on this page")

                for art_url in article_links:
                    if art_url in seen_urls:
                        continue
                    seen_urls.add(art_url)

                    print(f"  -> requesting article: {art_url}")
                    try:
                        art_html = fetch_html(art_url)
                    except Exception as e:
                        print("     Error fetching article:", e)
                        continue

                    headline, date_str, body = parse_variety_article(art_html)
                    if not body.strip():
                        print("     Skipped: empty body")
                        continue

                    article_obj = {
                        "id": str(uuid.uuid4()),
                        "url": art_url,
                        "date": date_str or today_str,
                        "headline": headline,
                        "body": body,
                    }

                    f.write(json.dumps(article_obj, ensure_ascii=False) + "\n")
                    collected += 1
                    print(f"     Saved. Total: {collected}")

                    if collected >= target_count:
                        print(f"\nCollected {collected} articles. Done.")
                        return

    print(f"\nFinished. Collected {collected} articles total.")


if __name__ == "__main__":
    main()
