# scraper/sources.py
from __future__ import annotations
from typing import List, Dict, Iterable
from datetime import datetime, timezone
import time, hashlib, logging, re
import requests, feedparser
from bs4 import BeautifulSoup
from newspaper import Article

# --------- CONFIG ---------
MAX_PER_SOURCE = int(re.sub(r"\D", "", str(20)))   # per-source cap (tune later)
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN = 0.5  # be polite between fetches
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-IN,en;q=0.9"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --------- HELPERS ---------
def _uid(url: str, title: str) -> str:
    m = hashlib.md5()
    m.update((url.strip() + "||" + title.strip()).encode("utf-8", "ignore"))
    return m.hexdigest()

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _fetch_ok(url: str) -> bool:
    # quick HEAD ping to avoid long download for dead links
    try:
        r = requests.head(url, timeout=REQUEST_TIMEOUT, headers={**HEADERS, "Accept": "*/*"}, allow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False

def _extract_article_text(url: str) -> str:
    """
    Prefer newspaper3k for readability; fallback to simple HTML extraction.
    Keep it robust (GitHub Actions has limited time).
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        txt = (art.text or "").strip()
        if len(txt) >= 400:  # good enough
            return txt
    except Exception:
        pass

    # fallback: basic HTML parse
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # generic content guesses:
        candidates = []
        for sel in [
            "article", ".story__content", ".content", ".article-content", ".row .col", "#content", ".Post-content",
            ".td-post-content", ".entry-content", ".story-content"
        ]:
            for node in soup.select(sel):
                txt = " ".join(p.get_text(" ", strip=True) for p in node.select("p"))
                if len(txt) > 400:
                    candidates.append(txt)
        if not candidates:
            # fallback to all <p>
            txt = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            return txt.strip()
        # pick the longest chunk
        return max(candidates, key=len).strip()
    except Exception:
        return ""

def _rss_items(feeds: Iterable[str], source_name: str) -> List[Dict]:
    items: List[Dict] = []
    seen = set()
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed)
            for e in parsed.entries[: MAX_PER_SOURCE*2]:  # overfetch; we filter below
                link = getattr(e, "link", "") or ""
                title = getattr(e, "title", "") or ""
                if not link or not title: 
                    continue
                k = (link.strip(), title.strip())
                if k in seen:
                    continue
                seen.add(k)
                items.append({
                    "uid": _uid(link, title),
                    "title": title.strip(),
                    "url": link.strip(),
                    "source": source_name,
                    # prefer e.published if present
                    "date": getattr(e, "published", "") or _utcnow_iso(),
                    "lang": "en",   # will be auto-detected later if missing
                })
                if len(items) >= MAX_PER_SOURCE:
                    break
        except Exception as ex:
            logging.warning(f"Feed error for {source_name}: {feed} -> {ex}")
        if len(items) >= MAX_PER_SOURCE:
            break
    return items

def _hydrate_fulltext(slim_items: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for it in slim_items:
        url = it["url"]
        title = it["title"]
        if not _fetch_ok(url):
            continue
        txt = _extract_article_text(url)
        if len(txt) < 400:
            # skip very short or broken pages
            continue
        out.append({
            "uid": it["uid"],
            "title": title,
            "text": txt,
            "lang": it.get("lang") or "en",
            "source": it["source"],
            "date": it.get("date") or _utcnow_iso(),
            "url": url
        })
        time.sleep(SLEEP_BETWEEN)
    return out

# --------- PER-SOURCE SCRAPERS (RSS → article text) ---------
def scrape_ndtv() -> List[Dict]:
    feeds = [
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        # add more NDTV feeds if you want
    ]
    logging.info("NDTV: fetching feeds")
    items = _rss_items(feeds, "NDTV")
    logging.info(f"NDTV: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

def scrape_thehindu() -> List[Dict]:
    feeds = [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://www.thehindu.com/business/feeder/default.rss",
        "https://www.thehindu.com/sci-tech/feeder/default.rss",
    ]
    logging.info("The Hindu: fetching feeds")
    items = _rss_items(feeds, "The Hindu")
    logging.info(f"The Hindu: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

def scrape_toi() -> List[Dict]:
    # TOI Top Stories feed id (commonly used):
    feeds = [
        "https://timesofindia.indiatimes.com/rssfeeds/-2128932452.cms",
    ]
    logging.info("TOI: fetching feeds")
    items = _rss_items(feeds, "TOI")
    logging.info(f"TOI: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

def scrape_jagran() -> List[Dict]:
    # Hindi; adjust feeds if any returns 404 (use RSS pages provided by Jagran)
    feeds = [
        "https://www.jagran.com/rss/national.xml",
        "https://www.jagran.com/rss/state/andhra-pradesh.xml",  # example; keep or remove
    ]
    logging.info("Jagran: fetching feeds")
    items = _rss_items(feeds, "Jagran")
    logging.info(f"Jagran: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

def scrape_eenadu() -> List[Dict]:
    # Telugu; if Eenadu RSS changes, you can swap to Google News RSS fallback:
    # f"https://news.google.com/rss/search?q=site:eenadu.net&hl=te-IN&gl=IN&ceid=IN:te"
    feeds = [
        "https://www.eenadu.net/rss",  # if 404, comment and use Google News fallback
    ]
    logging.info("Eenadu: fetching feeds")
    items = _rss_items(feeds, "Eenadu")
    logging.info(f"Eenadu: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

def scrape_abp() -> List[Dict]:
    feeds = [
        # If this 404s, use Google News fallback for abplive.com (below)
        "https://news.abplive.com/feeds/rss/india.xml",
    ]
    logging.info("ABP: fetching feeds")
    items = _rss_items(feeds, "ABP")
    logging.info(f"ABP: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)

# --------- GOOGLE NEWS FALLBACK (if any native feed breaks) ---------
def scrape_google_news_site(domain: str, label: str) -> List[Dict]:
    # English fallback; adjust hl/gl/ceid for regional languages if needed
    feed = f"https://news.google.com/rss/search?q=site:{domain}&hl=en-IN&gl=IN&ceid=IN:en"
    logging.info(f"Google News fallback for {label}")
    items = _rss_items([feed], label)
    return _hydrate_fulltext(items)

# --------- MAIN AGGREGATOR ---------
def scrape_all() -> List[Dict]:
    all_items: List[Dict] = []
    try:
        all_items.extend(scrape_ndtv())
    except Exception as ex:
        logging.warning(f"NDTV failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("ndtv.com", "NDTV-GNews"))

    try:
        all_items.extend(scrape_thehindu())
    except Exception as ex:
        logging.warning(f"The Hindu failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("thehindu.com", "Hindu-GNews"))

    try:
        all_items.extend(scrape_toi())
    except Exception as ex:
        logging.warning(f"TOI failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("indiatimes.com", "TOI-GNews"))

    try:
        all_items.extend(scrape_jagran())
    except Exception as ex:
        logging.warning(f"Jagran failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("jagran.com", "Jagran-GNews"))

    try:
        all_items.extend(scrape_eenadu())
    except Exception as ex:
        logging.warning(f"Eenadu failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("eenadu.net", "Eenadu-GNews"))

    try:
        all_items.extend(scrape_abp())
    except Exception as ex:
        logging.warning(f"ABP failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("abplive.com", "ABP-GNews"))

    logging.info(f"TOTAL articles scraped (pre-dedup by UID upstream): {len(all_items)}")
    return all_items
