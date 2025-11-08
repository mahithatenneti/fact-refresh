# scraper/sources.py
from __future__ import annotations
from typing import List, Dict, Iterable
from datetime import datetime, timezone
import time, hashlib, logging, re
import requests, feedparser
from bs4 import BeautifulSoup
from newspaper import Article

# --------- CONFIG ---------
MAX_PER_SOURCE = int(re.sub(r"\D", "", str(20)))  # per-source cap
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN = 0.4  # be polite between fetches
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
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
    """
    Some publishers block HEAD; use a tiny GET with stream=True.
    Only check reachability, not content.
    """
    try:
        r = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={**HEADERS, "Accept": "*/*"}, stream=True
        )
        return r.status_code < 400
    except Exception:
        return False


def _extract_article_text(url: str) -> str:
    """
    Prefer newspaper3k; fallback to soup-based extraction.
    """
    # newspaper3k
    try:
        art = Article(url, browser_user_agent=UA, request_timeout=REQUEST_TIMEOUT)
        art.download()
        art.parse()
        txt = (art.text or "").strip()
        if len(txt) >= 300:  # slightly lower threshold to avoid over-dropping
            return txt
    except Exception:
        pass

    # fallback: generic HTML
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        candidates = []
        for sel in [
            "article",
            ".story__content",
            ".content",
            ".article-content",
            ".row .col",
            "#content",
            ".Post-content",
            ".td-post-content",
            ".entry-content",
            ".story-content",
            "main",
        ]:
            for node in soup.select(sel):
                txt = " ".join(p.get_text(" ", strip=True) for p in node.select("p"))
                if len(txt) > 300:
                    candidates.append(txt)
        if not candidates:
            txt = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            return txt.strip()
        return max(candidates, key=len).strip()
    except Exception:
        return ""


def _rss_items(feeds: Iterable[str], source_name: str, default_lang: str = "en") -> List[Dict]:
    items: List[Dict] = []
    seen = set()
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed, request_headers=HEADERS)
            for e in parsed.entries[: MAX_PER_SOURCE * 3]:  # overfetch; we filter below
                link = getattr(e, "link", "") or ""
                title = getattr(e, "title", "") or ""
                if not link or not title:
                    continue
                k = (link.strip(), title.strip())
                if k in seen:
                    continue
                seen.add(k)
                items.append(
                    {
                        "uid": _uid(link, title),
                        "title": title.strip(),
                        "url": link.strip(),
                        "source": source_name,
                        "date": getattr(e, "published", "")
                        or getattr(e, "updated", "")
                        or _utcnow_iso(),
                        "lang": default_lang,
                    }
                )
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
        if len(txt) < 300:  # aligned with extractor
            continue
        out.append(
            {
                "uid": it["uid"],
                "title": title,
                "text": txt,
                "lang": it.get("lang") or "en",
                "source": it["source"],
                "date": it.get("date") or _utcnow_iso(),
                "url": url,
            }
        )
        time.sleep(SLEEP_BETWEEN)
    return out


# --------- GOOGLE NEWS HELPERS ---------
def _gnews_feed(domain: str, lang: str) -> str:
    """Language-aware Google News RSS feed."""
    hl_map = {"en": "en-IN", "hi": "hi-IN", "te": "te-IN"}
    ceid_map = {"en": "IN:en", "hi": "IN:hi", "te": "IN:te"}
    hl = hl_map.get(lang, "en-IN")
    ceid = ceid_map.get(lang, "IN:en")
    return f"https://news.google.com/rss/search?q=site:{domain}&hl={hl}&gl=IN&ceid={ceid}"


def scrape_google_news_site(domain: str, label: str, default_lang: str = "en") -> List[Dict]:
    feed = _gnews_feed(domain, default_lang)
    logging.info(f"Google News fallback for {label} ({default_lang})")
    items = _rss_items([feed], label, default_lang)
    return _hydrate_fulltext(items)


def _fallback_if_empty(items: List[Dict], domain: str, label: str, default_lang: str) -> List[Dict]:
    """Use Google News if a native feed returns 0 items."""
    if items:
        return items
    logging.warning(f"{label}: 0 native items. Falling back to Google News.")
    return _rss_items([_gnews_feed(domain, default_lang)], label, default_lang)


# --------- PER-SOURCE SCRAPERS (RSS → article text) ---------
def scrape_ndtv() -> List[Dict]:
    feeds = [
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://feeds.feedburner.com/ndtvnews-india-news",
        "https://feeds.feedburner.com/ndtvnews-world-news",
    ]
    logging.info("NDTV: fetching feeds")
    items = _rss_items(feeds, "NDTV", "en")
    items = _fallback_if_empty(items, "ndtv.com", "NDTV", "en")
    logging.info(f"NDTV: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


def scrape_thehindu() -> List[Dict]:
    feeds = [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://www.thehindu.com/business/feeder/default.rss",
        "https://www.thehindu.com/sci-tech/feeder/default.rss",
    ]
    logging.info("The Hindu: fetching feeds")
    items = _rss_items(feeds, "The Hindu", "en")
    items = _fallback_if_empty(items, "thehindu.com", "The Hindu", "en")
    logging.info(f"The Hindu: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


def scrape_toi() -> List[Dict]:
    feeds = [
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://timesofindia.indiatimes.com/rssfeeds/-2128932452.cms",  # India
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",    # World
    ]
    logging.info("TOI: fetching feeds")
    items = _rss_items(feeds, "TOI", "en")
    items = _fallback_if_empty(items, "indiatimes.com", "TOI", "en")
    logging.info(f"TOI: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


def scrape_jagran() -> List[Dict]:
    feeds = [
        "https://www.jagran.com/rss/news/national.xml",
        # add more regional feeds if you like
    ]
    logging.info("Jagran: fetching feeds")
    items = _rss_items(feeds, "Jagran", "hi")
    items = _fallback_if_empty(items, "jagran.com", "Jagran", "hi")
    logging.info(f"Jagran: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


def scrape_eenadu() -> List[Dict]:
    feeds = [
        "https://www.eenadu.net/rss/latest-news.xml",
    ]
    logging.info("Eenadu: fetching feeds")
    items = _rss_items(feeds, "Eenadu", "te")
    items = _fallback_if_empty(items, "eenadu.net", "Eenadu", "te")
    logging.info(f"Eenadu: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


def scrape_abp() -> List[Dict]:
    feeds = [
        "https://news.abplive.com/home/feed",
    ]
    logging.info("ABP: fetching feeds")
    items = _rss_items(feeds, "ABP", "hi")
    items = _fallback_if_empty(items, "abplive.com", "ABP", "hi")
    logging.info(f"ABP: {len(items)} feed items; hydrating")
    return _hydrate_fulltext(items)


# --------- MAIN AGGREGATOR ---------
def scrape_all() -> List[Dict]:
    all_items: List[Dict] = []
    try:
        all_items.extend(scrape_ndtv())
    except Exception as ex:
        logging.warning(f"NDTV failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("ndtv.com", "NDTV-GNews", "en"))

    try:
        all_items.extend(scrape_thehindu())
    except Exception as ex:
        logging.warning(f"The Hindu failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("thehindu.com", "Hindu-GNews", "en"))

    try:
        all_items.extend(scrape_toi())
    except Exception as ex:
        logging.warning(f"TOI failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("indiatimes.com", "TOI-GNews", "en"))

    try:
        all_items.extend(scrape_jagran())
    except Exception as ex:
        logging.warning(f"Jagran failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("jagran.com", "Jagran-GNews", "hi"))

    try:
        all_items.extend(scrape_eenadu())
    except Exception as ex:
        logging.warning(f"Eenadu failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("eenadu.net", "Eenadu-GNews", "te"))

    try:
        all_items.extend(scrape_abp())
    except Exception as ex:
        logging.warning(f"ABP failed, using fallback: {ex}")
        all_items.extend(scrape_google_news_site("abplive.com", "ABP-GNews", "hi"))

    logging.info(f"TOTAL articles scraped (pre-dedup by UID upstream): {len(all_items)}")
    return all_items
