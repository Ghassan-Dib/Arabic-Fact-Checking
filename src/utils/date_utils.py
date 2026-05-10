import json
import logging
import re
from datetime import datetime

import pytz
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as date_parse

from core.exceptions import WebScrapingError

logger = logging.getLogger(__name__)

_ARABIC_MONTHS = {
    "يناير": "January",
    "فبراير": "February",
    "مارس": "March",
    "أبريل": "April",
    "مايو": "May",
    "يونيو": "June",
    "يوليو": "July",
    "أغسطس": "August",
    "سبتمبر": "September",
    "أكتوبر": "October",
    "نوفمبر": "November",
    "ديسمبر": "December",
}

_ARABIC_AMPM = {"ص": "AM", "م": "PM"}

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_DATE_IN_URL_RE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def _make_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=pytz.UTC)
    return dt


def parse_arabic_date(date_str: str) -> datetime | None:
    """Translate Arabic month names / AM-PM markers then parse with dateutil."""
    for ar, en in _ARABIC_MONTHS.items():
        date_str = date_str.replace(ar, en)
    for ar, en in _ARABIC_AMPM.items():
        date_str = re.sub(rf"\b{ar}\b", en, date_str)
    date_str = re.sub(r"^[؀-ۿ]+،\s*", "", date_str)

    try:
        return _make_aware(date_parse(date_str, fuzzy=True))
    except Exception as exc:
        raise WebScrapingError(f"Failed to parse Arabic date: {date_str}") from exc


def extract_published_date(url: str) -> datetime | None:
    """Try URL pattern, schema.org JSON-LD, then meta tags."""
    m = _DATE_IN_URL_RE.search(url)
    if m:
        try:
            return _make_aware(date_parse("-".join(m.groups()), fuzzy=True))
        except Exception:
            pass

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        raise WebScrapingError(f"Failed to fetch page for date extraction: {url}") from exc

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items: list[dict[str, object]] = [data] if isinstance(data, dict) else data
            for item in items:
                raw_graph = item.get("@graph", [])
                graph: list[dict[str, object]] = raw_graph if isinstance(raw_graph, list) else []
                date_str = item.get("datePublished") or next(
                    (i.get("datePublished") for i in graph if "datePublished" in i),
                    None,
                )
                if isinstance(date_str, str):
                    dt = (
                        parse_arabic_date(date_str)
                        if _ARABIC_RE.search(date_str)
                        else _make_aware(date_parse(date_str, fuzzy=True))
                    )
                    if dt:
                        return dt
        except Exception:
            continue

    from bs4 import Tag

    for name in ["pubdate", "publish_date", "date", "dc.date", "dc.date.issued"]:
        tag = soup.find("meta", attrs={"name": name})
        if isinstance(tag, Tag):
            content = tag.get("content")
            if content:
                try:
                    return _make_aware(date_parse(str(content), fuzzy=True))
                except Exception:
                    pass

    for prop in ["article:published_time", "og:published_time", "og:updated_time"]:
        tag = soup.find("meta", attrs={"property": prop})
        if isinstance(tag, Tag):
            content = tag.get("content")
            if content:
                try:
                    return _make_aware(date_parse(str(content), fuzzy=True))
                except Exception:
                    pass

    return None


def find_published_date(url: str) -> datetime | None:
    """Try custom extraction first, fall back to htmldate."""
    try:
        dt = extract_published_date(url)
        if dt:
            return dt
    except WebScrapingError:
        pass

    try:
        from htmldate import find_date

        raw = find_date(url, extensive_search=True, original_date=True)
        if raw:
            return _make_aware(date_parse(raw))
    except Exception as exc:
        logger.debug("htmldate failed for %s: %s", url, exc)

    return None
