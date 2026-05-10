import logging
import re

from bs4 import BeautifulSoup, Tag

from models.evidence import Evidence, GoldEvidence
from utils.web_scraping import scrape_html

logger = logging.getLogger(__name__)


def _extract_button_sources(
    soup: BeautifulSoup, header_keywords: list[str], btn_class: str
) -> list[dict[str, str]]:
    header = soup.find(
        lambda tag: (
            tag.name in ["h2", "h3", "h4", "h5", "h6"]
            and any(kw in tag.get_text(strip=True) for kw in header_keywords)
        )
    )
    entries: list[dict[str, str]] = []
    if not isinstance(header, Tag):
        return entries
    row = header.find_next_sibling("div", class_="row") or header.find_next_sibling("div")
    if isinstance(row, Tag):
        for col in row.find_all("div", class_="col-md-6"):
            a = col.find("a", class_=btn_class, href=True)
            if isinstance(a, Tag):
                entries.append({"name": a.get_text(strip=True), "url": str(a["href"])})
    return entries


def _extract_carousel_sources(soup: BeautifulSoup) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    container = soup.find("div", class_="deep-dive--article_sources")
    if not isinstance(container, Tag):
        return sources
    for item in container.select(".owl-item a[href]"):
        name_tag = item.find("span", class_="name")
        name = name_tag.get_text(strip=True) if isinstance(name_tag, Tag) else None
        if name:
            sources.append({"name": name, "url": str(item["href"])})
    for a in container.select("div.section-body a[href]"):
        name_tag = a.find("span", class_="name")
        name = name_tag.get_text(strip=True) if isinstance(name_tag, Tag) else None
        if name:
            sources.append({"name": name, "url": str(a["href"])})
    return sources


def _extract_carousel_near_header(
    soup: BeautifulSoup, header_keywords: list[str]
) -> list[dict[str, str]]:
    header = soup.find(
        lambda tag: (
            tag.name in ["h2", "h3", "h4", "h5", "h6"]
            and any(kw in tag.get_text(strip=True) for kw in header_keywords)
        )
    )
    sources: list[dict[str, str]] = []
    if not header:
        return sources
    section = header.find_parent("div") or header.find_next("div")
    if isinstance(section, Tag):
        for item in section.select(".owl-item a[href]"):
            name_tag = item.find("span", class_="name")
            name = name_tag.get_text(strip=True) if isinstance(name_tag, Tag) else None
            if name:
                sources.append({"name": name, "url": str(item["href"])})
    return sources


def extract_sources(soup: BeautifulSoup) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []

    # Pattern 1: <h5>المصدر</h5> + .btn-success
    h = soup.find(
        lambda tag: (
            tag.name in ["h2", "h3", "h4", "h5", "h6"] and tag.get_text(strip=True) == "المصدر"
        )
    )
    if isinstance(h, Tag):
        row = h.find_next_sibling("div", class_="row")
        if isinstance(row, Tag):
            for a in row.find_all("a", class_="btn-success", href=True):
                sources.append({"name": a.get_text(strip=True), "url": str(a["href"])})

    # Pattern 2: section#resources
    section = soup.find(
        lambda tag: tag.name == "section" and tag.get("id", "").lower() == "resources"
    )
    if isinstance(section, Tag):
        for a in section.find_all("a", href=True):
            text_div = a.find("div", class_="font-bold")
            if isinstance(text_div, Tag):
                sources.append({"name": text_div.get_text(strip=True), "url": str(a["href"])})

    # Pattern 3 & 4: .cz_title_content and <p><strong>مصادر التحقق</strong>
    for div in soup.find_all("div", class_="cz_title_content"):
        h2 = div.find("h2")
        if isinstance(h2, Tag) and "مصادر التحقق" in h2.get_text(strip=True):
            for a in div.find_all("a", href=True):
                sources.append({"name": a.get_text(strip=True), "url": str(a["href"])})

    for p in soup.find_all("p"):
        strong = p.find("strong")
        if isinstance(strong, Tag) and "مصادر التحقق" in strong.get_text(strip=True):
            sib = p.find_next_sibling()
            while isinstance(sib, Tag) and sib.name == "p":
                a = sib.find("a", href=True)
                if isinstance(a, Tag):
                    sources.append({"name": a.get_text(strip=True), "url": str(a["href"])})
                sib = sib.find_next_sibling()
            break

    # Fallback: anchor text matching "مصدر N"
    for a in soup.find_all("a", href=True):
        if re.fullmatch(r"مصدر\s*\d*", a.get_text(strip=True)):
            sources.append({"name": a.get_text(strip=True), "url": str(a["href"])})

    sources += _extract_button_sources(soup, ["مصادر", "المصادر"], "btn-success")
    sources += _extract_carousel_sources(soup)
    sources += _extract_carousel_near_header(soup, ["مصادر", "المصادر"])

    # Deduplicate by URL while preserving order
    seen_urls: set[str] = set()
    unique: list[dict[str, str]] = []
    for s in sources:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique.append(s)
    return unique


class GoldEvidenceRetriever:
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def retrieve(self, source_url: str) -> GoldEvidence | None:
        soup, _ = scrape_html(source_url)
        if not soup:
            logger.warning("Failed to scrape %s", source_url)
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = extract_sources(soup)
                if raw:
                    items = [Evidence(title=s["name"], url=s["url"]) for s in raw if s.get("url")]
                    return GoldEvidence(sources=items)
                logger.debug("Empty evidence on attempt %d for %s", attempt, source_url)
            except Exception as exc:
                logger.warning("Error on attempt %d for %s: %s", attempt, source_url, exc)

        logger.warning("No evidence found after %d attempts: %s", self.max_retries, source_url)
        return None
