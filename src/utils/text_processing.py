import re
from typing import Any

import numpy as np

from utils.web_scraping import scrape_html

REMOVAL_KEYWORDS = [
    "مواضيع أخرى قد تهمك",
    "Related topics",
    "اقرأ/ي أيضًا",
    "اقرأ أيضاً",
    "قد يهمك",
    "مصادر الادعاء",
    "مصادر الادعاء:",
    "رابط الادعاء:",
    "المصادر",
    "Topic categories",
    "Claim sources",
]


def is_mostly_arabic(text: str, threshold: float = 0.5) -> bool:
    arabic_chars = re.findall(r"[؀-ۿ]", text)
    return (len(arabic_chars) / len(text)) >= threshold if text.strip() else False


def clean_text_block(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line.strip() for line in lines if line.strip() and is_mostly_arabic(line.strip())
    )


def remove_duplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


def extract_text_from_url(url: str) -> str:
    soup, _ = scrape_html(url)
    if soup is None:
        return ""

    for tag in soup(
        [
            "style",
            "script",
            "iframe",
            "noscript",
            "header",
            "footer",
            "nav",
            "aside",
            "form",
            "input",
            "button",
            "svg",
            "img",
            "video",
            "audio",
            "canvas",
            "object",
            "embed",
            "link",
            "meta",
            "title",
            "figure",
        ]
    ):
        tag.decompose()

    for div in soup.find_all(
        "div",
        class_=lambda c: (
            c
            and any(
                kw in c
                for kw in [
                    "footer",
                    "sidebar",
                    "related",
                    "advertisement",
                    "comments",
                    "share",
                    "social",
                    "navigation",
                    "consent",
                ]
            )
        ),
    ):
        div.decompose()

    texts: list[str] = []
    for section in soup.find_all("div"):
        for tag in section.find_all(["p", "h2", "h3", "span"]):
            text = tag.get_text(strip=True)
            if text:
                texts.append(text)

    texts = remove_duplicate_lines(texts)

    cut_idx = next(
        (i for i, t in enumerate(texts) if any(kw == t for kw in REMOVAL_KEYWORDS)),
        None,
    )
    if cut_idx is not None:
        texts = texts[:cut_idx]

    if "تحقيق مسبار" in texts:
        texts = texts[texts.index("تحقيق مسبار") :]

    return "\n".join(texts)


def concatenate_sources(urls: list[str], separator: str = "\n\n") -> str:
    parts: list[str] = []
    for i, url in enumerate(urls, start=1):
        text = clean_text_block(extract_text_from_url(url))
        if text:
            parts.append(f"{separator} {i}:\n{text}")
    return "\n\n".join(parts)


def concatenate_evidence(evi_pairs: list[tuple[str, str, str]]) -> str:
    parts: list[str] = []
    for i, (url, snippet, date) in enumerate(evi_pairs, start=1):
        text = clean_text_block(extract_text_from_url(url))
        if text:
            parts.append(f"EVIDENCE {i}:\npublished date: {date}\n{snippet}\n{text}")
    return "\n\n".join(parts)


def convert_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
