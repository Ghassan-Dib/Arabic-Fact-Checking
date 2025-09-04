import os
import re
import sys
import ast
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from .web_scraping import scrape_html
from src.config.constants import REMOVAL_KEYWORDS

tqdm.pandas()


def get_claim_domain(claim_url):
    pattern = r"https?://(?:www\.)?([^/]+)"
    match = re.search(pattern, claim_url)
    claim_domain = match.group(1) if match else ""
    return claim_domain


def remove_duplicate_lines(lines):
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line not in seen and line != "":
            seen.add(line)
            unique_lines.append(line)

    return unique_lines


def extract_text_from_url(url):
    soup, page_id = scrape_html(url)

    if soup is None:
        return ""

    # Remove unwanted tags
    for tag in soup(
        [
            "style",
            "script",
            'div[id*="ad"]',
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

    # Remove footer divs
    for div in soup.find_all(
        "div",
        class_=lambda c: c
        and (
            "footer" in c
            or "sidebar" in c
            or "related" in c
            or "advertisement" in c
            or "comments" in c
            or "share" in c
            or "social" in c
            or "navigation" in c
            or "consent" in c
        ),
    ):
        div.decompose()

    # Remove "آخر المقالات" element in "Fatabyyano" articles
    for h5 in soup.find_all("h5"):
        if "آخر المقالات" in h5.get_text(strip=True):
            # Get the parent container that wraps the whole sidebar
            parent_div = h5.find_parent("div", class_="vc_col-sm-2")
            if parent_div:
                parent_div.decompose()
                break

    pretty_soup = soup.prettify()
    with open(f"scraped_html/{page_id}*.html", "w", encoding="utf-8") as f:
        f.write(str(pretty_soup))

    # Extract candidate sections
    main_sections = soup.find_all(["div"])

    texts = []
    if main_sections:
        for section in main_sections:
            for tag in section.find_all(["p", "h2", "h3", "span"]):
                text = tag.get_text(strip=True)
                if text:
                    texts.append(text)

    # Remove duplicate lines
    texts = remove_duplicate_lines(texts)

    # Remove unwanted keywords
    cut_idx = None
    for i, text in enumerate(texts):
        if any(keyword == text for keyword in REMOVAL_KEYWORDS):
            cut_idx = i
            break

    # Keep only items before the keyword (if found)
    if cut_idx is not None:
        texts = texts[:cut_idx]

    # For Misbar, remove everything that comes before "تحقيق مسبار"
    if "تحقيق مسبار" in texts:
        cut_idx = texts.index("تحقيق مسبار")
        texts = texts[cut_idx:]

    # with open("extracted_text.txt", "w", encoding="utf-8") as f:
    #     for text in texts:
    #         f.write(text + "\n")

    return "\n".join(texts) if texts else ""


def is_mostly_arabic(text, threshold=0.5):
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    return (len(arabic_chars) / len(text)) >= threshold if text.strip() else False


def clean_text_block(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        # Keep lines that are mostly Arabic
        if stripped_line and is_mostly_arabic(stripped_line):
            cleaned_lines.append(stripped_line)

    return "\n".join(cleaned_lines)


def concatenate_sources(urls, separator="\n\n"):
    texts = []
    index = 1

    for url in urls:
        sources_text = "".join(clean_text_block(extract_text_from_url(url)))
        if sources_text:
            texts.append(f"{separator} {index}:\n{sources_text}")
            index += 1

    return "\n\n".join(texts)


def concatenate_evidence(evi_pairs):
    texts = []
    index = 1

    for url, snippet, date in evi_pairs:
        sources_text = "".join(clean_text_block(extract_text_from_url(url)))
        if sources_text:
            texts.append(
                f"EVIDENCE {index}:\npublished date: {date}\n{snippet}\n{sources_text}"
            )
            index += 1

    return "\n\n".join(texts)


def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
