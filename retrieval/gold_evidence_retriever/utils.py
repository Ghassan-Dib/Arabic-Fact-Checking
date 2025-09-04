import os
import re
import json
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


def create_sources_set(csv_path):
    df = pd.read_csv(csv_path)
    sources = df["source"].dropna().unique()

    with open("sources.txt", "w", encoding="utf-8") as f:
        for source in sources:
            f.write(f"{source}\n")
    return set(sources)


def generate_random_id():
    return "".join(random.choices("0123456789", k=16))


def is_error_page(soup):
    text = soup.get_text().lower()
    return (
        "404" in text
        or "page not found" in text
        or "not found" in text
        or "عذراً الصفحة المطلوبة غير موجودة" in text
        or "الصفحة غير موجودة" in text
        or "المعذرة، ليس لديك حق الوصول إلى هذه الصفحة يمكنك العودة للصفحة الرئيسية"
        in text
    )


def scrape_html(url):
    os.makedirs("scraped_html", exist_ok=True)
    os.makedirs("publishers", exist_ok=True)

    page_id = generate_random_id()

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
    except Exception as e:
        print(f"❌ Selenium error loading page {url}: {e}")
        driver.quit()
        return None, None

    driver.quit()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    if is_error_page(soup):
        print(f"⚠️ Detected 404 or error page: {url}")
        return None, None

    # Remove <style> tags
    for style in soup.find_all("style"):
        style.decompose()

    # Remove <link rel="stylesheet" ...> tags
    for link in soup.find_all("link", rel="stylesheet"):
        link.decompose()

    # Remove inline style attributes
    for tag in soup.find_all(True):  # True = all tags
        if tag.has_attr("style"):
            del tag["style"]

    # Save cleaned HTML to file
    pretty_soup = soup.prettify()
    with open(f"scraped_html/{page_id}.html", "w", encoding="utf-8") as f:
        f.write(str(pretty_soup))

    # print(f"✓ HTML saved to scraped_html/{page_id}.html")

    return soup, page_id


def extract_button_sources(soup, header_keywords, btn_class):
    header = soup.find(
        lambda tag: tag.name in ["h2", "h3", "h4", "h5", "h6"]
        and tag.get_text(strip=True)
        and any(kw in tag.get_text(strip=True) for kw in header_keywords)
    )
    entries = []

    if header:
        section_div = header.find_next_sibling(
            "div", class_="row"
        ) or header.find_next_sibling("div")
        if section_div:
            for col_div in section_div.find_all("div", class_="col-md-6"):
                a_tag = col_div.find("a", class_=btn_class, href=True)
                if a_tag:
                    name = a_tag.get_text(strip=True)
                    url = a_tag["href"]
                    entries.append({"name": name, "url": url})
    return entries


def extract_carousel_sources(soup):
    """Extract sources from the deep-dive article carousel"""
    sources = []
    container = soup.find("div", class_="deep-dive--article_sources")

    if container:
        for item in container.select(".owl-item a[href]"):
            name_tag = item.find("span", class_="name")
            name = name_tag.get_text(strip=True) if name_tag else None
            url = item["href"]
            if name and url:
                sources.append({"name": name, "url": url})

        for anchor in container.select("div.section-body a[href]"):
            name_tag = anchor.find("span", class_="name")
            name = name_tag.get_text(strip=True) if name_tag else None
            url = anchor["href"]
            if name and url:
                sources.append({"name": name, "url": url})
    return sources


def extract_carousel_sources_near_header(soup, header_keywords):
    """Extract sources from an owl-carousel near a matching header like 'الناشرون'"""
    header = soup.find(
        lambda tag: tag.name in ["h2", "h3", "h4", "h5", "h6"]
        and tag.get_text(strip=True)
        and any(kw in tag.get_text(strip=True) for kw in header_keywords)
    )
    sources = []

    if header:
        # Look ahead for the carousel container
        section_div = header.find_parent("div") or header.find_next("div")
        if section_div:
            for item in section_div.select(".owl-item a[href]"):
                name_tag = item.find("span", class_="name")
                name = name_tag.get_text(strip=True) if name_tag else None
                url = item["href"]
                if name and url:
                    sources.append({"name": name, "url": url})
    return sources


def extract_sources(soup):
    sources = []

    # === Pattern 1: Classic evidence under <h5>المصدر</h5> + .btn-success ===
    header = soup.find(
        lambda tag: tag.name in ["h2", "h3", "h4", "h5", "h6"]
        and tag.get_text(strip=True) == "المصدر"
    )
    if header:
        row_div = header.find_next_sibling("div", class_="row")
        if row_div:
            for a_tag in row_div.find_all("a", class_="btn-success", href=True):
                sources.append(
                    {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                )

    # === Pattern 2: Styled section with id="resources" or id="Resources"
    resources_section = soup.find(
        lambda tag: tag.name == "section" and tag.get("id", "").lower() == "resources"
    )
    if resources_section:
        for a_tag in resources_section.find_all("a", href=True):
            text_div = a_tag.find("div", class_="font-bold")
            if text_div:
                sources.append(
                    {"name": text_div.get_text(strip=True), "url": a_tag["href"]}
                )

    # === Pattern 3: ".cz_title_content" containing <h2>مصادر التحقق:</h2>
    cz_sections = soup.find_all("div", class_="cz_title_content")
    for section in cz_sections:
        h2 = section.find("h2")
        if h2 and "مصادر التحقق" in h2.get_text(strip=True):
            # Look for <a> tags inside this block
            for a_tag in section.find_all("a", href=True):
                sources.append(
                    {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                )

    # === Pattern 4: <p><strong>مصادر التحقق</strong></p> followed by multiple <p><a>...</a></p>
    for p in soup.find_all("p"):
        strong_tag = p.find("strong")
        if strong_tag and "مصادر التحقق" in strong_tag.get_text(strip=True):
            next_p = p.find_next_sibling()
            while next_p and next_p.name == "p":
                a_tag = next_p.find("a", href=True)
                if a_tag:
                    sources.append(
                        {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                    )
                next_p = next_p.find_next_sibling()
            break

    # === Fallback 1: Links with anchor text like "مصدر", "مصدر 1", etc.
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if re.fullmatch(r"مصدر\s*\d*", text):
            sources.append({"name": text, "url": a["href"]})

    # === Fallback 2: <p><strong>المصادر</strong></p> with links in next <p>
    for p in soup.find_all("p"):
        strong_tag = p.find("strong")
        if strong_tag and "المصادر" in strong_tag.get_text(strip=True):
            next_p = p.find_next_sibling("p")
            if next_p:
                for a_tag in next_p.find_all("a", href=True):
                    sources.append(
                        {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                    )
            break

    # 3. Button-style sources under headings like "مصادر" or "المصادر"
    sources += extract_button_sources(soup, ["مصادر", "المصادر"], "btn-success")

    # 4. Carousel sources in the deep-dive article section
    sources += extract_carousel_sources(soup)

    # 5. Carousel sources near headers like "مصادر" or "المصادر"
    sources += extract_carousel_sources_near_header(soup, ["مصادر", "المصادر"])

    return sources


def extract_publishers(soup):
    sources = []

    # 1. Extract <p><a> sources under the "الناشرون" heading
    publisher_heading = soup.find(
        lambda tag: tag.name in ["h2", "h3", "h4", "h5", "h6"]
        and "الناشرون" in tag.get_text(strip=True)
    )

    if publisher_heading:
        current = publisher_heading.find_next_sibling()
        while current:
            if current.name == "p":
                for a_tag in current.find_all("a", href=True):
                    sources.append(
                        {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                    )
            elif current.name and current.name.startswith("h"):
                break  # stop at next heading
            current = current.find_next_sibling()

    # 2. Pattern: ".cz_title_content" containing <h2>مصادر الادعاء:
    cz_sections = soup.find_all("div", class_="cz_title_content")
    for section in cz_sections:
        h2 = section.find("h2")
        if h2 and "مصادر الادعاء" in h2.get_text(strip=True):
            for a_tag in section.find_all("a", href=True):
                sources.append(
                    {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                )

    # 3. Pattern: <p><strong>مصادر الادعاء</strong></p> followed by multiple <p><a>...</a></p>
    for p in soup.find_all("p"):
        strong_tag = p.find("strong")
        if strong_tag and "مصادر الادعاء" in strong_tag.get_text(strip=True):
            next_p = p.find_next_sibling()
            while next_p and next_p.name == "p":
                # Stop if we hit the next section heading
                next_strong = next_p.find("strong")
                if next_strong and "مصادر التحقق" in next_strong.get_text(strip=True):
                    break
                a_tag = next_p.find("a", href=True)
                if a_tag:
                    sources.append(
                        {"name": a_tag.get_text(strip=True), "url": a_tag["href"]}
                    )
                next_p = next_p.find_next_sibling()
            break

    # 3. Extract from button-style publishers (e.g., btn-warning)
    sources += extract_button_sources(soup, ["ناشرون", "الناشرون"], "btn-warning")

    # 4. Extract from carousel sources near the header
    sources += extract_carousel_sources_near_header(soup, ["ناشرون", "الناشرون"])

    return sources


def extract_sources_and_publishers(soup, page_id):
    sources = extract_sources(soup)
    # publishers = extract_publishers(soup)

    gold_evidence = sources

    with open(f"publishers/{page_id}.json", "w", encoding="utf-8") as f:
        json.dump(gold_evidence, f, ensure_ascii=False, indent=4)

    # print(
    #     f"✓ Extracted publishers' links and names saved to publishers/{page_id}.json"
    # )

    return {"sources": sources}


def retrieve_gold_evidence(source_url, retries=3):
    soup, page_id = scrape_html(source_url)

    if not soup:
        print(f"❌ Failed to scrape HTML for {source_url}")
        return None

    for attempt in range(1, retries + 1):
        try:
            gold_evidence = extract_sources_and_publishers(soup, page_id)

            if gold_evidence and (
                gold_evidence.get("publishers") or gold_evidence.get("sources")
            ):
                return gold_evidence

            print(f"🔄 Empty evidence, retrying {source_url} ({attempt}/{retries})...")

        except Exception as e:
            print(f"❌ Error on attempt {attempt} for {source_url}: {e}")

    print(f"⚠️ Failed to retrieve evidence after {retries} attempts: {source_url}")
    return None
