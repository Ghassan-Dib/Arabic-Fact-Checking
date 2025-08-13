import json
import requests
from bs4 import BeautifulSoup
import pytz
import re
import os
import sys
import time
from htmldate import find_date

# from duckduckgo_search import DDGS
from ddgs import DDGS
from tqdm import tqdm
from dateutil.parser import parse as date_parse
from src.utils.text_pocessing import extract_text_from_url

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from src.utils.text_pocessing import convert_types, get_claim_domain
from src.utils.web_scraping import scrape_html


def save_to_file(data, output_filename, encoding="utf-8"):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        output_path = os.path.join(data_dir, output_filename)
        with open(output_path, "w", encoding=encoding) as fout:
            json.dump(data, fout, ensure_ascii=False, indent=4)
        print(
            f"✓ Successfully saved claims and their evidences to /data/'{output_filename}'"
        )
        print("_____________________________________________")
    except Exception as e:
        print(f"Error saving claims to file: {e}")


def extract_date_published(soup):
    # 1. Try <meta> tags
    meta_props = [
        ("property", "article:published_time"),
        ("name", "date"),
        ("name", "pubdate"),
        ("itemprop", "datePublished"),
        ("property", "og:published_time"),
        ("name", "publish-date"),
    ]
    for attr, value in meta_props:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return tag["content"]

    # 2. Try <time> tags
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag["datetime"]

    # 3. Try JSON-LD script block
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            # Handle both dict and list types
            if isinstance(data, dict) and "datePublished" in data:
                return data["datePublished"]
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "datePublished" in entry:
                        return entry["datePublished"]
        except (json.JSONDecodeError, TypeError):
            continue

    # 4. If nothing found
    return None


def find_published_date(url, max_date=None):
    dt = extract_published_date(url)
    if dt:
        return dt

    try:
        raw_date = find_date(
            url, extensive_search=True, original_date=True, max_date=max_date
        )
        if raw_date:
            dt = date_parse(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt
    except Exception as e:
        print(f"[htmldate] Error for {url}: {e}")

    return None


def extract_published_date(url):
    try:
        # 1. Try to extract date from URL
        date_match = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", url)
        if date_match:
            date_str = "-".join(date_match.groups())
            try:
                dt = date_parse(date_str, fuzzy=True)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                return dt
            except Exception as e:
                print(f"❌ Failed to parse date from URL: {date_str}: {e}")

        # headers = {"User-Agent": "Mozilla/5.0"}
        # response = requests.get(url, headers=headers, timeout=10)
        # soup, _ = BeautifulSoup(response.text, "html.parser")
        soup, _ = scrape_html(url)

        # 2. Check schema.org JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Direct datePublished
                    if "datePublished" in data:
                        data_str = data["datePublished"]
                        if re.search(r"[\u0600-\u06FF]", data_str):  # Arabic characters
                            dt = parse_arabic_date(data_str)
                        else:
                            dt = date_parse(data_str, fuzzy=True)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=pytz.UTC)
                        return dt
                    # Handle @graph
                    elif "@graph" in data and isinstance(data["@graph"], list):
                        for item in data["@graph"]:
                            if "datePublished" in item:
                                data_str = item["datePublished"]
                                if re.search(r"[\u0600-\u06FF]", data_str):
                                    dt = parse_arabic_date(data_str)
                                else:
                                    dt = date_parse(data_str, fuzzy=True)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=pytz.UTC)
                                return dt
                elif isinstance(data, list):  # Sometimes it's a list of dicts
                    for item in data:
                        if "datePublished" in item:
                            data_str = data["datePublished"]
                            if re.search(
                                r"[\u0600-\u06FF]", data_str
                            ):  # Arabic characters
                                dt = parse_arabic_date(data_str)
                            else:
                                dt = date_parse(data_str, fuzzy=True)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=pytz.UTC)
                            return dt
            except Exception:
                continue

        # 3. Check common meta tags
        meta_names = ["pubdate", "publish_date", "date", "dc.date", "dc.date.issued"]
        meta_props = ["article:published_time", "og:published_time", "og:updated_time"]
        for name in meta_names:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                dt = date_parse(tag["content"], fuzzy=True)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                return dt

        for prop in meta_props:
            tag = soup.find("meta", attrs={"property": prop})
            if tag and tag.get("content"):
                dt = date_parse(tag["content"], fuzzy=True)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                return dt

    except Exception as e:
        print(f"Error fetching/parsing {url}: {e}")

    return None


def parse_arabic_date(date_str):
    month_map = {
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

    am_pm_map = {"ص": "AM", "م": "PM"}

    for ar_month, en_month in month_map.items():
        date_str = date_str.replace(ar_month, en_month)

    for ar_ampm, en_ampm in am_pm_map.items():
        date_str = re.sub(rf"\b{ar_ampm}\b", en_ampm, date_str)

    date_str = re.sub(r"^[\u0600-\u06FF]+،\s*", "", date_str)

    try:
        dt = date_parse(date_str, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except Exception as e:
        print(f"❌ Failed to parse Arabic date: {date_str}: {e}")
        return None


def retrieve_potential_evidence(claim_reviews):
    print("\n🔍  Retrieving Evidence...\n")

    knowledge_store = []

    BATCH_SIZE = 10
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2

    search = DDGS()

    # Process claims in batches
    for batch_start in range(0, len(claim_reviews), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = claim_reviews[batch_start:batch_end]
        print(f"Processing batch {batch_start + 1} to {batch_end}...")

        for claim in tqdm.tqdm(batch):
            claim_text = claim.get("text")
            claim_date = (
                date_parse(claim.get("claimDate", ""), fuzzy=True)
                if claim.get("claimDate")
                else None
            )
            retries = 0
            claim_results = []

            # Skip claims with no date or text
            if not claim_date or not claim_text:
                print(
                    f"Skipping empty claim text at index {batch_start + len(claim_results)}"
                )
                continue

            while retries < MAX_RETRIES:
                try:
                    # Conduct Duckduckgo search
                    results = search.text(claim_text, max_results=5)

                    for result in results:
                        publish_date = find_published_date(result["href"])
                        print(f"‼️ Publish date: {publish_date}")
                        if publish_date and publish_date < claim_date:
                            claim_results.append(
                                {
                                    "title": result["title"],
                                    "url": result["href"],
                                    "snippet": result["body"],
                                }
                            )
                    break
                except Exception as e:
                    retries += 1
                    wait_time = BACKOFF_FACTOR**retries
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            # Add the results for the current claim to the knowledge store
            knowledge_store.append(
                {
                    "claim": claim,
                    "evidences": claim_results,
                }
            )

            time.sleep(2)  # Wait between claims to avoid rate limiting

    # Save progress after each batch
    save_to_file(knowledge_store, "knowledge_store.json")
    print(f"✓ Batch {batch_start + 1} to {batch_end} processed and saved.")

    print("✓ Knowledge store creation complete.")


def retrieve_external_evidence_lc(claim_text, claim_date=None):
    """
    Retrieve external evidence for a given claim text using DuckDuckGo search.
    """
    print(f"\n🔍 Searching for evidence for claim: {claim_text}\n")

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=4)
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper, source="news", output_format="list"
    )

    results = []
    try:
        results = search.invoke(claim_text)
        for result in results:
            print(f"result : {result}")
            # publish_date = find_published_date(result["href"])
            # if publish_date and publish_date < claim_date:
            if not is_relevant_result(result, claim_text):
                continue

            results.append(
                {
                    "claim": claim_text,
                    "title": result["title"],
                    "url": result.get("link"),
                    "snippet": result["snippet"],
                }
            )
        print(f"✓ Found {len(results)} external sources for the claim.")
        with open("retrieved_evidence.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ Error retrieving evidence: {e}", file=sys.stderr)
    return results


def retrieve_external_evidence(claim_text, claim_date=None):
    """
    Retrieve external evidence for a given claim text using DuckDuckGo search.
    """
    # print(f"\n🔍 Searching for evidence for claim: {claim_text}\n")

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    results = []

    retries = 0
    while retries < MAX_RETRIES:
        try:
            with DDGS() as search:
                for result in search.text(
                    query=claim_text,
                    # region="xa-ar",
                    safesearch="off",
                    max_results=5,
                ):
                    publish_date = find_published_date(result["href"])
                    if not is_relevant_result(result, claim_text):
                        continue

                    results.append(
                        {
                            "claim": claim_text,
                            "title": result["title"],
                            "url": result["href"],
                            "snippet": result["body"],
                            "date": publish_date.isoformat() if publish_date else None,
                        }
                    )
                print(f"✓ Found {len(results)} external sources for the claim.")
                break
        except Exception as e:
            retries += 1
            wait_time = BACKOFF_FACTOR**retries
            print(f"Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return results


def extract_evidence_from_claim(claim_text, claim_domain, qa_generator):
    """returns a list of evidence text concatenated into one object"""
    retrieved_evidence = retrieve_external_evidence(claim_text)

    if not retrieved_evidence:
        # retry twice
        print("\nNo evidence found, retrying...\n")
        for _ in range(2):
            retrieved_evidence = retrieve_external_evidence(claim_text)
            if retrieved_evidence:
                break

    evidence_text = []
    for i, evidence in enumerate(retrieved_evidence, start=1):
        url = evidence["url"] or evidence.get("link")
        snippet = evidence["snippet"]
        if claim_domain and claim_domain in url:
            print(f"Skipping evidence from fact-checking: {url}")
            continue

        text = extract_text_from_url(url)
        # TODO handle cases where text is Noisy, None or empty

        if text:
            evidence_text.append(f"EVIDENCE {i}:\n" + snippet + "\n\n" + text + "\n\n")

    if evidence_text:
        evidence_text_str = "".join(evidence_text)
        qa_result = qa_generator.generate_evidence_qa_pairs(
            claim_text, evidence_text_str
        )

        return qa_result

    return {"qa_pairs": []}


def extract_evidence_from_claim_lc(claim_text, claim_domain, qa_generator):
    """returns a list of evidence text concatenated into one object"""
    retrieved_evidence = retrieve_external_evidence_lc(claim_text)

    evidence_text = []
    for i, evidence in enumerate(retrieved_evidence, start=1):
        print(f"evidence: {evidence}")
        url = evidence.get("link")
        snippet = evidence.get("snippet")

        if url is None:
            print(f"❌ No URL found for evidence {i}, skipping...")
            continue
        if claim_domain and claim_domain in url:
            print(f"Skipping evidence from fact-checking: {url}")
            continue

        text = extract_text_from_url(url)
        # TODO handle cases where text is Noisy, None or empty

        if text:
            evidence_text.append(f"EVIDENCE {i}:\n" + snippet + "\n\n" + text + "\n\n")

    if evidence_text:
        evidence_text_str = "".join(evidence_text)
        qa_result = qa_generator.generate_evidence_qa_pairs(
            claim_text, evidence_text_str
        )

        return qa_result

    return {"qa_pairs": []}


def is_relevant_result(result, claim_text):
    """Filter out irrelevant search results"""

    # Extract key terms from claim
    claim_keywords = set(process_arabic_claim_for_search(claim_text).split())

    # Check title and snippet for relevance
    title = result.get("title", "").lower()
    snippet = result.get("body", "").lower()
    url = result.get("href", "").lower()

    # Filter out obvious irrelevant results
    irrelevant_indicators = [
        "google",
        "search",
        "homepage",
        "privacy",
        "terms",
        "settings",
        "advertising",
        "about google",
        "carbon neutral",
    ]

    for indicator in irrelevant_indicators:
        if indicator in title or indicator in snippet or indicator in url:
            return False

    # Check for Arabic content presence
    arabic_pattern = r"[\u0600-\u06FF]"
    if not re.search(arabic_pattern, title + snippet):
        return False

    # Check for keyword overlap
    combined_text = title + " " + snippet
    matches = sum(1 for keyword in claim_keywords if keyword in combined_text)

    # Require at least 2 keyword matches
    return matches >= 2


def process_arabic_claim_for_search(claim_text, max_keywords=5):
    """Extract key terms from Arabic claim for better search results"""

    # Remove common Arabic stop words and filler phrases
    stop_words = [
        "خبر",
        "زعم",
        "ناشروه",
        "قيام",
        "حتى",
        "إشعار",
        "آخر",
        "الذي",
        "التي",
        "الذين",
        "اللذين",
        "اللتين",
        "اللواتي",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "من",
        "في",
        "على",
        "إلى",
        "عن",
        "مع",
        "بين",
        "تحت",
        "فوق",
        "أمام",
        "خلف",
        "بعد",
        "قبل",
    ]

    # Extract key entities and terms
    # Remove the introductory phrase "خبر زعم ناشروه"
    cleaned_claim = re.sub(r"^خبر زعم ناشروه\s*", "", claim_text)

    # Split into words and filter
    words = cleaned_claim.split()
    keywords = []

    for word in words:
        # Clean punctuation
        clean_word = re.sub(r"[،؛:.!؟]", "", word)

        # Skip stop words and short words
        if clean_word and len(clean_word) > 2 and clean_word not in stop_words:
            keywords.append(clean_word)

    # Return top keywords
    return " ".join(keywords[:max_keywords])


def retrieve_qa_evidence(df, qa_generator):
    samples = []
    for i in tqdm(range(len(df))):
        claim_domain = get_claim_domain(df["source_url"].iloc[i])

        retrieved_evidence_qa = extract_evidence_from_claim(
            df["claim"].iloc[i], claim_domain, qa_generator
        )

        sample = {
            "id": df["ClaimID"].iloc[i],
            "claim": df["claim"].iloc[i],
            "description": df["description"].iloc[i],
            "date": df["date"].iloc[i],
            "source_label": df["source_label"].iloc[i],
            "normalized_label": df["normalized_label"].iloc[i],
            "source_url": df["source_url"].iloc[i],
            "retrieved_evidence": retrieved_evidence_qa["qa_pairs"],
        }
        samples.append(sample)

    with open("data/evidence/retrieved_evidence_100.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4, default=convert_types)
