import json
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as date_parse
import pytz
import re
import os
from htmldate import find_date


def save_to_file(data, output_filename, encoding="utf-8"):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        output_path = os.path.join(data_dir, output_filename)
        with open(output_path, "w", encoding=encoding) as fout:
            json.dump(data, fout, ensure_ascii=False, indent=4)
        print(f"✅ Successfully saved claims and their evidences to '{output_path}'")
    except Exception as e:
        print(f"Error saving claims to file: {e}")


def find_published_date(url):
    try:
        publish_date = date_parse(
            find_date(
                url,
                extensive_search=True,
            )
        )
        if publish_date.tzinfo is None:
            publish_date = publish_date.replace(tzinfo=pytz.UTC)
        return publish_date

    except Exception as e:
        print(f"Error fetching/parsing {url}: {e}")

    return None


def extract_published_date(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. Check common meta tags
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

        # 2. Check schema.org JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "datePublished" in data:
                    data_str = data["datePublished"]
                    if re.search(r"[\u0600-\u06FF]", data_str):  # Arabic characters
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
