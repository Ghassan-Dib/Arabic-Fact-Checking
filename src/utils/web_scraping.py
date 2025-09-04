import os
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


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
