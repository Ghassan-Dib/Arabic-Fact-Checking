import logging
import random
import time
from pathlib import Path

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_DEFAULT_SCRAPED_HTML_DIR = Path("scraped_html")


def _generate_page_id() -> str:
    return "".join(random.choices("0123456789", k=16))


def is_error_page(soup: BeautifulSoup) -> bool:
    text = soup.get_text().lower()
    return (
        "404" in text
        or "page not found" in text
        or "not found" in text
        or "عذراً الصفحة المطلوبة غير موجودة" in text
        or "الصفحة غير موجودة" in text
        or "المعذرة، ليس لديك حق الوصول إلى هذه الصفحة يمكنك العودة للصفحة الرئيسية" in text
    )


def scrape_html(
    url: str, output_dir: Path | None = None
) -> tuple[BeautifulSoup | None, str | None]:
    """Fetch a URL via headless Chrome and return a cleaned BeautifulSoup tree."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as exc:
        raise ImportError("selenium is required for web scraping") from exc

    save_dir = output_dir if output_dir is not None else _DEFAULT_SCRAPED_HTML_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    page_id = _generate_page_id()

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
    except Exception:
        logger.warning("Selenium failed to load %s", url)
        return None, None
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    if is_error_page(soup):
        logger.warning("Error page detected: %s", url)
        return None, None

    for tag in soup.find_all("style"):
        tag.decompose()
    for tag in soup.find_all("link", rel="stylesheet"):
        tag.decompose()
    for tag in soup.find_all(True):
        if isinstance(tag, Tag) and tag.has_attr("style"):
            del tag["style"]

    with open(save_dir / f"{page_id}.html", "w", encoding="utf-8") as fh:
        fh.write(soup.prettify())

    return soup, page_id
