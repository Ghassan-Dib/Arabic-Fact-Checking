from bs4 import BeautifulSoup

from retrieval.gold_retriever import extract_sources
from utils.web_scraping import is_error_page


class TestIsErrorPage:
    def test_404_text_detected(self) -> None:
        soup = BeautifulSoup("<html><body>404 page not found</body></html>", "html.parser")
        assert is_error_page(soup) is True

    def test_normal_page_not_error(self) -> None:
        soup = BeautifulSoup("<html><body><h1>مقال صحفي</h1></body></html>", "html.parser")
        assert is_error_page(soup) is False

    def test_arabic_error_page(self) -> None:
        soup = BeautifulSoup(
            "<html><body>عذراً الصفحة المطلوبة غير موجودة</body></html>", "html.parser"
        )
        assert is_error_page(soup) is True


class TestExtractSources:
    def test_btn_success_pattern(self) -> None:
        html = """
        <html><body>
          <h5>المصدر</h5>
          <div class="row">
            <div class="col-md-6">
              <a class="btn-success" href="https://example.com">مصدر 1</a>
            </div>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        sources = extract_sources(soup)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com"
        assert sources[0]["name"] == "مصدر 1"

    def test_no_sources_returns_empty(self) -> None:
        soup = BeautifulSoup("<html><body><p>نص عادي</p></body></html>", "html.parser")
        sources = extract_sources(soup)
        assert sources == []

    def test_fallback_anchor_pattern(self) -> None:
        html = '<html><body><a href="https://src.com">مصدر 1</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        sources = extract_sources(soup)
        assert any(s["url"] == "https://src.com" for s in sources)
