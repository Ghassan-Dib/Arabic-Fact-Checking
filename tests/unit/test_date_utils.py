import pytest

from core.exceptions import WebScrapingError
from utils.date_utils import parse_arabic_date


class TestParseArabicDate:
    def test_arabic_month_name(self) -> None:
        dt = parse_arabic_date("15 يناير 2024")
        assert dt is not None
        assert dt.month == 1
        assert dt.year == 2024

    def test_all_arabic_months(self) -> None:
        months = [
            ("يناير", 1),
            ("فبراير", 2),
            ("مارس", 3),
            ("أبريل", 4),
            ("مايو", 5),
            ("يونيو", 6),
            ("يوليو", 7),
            ("أغسطس", 8),
            ("سبتمبر", 9),
            ("أكتوبر", 10),
            ("نوفمبر", 11),
            ("ديسمبر", 12),
        ]
        for ar_month, expected_month in months:
            dt = parse_arabic_date(f"1 {ar_month} 2024")
            assert dt is not None, f"Failed to parse month: {ar_month}"
            assert dt.month == expected_month

    def test_invalid_date_raises(self) -> None:
        with pytest.raises(WebScrapingError):
            parse_arabic_date("not a date at all xyz")

    def test_returns_timezone_aware(self) -> None:
        dt = parse_arabic_date("10 مارس 2023")
        assert dt is not None
        assert dt.tzinfo is not None


class TestExtractPublishedDateFromUrl:
    def test_url_with_date_pattern(self) -> None:
        from utils.date_utils import extract_published_date

        with pytest.MonkeyPatch().context() as mp:
            import requests

            mp.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(Exception("no net")))
            # Should still get date from URL itself
            dt = extract_published_date("https://example.com/2023/05/15/article")
            assert dt is not None
            assert dt.year == 2023
            assert dt.month == 5
            assert dt.day == 15
