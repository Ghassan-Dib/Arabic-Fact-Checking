import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
FACT_CHECK_TOOLS_URL = os.getenv("FACT_CHECK_TOOLS_URL")
OUTPUT_FILENAME = "claim_reviews.json"
DEFAULT_LANGUAGE = "ar"
DEFAULT_MAX_AGE_DAYS = 365
DEFAULT_PAGE_SIZE = 50
