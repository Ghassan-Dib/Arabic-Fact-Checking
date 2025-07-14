import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
API_KEY = os.getenv("API_KEY")
FACT_CHECK_TOOLS_URL = os.getenv("FACT_CHECK_TOOLS_URL")

# Paths to data files
CLAIMS_PATH = "data/claims.csv"
QUERIES_PATH = "data/queries.txt"

# Default configurations
DEFAULT_LANGUAGE = "ar"
DEFAULT_MAX_AGE_DAYS = 365
DEFAULT_PAGE_SIZE = 50
