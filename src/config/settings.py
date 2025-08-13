"""Configuration settings for the fact checker system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVIDENCE_DIR = DATA_DIR / "evidence"

# Environment variables
API_KEY = os.getenv("API_KEY")
FACT_CHECK_TOOLS_URL = os.getenv("FACT_CHECK_TOOLS_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths to data files
CLAIMS_PATH = "claims.csv"
QUERIES_PATH = "data/queries.txt"

# Default configurations
DEFAULT_LANGUAGE = "ar"
DEFAULT_MAX_AGE_DAYS = 365
DEFAULT_PAGE_SIZE = 50

# Claude Sonnet 4
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"

# File paths (using pathlib for better path handling)
CLAIMS_FILE = RAW_DATA_DIR / "claims.csv"
GOLD_EVIDENCE_FILE = EVIDENCE_DIR / "gold_evidence.json"
RETRIEVED_EVIDENCE_FILE = EVIDENCE_DIR / "retrieved_evidence.json"
