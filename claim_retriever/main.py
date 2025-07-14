import sys
import os
import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from claim_retriever.api import query_api
from claim_retriever.utils import (
    load_queries,
    save_to_csv,
    remove_duplicates,
    normalize_claims,
    get_label_distribution,
)
from claim_retriever.config import (
    API_KEY,
    QUERIES_PATH,
    CLAIMS_PATH,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_PAGE_SIZE,
)
from claim_retriever.constants import labels_map


def main():
    """
    Main function to retrieve claims based on predefined queries.
    It loads queries from a file, queries the API for claims, cleans the data by removing duplicates,
    normalizes the claims, and creates a csv dataset.
    """
    queries = load_queries(QUERIES_PATH)

    all_claims = []

    for query in tqdm.tqdm(queries):
        params = {
            "query": query,
            "languageCode": DEFAULT_LANGUAGE,
            "maxAgeDays": DEFAULT_MAX_AGE_DAYS,
            "pageSize": DEFAULT_PAGE_SIZE,
            "key": API_KEY,
        }
        claims = query_api(params)
        all_claims.extend(claims)

    # Clean the claims data
    if all_claims:
        # Remove duplicates from the claims
        all_claims = remove_duplicates(all_claims)

        # Count unique textual ratings and save to a file
        get_label_distribution(all_claims, "textualRating", "data/source_labels.json")

        # Normalize and translate textual ratings
        all_claims = normalize_claims(
            all_claims, labels_map, translate_labels=True, remove_noisy_labels=True
        )

        # Get the label distribution for normalized labels
        get_label_distribution(
            all_claims, "normalizedTextualRating", "data/normalized_labels.json"
        )

        # Create a CSV dataset
        save_to_csv(all_claims, CLAIMS_PATH)


if __name__ == "__main__":
    start_time = time.time()
    print("\n⏱ Starting claim retrieval process...")
    main()
    end_time = time.time()
    print(f"✅ Claim retrieval completed in {end_time - start_time:.2f} seconds.")
