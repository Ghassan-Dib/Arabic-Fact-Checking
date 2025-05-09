import sys
import os
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from claim_retriever.api import query_api
from claim_retriever.utils import load_queries, save_claims_to_file
from claim_retriever.config import (
    API_KEY,
    OUTPUT_FILENAME,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_PAGE_SIZE,
)


def main():
    queries = load_queries("claim_retriever/queries.txt")

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

    if all_claims:
        save_claims_to_file(all_claims, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
