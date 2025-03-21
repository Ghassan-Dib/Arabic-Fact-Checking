import json
import requests
import time
import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
FACT_CHECK_TOOLS_URL = os.getenv("FACT_CHECK_TOOLS_URL")


def query_api(url, data):
    response = requests.get(url, params=data)

    if response.status_code != 200:
        response.raise_for_status()

    response = response.json()

    sleep_time = 1

    while "error" in response and response["error"]["code"] == 503:
        print("Service unavailable. Trying again in " + str(sleep_time) + " sec.")
        time.sleep(sleep_time)
        r = requests.get(url, params=data)
        if r.status_code != 200:
            r.raise_for_status()
        response = r.json()
        sleep_time *= 2

    return response


def get_claim_reviews(
    query, language="ar", max_age_days=365, output_filename="claim_reviews.json"
):
    """
    Retrieves ClaimReview data using the Google Fact Check Tools API.

    Args:
        query: The search query to find claims about.
        languageCode: The language of the query (e.g., "en", "es", "fr"). Defaults to English.
        max_age_days: The maximum age of the claim in days. Defaults to 365 (1 year).

    Returns:
        A list of ClaimReview dictionaries, or None if there's an error.
        Returns an empty list if no results are found.
    """

    params = {
        "query": query,
        "languageCode": language,
        "maxAgeDays": max_age_days,
        "pageSize": 50,
        "key": API_KEY,
    }

    try:
        data = query_api(FACT_CHECK_TOOLS_URL, params)

        if "claims" in data:
            # if output_filename is not None:
            #     with open(output_filename, "w") as fout:
            #         fout.write(json.dumps(data["claims"]))

            return data["claims"]
        else:
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response text: {data}")
        return None


# try different variations of the query
query = [
    "الأخبار",
    "منشورات",
    "تغريدة",
    "صحفي",
    "اشتباكات",
    "مظاهرة",
    "اعلام",
    "تويتر",
    "فيسبوك",
    "موقع",
    "امراة",
    "رجل",
    "شاب",
]

claims_count = 0

for q in tqdm.tqdm(query):
    claim_reviews = get_claim_reviews(q)
    claims_count += len(claim_reviews)
    # if claim_reviews is not None:
    #     if claim_reviews:  # Checks if the list is not empty
    #         print(f"Found {len(claim_reviews)} claim review(s) for: '{q}'")
    #     else:
    #         print(f"No claim reviews found for: '{q}'")

print(f"Total number of claim reviews found: {claims_count}")
