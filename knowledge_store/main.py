from duckduckgo_search import DDGS
import sys
import os
import json
import tqdm
import time
from dateutil.parser import parse as date_parse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge_store.utils import save_to_file, find_published_date

search = DDGS()

print("🧠 Creating knowledge store...")

knowledge_store = []

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
claim_reviews_path = os.path.join(data_dir, "claim_reviews.json")

with open(claim_reviews_path, "r") as f:
    claim_reviews = json.load(f)

BATCH_SIZE = 100
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# Process claims in batches
for batch_start in range(0, len(claim_reviews), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = claim_reviews[batch_start:batch_end]
    print(f"Processing batch {batch_start + 1} to {batch_end}...")

    for claim in tqdm.tqdm(batch):
        claim_text = claim.get("text")
        claim_date = (
            date_parse(claim.get("claimDate", ""), fuzzy=True)
            if claim.get("claimDate")
            else None
        )
        retries = 0
        claim_results = []

        # Skip claims with no date or text
        if not claim_date or not claim_text:
            print(
                f"Skipping empty claim text at index {batch_start + len(claim_results)}"
            )
            continue

        while retries < MAX_RETRIES:
            try:
                # Conduct Duckduckgo search
                results = search.text(claim_text, max_results=10)
                for result in results:
                    publish_date = find_published_date(result["href"])
                    if publish_date and publish_date < claim_date:
                        claim_results.append(
                            {
                                "title": result["title"],
                                "url": result["href"],
                                "snippet": result["body"],
                            }
                        )
                break
            except Exception as e:
                retries += 1
                wait_time = BACKOFF_FACTOR**retries
                print(f"Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # Add the results for the current claim to the knowledge store
        knowledge_store.append(
            {
                "claim": claim,
                "evidences": claim_results,
            }
        )

        time.sleep(2)  # Wait between claims to avoid rate limiting

# Save progress after each batch
save_to_file(knowledge_store, "knowledge_store.json")
print(f"✅ Batch {batch_start + 1} to {batch_end} processed and saved.")

print("✅ Knowledge store creation complete.")
