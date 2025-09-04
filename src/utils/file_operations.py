import json
import csv
from collections import Counter


def load_queries(filename="queries.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            queries = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        return queries
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please make sure it exists.")
        return []


def save_to_file(data, output_filename, encoding="utf-8"):
    try:
        with open(output_filename, "w", encoding=encoding) as fout:
            json.dump(data, fout, ensure_ascii=False, indent=4)
        print(f"✓ Successfully saved {len(data)} claims to '{output_filename}'")
    except Exception as e:
        print(f"Error saving claims to file: {e}")


def save_df(df, output_filename):
    df.to_json(
        output_filename,
        orient="records",
        force_ascii=False,
        indent=4,
        date_format="iso",
    )


def get_label_distribution(claims, fieldName, output_path):
    ratings = []
    for claim in claims:
        for review in claim.get("claimReview", []):
            rating = review.get(fieldName)
            if rating:
                ratings.append(rating)
    ratings_count = dict(Counter(ratings))

    # Write the ratings_count to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ratings_count, f, ensure_ascii=False, indent=2)

    print(f"✓ Label distribution saved to '{output_path}'")
    return ratings_count


def filter_claims_by_source(claims, whitelisted_sites):
    """
    Filters claims to only include those from whitelisted sites.
    """
    filtered_claims = []
    for claim in claims:
        claim_review = claim.get("claimReview", [])
        if any(
            review.get("publisher", {}).get("site", "").lower() in whitelisted_sites
            for review in claim_review
        ):
            filtered_claims.append(claim)
    print(f"✓ Filtered claims to {len(filtered_claims)} from whitelisted sites.")

    return filtered_claims


def remove_duplicates(claims):
    seen_claims = set()
    unique_claims = []
    for claim in claims:
        claim_text = claim.get("text")
        if claim_text:
            if claim_text not in seen_claims:
                unique_claims.append(claim)
                seen_claims.add(claim_text)
    print(f"✓ Removed {len(claims) - len(unique_claims)} duplicate claims.")
    print(f"✓ {len(unique_claims)} unique claims returned.")
    return unique_claims


def normalize_claims(
    claims, labels_map, translate_labels=True, remove_noisy_labels=True
):
    if translate_labels:
        # Create a reverse map from all known translations to their normalized label
        translation_to_label = {}
        for normalized_label, variants in labels_map.items():
            if isinstance(variants, list):
                for variant in variants:
                    translation_to_label[variant.strip().lower()] = normalized_label
            else:
                translation_to_label[variants.strip().lower()] = normalized_label

        # Update the textualRating fields
        for claim in claims:
            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "").strip().lower()
                if rating in translation_to_label:
                    review["normalizedTextualRating"] = translation_to_label[rating]

    if remove_noisy_labels:
        # Remove claims with non-standard textual ratings
        claims = [
            claim
            for claim in claims
            if all(
                review.get("normalizedTextualRating") in labels_map
                for review in claim.get("claimReview", [])
            )
        ]

    save_to_file(claims, "data/normalized_claims.json")
    print(f"✓ Normalized and translated textual ratings in {len(claims)} claims")
    return claims


def save_to_csv(data, csv_path):
    headers = [
        "ClaimID",
        "claim",
        "description",
        "source",
        "date",
        "source_label",
        "normalized_label",
        "source_url",
        "claimant",
    ]

    rows = []
    for idx, item in enumerate(data, start=1):
        claim_text = item.get("text", "")
        claimant = item.get("claimant", "")
        claim_reviews = item.get("claimReview", [])

        if not claim_reviews:
            continue  # Skip if there's no review

        review = claim_reviews[0]
        source_name = review.get("publisher", {}).get("name", "")
        review_url = review.get("url", "")
        review_title = review.get("title", "")
        review_date = review.get("reviewDate", "")
        label = review.get("textualRating", "")
        normalized_label = review.get("normalizedTextualRating", "")

        rows.append(
            {
                "ClaimID": idx,
                "claim": claim_text,
                "description": review_title,
                "source": source_name,
                "date": review_date,
                "source_label": label,
                "normalized_label": normalized_label,
                "source_url": review_url,
                "claimant": claimant,
            }
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ CSV file created successfully at: {csv_path}")
