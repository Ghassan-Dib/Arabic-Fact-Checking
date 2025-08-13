import json
import pandas as pd
from tqdm import tqdm


from .utils import extract_evidence_from_claim_lc
from src.config.settings import ANTHROPIC_API_KEY
from verification.qa_generator.fact_check_qa_generator import FactCheckQAGenerator
from src.utils.text_pocessing import convert_types, get_claim_domain

tqdm.pandas()


def main():
    df = pd.read_csv("data/processed/claims_with_100_text.csv", encoding="utf-8")
    df.dropna(subset=["gold_evidence_text"], inplace=True)

    qa_generator = FactCheckQAGenerator(ANTHROPIC_API_KEY)

    samples = []
    for i in tqdm(range(10)):
        claim_domain = get_claim_domain(df["source_url"].iloc[i])

        retrieved_evidence_qa = extract_evidence_from_claim_lc(
            df["claim"].iloc[i], claim_domain, qa_generator
        )

        sample = {
            "id": df["ClaimID"].iloc[i],
            "claim": df["claim"].iloc[i],
            "description": df["description"].iloc[i],
            "date": df["date"].iloc[i],
            "source_label": df["source_label"].iloc[i],
            "normalized_label": df["normalized_label"].iloc[i],
            "source_url": df["source_url"].iloc[i],
            "retrieved_evidence": retrieved_evidence_qa["qa_pairs"],
        }
        samples.append(sample)

    with open("data/evidence/retrieved_evidence_100.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4, default=convert_types)

    print(
        f"✓ Retrieved evidence for {len(samples)} samples saved to data/evidence/retrieved_evidence_100.json"
    )

    return df


if __name__ == "__main__":
    main()
