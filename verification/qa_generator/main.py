import os
import sys
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.settings import ANTHROPIC_API_KEY
from .url2text import extract_text_from_url
from src.utils.text_pocessing import clean_text_block, convert_types
from .fact_check_qa_generator import FactCheckQAGenerator


tqdm.pandas()


def main():
    df = pd.read_csv("data/train/claims.json", encoding="utf-8")

    print(f"\nGenerating QA pairs for {len(df)} fact checking articles...\n")

    samples = []
    for i in tqdm(range(2)):
        qa_generator = FactCheckQAGenerator(ANTHROPIC_API_KEY)
        fact_check_text = df["fact_checking_text"].iloc[i]  # Fact Checking article text
        sources_text = df["gold_evidence_text"].iloc[i]  # Sources text

        gold_evidence_qa_pairs = qa_generator.generate_qa_pairs(
            fact_check_text, sources_text
        )

        sample = {
            "id": df["ClaimID"].iloc[i],
            "claim": df["claim"].iloc[i],
            "description": df["description"].iloc[i],
            "date": df["date"].iloc[i],
            "source_label": df["source_label"].iloc[i],
            "normalized_label": df["normalized_label"].iloc[i],
            "source_url": df["source_url"].iloc[i],
            "fact_checking_text": df["fact_checking_text"].iloc[i],
            "gold_evidence": df["gold_evidence"].iloc[i],
            "gold_evidence_text": df["gold_evidence_text"].iloc[i],
            "gold_evidence_qa_pairs": gold_evidence_qa_pairs["qa_pairs"],
        }
        samples.append(sample)

    with open("data/processed/gold_evidence_100.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4, default=convert_types)

    print(f"✓ qa pairs generated for {len(samples)} samples.")


if __name__ == "__main__":
    main()
