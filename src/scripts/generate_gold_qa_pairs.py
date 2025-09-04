import json
import pandas as pd
from tqdm import tqdm

from src.verification.fact_check_qa_generator import FactCheckQAGenerator
from src.utils.file_operations import save_df


def main():
    df = pd.read_json("data/processed/claims14.json")

    if "qa_pairs" not in df.columns:
        df["qa_pairs"] = None

    print(f"\ngenerating QA pairs for {len(df)} claims..\n")

    samples = []
    qa_generator = FactCheckQAGenerator()
    
    for index, row in tqdm(df.iterrows()):
        claim = row.get("claim", "")
        fact_checking_text = row.get("fact_checking_text", "")
        gold_text = row.get("gold_evidence_text", "")

        if gold_text:
            qa_pairs = qa_generator.generate_qa_pairs(
                claim, fact_checking_text, gold_text
            )
            if qa_pairs:
                samples.append(qa_pairs)
                df.at[index, "qa_pairs"] = qa_pairs["qa_pairs"]
            else:
                df.at[index, "qa_pairs"] = None

        save_df(df, "data/train/claims_test.json")

    with open("qa_pairs.json", "w", encoding="utf-8") as fout:
        json.dump(samples, fout, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
