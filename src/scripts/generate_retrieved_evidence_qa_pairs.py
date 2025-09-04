import json
import pandas as pd
from tqdm import tqdm

from src.verification.fact_check_qa_generator import FactCheckQAGenerator
from src.utils.file_operations import save_df


def main():
    df = pd.read_json("data/train/retrieved_evidence.json")

    if "retrieved_qa_pairs" not in df.columns:
        df["retrieved_qa_pairs"] = None

    print(f"\ngenerating QA pairs for {len(df)} evidence..\n")

    qa_generator = FactCheckQAGenerator()

    samples = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        retrieved_evidence = row.get("retrieved_evidence")

        # If evidence already contains QA pairs
        if (
            isinstance(retrieved_evidence, list)
            and len(retrieved_evidence) > 0
            and isinstance(retrieved_evidence[0], dict)
            and "question" in retrieved_evidence[0]
            and "answer" in retrieved_evidence[0]
        ):
            df.at[index, "retrieved_qa_pairs"] = retrieved_evidence
            continue

        claim = row.get("claim", "")
        claim_date = row.get("date", "")
        retrieved_evidence_text = row.get("retrieved_evidence_text", "")

        if retrieved_evidence_text:
            qa_pairs = qa_generator.generate_evidence_qa_pairs(
                claim, claim_date, retrieved_evidence_text
            )
            if qa_pairs:
                samples.append(qa_pairs)
                df.at[index, "retrieved_qa_pairs"] = qa_pairs["qa_pairs"]
            else:
                df.at[index, "retrieved_qa_pairs"] = None

        save_df(df, "data/train/evidence_test.json")

    with open("retrieved_qa_pairs.json", "w", encoding="utf-8") as fout:
        json.dump(samples, fout, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
