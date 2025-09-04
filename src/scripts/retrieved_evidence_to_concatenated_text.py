from tqdm import tqdm
import pandas as pd

from src.utils.text_pocessing import concatenate_evidence
from src.utils.file_operations import save_df


def main():
    df = pd.read_json("data/train/evi04.json")

    if "retrieved_evidence_text" not in df.columns:
        df["retrieved_evidence_text"] = None

    print(f"\nextracting retrieved evidence text for {len(df)} claims..\n")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        evi_pairs = [
            (item["url"], item["snippet"], item["date"])
            for item in row["retrieved_evidence"]
            if item and "url" in item and "snippet" in item and "date" in item
        ]

        if evi_pairs:
            text = concatenate_evidence(evi_pairs)
            if text:
                df.at[index, "retrieved_evidence_text"] = text
            else:
                df.at[index, "retrieved_evidence_text"] = None

    save_df(df, "data/train/evi05.json")


if __name__ == "__main__":
    main()
