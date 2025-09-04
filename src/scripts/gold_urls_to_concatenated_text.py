from tqdm import tqdm

import pandas as pd

from src.utils.text_pocessing import concatenate_sources
from src.utils.file_operations import save_df


def main():
    df = pd.read_json("data/train/claims12.json")

    print(f"\nextracting gold evidence text from {len(df)} claims..\n")

    for index, row in tqdm(df.iterrows()):
        gold_urls = [item["url"] for item in row["gold_evidence_urls"] if item]
        if gold_urls:
            text = concatenate_sources(gold_urls, "SOURCE")
            if text:
                df.at[index, "gold_evidence_text"] = text
            else:
                df.at[index, "gold_evidence_text"] = None

    save_df(df, "data/train/claims12.json")


if __name__ == "__main__":
    main()
