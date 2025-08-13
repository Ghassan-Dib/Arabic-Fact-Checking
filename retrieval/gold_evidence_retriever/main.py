import pandas as pd
from tqdm import tqdm
from utils import retrieve_gold_evidence

tqdm.pandas()


def safe_retrieve(url):
    try:
        return retrieve_gold_evidence(url)
    except Exception as e:
        print(f"❌ Error retrieving evidence for URL: {url}\n{e}")
        return None


def main():
    # Load the dataset
    df = pd.read_csv("data/raw/draft1.csv", encoding="utf-8")

    # Missing gold evidence IDs
    missing_evidence_ids = [
        17,
        49,
        99,
        114,
        115,
        173,
        186,
        188,
        194,
        220,
        221,
        222,
        223,
        236,
        281,
        289,
        292,
        321,
        332,
        334,
        354,
        363,
        365,
        368,
        369,
        396,
        464,
        465,
        495,
        498,
        613,
        634,
        640,
        678,
        776,
        801,
        802,
        803,
        804,
        816,
        835,
        857,
        941,
        942,
        943,
        946,
        947,
        950,
        955,
        974,
        975,
        982,
        988,
        993,
        1002,
        1017,
        1022,
        1051,
        1060,
        1065,
        1069,
        1071,
        1073,
        1076,
        1077,
        1080,
        1084,
        1093,
        1094,
        1106,
        1113,
        1123,
        1125,
        1126,
        1127,
        1128,
        1130,
        1132,
        1133,
        1138,
        1139,
        1143,
    ]
    print(f"\n✅ Found {len(missing_evidence_ids)} missing evidence entries")
    print("\n🔍 Retrieving gold evidence for claims...\n")

    # Filter to only process missing entries
    missing_mask = df["ClaimID"].isin(missing_evidence_ids)

    # Update only the missing entries
    df.loc[missing_mask, "gold_evidence"] = df.loc[
        missing_mask, "source_url"
    ].progress_apply(safe_retrieve)

    # Save updated dataset
    output_path = "data/raw/draft2.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✓ Gold evidence retrieved and saved to {output_path}")

    return df

    # df["gold_evidence"] = df["source_url"].progress_apply(safe_retrieve)
    # df.to_csv("data/claims_with_gold_evidence.csv", index=False, encoding="utf-8")
    # print("✓ Gold evidence retrieved and saved to data/claims_with_gold_evidence.csv.")
    # return df


if __name__ == "__main__":
    main()
