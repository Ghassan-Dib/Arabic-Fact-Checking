import os
import pytz
import datetime
import ast
import json
import pandas as pd
from tqdm import tqdm

# from duckduckgo_search import DDGS
from ddgs import DDGS

import requests
from .utils import extract_evidence_from_claim, retrieve_external_evidence
from src.utils.text_pocessing import extract_text_from_url


from src.config.settings import ANTHROPIC_API_KEY
from verification.qa_generator.fact_check_qa_generator import FactCheckQAGenerator

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


search = DDGS()


# dt = datetime.datetime.strptime(claim_date, "%Y-%m-%dT%H:%M:%SZ")
# dt = dt.replace(tzinfo=pytz.UTC)


def main():
    # df = pd.read_csv("data/claims_with_gold_evidence.csv", encoding="utf-8")
    # df["evidence"] = df["evidence"].progress_apply(
    #     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    # )
    # df["evidence"] = df["evidence"].apply(
    #     lambda x: x["sources"] if isinstance(x, dict) else x
    # )

    # qa_generator = FactCheckQAGenerator(ANTHROPIC_API_KEY)

    # samples = []
    # for i in range(5):
    #     retrieved_evidence_qa = extract_evidence_from_claim(
    #         df["claim"].iloc[i], df["source_url"].iloc[i], qa_generator
    #     )

    #     sample = {
    #         "claim": df["claim"].iloc[i],
    #         "source": df["source"].iloc[i],
    #         "date": df["date"].iloc[i],
    #         "source_label": df["source_label"].iloc[i],
    #         "normalized_label": df["normalized_label"].iloc[i],
    #         "source_url": df["source_url"].iloc[i],
    #         "questions": retrieved_evidence_qa["qa_pairs"],
    #     }
    #     samples.append(sample)

    # with open("data/evidence/retrieved_evidence.json", "w", encoding="utf-8") as f:
    #     json.dump(samples, f, ensure_ascii=False, indent=4)

    claim_text = "خبر نشرته قناة الجزيرة عن صحيفة يديعوت أحرونوت العبرية، مفاده أن المغرب أرسل تعزيزات أمنية مكثفة إلى إسرائيل على حدود الجولان."
    retrieved_evidence = retrieve_external_evidence(claim_text)

    with open("retrieved_evidence.json", "w", encoding="utf-8") as f:
        json.dump(retrieved_evidence, f, ensure_ascii=False, indent=4)


# if __name__ == "__main__":
#     main()


"""
unresolved: 
"https://ghrannews.com/65993/",
"https://www.alyaum.com/articles/1143322/المملكة-اليوم/",
"https://www.pressreader.com/germany/deutsche-welle-arabic-edition/20241104/281797109503362", -> directs to the main page, not the article
"https://nrttv.com/ar/detail3/38875",
"https://nrttv.com/ar/detail3/38875",
"https://nrttv.com/AR/detail3/38891",
"https://gate.ahram.org.eg/News/5075442.aspx",
"https://www.ohchr.org/ar/stories/2022/07/un-expert-warns-dangerous-decline-media-freedom",
"https://24.ae/article/863712/ما-حقيقة-العثور-على-صدام-حسين-في-سجن-صيدنايا-",
"""


def duckduckgo_search(query):
    ddgs = DDGS()
    results = []

    for result in ddgs.text(query, max_results=5):
        results.append(
            {
                "title": result["title"],
                "link": result["href"],
                "snippet": result["body"],
            }
        )

    return results

    # Example usage


claim_text = "خبر نشرته قناة الجزيرة عن صحيفة يديعوت أحرونوت العبرية، مفاده أن المغرب أرسل تعزيزات أمنية مكثفة إلى إسرائيل على حدود الجولان."
search_results = duckduckgo_search(claim_text)
with open("retrieved_evidence.json", "w", encoding="utf-8") as f:
    json.dump(search_results, f, ensure_ascii=False, indent=4)
