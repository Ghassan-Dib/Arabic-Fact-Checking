import json
from .utils import retrieve_gold_evidence
# from retrieval.qa_generator.url2text import extract_text_from_url

url = "https://akhbarmeter.org/topics/2341"

gold_evidence = retrieve_gold_evidence(url)

with open("gold_evidence.json", "w", encoding="utf-8") as f:
    json.dump(gold_evidence, f, ensure_ascii=False, indent=4)

# url = "https://yoopyup.com/yoopyups/157.html"
# print("\n🔍 Extracting text:")
# extracted_text = extract_text_from_url(url)
# with open("extracted_text.txt", "w", encoding="utf-8") as f:
#     f.write(extracted_text)
# print("✓ Extracted text saved to extracted_text.txt.")
