import json
from itertools import islice
import pandas as pd
from tqdm import tqdm

from src.clients import ClaudeSonnet4Client
from src.utils.file_operations import save_df


def generate_prompt(claim, evidence):
    return f"""
    سوف تحصل على ادعاء ومجموعة من الأدلة. حدد الحكم على الادعاء بناءً على الأدلة فقط، دون الاعتماد على أي معرفة خارجية.

    التصنيفات المتاحة:
    1. SUPPORTED - الأدلة تدعم وتؤكد حصول الادعاء بشكل واضح وصريح.
    2. REFUTED - الأدلة تناقض الادعاء بشكل مباشر أو تجعله غير مرجح. يُستخدم أيضًا في الحالات التي يمكن أن يكون فيها أدلة صحيحة جزئيًا لكنها تُستخدم في غير موضعها أو زمانها بشكل يضلل القارئ، حتى إن لم توجد أدلة داعمة.
    3. CONFLICTING_EVIDENCE - توجد أدلة متناقضة أو انتقائية (مثل عرض جزء من الحقيقة أو تغيّر في الموقف لم يُذكر في الادعاء).
     
    تعليمات:
    - اعتمد فقط على الأدلة المقدمة، ولا تستخدم أي معرفة خارجية.
    - حلل الأدلة بعناية وفقًا للتصنيفات المذكورة أعلاه.
    - أعد فقط كائن JSON يحتوي على المفتاح "predicted_label" وقيمته واحدة من التصنيفات الأربعة أعلاه.
    - لا تضف أي تفسير أو معلومات إضافية.

    الادعاء: {claim}
    الأدلة: {evidence}

    أعد الإجابة بهذا الشكل فقط (مع استبدال القيمة بالقيمة المناسبة):

    {{ "predicted_label": "REFUTED" }}
    """


def extract_label(response_text):
    try:
        data = json.loads(response_text)
        return data.get("predicted_label")
    except json.JSONDecodeError:
        return None


def normalize_label(label):
    labels_map = {
        "SUPPORTED": "supported",
        "REFUTED": "refuted",
        "NOT_ENOUGH_EVIDENCE": "Not Enough Evidence",
        "CONFLICTING_EVIDENCE": "Conflicting Evidence/Cherrypicking",
    }
    return labels_map.get(label, None)


def main():
    llm_client = ClaudeSonnet4Client()

    df = pd.read_json("data/train/evidence_test.json")

    print(f"\npredicting labels for {len(df)} claims..\n")

    if "predicted_label" not in df.columns:
        df["predicted_label"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim = row.get("claim")
        evidence = row.get("retrieved_qa_pairs")

        prompt = generate_prompt(claim, evidence)

        llm_response = llm_client.generate_response(prompt)
        label = extract_label(llm_response)
        df.at[idx, "predicted_label"] = normalize_label(label)

    save_df(df, "data/train/evi06.json")


if __name__ == "__main__":
    main()
