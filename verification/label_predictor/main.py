import json
import pandas as pd
from tqdm import tqdm

from src.clients import ClaudeSonnet4Client


def generate_prompt(claim, evidence):
    return f"""
    سوف تحصل على ادعاء ومجموعة من الأدلة. حدد الحكم على الادعاء بناءً على الأدلة فقط، دون الاعتماد على أي معرفة خارجية.

    التصنيفات المتاحة:
    1. SUPPORTED - الأدلة تدعم الادعاء بشكل واضح.
    2. REFUTED - الأدلة تناقض الادعاء بشكل مباشر أو تجعله غير مرجح.
    3. NOT_ENOUGH_EVIDENCE - لا توجد أدلة كافية لدعم أو نفي الادعاء. استخدم هذا التصنيف إذا لم تتمكن من إثبات أو دحض الادعاء بشكل واضح بناءً على الأدلة، حتى إن لم توجد أدلة داعمة.
    4. CONFLICTING_EVIDENCE - توجد أدلة متناقضة أو انتقائية (مثل عرض جزء من الحقيقة بشكل مضلل أو تغيّر في الموقف لم يُذكر في الادعاء). يُستخدم أيضًا في الحالات التي تكون فيها الأدلة صحيحة جزئيًا لكنها تُستخدم بشكل يضلل القارئ.

    تعليمات:
    - اعتمد فقط على الأدلة المقدمة، ولا تستخدم أي معرفة خارجية.
    - حلل الأدلة بعناية وفقًا للتصنيفات المذكورة أعلاه.
    - أعد فقط كائن JSON يحتوي على المفتاح "predicted_label" وقيمته واحدة من التصنيفات الأربعة أعلاه.
    - لا تضف أي تفسير أو معلومات إضافية.

    الادعاء: {claim}
    الأدلة: {evidence}

    أعد الإجابة بهذا الشكل فقط (مع استبدال القيمة بالقيمة المناسبة):

    {{ "predicted_label": "SUPPORTED" }}
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

    # predict label, given a claim and a set of evidence
    df = pd.read_json("data/evidence/retrieved_evidence_45.json", encoding="utf-8")

    for i in tqdm(range(len(df))):
        claim = df["claim"].iloc[i]
        evidence = df["questions"].iloc[i]

        prompt = generate_prompt(claim, evidence)

        llm_response = llm_client.generate_response(prompt)
        label = extract_label(llm_response)
        df.at[i, "predicted_label"] = normalize_label(label)

    # save the updated DataFrame to a new JSON file
    df.to_json(
        "data/evidence/predicted_labels.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    main()
