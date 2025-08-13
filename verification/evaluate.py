import os
import sys
import argparse
import importlib
import pandas as pd
from verification.hungarian_meteor import AVeriTeCEvaluator
from verification.ev2r_recall import EV2REvaluator
from nltk.downloader import download as download_nltk_data

from .utils import setup_nltk_arabic

{
    "facts in predicted evidence": "1. نفت نقابة الفنانين السوريين خبر وفاة أسعد فضة وأصدرت بياناً أكدت فيه أن 'الزميل الفنان القدير أ. أسعد فضة بخير والحمد لله'. 2. انتشرت الشائعة بعدما قامت منصة 'ويكيبيديا' بتحديث صفحتها الخاصة بالفنان وأدرجت تاريخ 29 حزيران/يونيو 2025 كموعد لوفاته، ناسبةً إلى نقابة الفنانين السوريين مصدر المعلومة. 3. تداولت الشائعات أن أسعد فضة توفي عن عمر ناهز 87",
    "fact check predicted evidence": "1. نفت نقابة الفنانين السوريين خبر وفاة أسعد فضة وأصدرت بياناً أكدت فيه أن 'الزميل الفنان القدير أ. أسعد فضة بخير والحمد لله'. هذه الحقيقة مدعومة بالدليل المرجعي الذي يؤكد نفي النقابة للخبر وإصدارها بياناً بنفس المضمون. 2. انتشرت الشائعة بعدما قامت منصة 'ويكيبيديا' بتحديث صفحتها الخاصة بالفنان وأدرجت تاريخ 29 حزيران/يونيو 2025 كموعد لوفاته، ناسبةً إلى نقابة الفنانين السوريين مصدر المعلومة. الدليل المرجعي لا يذكر أي شيء عن ويكيبيديا أو تاريخ 29 يونيو 2025. معلومات غير كافية. 3. تداولت الشائعات أن أسعد فضة توفي عن عمر ناهز 87 عاماً إثر نوبة قلبية في 29 يونيو 2025. الدليل المرجعي لا يذكر تفاصيل الشائعة المزعومة. معلومات غير كافية. 4. قالت النقابة: 'نعود ونكرر ونتمنى على جميع المواقع الفنية والصفحات الصفراء التأكد قبل نشر أي خبر، وإذا لم يصدر من نقابة الفنانين فهو خبر منفي حكماً'. هذه الحقيقة مدعومة بالدليل المرجعي الذي يحتوي على نفس التصريح. 5. تأتي هذه الشائعة ضمن موجة متكررة من الأخبار الكاذبة التي طالت عدداً من الفنانين السوريين، من بينهم منى واصف، ودريد لحام، ورشيد عساف، قبل أن يتم نفيها جميعاً بشكل رسمي. الدليل المرجعي لا يذكر أي شيء عن شائعات أخرى طالت فنانين آخرين. معلومات غير كافية. 6. طمأن الفنان السوري جمهوره في تصريح لموقع 'فوشيا' مؤكداً رغبته في العودة إلى الساحة الفنية قريباً، وأشاد بأداء نقابة الفنانين السوريين تحت إدارتها الجديدة. الدليل المرجعي يذكر مقابلة مع موقع فوشيا ورغبته في العودة، لكن لا يذكر إشادته بأداء النقابة تحت إدارتها الجديدة. مدعوم جزئياً.",
    "facts count predicted evidence": 6,
    "support predicted evidence": 3,
    "facts in reference evidence": "1. نقابة الفنانين السوريين نفت إصدار أي بيان يخص وفاة الفنان. 2. نقابة الفنانين السوريين نشرت بياناً تنويهياً تؤكد فيه أن 'الزميل الفنان القدير أ. أسعد فضة بخير والحمد لله'. 3. النقابة أكدت أن 'إذا لم يصدر من نقابة الفنانين فهو خبر منفي حكماً'. 4. النقابة نفت الشائعة رسمياً عبر صفحتها على فيسبوك. 5. النقابة طالبت 'جميع المواقع الفنية والصفحات الصفراء التأكد قبل نشر أي خبر'. 6. أسعد فضة ظهر في مقابلة مع موقع 'فوشيا' في 13 مايو. 7. أسعد فضة تم تكريمه في النسخة الثانية من جائزة قادة العمل الإنساني في دبي. 8. أسعد فضة تحدث عن غيابه وخطط عودته للفن. 9. غياب أسعد فضة يعود إلى 'فترة راحة'. 10. أسعد فضة قال أنه 'مشتاق للعودة وللجمهور'. 11. أسعد فضة يرغب في أن 'تتم عودتي بشكل مُرتب'.",
    "fact check reference evidence": "1. نقابة الفنانين السوريين نفت إصدار أي بيان يخص وفاة الفنان. الدليل المتوقع يؤكد نفي النقابة للخبر. مدعوم. 2. نقابة الفنانين السوريين نشرت بياناً تنويهياً تؤكد فيه أن 'الزميل الفنان القدير أ. أسعد فضة بخير والحمد لله'. الدليل المتوقع يحتوي على نفس النص. مدعوم. 3. النقابة أكدت أن 'إذا لم يصدر من نقابة الفنانين فهو خبر منفي حكماً'. الدليل المتوقع يحتوي على نفس التصريح. مدعوم. 4. النقابة نفت الشائعة رسمياً عبر صفحتها على فيسبوك. الدليل المتوقع لا يذكر تحديداً منصة فيسبوك. معلومات غير كافية. 5. النقابة طالبت 'جميع المواقع الفنية والصفحات الصفراء التأكد قبل نشر أي خبر'. الدليل المتوقع يحتوي على نفس التصريح. مدعوم. 6. أسعد فضة ظهر في مقابلة مع موقع 'فوشيا' في 13 مايو. الدليل المتوقع يذكر تصريحاً لموقع فوشيا لكن لا يذكر التاريخ المحدد. مدعوم جزئياً. 7. أسعد فضة تم تكريمه في النسخة الثانية من جائزة قادة العمل الإنساني في دبي. الدليل المتوقع لا يذكر هذا التكريم. معلومات غير كافية. 8. أسعد فضة تحدث عن غيابه وخطط عودته للفن. الدليل المتوقع يذكر رغبته في العودة إلى الساحة الفنية. مدعوم. 9. غياب أسعد فضة يعود إلى 'فترة راحة'. الدليل المتوقع لا يذكر سبب الغياب. معلومات غير كافية. 10. أسعد فضة قال أنه 'مشتاق للعودة وللجمهور'. الدليل المتوقع لا يحتوي على هذا التصريح المحدد. معلومات غير كافية. 11. أسعد فضة يرغب في أن 'تتم عودتي بشكل مُرتب'. الدليل المتوقع لا يحتوي على هذا التصريح المحدد. معلومات غير كافية.",
    "facts count reference evidence": 11,
    "support reference evidence": 5,
}


def compute_averitec_scores(gold_df, predicted_df):
    """
    # @0.25
    {
        "public_score": {
            "Q only (Hungarian meteor)": "0.2035",
            "Q + A (Hungarian meteor)": "0.13",
            "old AVeriTeC Score (Hungarian meteor)": "0.0444",
        }
    }
    # @0.1
    {
        "public_score": {
            "Q only (Hungarian meteor)": "0.2035",
            "Q + A (Hungarian meteor)": "0.13",
            "old AVeriTeC Score (Hungarian meteor)": "0.2222",
        }
    }
    # @0.3
    {
        "public_score": {
            "Q only (Hungarian meteor)": "0.2035",
            "Q + A (Hungarian meteor)": "0.13",
            "old AVeriTeC Score (Hungarian meteor)": "0.0222",
        }
    }"""
    averitec_scorer = AVeriTeCEvaluator()
    # Q only
    Q_evidence_score, _ = averitec_scorer.evaluate_questions_only(predicted_df, gold_df)

    # Q + A
    QA_evidence_score, _ = averitec_scorer.evaluate_questions_and_answers(
        predicted_df, gold_df
    )

    # AVeriTeC Score
    averitec_scores, _ = averitec_scorer.evaluate_averitec_score(predicted_df, gold_df)

    print()
    print("✓ Q only (Hungarian meteor) score: ", Q_evidence_score)
    print("✓ Q + A (Hungarian meteor) score: ", QA_evidence_score)
    print("✓ Old AVeriTeC Score (Hungarian meteor): ", averitec_scores)
    print("_____________________________________________________________________")

    return Q_evidence_score, QA_evidence_score, averitec_scores


def compute_ev2r_score(properties, gold_df, predicted_df):
    """{
        "Q only (Hungarian meteor)": "0.2035",
        "Q + A (Hungarian meteor)": "0.13",
        "old AVeriTeC Score (Hungarian meteor)": "0.0444",
        "Q only (Ev2R recall)": "0.2119",
        "Q + A (Ev2R recall)": "0.1691",
        "new AVeriTeC score (Ev2R recall)": "0.2",  # (recall @ 0.25)
    }"""
    EV2R_scorer = EV2REvaluator(properties)
    pred_questions, ref_questions, pred_qa_pairs, ref_qa_pairs = (
        EV2R_scorer.prepare_dataset(predicted_df, gold_df)
    )

    # Q only
    print("\n⚙️ Evaluating Q only (Ev2R recall)\n")
    q_responses = EV2R_scorer.prompt_api_model(
        pred_questions, ref_questions, input_type="question"
    )

    q_evi_scores = EV2R_scorer.calculate_question_scores(q_responses)
    ev2r_q_recall, _ = EV2R_scorer.extract_recall_score(q_evi_scores)

    # Q + A
    print("\n⚙️ Evaluating Q + A (Ev2R recall)\n")
    qa_responses = EV2R_scorer.prompt_api_model(
        pred_qa_pairs, ref_qa_pairs, input_type="qa_pair"
    )

    qa_evi_scores = EV2R_scorer.calculate_prediction_scores(qa_responses)
    ev2r_qa_scores, _, ev2r_qa_recall, _ = EV2R_scorer.extract_ev2r_score(
        predicted_df, gold_df, qa_evi_scores
    )

    print()
    print("✓ Q only (Ev2R recall) score: ", ev2r_q_recall)
    print("✓ Q + A (Ev2R recall) score: ", ev2r_qa_recall)
    print("✓ New AVeriTeC score (Ev2R recall): ", ev2r_qa_scores)
    print("_____________________________________________________________________")

    return ev2r_q_recall, ev2r_qa_recall, ev2r_qa_scores


def compute(gold_df, predicted_df, properties=None):
    """Evaluation on old AVeriTeC score (Hungarian meteor) and new AVeriTeC score (EV2R recall)"""

    # AVeriTeC Score
    print("\n\n*****Evaluating AVeriTeC Scores *****\n")
    Q_evidence_score, QA_evidence_score, averitec_scores = compute_averitec_scores(
        gold_df, predicted_df
    )

    # EV2R Score
    print("\n\n*****Evaluating EV2R Scores *****\n")
    ev2r_q_recall, ev2r_qa_recall, ev2r_qa_scores = compute_ev2r_score(
        properties, gold_df.iloc[:5], predicted_df.iloc[:5]
    )

    evaluation = {
        "Q only (Hungarian meteor)": "{}".format(round(Q_evidence_score, 4)),
        "Q + A (Hungarian meteor)": "{}".format(round(QA_evidence_score, 4)),
        "old AVeriTeC Score (Hungarian meteor)": "{}".format(
            round(averitec_scores[0], 4)
        ),  # (meteor @ 0.25)
        "Q only (Ev2R recall)": "{}".format(round(ev2r_q_recall, 4)),
        "Q + A (Ev2R recall)": "{}".format(round(ev2r_qa_recall, 4)),
        "new AVeriTeC score (Ev2R recall)": "{}".format(
            round(ev2r_qa_scores[0], 4)
        ),  # (recall @ 0.5)
    }

    print("\n*****Results of Submission *****\n")
    print(evaluation)

    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Process annotation files")
    parser.add_argument(
        "--label_file",
        type=str,
        default="data/evidence/gold_evidence.json",
        help="Golden data filename.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="data/evidence/retrieved_evidence.json",
        help="Predicted data filename",
    )

    # Parse arguments
    args = parser.parse_args()

    download_nltk_data("punkt")
    download_nltk_data("punkt_tab")
    download_nltk_data("wordnet")

    print("Setting up NLTK for Arabic...")
    setup_nltk_arabic()

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    properties = importlib.import_module("properties")

    # load golden and predicted file
    gold_df = pd.read_json("data/evidence/gold_evidence_45.json")
    predicted_df = pd.read_json("data/evidence/predicted_labels_45_normalized.json")

    # compute(args.label_file, args.prediction_file)
    compute(gold_df, predicted_df, properties)


if __name__ == "__main__":
    main()
