import os
import sys
import importlib
import pandas as pd
from verification.evaluation.hungarian_meteor import AVeriTeCEvaluator
from verification.evaluation.ev2r_recall import EV2REvaluator


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
    EV2R_scorer = EV2REvaluator(properties)
    pred_questions, ref_questions, pred_qa_pairs, ref_qa_pairs = (
        EV2R_scorer.prepare_dataset(predicted_df, gold_df)
    )

    # Q only
    print("\nEvaluating Q only (Ev2R recall)\n")
    q_responses = EV2R_scorer.prompt_api_model(
        pred_questions, ref_questions, input_type="question"
    )

    q_evi_scores = EV2R_scorer.calculate_question_scores(q_responses)
    ev2r_q_recall, _ = EV2R_scorer.extract_recall_score(q_evi_scores)

    # Q + A
    print("\nEvaluating Q + A (Ev2R recall)\n")
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
    """Evaluation on old AVeriTeC score (Hungarian meteor) and new AVeriTeC score (EV2R recall)
    {
        "Q only (Hungarian meteor)": "0.2035",
        "Q + A (Hungarian meteor)": "0.13",
        "old AVeriTeC Score (Hungarian meteor)": "0.0444",
        "Q only (Ev2R recall)": "0.2119",
        "Q + A (Ev2R recall)": "0.1691",
        "new AVeriTeC score (Ev2R recall)": "0.2",  # (recall @ 0.25)
    }
    {
        "Q only (Hungarian meteor)": "0.2115",
        "Q + A (Hungarian meteor)": "0.1448",
        "old AVeriTeC Score (Hungarian meteor)": "0.0526",
        "Q only (Ev2R recall)": "0.2669",
        "Q + A (Ev2R recall)": "0.0355",
        "new AVeriTeC score (Ev2R recall)": "0.0226",
    }
    """

    # AVeriTeC Score
    print("\n\n*****Evaluating AVeriTeC Scores *****\n")
    Q_evidence_score, QA_evidence_score, averitec_scores = compute_averitec_scores(
        gold_df, predicted_df
    )

    # EV2R Score
    print("\n\n*****Evaluating EV2R Scores *****\n")
    ev2r_q_recall, ev2r_qa_recall, ev2r_qa_scores = compute_ev2r_score(
        properties, gold_df, predicted_df
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
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    properties = importlib.import_module("properties")

    # load golden and predicted file
    gold_df = pd.read_json("data/train/claims_test2.json")
    predicted_df = pd.read_json("data/train/evidence_test2.json")

    compute(gold_df, predicted_df, properties)


if __name__ == "__main__":
    main()
