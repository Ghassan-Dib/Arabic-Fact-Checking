import numpy as np
import scipy.optimize
import tqdm

from .utils import pairwise_meteor, compute_all_pairwise_scores


class AVeriTeCEvaluator:
    pairwise_metric = None
    max_questions = 10
    metric = None
    averitec_reporting_levels = [0.3]

    def __init__(self, metric="meteor"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        for i in tqdm.tqdm(range(len(srcs))):
            score = self.compute_pairwise_evidence_score(srcs.iloc[i], tgts.iloc[i])

            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            for j, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[j] = (
                        1.0
                        if srcs.iloc[i]["predicted_label"]
                        == tgts.iloc[i]["normalized_label"]
                        else 0.0
                    )

            scores.append(this_example_scores)

        return np.mean(np.array(scores), axis=0), scores

    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []

        for i in tqdm.tqdm(range(len(srcs))):
            src_questions, tgt_questions = [], []
            # prediction
            pred_evidence = srcs.iloc[i]["retrieved_qa_pairs"]

            if pred_evidence:
                for pred_qa in pred_evidence:
                    if pred_qa != "":
                        pred_question = pred_qa["question"]
                        if pred_question not in src_questions:
                            src_questions.append(pred_question)

            src_questions = src_questions[: self.max_questions]

            # gold
            gold_evidence = tgts.iloc[i]["qa_pairs"]

            for gold_qa in gold_evidence:
                if gold_qa != "":
                    gold_question = gold_qa["question"]
                    if gold_question not in tgt_questions:
                        tgt_questions.append(gold_question)

            #
            pairwise_scores = compute_all_pairwise_scores(
                src_questions, tgt_questions, self.pairwise_metric
            )
            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils), all_utils

    def compute_pairwise_evidence_score(self, src, tgt):
        """Different key is used for reference_data and prediction.
        For the prediction, the format is
        {"evidence": [
            {
                "question": "What does the increased federal medical assistance percentage mean for you?",
                "answer": "Appendix A: Applicability of the Increased Federal Medical Assistance Percentage ",
                "url": "https://www.medicaid.gov/federal-policy-guidance/downloads/smd21003.pdf"
            }],
        "pred_label": "Supported"}
        And for the data with fold label:
        {"questions": [
            {
                "question": "Where was the claim first published",
                "answers": [
                    {
                        "answer": "It was first published on Sccopertino",
                        "answer_type": "Abstractive",
                        "source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
                        "source_medium": "Web text",
                        "cached_source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"
                    }
                ]
            }]
        "label": "Refuted"}
        """
        # prediction
        src_strings = []
        pred_evidence = src["retrieved_qa_pairs"]

        if pred_evidence:
            for qa_pair in pred_evidence:
                if qa_pair != "":
                    pred_question, pred_answer = qa_pair["question"], qa_pair["answer"]
                    pred_qa_pairs = pred_question + " " + pred_answer
                    src_strings.append(pred_qa_pairs)

        src_strings = src_strings[: self.max_questions]

        # gold
        tgt_strings = []
        gold_evidence = tgt["qa_pairs"]

        for qa_pair in gold_evidence:
            if qa_pair != "":
                gold_question, gold_answer = qa_pair["question"], qa_pair["answer"]
                gold_qa_pairs = gold_question + " " + gold_answer
                tgt_strings.append(gold_qa_pairs)

        #
        pairwise_scores = compute_all_pairwise_scores(
            src_strings, tgt_strings, self.pairwise_metric
        )
        assignment = scipy.optimize.linear_sum_assignment(
            pairwise_scores, maximize=True
        )
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

        # Reweight to account for unmatched target questions
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility

    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []

        for i in tqdm.tqdm(range(len(srcs))):
            # pred
            src_strings = []
            pred_evidence = srcs.iloc[i]["retrieved_qa_pairs"]

            if pred_evidence:
                for qa_pair in pred_evidence:
                    if qa_pair != "":
                        pred_question, pred_answer = (
                            qa_pair["question"],
                            qa_pair["answer"],
                        )
                        pred_qa_pairs = pred_question + " " + pred_answer
                        src_strings.append(pred_qa_pairs)

            src_strings = src_strings[: self.max_questions]

            # gold
            tgt_strings = []
            gold_evidence = tgts.iloc[i]["qa_pairs"]

            for qa_pair in gold_evidence:
                if qa_pair != "":
                    gold_question, gold_answer = qa_pair["question"], qa_pair["answer"]
                    gold_qa_pair = gold_question + " " + gold_answer
                    tgt_strings.append(gold_qa_pair)

            pairwise_scores = compute_all_pairwise_scores(
                src_strings, tgt_strings, self.pairwise_metric
            )
            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils), all_utils

    def extract_full_comparison_strings(self, example, is_target=True):
        example_strings = []

        if is_target:
            if "questions" in example:
                for evidence in example["questions"]:
                    # If the answers is not a list, make them a list:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]

                    for answer in evidence["answers"]:
                        example_strings.append(
                            evidence["question"] + " " + answer["answer"]
                        )
                        if (
                            "answer_type" in answer
                            and answer["answer_type"] == "Boolean"
                            and "boolean_explanation" in answer
                        ):
                            example_strings[-1] += ". " + answer["boolean_explanation"]
                    if len(evidence["answers"]) == 0:
                        example_strings.append(
                            evidence["question"] + " No answer could be found."
                        )
        else:
            if "evidence" in example:
                for evidence in example["evidence"]:
                    example_strings.append(
                        evidence["question"] + " " + evidence["answer"]
                    )

        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)
        return example_strings
