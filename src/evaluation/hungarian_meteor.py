import logging

import numpy as np
import scipy.optimize
import tqdm
from pandas import DataFrame

from evaluation.utils import compute_all_pairwise_scores, pairwise_meteor

logger = logging.getLogger(__name__)


class AVeriTeCEvaluator:
    max_questions: int = 10
    averitec_reporting_levels: list[float] = [0.3]

    def __init__(self) -> None:
        self.pairwise_metric = pairwise_meteor

    def _assignment_score(self, src_strings: list[str], tgt_strings: list[str]) -> float:
        if not src_strings or not tgt_strings:
            return 0.0
        pairwise = compute_all_pairwise_scores(src_strings, tgt_strings, self.pairwise_metric)
        assignment = scipy.optimize.linear_sum_assignment(pairwise, maximize=True)
        utility = float(pairwise[assignment[0], assignment[1]].sum())
        return utility / float(len(tgt_strings))

    def evaluate_questions_only(
        self, srcs: DataFrame, tgts: DataFrame
    ) -> tuple[float, list[float]]:
        all_utils: list[float] = []
        for i in tqdm.tqdm(range(len(srcs))):
            src_qs = [
                qa["question"]
                for qa in (srcs.iloc[i]["retrieved_qa_pairs"] or [])
                if qa and qa.get("question")
            ][: self.max_questions]
            tgt_qs = [
                qa["question"]
                for qa in (tgts.iloc[i]["qa_pairs"] or [])
                if qa and qa.get("question")
            ]
            all_utils.append(self._assignment_score(src_qs, tgt_qs))
        return float(np.mean(all_utils)), all_utils

    def evaluate_questions_and_answers(
        self, srcs: DataFrame, tgts: DataFrame
    ) -> tuple[float, list[float]]:
        all_utils: list[float] = []
        for i in tqdm.tqdm(range(len(srcs))):
            src_strs = [
                qa["question"] + " " + qa["answer"]
                for qa in (srcs.iloc[i]["retrieved_qa_pairs"] or [])
                if qa and qa.get("question") and qa.get("answer")
            ][: self.max_questions]
            tgt_strs = [
                qa["question"] + " " + qa["answer"]
                for qa in (tgts.iloc[i]["qa_pairs"] or [])
                if qa and qa.get("question") and qa.get("answer")
            ]
            all_utils.append(self._assignment_score(src_strs, tgt_strs))
        return float(np.mean(all_utils)), all_utils

    def evaluate_averitec_score(
        self, srcs: DataFrame, tgts: DataFrame
    ) -> tuple[np.ndarray, list[list[float]]]:
        scores: list[list[float]] = []
        for i in tqdm.tqdm(range(len(srcs))):
            src_strs = [
                qa["question"] + " " + qa["answer"]
                for qa in (srcs.iloc[i]["retrieved_qa_pairs"] or [])
                if qa and qa.get("question") and qa.get("answer")
            ][: self.max_questions]
            tgt_strs = [
                qa["question"] + " " + qa["answer"]
                for qa in (tgts.iloc[i]["qa_pairs"] or [])
                if qa and qa.get("question") and qa.get("answer")
            ]
            evidence_score = self._assignment_score(src_strs, tgt_strs)
            row = [0.0] * len(self.averitec_reporting_levels)
            for j, level in enumerate(self.averitec_reporting_levels):
                if evidence_score > level:
                    row[j] = (
                        1.0
                        if srcs.iloc[i]["predicted_label"] == tgts.iloc[i]["normalized_label"]
                        else 0.0
                    )
            scores.append(row)
        return np.mean(np.array(scores), axis=0), scores
