import logging
from pathlib import Path

import pandas as pd

from core.exceptions import EvaluationError
from evaluation.hungarian_meteor import AVeriTeCEvaluator
from models.evaluation import EvaluationResult

logger = logging.getLogger(__name__)


def evaluate_from_files(
    predicted_path: Path,
    gold_path: Path,
) -> EvaluationResult:
    try:
        predicted_df = pd.read_json(predicted_path)
        gold_df = pd.read_json(gold_path)
    except (ValueError, FileNotFoundError, Exception) as exc:
        raise EvaluationError("Failed to load evaluation files") from exc

    scorer = AVeriTeCEvaluator()

    q_score, _ = scorer.evaluate_questions_only(predicted_df, gold_df)
    qa_score, _ = scorer.evaluate_questions_and_answers(predicted_df, gold_df)
    averitec_scores, _ = scorer.evaluate_averitec_score(predicted_df, gold_df)

    logger.info("Q-only METEOR: %.4f", q_score)
    logger.info("Q+A METEOR:    %.4f", qa_score)
    logger.info("AVeriTeC score: %s", averitec_scores)

    a_score = (
        float(averitec_scores[0]) if hasattr(averitec_scores, "__len__") else float(averitec_scores)
    )
    return EvaluationResult(
        q_only_meteor=float(q_score),
        qa_meteor=float(qa_score),
        averitec_score=a_score,
    )
