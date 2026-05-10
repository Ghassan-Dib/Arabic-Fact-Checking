from pydantic import BaseModel


class EvaluationResult(BaseModel):
    q_only_meteor: float
    qa_meteor: float
    averitec_score: float
    ev2r_q_recall: float | None = None
    ev2r_qa_score: float | None = None
