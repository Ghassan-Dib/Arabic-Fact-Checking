from functools import cache
from typing import Annotated

from fastapi import Depends

from core.config import Settings, get_settings
from pipeline.runner import FactCheckingPipeline
from retrieval.claim_retriever import ClaimRetriever
from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.gold_retriever import GoldEvidenceRetriever
from verification.label_predictor import LabelPredictor
from verification.qa_generator import QAGenerator

SettingsDep = Annotated[Settings, Depends(get_settings)]


@cache
def _evidence_retriever() -> EvidenceRetriever:
    return EvidenceRetriever()


@cache
def _gold_retriever() -> GoldEvidenceRetriever:
    return GoldEvidenceRetriever()


def get_claim_retriever(settings: SettingsDep) -> ClaimRetriever:
    return ClaimRetriever(api_url=settings.fact_check_tools_url, api_key=settings.api_key)


def get_evidence_retriever() -> EvidenceRetriever:
    return _evidence_retriever()


def get_gold_retriever() -> GoldEvidenceRetriever:
    return _gold_retriever()


def get_qa_generator(settings: SettingsDep) -> QAGenerator:
    return QAGenerator(api_key=settings.anthropic_api_key, model=settings.claude_model)


def get_label_predictor(settings: SettingsDep) -> LabelPredictor:
    return LabelPredictor(api_key=settings.anthropic_api_key, model=settings.claude_model)


_pipeline: FactCheckingPipeline | None = None


def get_pipeline(settings: SettingsDep) -> FactCheckingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FactCheckingPipeline(settings)
    return _pipeline
