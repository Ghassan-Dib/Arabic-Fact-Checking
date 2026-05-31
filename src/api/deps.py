from functools import cache
from typing import Annotated

from fastapi import Depends

from agent.agent import Agent
from agent.factory import create_client
from agent.types import AgentConfig, Provider
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


@cache
def _claim_retriever(api_url: str, api_key: str) -> ClaimRetriever:
    return ClaimRetriever(api_url=api_url, api_key=api_key)


@cache
def _agent(api_key: str, model: str) -> Agent:
    config = AgentConfig(
        provider=Provider.ANTHROPIC,
        model=model,
        api_key=api_key,
    )
    client = create_client(config)
    return Agent(client=client, config=config)


@cache
def _qa_generator(api_key: str, model: str) -> QAGenerator:
    return QAGenerator(agent=_agent(api_key, model))


@cache
def _label_predictor(api_key: str, model: str) -> LabelPredictor:
    return LabelPredictor(agent=_agent(api_key, model))


@cache
def _pipeline() -> FactCheckingPipeline:
    settings = get_settings()
    return FactCheckingPipeline(
        settings=settings,
        qa_generator=_qa_generator(settings.anthropic_api_key, settings.claude_model),
        label_predictor=_label_predictor(settings.anthropic_api_key, settings.claude_model),
    )


def get_claim_retriever(settings: SettingsDep) -> ClaimRetriever:
    return _claim_retriever(settings.fact_check_tools_url, settings.api_key)


def get_evidence_retriever() -> EvidenceRetriever:
    return _evidence_retriever()


def get_gold_retriever() -> GoldEvidenceRetriever:
    return _gold_retriever()


def get_qa_generator(settings: SettingsDep) -> QAGenerator:
    return _qa_generator(settings.anthropic_api_key, settings.claude_model)


def get_label_predictor(settings: SettingsDep) -> LabelPredictor:
    return _label_predictor(settings.anthropic_api_key, settings.claude_model)


def get_pipeline() -> FactCheckingPipeline:
    return _pipeline()
