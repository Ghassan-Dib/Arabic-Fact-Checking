import logging
from typing import Any

from core.config import Settings
from models.pipeline import JobStatus, PipelineConfig
from pipeline import job_store
from queries import QUERIES
from retrieval.claim_retriever import ClaimRetriever
from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.gold_retriever import GoldEvidenceRetriever
from verification.label_predictor import LabelPredictor
from verification.qa_generator import QAGenerator

logger = logging.getLogger(__name__)


class FactCheckingPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.claim_retriever = ClaimRetriever(
            api_url=settings.fact_check_tools_url, api_key=settings.api_key
        )
        self.evidence_retriever = EvidenceRetriever()
        self.gold_retriever = GoldEvidenceRetriever()
        self.qa_generator = QAGenerator(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )
        self.label_predictor = LabelPredictor(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )

    def run(self, job_id: str, config: PipelineConfig) -> None:
        """Entry point for a BackgroundTask. Updates job_store throughout."""
        job_store.update_job(job_id, status=JobStatus.RUNNING)

        try:
            results: dict[str, Any] = {}

            if config.collect_claims:
                job_store.update_job(job_id, current_step="collect_claims")
                claims = []
                for query in QUERIES:
                    claims.extend(
                        self.claim_retriever.retrieve_by_query(
                            query,
                            languageCode=self.settings.default_language,
                        )
                    )
                if config.max_claims:
                    claims = claims[: config.max_claims]
                results["claims_count"] = len(claims)
                logger.info("Collected %d claims", len(claims))

            job_store.complete_job(job_id, results)

        except Exception as exc:
            logger.exception("Pipeline job %s failed", job_id)
            job_store.fail_job(job_id, str(exc))
