import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import get_data_file_path, validate_config
from core.exceptions import FactCheckerError, ConfigurationError
from retrieval.claim_retriever import ClaimRetriever
from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.gold_retriever import GoldEvidenceRetriever
from verification.label_predictor import LabelPredictor
from verification.evaluator import Evaluator
from utils.data_processing import ClaimDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the fact-checking pipeline."""

    # Input/Output settings
    queries_file: str = "queries.txt"
    output_dir: str = "pipeline_output"
    run_id: Optional[str] = None

    # Processing settings
    max_claims: Optional[int] = None
    batch_size: int = 10

    # Pipeline step controls
    collect_claims: bool = True
    extract_gold_evidence: bool = True
    retrieve_evidence: bool = True
    predict_labels: bool = True
    evaluate_results: bool = True

    # Step-specific configs
    claim_retrieval_config: Dict[str, Any] = field(default_factory=dict)
    evidence_config: Dict[str, Any] = field(default_factory=dict)
    prediction_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize run ID if not provided."""
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class FactCheckingPipeline:
    """Main fact-checking pipeline that orchestrates all steps."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.results = {}
        self.timing = {}

        # Initialize output directory
        self.output_dir = Path(get_data_file_path(config.output_dir, "processed"))
        self.run_dir = self.output_dir / config.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_processor = ClaimDataProcessor()
        self.claim_retriever = ClaimRetriever(**config.claim_retrieval_config)
        self.evidence_retriever = EvidenceRetriever(**config.evidence_config)
        self.gold_retriever = GoldEvidenceRetriever(**config.evidence_config)
        self.label_predictor = LabelPredictor(**config.prediction_config)
        self.evaluator = Evaluator(**config.evaluation_config)

        logger.info(f"Initialized pipeline with run ID: {config.run_id}")
        logger.info(f"Output directory: {self.run_dir}")

    def run(self) -> Dict[str, Any]:
        """Run the complete fact-checking pipeline.

        Returns:
            Dictionary containing pipeline results and metrics.

        Raises:
            FactCheckerError: If any pipeline step fails.
        """
        logger.info("Starting fact-checking pipeline")
        pipeline_start = time.time()

        try:
            # Validate configuration
            validate_config()

            # Step 1: Collect claims
            if self.config.collect_claims:
                claims = self._collect_claims()
                self.results["claims"] = claims
            else:
                claims = self._load_existing_claims()
                self.results["claims"] = claims

            # Step 2: Extract gold evidence
            if self.config.extract_gold_evidence:
                gold_evidence = self._extract_gold_evidence(claims)
                self.results["gold_evidence"] = gold_evidence
            else:
                gold_evidence = self._load_existing_gold_evidence()
                self.results["gold_evidence"] = gold_evidence

            # Step 3: Retrieve evidence
            if self.config.retrieve_evidence:
                retrieved_evidence = self._retrieve_evidence(claims)
                self.results["retrieved_evidence"] = retrieved_evidence
            else:
                retrieved_evidence = self._load_existing_retrieved_evidence()
                self.results["retrieved_evidence"] = retrieved_evidence

            # Step 4: Predict labels
            if self.config.predict_labels:
                predictions = self._predict_labels(claims, retrieved_evidence)
                self.results["predictions"] = predictions
            else:
                predictions = self._load_existing_predictions()
                self.results["predictions"] = predictions

            # Step 5: Evaluate
            if self.config.evaluate_results:
                evaluation_results = self._evaluate(
                    claims, gold_evidence, retrieved_evidence, predictions
                )
                self.results["evaluation"] = evaluation_results

            # Save final results
            self._save_pipeline_results()

            pipeline_time = time.time() - pipeline_start
            self.timing["total_pipeline"] = pipeline_time

            logger.info(
                f"Pipeline completed successfully in {pipeline_time:.2f} seconds"
            )
            return self.results

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self._save_error_state(str(e))
            raise FactCheckerError(error_msg) from e

    def _collect_claims(self) -> List[Dict[str, Any]]:
        """Step 1: Collect claims from various sources."""
        logger.info("Step 1: Collecting claims")
        step_start = time.time()

        try:
            # Load queries
            queries = self.data_processor.load_queries(self.config.queries_file)
            logger.info(f"Loaded {len(queries)} queries")

            # Collect claims for each query
            all_claims = []
            for i, query in enumerate(queries):
                logger.info(f"Processing query {i + 1}/{len(queries)}: {query}")

                # Retrieve claims for this query
                query_claims = self.claim_retriever.retrieve_claims_by_query(query)
                all_claims.extend(query_claims)

                if self.config.max_claims and len(all_claims) >= self.config.max_claims:
                    all_claims = all_claims[: self.config.max_claims]
                    logger.info(f"Reached max claims limit: {self.config.max_claims}")
                    break

            # Remove duplicates and process
            unique_claims = self.data_processor.remove_duplicates(all_claims)

            # Save collected claims
            output_file = self.run_dir / "collected_claims.json"
            self.data_processor.save_claims_to_json(unique_claims, str(output_file))

            step_time = time.time() - step_start
            self.timing["collect_claims"] = step_time

            logger.info(
                f"Collected {len(unique_claims)} unique claims in {step_time:.2f}s"
            )
            return unique_claims

        except Exception as e:
            raise FactCheckerError(f"Failed to collect claims: {str(e)}") from e

    def _extract_gold_evidence(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 2: Extract gold evidence for claims."""
        logger.info("Step 2: Extracting gold evidence")
        step_start = time.time()

        try:
            gold_evidence = {}

            # Process claims in batches
            for i in range(0, len(claims), self.config.batch_size):
                batch = claims[i : i + self.config.batch_size]
                logger.info(f"Processing batch {i // self.config.batch_size + 1}")

                batch_evidence = self.gold_retriever.extract_evidence_batch(batch)
                gold_evidence.update(batch_evidence)

            # Save gold evidence
            output_file = self.run_dir / "gold_evidence.json"
            with open(output_file, "w", encoding="utf-8") as f:
                import json

                json.dump(gold_evidence, f, ensure_ascii=False, indent=2)

            step_time = time.time() - step_start
            self.timing["extract_gold_evidence"] = step_time

            logger.info(
                f"Extracted gold evidence for {len(gold_evidence)} claims in {step_time:.2f}s"
            )
            return gold_evidence

        except Exception as e:
            raise FactCheckerError(f"Failed to extract gold evidence: {str(e)}") from e

    def _retrieve_evidence(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Retrieve evidence for claims."""
        logger.info("Step 3: Retrieving evidence")
        step_start = time.time()

        try:
            retrieved_evidence = {}

            # Process claims in batches
            for i in range(0, len(claims), self.config.batch_size):
                batch = claims[i : i + self.config.batch_size]
                logger.info(f"Processing batch {i // self.config.batch_size + 1}")

                batch_evidence = self.evidence_retriever.retrieve_evidence_batch(batch)
                retrieved_evidence.update(batch_evidence)

            # Save retrieved evidence
            output_file = self.run_dir / "retrieved_evidence.json"
            with open(output_file, "w", encoding="utf-8") as f:
                import json

                json.dump(retrieved_evidence, f, ensure_ascii=False, indent=2)

            step_time = time.time() - step_start
            self.timing["retrieve_evidence"] = step_time

            logger.info(
                f"Retrieved evidence for {len(retrieved_evidence)} claims in {step_time:.2f}s"
            )
            return retrieved_evidence

        except Exception as e:
            raise FactCheckerError(f"Failed to retrieve evidence: {str(e)}") from e

    def _predict_labels(
        self, claims: List[Dict[str, Any]], evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 4: Predict labels for claims."""
        logger.info("Step 4: Predicting labels")
        step_start = time.time()

        try:
            predictions = {}

            # Process claims in batches
            for i in range(0, len(claims), self.config.batch_size):
                batch = claims[i : i + self.config.batch_size]
                logger.info(f"Processing batch {i // self.config.batch_size + 1}")

                batch_predictions = self.label_predictor.predict_batch(batch, evidence)
                predictions.update(batch_predictions)

            # Save predictions
            output_file = self.run_dir / "predictions.json"
            with open(output_file, "w", encoding="utf-8") as f:
                import json

                json.dump(predictions, f, ensure_ascii=False, indent=2)

            step_time = time.time() - step_start
            self.timing["predict_labels"] = step_time

            logger.info(
                f"Generated predictions for {len(predictions)} claims in {step_time:.2f}s"
            )
            return predictions

        except Exception as e:
            raise FactCheckerError(f"Failed to predict labels: {str(e)}") from e

    def _evaluate(
        self,
        claims: List[Dict[str, Any]],
        gold_evidence: Dict[str, Any],
        retrieved_evidence: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Step 5: Evaluate pipeline results."""
        logger.info("Step 5: Evaluating results")
        step_start = time.time()

        try:
            evaluation_results = self.evaluator.evaluate_pipeline(
                claims=claims,
                gold_evidence=gold_evidence,
                retrieved_evidence=retrieved_evidence,
                predictions=predictions,
            )

            # Save evaluation results
            output_file = self.run_dir / "evaluation_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                import json

                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

            step_time = time.time() - step_start
            self.timing["evaluate"] = step_time

            logger.info(f"Evaluation completed in {step_time:.2f}s")
            return evaluation_results

        except Exception as e:
            raise FactCheckerError(f"Failed to evaluate results: {str(e)}") from e

    def _load_existing_claims(self) -> List[Dict[str, Any]]:
        """Load existing claims from previous run."""
        claims_file = self.run_dir / "collected_claims.json"
        if not claims_file.exists():
            raise ConfigurationError(f"Claims file not found: {claims_file}")

        import json

        with open(claims_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing_gold_evidence(self) -> Dict[str, Any]:
        """Load existing gold evidence."""
        evidence_file = self.run_dir / "gold_evidence.json"
        if not evidence_file.exists():
            raise ConfigurationError(f"Gold evidence file not found: {evidence_file}")

        import json

        with open(evidence_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing_retrieved_evidence(self) -> Dict[str, Any]:
        """Load existing retrieved evidence."""
        evidence_file = self.run_dir / "retrieved_evidence.json"
        if not evidence_file.exists():
            raise ConfigurationError(
                f"Retrieved evidence file not found: {evidence_file}"
            )

        import json

        with open(evidence_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing_predictions(self) -> Dict[str, Any]:
        """Load existing predictions."""
        predictions_file = self.run_dir / "predictions.json"
        if not predictions_file.exists():
            raise ConfigurationError(f"Predictions file not found: {predictions_file}")

        import json

        with open(predictions_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_pipeline_results(self):
        """Save complete pipeline results and metadata."""
        # Save summary
        summary = {
            "run_id": self.config.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "timing": self.timing,
            "results_summary": {
                "total_claims": len(self.results.get("claims", [])),
                "gold_evidence_count": len(self.results.get("gold_evidence", {})),
                "retrieved_evidence_count": len(
                    self.results.get("retrieved_evidence", {})
                ),
                "predictions_count": len(self.results.get("predictions", {})),
            },
        }

        if "evaluation" in self.results:
            summary["evaluation_summary"] = self.results["evaluation"].get(
                "summary", {}
            )

        summary_file = self.run_dir / "pipeline_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            import json

            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Pipeline summary saved to: {summary_file}")

    def _save_error_state(self, error_message: str):
        """Save error state for debugging."""
        error_info = {
            "run_id": self.config.run_id,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "completed_steps": list(self.results.keys()),
            "timing": self.timing,
        }

        error_file = self.run_dir / "error_state.json"
        with open(error_file, "w", encoding="utf-8") as f:
            import json

            json.dump(error_info, f, ensure_ascii=False, indent=2)


def create_pipeline(
    config_dict: Optional[Dict[str, Any]] = None,
) -> FactCheckingPipeline:
    """Create a fact-checking pipeline with configuration.

    Args:
        config_dict: Configuration dictionary. Uses defaults if None.

    Returns:
        Configured pipeline instance.
    """
    if config_dict:
        config = PipelineConfig(**config_dict)
    else:
        config = PipelineConfig()

    return FactCheckingPipeline(config)


def run_full_pipeline(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the complete fact-checking pipeline.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        Pipeline results.
    """
    pipeline = create_pipeline(config_dict)
    return pipeline.run()


def run_partial_pipeline(
    steps: List[str], config_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run specific steps of the pipeline.

    Args:
        steps: List of steps to run ('collect', 'extract', 'retrieve', 'predict', 'evaluate').
        config_dict: Configuration dictionary.

    Returns:
        Pipeline results.
    """
    if not config_dict:
        config_dict = {}

    # Disable all steps first
    config_dict.update(
        {
            "collect_claims": False,
            "extract_gold_evidence": False,
            "retrieve_evidence": False,
            "predict_labels": False,
            "evaluate_results": False,
        }
    )

    # Enable requested steps
    step_mapping = {
        "collect": "collect_claims",
        "extract": "extract_gold_evidence",
        "retrieve": "retrieve_evidence",
        "predict": "predict_labels",
        "evaluate": "evaluate_results",
    }

    for step in steps:
        if step in step_mapping:
            config_dict[step_mapping[step]] = True
        else:
            raise ValueError(f"Unknown step: {step}")

    pipeline = create_pipeline(config_dict)
    return pipeline.run()
