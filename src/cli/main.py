# src/fact_checker/cli/main.py
"""Command-line interface for the fact-checking pipeline."""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from pipeline.main import run_full_pipeline, run_partial_pipeline, PipelineConfig
from core.exceptions import FactCheckerError, ConfigurationError
from config.settings import LOG_LEVEL, LOG_FORMAT


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else LOG_LEVEL
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("fact_checker.log"),
        ],
    )


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If config file cannot be loaded.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}")


def create_sample_config(output_path: str):
    """Create a sample configuration file.

    Args:
        output_path: Path where to save the sample config.
    """
    sample_config = {
        "queries_file": "queries.txt",
        "output_dir": "pipeline_output",
        "max_claims": 100,
        "batch_size": 10,
        "collect_claims": True,
        "extract_gold_evidence": True,
        "retrieve_evidence": True,
        "predict_labels": True,
        "evaluate_results": True,
        "claim_retrieval_config": {"max_retries": 3, "initial_retry_delay": 1.0},
        "evidence_config": {"max_evidence_per_claim": 5},
        "prediction_config": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.1,
        },
        "evaluation_config": {"metrics": ["accuracy", "precision", "recall", "f1"]},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample configuration saved to: {output_path}")


def run_pipeline_command(args):
    """Run the full pipeline."""
    config_dict = {}

    # Load config file if provided
    if args.config:
        config_dict = load_config_file(args.config)

    # Override with command line arguments
    if args.queries_file:
        config_dict["queries_file"] = args.queries_file
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.max_claims:
        config_dict["max_claims"] = args.max_claims
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.run_id:
        config_dict["run_id"] = args.run_id

    try:
        print("Starting fact-checking pipeline...")
        results = run_full_pipeline(config_dict)

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)

        # Print summary
        if "evaluation" in results:
            eval_summary = results["evaluation"].get("summary", {})
            print(f"Total claims processed: {len(results.get('claims', []))}")
            print(f"Accuracy: {eval_summary.get('accuracy', 'N/A')}")
            print(f"F1 Score: {eval_summary.get('f1_score', 'N/A')}")

        print(f"Results saved to: {config_dict.get('output_dir', 'pipeline_output')}")

    except FactCheckerError as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def run_steps_command(args):
    """Run specific pipeline steps."""
    config_dict = {}

    # Load config file if provided
    if args.config:
        config_dict = load_config_file(args.config)

    # Override with command line arguments
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.run_id:
        config_dict["run_id"] = args.run_id

    try:
        print(f"Running pipeline steps: {', '.join(args.steps)}")
        results = run_partial_pipeline(args.steps, config_dict)

        print("\n" + "=" * 50)
        print("PIPELINE STEPS COMPLETED")
        print("=" * 50)

        for step in args.steps:
            if step in results:
                print(f"✓ {step}: completed")
            else:
                print(f"✗ {step}: failed or skipped")

    except FactCheckerError as e:
        print(f"Pipeline steps failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def status_command(args):
    """Show status of a pipeline run."""
    output_dir = Path(args.output_dir or "pipeline_output")
    run_dir = output_dir / args.run_id

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Check for summary file
    summary_file = run_dir / "pipeline_summary.json"
    error_file = run_dir / "error_state.json"

    if error_file.exists():
        with open(error_file, "r", encoding="utf-8") as f:
            error_info = json.load(f)

        print(f"Pipeline run {args.run_id} FAILED")
        print(f"Error: {error_info['error']}")
        print(f"Completed steps: {', '.join(error_info['completed_steps'])}")
        return

    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"Pipeline run {args.run_id} COMPLETED")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total claims: {summary['results_summary']['total_claims']}")
        print(f"Total time: {summary['timing']['total_pipeline']:.2f}s")

        if "evaluation_summary" in summary:
            eval_summary = summary["evaluation_summary"]
            print("\nEvaluation Results:")
            for metric, value in eval_summary.items():
                print(f"  {metric}: {value}")
    else:
        print(f"Pipeline run {args.run_id} is in progress or incomplete")

        # Check which files exist
        files = {
            "collected_claims.json": "Claims collection",
            "gold_evidence.json": "Gold evidence extraction",
            "retrieved_evidence.json": "Evidence retrieval",
            "predictions.json": "Label prediction",
            "evaluation_results.json": "Evaluation",
        }

        print("\nCompleted steps:")
        for filename, step_name in files.items():
            if (run_dir / filename).exists():
                print(f"  ✓ {step_name}")
            else:
                print(f"  ✗ {step_name}")


def list_runs_command(args):
    """List all pipeline runs."""
    output_dir = Path(args.output_dir or "pipeline_output")

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    runs = []
    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            summary_file = run_dir / "pipeline_summary.json"
            error_file = run_dir / "error_state.json"

            status = "UNKNOWN"
            timestamp = "N/A"
            claims_count = "N/A"

            if error_file.exists():
                status = "FAILED"
                with open(error_file, "r", encoding="utf-8") as f:
                    error_info = json.load(f)
                    timestamp = error_info["timestamp"]
            elif summary_file.exists():
                status = "COMPLETED"
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    timestamp = summary["timestamp"]
                    claims_count = summary["results_summary"]["total_claims"]
            else:
                status = "IN_PROGRESS"

            runs.append(
                {
                    "run_id": run_dir.name,
                    "status": status,
                    "timestamp": timestamp,
                    "claims": claims_count,
                }
            )

    if not runs:
        print("No pipeline runs found")
        return

    # Sort by timestamp
    runs.sort(key=lambda x: x["timestamp"], reverse=True)

    print(f"{'Run ID':<20} {'Status':<12} {'Claims':<8} {'Timestamp'}")
    print("-" * 60)
    for run in runs:
        print(
            f"{run['run_id']:<20} {run['status']:<12} {run['claims']:<8} {run['timestamp']}"
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fact-checking pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  fact-checker run --queries-file queries.txt --max-claims 100

  # Run with config file
  fact-checker run --config pipeline_config.json

  # Run specific steps
  fact-checker steps collect extract --run-id my_run

  # Check status
  fact-checker status my_run_20241201_143022

  # List all runs
  fact-checker list-runs
        """,
    )

    # Global arguments
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", type=str, help="Configuration file path")
    run_parser.add_argument("--queries-file", type=str, help="Queries file name")
    run_parser.add_argument("--output-dir", type=str, help="Output directory")
    run_parser.add_argument(
        "--max-claims", type=int, help="Maximum number of claims to process"
    )
    run_parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    run_parser.add_argument("--run-id", type=str, help="Custom run ID")
    run_parser.set_defaults(func=run_pipeline_command)

    # Steps command
    steps_parser = subparsers.add_parser
