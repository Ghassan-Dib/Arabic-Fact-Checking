# src/fact_checker/utils/data_processing.py
"""Data processing utilities for claims and evidence."""

import json
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from src.config.settings import get_data_file_path
from src.core.exceptions import DataProcessingError, ValidationError

logger = logging.getLogger(__name__)


class ClaimDataProcessor:
    """Handles processing and manipulation of claim data."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data processor.

        Args:
            data_dir: Base directory for data files. Uses config default if None.
        """
        self.data_dir = data_dir

    def load_queries(self, filename: str = "queries.txt") -> List[str]:
        """Load queries from a text file.

        Args:
            filename: Name of the queries file.

        Returns:
            List of query strings.

        Raises:
            DataProcessingError: If file cannot be loaded.
        """
        try:
            file_path = self._get_file_path(filename, "raw")

            with open(file_path, "r", encoding="utf-8") as f:
                queries = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            logger.info(f"Loaded {len(queries)} queries from '{file_path}'")
            return queries

        except FileNotFoundError:
            error_msg = f"Queries file '{filename}' not found"
            logger.error(error_msg)
            raise DataProcessingError(
                error_msg, file_path=str(file_path), data_format="txt"
            )
        except Exception as e:
            error_msg = f"Failed to load queries from '{filename}'"
            logger.error(f"{error_msg}: {e}")
            raise DataProcessingError(error_msg, file_path=str(file_path)) from e

    def save_claims_to_json(
        self,
        data: List[Dict[str, Any]],
        output_filename: str,
        data_type: str = "processed",
    ) -> None:
        """Save claims data to JSON file.

        Args:
            data: List of claim dictionaries.
            output_filename: Output file name.
            data_type: Type of data directory ('raw', 'processed', 'evidence').

        Raises:
            DataProcessingError: If file cannot be saved.
        """
        try:
            file_path = self._get_file_path(output_filename, data_type)

            with open(file_path, "w", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False, indent=4)

            logger.info(f"Successfully saved {len(data)} claims to '{file_path}'")

        except Exception as e:
            error_msg = f"Failed to save claims to '{output_filename}'"
            logger.error(f"{error_msg}: {e}")
            raise DataProcessingError(
                error_msg, file_path=str(file_path), data_format="json"
            ) from e

    def get_label_distribution(
        self, claims: List[Dict[str, Any]], field_name: str, output_filename: str
    ) -> Dict[str, int]:
        """Analyze and save label distribution from claims.

        Args:
            claims: List of claim dictionaries.
            field_name: Field name to analyze (e.g., 'textualRating').
            output_filename: Output file for distribution data.

        Returns:
            Dictionary with label counts.

        Raises:
            DataProcessingError: If analysis fails.
        """
        try:
            ratings = []

            for claim in claims:
                for review in claim.get("claimReview", []):
                    rating = review.get(field_name)
                    if rating:
                        ratings.append(rating)

            ratings_count = dict(Counter(ratings))

            # Save distribution to file
            output_path = self._get_file_path(output_filename, "processed")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(ratings_count, f, ensure_ascii=False, indent=2)

            logger.info(f"Label distribution saved to '{output_path}'")
            logger.info(
                f"Found {len(ratings_count)} unique labels in {len(ratings)} reviews"
            )

            return ratings_count

        except Exception as e:
            error_msg = f"Failed to analyze label distribution for field '{field_name}'"
            logger.error(f"{error_msg}: {e}")
            raise DataProcessingError(error_msg) from e

    def filter_claims_by_source(
        self, claims: List[Dict[str, Any]], whitelisted_sites: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter claims to only include those from whitelisted sites.

        Args:
            claims: List of claim dictionaries.
            whitelisted_sites: List of allowed site names (case-insensitive).

        Returns:
            Filtered list of claims.
        """
        if not whitelisted_sites:
            logger.warning("No whitelisted sites provided, returning all claims")
            return claims

        # Convert to lowercase for case-insensitive comparison
        whitelisted_lower = [site.lower() for site in whitelisted_sites]

        filtered_claims = []
        for claim in claims:
            claim_review = claim.get("claimReview", [])
            if any(
                review.get("publisher", {}).get("site", "").lower() in whitelisted_lower
                for review in claim_review
            ):
                filtered_claims.append(claim)

        logger.info(
            f"Filtered claims from {len(claims)} to {len(filtered_claims)} "
            f"using {len(whitelisted_sites)} whitelisted sites"
        )

        return filtered_claims

    def remove_duplicates(
        self, claims: List[Dict[str, Any]], key_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """Remove duplicate claims based on a key field.

        Args:
            claims: List of claim dictionaries.
            key_field: Field to use for duplicate detection.

        Returns:
            List of unique claims.
        """
        seen_values: Set[str] = set()
        unique_claims = []

        for claim in claims:
            key_value = claim.get(key_field)
            if key_value and key_value not in seen_values:
                unique_claims.append(claim)
                seen_values.add(key_value)

        duplicates_removed = len(claims) - len(unique_claims)
        logger.info(
            f"Removed {duplicates_removed} duplicate claims based on '{key_field}'"
        )
        logger.info(f"Returned {len(unique_claims)} unique claims")

        return unique_claims

    def normalize_claims(
        self,
        claims: List[Dict[str, Any]],
        labels_map: Dict[str, Any],
        translate_labels: bool = True,
        remove_noisy_labels: bool = True,
        output_filename: str = "normalized_claims.json",
    ) -> List[Dict[str, Any]]:
        """Normalize claim labels using a mapping dictionary.

        Args:
            claims: List of claim dictionaries.
            labels_map: Mapping from original labels to normalized labels.
            translate_labels: Whether to translate labels using the map.
            remove_noisy_labels: Whether to remove claims with unmapped labels.
            output_filename: Output file for normalized claims.

        Returns:
            List of normalized claims.

        Raises:
            ValidationError: If labels_map is invalid.
        """
        if not isinstance(labels_map, dict):
            raise ValidationError(
                "labels_map must be a dictionary",
                field_name="labels_map",
                invalid_value=type(labels_map),
                expected_type="dict",
            )

        processed_claims = claims.copy()

        if translate_labels:
            processed_claims = self._translate_labels(processed_claims, labels_map)

        if remove_noisy_labels:
            processed_claims = self._remove_noisy_labels(processed_claims, labels_map)

        # Save normalized claims
        self.save_claims_to_json(processed_claims, output_filename, "processed")

        logger.info(f"Normalized labels in {len(processed_claims)} claims")
        return processed_claims

    def save_to_csv(self, data: List[Dict[str, Any]], csv_filename: str) -> None:
        """Save claims data to CSV format.

        Args:
            data: List of claim dictionaries.
            csv_filename: Output CSV filename.

        Raises:
            DataProcessingError: If CSV creation fails.
        """
        headers = [
            "ClaimID",
            "claim",
            "description",
            "source",
            "date",
            "source_label",
            "normalized_label",
            "source_url",
            "claimant",
        ]

        try:
            csv_path = self._get_file_path(csv_filename, "processed")
            rows = self._prepare_csv_rows(data)

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"CSV file created successfully at: {csv_path}")
            logger.info(f"Exported {len(rows)} claim records")

        except Exception as e:
            error_msg = f"Failed to create CSV file '{csv_filename}'"
            logger.error(f"{error_msg}: {e}")
            raise DataProcessingError(
                error_msg, file_path=str(csv_path), data_format="csv"
            ) from e

    def _translate_labels(
        self, claims: List[Dict[str, Any]], labels_map: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Translate labels using the provided mapping."""
        # Create reverse mapping from all translations to normalized labels
        translation_to_label = {}
        for normalized_label, variants in labels_map.items():
            if isinstance(variants, list):
                for variant in variants:
                    translation_to_label[variant.strip().lower()] = normalized_label
            else:
                translation_to_label[str(variants).strip().lower()] = normalized_label

        # Update textualRating fields
        translated_count = 0
        for claim in claims:
            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "").strip().lower()
                if rating in translation_to_label:
                    review["normalizedTextualRating"] = translation_to_label[rating]
                    translated_count += 1

        logger.info(f"Translated {translated_count} label ratings")
        return claims

    def _remove_noisy_labels(
        self, claims: List[Dict[str, Any]], labels_map: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Remove claims with labels not in the mapping."""
        original_count = len(claims)

        filtered_claims = [
            claim
            for claim in claims
            if all(
                review.get("normalizedTextualRating") in labels_map
                for review in claim.get("claimReview", [])
                if review.get("normalizedTextualRating")  # Only check if field exists
            )
        ]

        removed_count = original_count - len(filtered_claims)
        logger.info(f"Removed {removed_count} claims with noisy/unmapped labels")

        return filtered_claims

    def _prepare_csv_rows(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare data rows for CSV export."""
        rows = []

        for idx, item in enumerate(data, start=1):
            claim_text = item.get("text", "")
            claimant = item.get("claimant", "")
            claim_reviews = item.get("claimReview", [])

            if not claim_reviews:
                continue  # Skip if there's no review

            # Use first review for CSV (could be extended to handle multiple reviews)
            review = claim_reviews[0]

            rows.append(
                {
                    "ClaimID": str(idx),
                    "claim": claim_text,
                    "description": review.get("title", ""),
                    "source": review.get("publisher", {}).get("name", ""),
                    "date": review.get("reviewDate", ""),
                    "source_label": review.get("textualRating", ""),
                    "normalized_label": review.get("normalizedTextualRating", ""),
                    "source_url": review.get("url", ""),
                    "claimant": claimant,
                }
            )

        return rows

    def _get_file_path(self, filename: str, data_type: str) -> Path:
        """Get full file path for data operations."""
        if self.data_dir:
            # Use custom data directory
            type_dirs = {
                "raw": self.data_dir / "raw",
                "processed": self.data_dir / "processed",
                "evidence": self.data_dir / "evidence",
            }
            return type_dirs.get(data_type, self.data_dir) / filename
        else:
            # Use config-based path
            return get_data_file_path(filename, data_type)


# Convenience functions for backward compatibility
def load_queries(filename: str = "queries.txt") -> List[str]:
    """Load queries from file (backward compatibility function).

    Args:
        filename: Name of queries file.

    Returns:
        List of query strings.
    """
    processor = ClaimDataProcessor()
    return processor.load_queries(filename)


def save_to_file(
    data: List[Dict[str, Any]], output_filename: str, encoding: str = "utf-8"
) -> None:
    """Save data to JSON file (backward compatibility function).

    Args:
        data: Data to save.
        output_filename: Output filename.
        encoding: File encoding (ignored, always uses utf-8).
    """
    processor = ClaimDataProcessor()
    processor.save_claims_to_json(data, output_filename)


def get_label_distribution(
    claims: List[Dict[str, Any]], field_name: str, output_path: str
) -> Dict[str, int]:
    """Get label distribution (backward compatibility function)."""
    processor = ClaimDataProcessor()
    return processor.get_label_distribution(claims, field_name, output_path)


def filter_claims_by_source(
    claims: List[Dict[str, Any]], whitelisted_sites: List[str]
) -> List[Dict[str, Any]]:
    """Filter claims by source (backward compatibility function)."""
    processor = ClaimDataProcessor()
    return processor.filter_claims_by_source(claims, whitelisted_sites)


def remove_duplicates(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates (backward compatibility function)."""
    processor = ClaimDataProcessor()
    return processor.remove_duplicates(claims)


def normalize_claims(
    claims: List[Dict[str, Any]],
    labels_map: Dict[str, Any],
    translate_labels: bool = True,
    remove_noisy_labels: bool = True,
) -> List[Dict[str, Any]]:
    """Normalize claims (backward compatibility function)."""
    processor = ClaimDataProcessor()
    return processor.normalize_claims(
        claims, labels_map, translate_labels, remove_noisy_labels
    )


def save_to_csv(data: List[Dict[str, Any]], csv_path: str) -> None:
    """Save to CSV (backward compatibility function)."""
    processor = ClaimDataProcessor()
    processor.save_to_csv(data, csv_path)
