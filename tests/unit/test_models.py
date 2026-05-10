import pytest
from pydantic import ValidationError

from models.claim import Claim, ClaimLabel
from models.evidence import Evidence, GoldEvidence
from models.pipeline import JobStatus, PipelineConfig, PipelineJobState
from models.verification import QAPair, VerificationResult


class TestClaimLabel:
    def test_all_values_defined(self) -> None:
        assert ClaimLabel.SUPPORTED == "supported"
        assert ClaimLabel.REFUTED == "refuted"
        assert ClaimLabel.NOT_ENOUGH_EVIDENCE == "Not Enough Evidence"
        assert ClaimLabel.CONFLICTING_EVIDENCE == "Conflicting Evidence/Cherrypicking"

    def test_is_str_enum(self) -> None:
        assert isinstance(ClaimLabel.SUPPORTED, str)


class TestClaim:
    def test_minimal_claim(self) -> None:
        c = Claim(id="1", text="ادعاء")
        assert c.id == "1"
        assert c.date is None
        assert c.normalized_label is None

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(ValidationError):
            Claim(id="1", text="x", normalized_label="unknown_label")  # type: ignore[arg-type]


class TestEvidence:
    def test_valid_evidence(self) -> None:
        e = Evidence(title="مقال", url="https://example.com")  # type: ignore[arg-type]
        assert e.title == "مقال"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(title="x", url="not-a-url")  # type: ignore[arg-type]


class TestGoldEvidence:
    def test_empty_sources(self) -> None:
        ge = GoldEvidence(sources=[])
        assert ge.sources == []


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.batch_size == 10
        assert cfg.collect_claims is True

    def test_batch_size_minimum(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(batch_size=0)


class TestPipelineJobState:
    def test_default_status(self) -> None:
        job = PipelineJobState(job_id="abc")
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None


class TestQAPair:
    def test_creation(self) -> None:
        qa = QAPair(question="ما هو؟", answer="هذا")
        assert qa.question == "ما هو؟"


class TestVerificationResult:
    def test_creation(self) -> None:
        vr = VerificationResult(
            claim_id="1",
            qa_pairs=[QAPair(question="q", answer="a")],
            predicted_label=ClaimLabel.SUPPORTED,
        )
        assert vr.predicted_label == ClaimLabel.SUPPORTED
