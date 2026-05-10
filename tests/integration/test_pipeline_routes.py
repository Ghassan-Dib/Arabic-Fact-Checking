from unittest.mock import patch

from fastapi.testclient import TestClient

from models.pipeline import JobStatus


def test_run_returns_job_id(client: TestClient) -> None:
    with patch(
        "pipeline.runner.FactCheckingPipeline.run",
        return_value=None,
    ):
        response = client.post("/api/v1/pipeline/run", json={})
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.QUEUED


def test_status_returns_job(client: TestClient) -> None:
    with patch(
        "pipeline.runner.FactCheckingPipeline.run",
        return_value=None,
    ):
        run_resp = client.post("/api/v1/pipeline/run", json={})
    job_id = run_resp.json()["job_id"]

    status_resp = client.get(f"/api/v1/pipeline/{job_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id


def test_status_404_for_unknown_job(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/nonexistent-id/status")
    assert response.status_code == 404


def test_result_404_when_not_ready(client: TestClient) -> None:
    with patch(
        "pipeline.runner.FactCheckingPipeline.run",
        return_value=None,
    ):
        run_resp = client.post("/api/v1/pipeline/run", json={})
    job_id = run_resp.json()["job_id"]

    result_resp = client.get(f"/api/v1/pipeline/{job_id}/result")
    assert result_resp.status_code == 404
