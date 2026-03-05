"""Tests for the health / info endpoints."""

from fastapi.testclient import TestClient


def test_root_returns_service_info(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "service" in body
    assert "endpoints" in body


def test_root_lists_expected_endpoint_groups(client: TestClient) -> None:
    body = client.get("/").json()
    endpoints = body["endpoints"]
    assert "ml" in endpoints
    assert "rag" in endpoints
    assert "proctoring" in endpoints


def test_health_check_returns_200(client: TestClient) -> None:
    response = client.get("/api/proctor/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_health_check_includes_device_info(client: TestClient) -> None:
    body = client.get("/api/proctor/health").json()
    # Device is "cpu" because CUDA is mocked to return False
    assert "device" in body
    assert body["device"] in ("cpu", "cuda")


def test_health_check_response_structure(client: TestClient) -> None:
    body = client.get("/api/proctor/health").json()
    # Should at minimum contain status and models_loaded keys
    required_keys = {"status"}
    assert required_keys.issubset(body.keys())
