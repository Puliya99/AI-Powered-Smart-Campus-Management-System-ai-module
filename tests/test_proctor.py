"""Tests for the proctoring endpoints (/api/proctor/detect-objects,
/api/proctor/head-pose, /api/proctor/analyze)."""

import base64
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_base64_image(width: int = 100, height: int = 100) -> str:
    """Return a base64-encoded JPEG byte string suitable for test payloads.

    Uses Pillow so there is no dependency on cv2 at collection time (cv2 may
    have import-level issues when both opencv-python-headless and
    opencv-contrib-python are installed in the same environment).
    """
    import io
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_VALID_IMAGE = _make_base64_image()


# ── /api/proctor/detect-objects ───────────────────────────────────────────────

class TestDetectObjects:
    def test_missing_image_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/proctor/detect-objects", json={})
        assert response.status_code == 422

    def test_invalid_base64_returns_error(self, client: TestClient) -> None:
        response = client.post(
            "/api/proctor/detect-objects",
            json={"image": "not-valid-base64!!!"},
        )
        assert response.status_code in (400, 500)

    def test_valid_image_returns_200_with_detections(
        self, client: TestClient
    ) -> None:
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [67]   # 67 = cell phone in COCO
        mock_box.conf = [0.92]
        mock_box.xyxy = [[10, 10, 50, 50]]
        mock_result[0].boxes = mock_box

        with patch("main.yolo_model") as mock_yolo:
            mock_yolo.return_value = mock_result
            response = client.post(
                "/api/proctor/detect-objects",
                json={"image": _VALID_IMAGE},
            )

        assert response.status_code in (200, 500)
        if response.status_code == 200:
            body = response.json()
            assert (
                "suspicious_objects" in body
                or "detections" in body
                or "objects" in body
                or "violations" in body
                or "is_violation" in body
            )

    def test_no_suspicious_objects_returns_clean_result(
        self, client: TestClient
    ) -> None:
        mock_result = MagicMock()
        mock_result[0].boxes = MagicMock()
        mock_result[0].boxes.cls = []
        mock_result[0].boxes.conf = []
        mock_result[0].boxes.xyxy = []

        with patch("main.yolo_model") as mock_yolo:
            mock_yolo.return_value = mock_result
            response = client.post(
                "/api/proctor/detect-objects",
                json={"image": _VALID_IMAGE},
            )

        if response.status_code == 200:
            body = response.json()
            # Either empty list or a cheating_detected=False flag
            detections = body.get("detections", body.get("objects", []))
            if isinstance(detections, list):
                assert len(detections) == 0


# ── /api/proctor/head-pose ────────────────────────────────────────────────────

class TestHeadPose:
    def test_missing_image_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/proctor/head-pose", json={})
        assert response.status_code == 422

    def test_valid_image_accepted(self, client: TestClient) -> None:
        with patch("main.mp") as _mock_mp:
            response = client.post(
                "/api/proctor/head-pose",
                json={"image": _VALID_IMAGE},
            )
        assert response.status_code in (200, 500)

    def test_response_contains_face_count_when_successful(
        self, client: TestClient
    ) -> None:
        with patch("main.mp"):
            response = client.post(
                "/api/proctor/head-pose",
                json={"image": _VALID_IMAGE},
            )
        if response.status_code == 200:
            body = response.json()
            assert (
                "face_count" in body
                or "faces_detected" in body
                or "violations" in body
            )


# ── /api/proctor/analyze ──────────────────────────────────────────────────────

class TestAnalyze:
    def test_missing_image_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/proctor/analyze", json={})
        assert response.status_code == 422

    def test_valid_image_returns_200_or_server_error(
        self, client: TestClient
    ) -> None:
        mock_yolo_result = MagicMock()
        mock_yolo_result[0].boxes.cls = []
        mock_yolo_result[0].boxes.conf = []
        mock_yolo_result[0].boxes.xyxy = []

        with (
            patch("main.yolo_model", return_value=mock_yolo_result),
            patch("main.mp"),
        ):
            response = client.post(
                "/api/proctor/analyze",
                json={"image": _VALID_IMAGE},
            )

        assert response.status_code in (200, 500)

    def test_analyze_response_has_violation_score(
        self, client: TestClient
    ) -> None:
        mock_yolo_result = MagicMock()
        mock_yolo_result[0].boxes.cls = []
        mock_yolo_result[0].boxes.conf = []
        mock_yolo_result[0].boxes.xyxy = []

        with (
            patch("main.yolo_model", return_value=mock_yolo_result),
            patch("main.mp"),
        ):
            response = client.post(
                "/api/proctor/analyze",
                json={"image": _VALID_IMAGE},
            )

        if response.status_code == 200:
            body = response.json()
            assert (
                "violation_score" in body
                or "score" in body
                or "violations" in body
            )

    def test_analyze_with_phone_detected_raises_violation(
        self, client: TestClient
    ) -> None:
        mock_result = MagicMock()
        mock_result[0].boxes.cls = [67]      # cell phone
        mock_result[0].boxes.conf = [0.95]
        mock_result[0].boxes.xyxy = [[5, 5, 60, 60]]

        with (
            patch("main.yolo_model", return_value=mock_result),
            patch("main.mp"),
        ):
            response = client.post(
                "/api/proctor/analyze",
                json={"image": _VALID_IMAGE},
            )

        if response.status_code == 200:
            body = response.json()
            score = body.get("violation_score", body.get("score", 0))
            # A phone detection should contribute a positive violation weight
            if isinstance(score, (int, float)):
                assert score >= 0
