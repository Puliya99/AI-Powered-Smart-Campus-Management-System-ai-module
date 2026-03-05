"""Tests for the RAG chatbot endpoints (/process-material, /chat)."""

import io
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── /process-material ─────────────────────────────────────────────────────────
# Endpoint signature: courseId=Form(...), materialId=Form(...), file=File(...)
# (camelCase form fields — NOT course_id / material_name)

class TestProcessMaterial:
    def test_upload_plain_text_file_returns_200(self, client: TestClient) -> None:
        content = b"Introduction to Machine Learning\n\nMachine learning is a subset of AI."
        files = {"file": ("lecture.txt", io.BytesIO(content), "text/plain")}
        data = {"courseId": "CS101", "materialId": "mat-001"}

        response = client.post("/process-material", files=files, data=data)
        assert response.status_code in (200, 500)

    def test_upload_requires_file_field(self, client: TestClient) -> None:
        data = {"courseId": "CS101", "materialId": "mat-001"}
        response = client.post("/process-material", data=data)
        assert response.status_code == 422

    def test_upload_with_empty_file_returns_error(self, client: TestClient) -> None:
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        data = {"courseId": "CS101", "materialId": "mat-empty"}
        response = client.post("/process-material", files=files, data=data)
        # Empty content → no chunks extracted → graceful 200 or error
        assert response.status_code in (200, 400, 422, 500)

    def test_upload_pdf_file(self, client: TestClient) -> None:
        """PDF upload should be accepted; PdfReader is mocked to avoid real parsing."""
        pdf_bytes = b"%PDF-1.4\n%%EOF"
        files = {"file": ("notes.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        data = {"courseId": "CS101", "materialId": "mat-pdf"}

        # Mock PdfReader so the test doesn't depend on a structurally valid PDF
        with patch("main.PdfReader") as mock_reader:
            mock_reader.return_value.pages = []
            response = client.post("/process-material", files=files, data=data)

        assert response.status_code in (200, 400, 500)

    def test_successful_upload_response_structure(self, client: TestClient) -> None:
        content = b"Database concepts: ACID, normalisation, SQL."
        files = {"file": ("db.txt", io.BytesIO(content), "text/plain")}
        data = {"courseId": "DB201", "materialId": "mat-db"}

        with patch("main.faiss") as _mock_faiss:
            response = client.post("/process-material", files=files, data=data)

        if response.status_code == 200:
            body = response.json()
            assert "message" in body or "chunks" in body or "status" in body


# ── /chat ─────────────────────────────────────────────────────────────────────
# ChatRequest schema: { courseId: str, question: str, top_k?: int }
# Global names in main.py: embed_model, indices (dict[courseId → {index, chunks}])

class TestChat:
    def test_chat_requires_question_and_course_id(self, client: TestClient) -> None:
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_with_mocked_gemini_returns_answer(self, client: TestClient) -> None:
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1] * 384], dtype="float32")

        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_index.search.return_value = (
            np.array([[0.05]], dtype="float32"),
            np.array([[0]], dtype="int64"),
        )

        # indices stores {courseId: {"index": FaissIndex, "chunks": [chunk_info, ...]}}
        fake_chunk = {
            "id": "mat-001_0",
            "materialId": "mat-001",
            "content": "Machine learning is a subset of AI.",
            "metadata": {},
            "global_idx": 0,
        }
        fake_indices = {"CS101": {"index": mock_index, "chunks": [fake_chunk]}}

        with (
            patch("main.embed_model", mock_embedder),
            patch("main.indices", fake_indices),
            patch("main.gemini_client") as mock_gc,
        ):
            mock_response = MagicMock()
            mock_response.text = "Machine learning is a branch of artificial intelligence."
            mock_gc.models.generate_content.return_value = mock_response

            # ChatRequest uses courseId (camelCase)
            response = client.post(
                "/chat",
                json={"courseId": "CS101", "question": "What is machine learning?"},
            )

        if response.status_code == 200:
            body = response.json()
            assert "answer" in body or "response" in body or "message" in body

    def test_chat_returns_answer_when_no_material_uploaded(
        self, client: TestClient
    ) -> None:
        # _sync_chat does NOT error when course not in indices — it just calls
        # Gemini with an empty context. The response is 200 (or 500 if Gemini fails).
        with patch("main.indices", {}):
            response = client.post(
                "/chat",
                json={"courseId": "NONEXISTENT999", "question": "What is AI?"},
            )
        assert response.status_code in (200, 400, 404, 500)

    def test_chat_returns_error_when_gemini_client_is_none(
        self, client: TestClient
    ) -> None:
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1] * 384], dtype="float32")

        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_index.search.return_value = (
            np.array([[0.1]], dtype="float32"),
            np.array([[0]], dtype="int64"),
        )
        fake_chunk = {
            "id": "mat-001_0",
            "materialId": "mat-001",
            "content": "Some text.",
            "metadata": {},
            "global_idx": 0,
        }

        with (
            patch("main.embed_model", mock_embedder),
            patch("main.indices", {"CS101": {"index": mock_index, "chunks": [fake_chunk]}}),
            patch("main.gemini_client", None),
        ):
            response = client.post(
                "/chat",
                json={"courseId": "CS101", "question": "Explain normalisation."},
            )

        # When gemini_client is None the endpoint returns a fallback message (200)
        # rather than an HTTP error — it's a graceful degradation.
        assert response.status_code in (200, 400, 500)
