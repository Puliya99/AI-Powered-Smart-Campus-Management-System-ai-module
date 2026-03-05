"""
Pytest configuration and shared fixtures.

Heavy ML dependencies (YOLO, MediaPipe, PyTorch, FAISS, Sentence-Transformers,
Google GenAI) are replaced with MagicMock objects in sys.modules BEFORE main.py
is imported so that:
  - No model files need to be present.
  - No GPU / CUDA drivers are required.
  - Tests run quickly in any CI environment.
"""

import os
import sys
import numpy as np
from unittest.mock import MagicMock

# ── Environment ────────────────────────────────────────────────────────────────
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")

# ── torch mock ────────────────────────────────────────────────────────────────
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False  # force CPU path
mock_torch.device.return_value = "cpu"
# scipy.stats (imported by sklearn) calls issubclass(x, torch.Tensor).
# MagicMock attributes are not real classes, causing TypeError.
# Assign a real class so the issubclass check passes cleanly.
mock_torch.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torchvision", MagicMock())

# ── cv2 ───────────────────────────────────────────────────────────────────────
sys.modules.setdefault("cv2", MagicMock())

# ── ultralytics / YOLO ────────────────────────────────────────────────────────
mock_ultralytics = MagicMock()
sys.modules.setdefault("ultralytics", mock_ultralytics)

# ── mediapipe ────────────────────────────────────────────────────────────────
sys.modules.setdefault("mediapipe", MagicMock())
sys.modules.setdefault("mediapipe.tasks", MagicMock())
sys.modules.setdefault("mediapipe.tasks.python", MagicMock())
sys.modules.setdefault("mediapipe.tasks.python.vision", MagicMock())
sys.modules.setdefault("mediapipe.tasks.python.core", MagicMock())
sys.modules.setdefault("mediapipe.tasks.python.components", MagicMock())

# ── sentence_transformers ─────────────────────────────────────────────────────
mock_st = MagicMock()
# Return a numpy array so that embeddings.shape[1] works in _sync_process_material
mock_st.SentenceTransformer.return_value.encode.return_value = np.array([[0.1] * 384], dtype="float32")
sys.modules.setdefault("sentence_transformers", mock_st)

# ── faiss ─────────────────────────────────────────────────────────────────────
mock_faiss = MagicMock()
mock_faiss_index = MagicMock()
mock_faiss_index.ntotal = 1
# Simulate a single result (distance, index)
mock_faiss_index.search.return_value = (
    np.array([[0.1]], dtype="float32"),
    np.array([[0]], dtype="int64"),
)
mock_faiss.IndexFlatL2.return_value = mock_faiss_index
sys.modules.setdefault("faiss", mock_faiss)

# ── google.genai ──────────────────────────────────────────────────────────────
mock_genai = MagicMock()
mock_genai_response = MagicMock()
mock_genai_response.text = "This is a mocked Gemini response."
mock_genai.Client.return_value.models.generate_content.return_value = mock_genai_response

mock_google = MagicMock()
mock_google.genai = mock_genai
sys.modules.setdefault("google", mock_google)
sys.modules.setdefault("google.genai", mock_genai)

# ── PIL / deepface / transformers ─────────────────────────────────────────────
# PIL must NOT be mocked: test_proctor._make_base64_image() uses real Pillow
# to build a valid JPEG without depending on cv2.  Only mock deepface/transformers.
sys.modules.setdefault("deepface", MagicMock())
sys.modules.setdefault("deepface", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

# ── pytest fixtures ───────────────────────────────────────────────────────────
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a FastAPI TestClient backed by the real (but mock-dep) app."""
    from main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def sample_predict_payload() -> dict:
    # StudentData schema: student_id and subject_id are required strings.
    # Numeric fields use the exact names from the Pydantic model in main.py.
    return {
        "student_id": "STU001",
        "subject_id": "CS101",
        "attendance_percentage": 75.0,
        "average_assignment_score": 60.0,
        "quiz_average": 55.0,
        "gpa": 2.5,
        "classes_missed": 5,
        "late_submissions": 3,
        "face_violation_count": 2,
        "payment_delay_days": 10,
        "previous_exam_score": 50.0,
    }


@pytest.fixture(scope="session")
def sample_train_payload() -> list:
    """Minimum of 10 StudentData records required to train.

    /train takes List[StudentData] directly (not wrapped in {"students": [...]}).
    exam_result must be present (0=PASS, 1=FAIL) — None values cause a 400.
    """
    records = []
    for i in range(10):
        records.append({
            "student_id": f"STU{i:03d}",
            "subject_id": "CS101",
            "attendance_percentage": 70.0 + i,
            "average_assignment_score": 60.0 + i,
            "quiz_average": 55.0 + i,
            "gpa": 2.5 + (i * 0.05),
            "classes_missed": max(0, 5 - i),
            "late_submissions": max(0, 3 - i),
            "face_violation_count": i % 3,
            "payment_delay_days": i * 2,
            "previous_exam_score": 50.0 + i,
            "exam_result": 1 if i < 4 else 0,
        })
    return records
