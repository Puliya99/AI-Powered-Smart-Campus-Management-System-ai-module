# AI-Powered Smart Campus Management System — AI Module

## Overview

A standalone Python FastAPI microservice providing ML-based student risk prediction, RAG-powered lecture material Q&A, and real-time AI proctoring for online exams.

This module is consumed by the Node.js backend via HTTP and can also be called directly by the frontend for proctoring endpoints.

---

## Core Features

- **Student Risk Prediction** — RandomForest classifier trained on attendance, GPA, quiz scores, assignments, and behavioural data
- **RAG Chatbot** — Upload course materials (PDF/DOCX/PPTX/TXT), embed with Sentence-Transformers, retrieve via FAISS, answer via Google Gemini
- **Object Detection** — YOLOv8m detects suspicious objects during exams (phones, books, laptops, remotes)
- **Head Pose Estimation** — MediaPipe face mesh for gaze tracking, eye closure detection, and face count verification
- **Combined Proctoring** — Unified analysis endpoint with weighted violation scoring

---

## Technology Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Framework | FastAPI | 0.128 |
| ASGI Server | Uvicorn | 0.40 |
| Validation | Pydantic | 2.12 |
| ML | Scikit-learn (RandomForestClassifier) | 1.8 |
| Data | Pandas, NumPy, SciPy | 2.3 / ≥1.26 / 1.17 |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) | 5.2 |
| Vector Search | FAISS (faiss-cpu) | 1.13 |
| LLM | Google Gemini (google-genai SDK) | 1.59 |
| Object Detection | Ultralytics YOLOv8m | 8.4 |
| Pose / Face | MediaPipe, DeepFace, OpenCV | 0.10 / 0.0.93 / ≥4.6 |
| Deep Learning | PyTorch (CPU), Torchvision | — |
| NLP | Hugging Face Transformers | 4.57 |
| Document Parsing | pypdf, python-docx, python-pptx | 6.6 / 1.2 / 1.0 |
| HTTP | httpx, requests | 0.28 / 2.32 |
| Retry Logic | Tenacity | 9.1 |
| LLM (Groq) | groq SDK | 1.0 |

---

## Prerequisites

- **Python** 3.12
- **pip** (latest)
- **~2 GB disk space** for Python dependencies
- **~500 MB additional** for model weights downloaded on first run (YOLOv8, Sentence-Transformers, DeepFace)
- **Google Gemini API key** — required for RAG chatbot and LLM features

---

## Getting Started

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

> The `--extra-index-url` flag pulls CPU-only PyTorch wheels, keeping install size smaller. Omit it to let pip resolve from PyPI (may pull CUDA builds).

### 3. Environment Configuration

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` to your Google Gemini API key.

### 4. Run the Server

```bash
python main.py
```

The server starts on `http://localhost:8001` by default.

> **First run:** model weights (YOLOv8m, Sentence-Transformers, DeepFace) are downloaded automatically on startup. Allow a few minutes and ensure internet access.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (also accepted as `GOOGLE_API_KEY`) | **required** |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `AI_SERVICE_URL` | Self-referencing service URL | `http://localhost:8001` |

---

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info and available endpoint listing |
| GET | `/api/proctor/health` | Health check with device info (CPU/GPU) |

### ML — Student Risk Prediction

| Method | Endpoint | Description |
|---|---|---|
| POST | `/train` | Train RandomForest on historical student data (min 10 records) |
| POST | `/predict` | Predict exam failure risk for a single student |

**Input features:** `attendance_percentage`, `assignment_score`, `quiz_average`, `gpa`, `missed_classes`, `late_submissions`, `face_violations`, `payment_delay_days`, `previous_exam_score`, `total_modules`, `modules_at_risk`

**Output:** `risk_score` (0–1), `risk_level` (HIGH / MEDIUM / LOW), `reasons` list

**Risk thresholds:** LOW < 0.4 · MEDIUM 0.4–0.7 · HIGH ≥ 0.7

### RAG — Course Material Chatbot

| Method | Endpoint | Description |
|---|---|---|
| POST | `/process-material` | Upload file (PDF/DOCX/PPTX/TXT), chunk, embed, store in per-course FAISS index |
| POST | `/chat` | Query — embed question → FAISS similarity search → Gemini generation |

**Supported formats:** PDF, DOCX, PPTX, TXT

**Embedding model:** `all-MiniLM-L6-v2` (Sentence-Transformers)

### Proctoring — Exam Surveillance

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/proctor/detect-objects` | YOLOv8m object detection on base64 image |
| POST | `/api/proctor/head-pose` | MediaPipe face detection + head pose + eye tracking |
| POST | `/api/proctor/analyze` | Combined proctoring analysis with weighted violation score |

**Detected objects:** cell phone, book, laptop, remote, clock, person

**Violation weights:**

| Violation | Weight |
|---|---|
| NO_FACE | 1 |
| MULTIPLE_FACES | 2 |
| LOOKING_AWAY | 1 |
| EYES_CLOSED | varies |
| CHEATING_OBJECT | 2–3 |

---

## Project Structure

```
ai-module/
├── main.py              # FastAPI application (all endpoints, models, services)
├── yolov8m.pt           # YOLOv8 Medium pre-trained model (~52 MB)
├── requirements.txt     # Python dependencies
├── pytest.ini           # pytest configuration
├── .env.example         # Environment variable template
├── .env                 # Local environment (git-ignored)
├── .gitignore
├── Dockerfile           # Python 3.12-slim + OpenCV/MediaPipe system deps
├── render.yaml          # Render.com deployment config
├── models/              # Trained RandomForest models (git-ignored, created at runtime)
│   ├── model_v1.pkl     # Versioned model files
│   └── metadata.json    # Model version tracking
├── indices/             # FAISS vector indices per course (git-ignored, in-memory)
├── task_models/         # Pre-trained task-specific model files
├── face_embeddings/     # Cached face embedding data (git-ignored)
├── tests/               # pytest test suite
│   ├── conftest.py      # Shared fixtures (TestClient, mock data)
│   ├── test_health.py   # Health/info endpoint tests
│   ├── test_predict.py  # ML risk prediction tests
│   ├── test_chat.py     # RAG chatbot tests
│   └── test_proctor.py  # Proctoring endpoint tests
└── ai_service.log       # Runtime log file (git-ignored)
```

---

## Docker

### Build

```bash
docker build -t smart-campus-ai .
```

### Run

```bash
docker run -p 8001:10000 \
  -e GEMINI_API_KEY=your_key \
  smart-campus-ai
```

The Dockerfile uses `python:3.12-slim`, installs system libraries for OpenCV/MediaPipe (`libgl1`, `libglib2.0-0`, etc.), and uses CPU-only PyTorch for a smaller image.

> The container exposes port `10000` internally; the run command maps it to `8001` on the host.

---

## Deployment

### Render.com

The `render.yaml` defines a Docker web service:

- **Region:** Oregon
- **Plan:** Starter
- **Health check:** `/api/proctor/health`
- **Auto-deploy:** Enabled

Set `GEMINI_API_KEY` in Render's environment variable settings before deploying.

---

## Architecture Notes

- All blocking ML operations (training, inference, YOLO detection) run in executor threads via `asyncio.get_event_loop().run_in_executor()` to keep the async API non-blocking
- FAISS indices are stored **in-memory per course** — they are not persisted to disk by default; re-upload materials after a restart
- Trained models are versioned (`model_v1.pkl`, `model_v2.pkl`, …) with a `metadata.json` version tracker
- Auto-detects CUDA/GPU availability; falls back to CPU
- CORS is open (`*`) — restrict in production via a reverse proxy or FastAPI middleware

---

## Testing

- **Framework:** pytest + httpx (async endpoint testing)
- **Run:** `pytest` (in `ai-module/` with venv active)
- **Directory:** `tests/` — `test_health.py`, `test_predict.py`, `test_chat.py`, `test_proctor.py`
- **Configuration:** `pytest.ini`

```bash
# Run all tests
pytest

# Run a specific file
pytest tests/test_health.py -v

# Manual smoke tests
curl http://localhost:8001/api/proctor/health
curl http://localhost:8001/
```

---

## License

<!-- TODO: Add a LICENSE file to ai-module/ -->

ISC (root project). See root `README.md` for details.
