# AI-Powered Smart Campus Management System - AI Module

## Overview

A standalone Python FastAPI microservice providing ML-based student risk prediction, RAG-powered lecture material Q&A, and real-time AI proctoring for online exams.

This module is consumed by the Node.js backend via HTTP and can also be called directly by the frontend for proctoring endpoints.

---

## Core Features

- **Student Risk Prediction** - RandomForest classifier trained on attendance, GPA, quiz scores, assignments, and behavioral data
- **RAG Chatbot** - Upload course materials (PDF/DOCX/PPTX/TXT), embed with Sentence-Transformers, retrieve via FAISS, and answer via Google Gemini
- **Object Detection** - YOLOv8m detects suspicious objects during exams (phones, books, laptops, remotes)
- **Head Pose Estimation** - MediaPipe face mesh for gaze tracking, eye closure detection, and face count verification
- **Combined Proctoring** - Unified analysis endpoint with weighted violation scoring

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Framework | FastAPI 0.128 |
| ASGI Server | Uvicorn 0.40 |
| Validation | Pydantic 2.12 |
| ML | Scikit-learn 1.8 (RandomForestClassifier) |
| Data | Pandas 2.3, NumPy, SciPy 1.17 |
| Embeddings | Sentence-Transformers 5.2 (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu 1.13) |
| LLM | Google Gemini (google-genai 1.59) |
| Object Detection | Ultralytics 8.4 (YOLOv8m) |
| Pose Estimation | MediaPipe 0.10, OpenCV |
| Deep Learning | PyTorch (CPU), Torchvision |
| Document Parsing | pypdf 6.6, python-docx 1.2, python-pptx 1.0 |
| NLP | Hugging Face Transformers 4.57 |
| HTTP | httpx 0.28, requests 2.32 |

---

## Prerequisites

- **Python** 3.12
- **pip**
- ~2 GB disk space (for dependencies + YOLOv8m model)
- **Google Gemini API key** (for RAG chatbot features)

---

## Getting Started

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: PyTorch CPU wheels are pulled from `https://download.pytorch.org/whl/cpu` automatically via the Dockerfile. For local install, pip resolves dependencies from PyPI.

### 3. Environment Configuration

```bash
cp .env.example .env
```

Update with your Gemini API key.

### 4. Run the Server

```bash
python main.py
```

The server starts on `http://localhost:8001` by default. Pass a port argument to override: `python main.py 9000`.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | **required** |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |

---

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info and available endpoint listing |
| GET | `/api/proctor/health` | Health check with device info (CPU/GPU) |

### ML - Student Risk Prediction

| Method | Endpoint | Description |
|---|---|---|
| POST | `/train` | Train RandomForest on historical student data (min 10 records) |
| POST | `/predict` | Predict exam failure risk for a single student |

**Input features:** attendance_percentage, assignment_score, quiz_average, gpa, missed_classes, late_submissions, face_violations, payment_delay_days, previous_exam_score, total_modules, modules_at_risk

**Output:** risk_score (0-1), risk_level (HIGH/MEDIUM/LOW), reasons list

**Risk thresholds:** LOW < 0.4, MEDIUM 0.4-0.7, HIGH >= 0.7

### RAG - Course Material Chatbot

| Method | Endpoint | Description |
|---|---|---|
| POST | `/process-material` | Upload file (PDF/DOCX/PPTX/TXT), extract text, embed, store in FAISS |
| POST | `/chat` | Query course materials — embed question, FAISS search, Gemini generation |

**Process flow:** Upload file -> chunk text -> embed with all-MiniLM-L6-v2 -> store in per-course FAISS index -> query with similarity search -> generate answer with Gemini using retrieved context

### Proctoring - Exam Surveillance

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/proctor/detect-objects` | YOLOv8m object detection on base64 image |
| POST | `/api/proctor/head-pose` | MediaPipe face detection + head pose + eye tracking |
| POST | `/api/proctor/analyze` | Combined proctoring analysis with weighted violations |

**Detected objects:** Cell phone, book, laptop, remote, clock, person

**Violation types and weights:**
| Violation | Weight |
|---|---|
| NO_FACE | 1 |
| MULTIPLE_FACES | 2 |
| LOOKING_AWAY | 1 |
| EYES_CLOSED | varies |
| CHEATING_OBJECT | 2-3 |

---

## Project Structure

```
ai-module/
├── main.py              # FastAPI application (all endpoints, models, services)
├── yolov8m.pt           # YOLOv8 Medium pre-trained model (~52MB)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Local environment (git-ignored)
├── .gitignore
├── Dockerfile           # Python 3.12-slim with system deps for OpenCV/MediaPipe
├── render.yaml          # Render.com deployment config (Oregon, Starter plan)
├── models/              # Trained ML models (git-ignored, created at runtime)
│   ├── model_v1.pkl     # Versioned RandomForest models
│   └── metadata.json    # Model version tracking
├── indices/             # FAISS vector indices (git-ignored, in-memory at runtime)
└── ai_service.log       # Runtime logs (git-ignored)
```

---

## Docker

### Build

```bash
docker build -t smart-campus-ai .
```

### Run

```bash
docker run -p 8001:10000 -e GEMINI_API_KEY=your_key smart-campus-ai
```

The Dockerfile uses `python:3.12-slim`, installs system dependencies for OpenCV/MediaPipe, and uses CPU-only PyTorch for smaller image size.

---

## Deployment

### Render.com

The `render.yaml` defines a Docker web service:

- **Region:** Oregon
- **Plan:** Starter
- **Health check:** `/api/proctor/health`
- **Auto-deploy:** Enabled

Ensure `GEMINI_API_KEY` is set in Render's environment variables.

---

## Architecture Notes

- All blocking ML operations (training, prediction, inference) run in executor threads via `asyncio.get_event_loop().run_in_executor()` for non-blocking API responses
- FAISS indices are stored in-memory per course (not persisted to disk by default)
- Trained models are versioned (`model_v1.pkl`, `model_v2.pkl`, ...) with metadata tracking
- Auto-detects CUDA/GPU availability; falls back to CPU
- CORS is configured to allow all origins (`*`)

---

## Testing

<!-- TODO: Add test suite (pytest + httpx for async endpoint testing) -->

Run the server and test endpoints manually:

```bash
# Health check
curl http://localhost:8001/api/proctor/health

# Service info
curl http://localhost:8001/
```

---

## License

ISC
