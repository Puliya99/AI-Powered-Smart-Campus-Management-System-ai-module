import os
import io
import json
import math
import base64
import asyncio
import datetime
from typing import List, Optional, Dict
from functools import partial

import joblib
import pandas as pd
import numpy as np
import faiss
import cv2
from PIL import Image
import torch
import mediapipe as mp

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from ultralytics import YOLO

from dotenv import load_dotenv

# ✅ New Gemini SDK (replaces deprecated google.generativeai)
from google import genai

# ----------------------------
# Env
# ----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # start with flash (fast + usually available)

# Create Gemini client only if key exists (so server can still run without it)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Campus AI Analytics & RAG API")

# CORS - allow frontend to call this service directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# GPU / Device Setup
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


@app.get("/")
async def root():
    return {
        "service": "Campus AI Analytics & RAG API",
        "status": "running",
        "device": DEVICE,
        "endpoints": {
            "ml": ["/train", "/predict"],
            "rag": ["/process-material", "/chat"],
            "proctoring": [
                "/api/proctor/detect-objects",
                "/api/proctor/head-pose",
                "/api/proctor/analyze",
                "/api/proctor/health",
            ],
        },
    }

# ----------------------------
# YOLOv8m Model for Object Detection (Proctoring)
# ----------------------------
print("Loading YOLOv8m model for proctoring (balanced speed/accuracy)...")
yolo_model = YOLO("yolov8m.pt")  # medium model - best balance for MX150

# COCO class IDs for suspicious objects during exams
SUSPICIOUS_OBJECTS = {
    67: "cell phone",
    73: "book",
    63: "laptop",
    65: "remote",
    74: "clock",
    0: "person",  # tracked separately for multiple-person detection
}

# ----------------------------
# MediaPipe Face Detection + FaceMesh (Head Pose)
# ----------------------------
print("Loading MediaPipe models...")
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,  # full-range model
    min_detection_confidence=0.7,
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
)
print("MediaPipe models loaded.")

# ----------------------------
# Embedding Model
# ----------------------------
print("Loading Embedding Model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_DIR = "models"
INDEX_DIR = "indices"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# FAISS Indices storage (in-memory)
# courseId -> {"index": faiss.IndexFlatL2(dim), "chunks": [chunk_info, ...]}
indices: Dict[str, Dict] = {}


# ----------------------------
# Helpers: metadata
# ----------------------------
def save_metadata(metadata: dict) -> None:
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)


def load_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {"current_version": "v1", "models": {}}


# ----------------------------
# ML: Risk prediction schemas
# ----------------------------
class StudentData(BaseModel):
    student_id: str
    subject_id: str
    attendance_percentage: Optional[float] = 0
    average_assignment_score: Optional[float] = 0
    quiz_average: Optional[float] = 0
    gpa: Optional[float] = 0
    classes_missed: Optional[int] = 0
    late_submissions: Optional[int] = 0
    quiz_attempts: Optional[int] = 0
    face_violation_count: Optional[int] = 0
    payment_delay_days: Optional[int] = 0
    previous_exam_score: Optional[float] = 0
    exam_result: Optional[int] = None  # 1 FAIL, 0 PASS


class PredictionResponse(BaseModel):
    student_id: str
    risk_score: float
    risk_level: str
    reasons: List[str]
    version: Optional[str] = None


def _sync_train(data: List[StudentData]):
    """Run blocking ML training in a worker thread."""
    df = pd.DataFrame([d.dict() for d in data])

    # Feature Engineering
    df["attendance_risk"] = (df["attendance_percentage"] < 75).astype(int)
    df["low_gpa"] = (df["gpa"] < 2.5).astype(int)

    X = df.drop(columns=["student_id", "subject_id", "exam_result"])
    y = df["exam_result"]

    if y.isnull().any():
        raise HTTPException(status_code=400, detail="Target variable 'exam_result' is missing in some records.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    metadata = load_metadata()
    version_num = len(metadata["models"]) + 1
    version = f"v{version_num}"
    model_filename = f"model_{version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(model, model_path)

    metadata["current_version"] = version
    metadata["models"][version] = {
        "path": model_path,
        "trained_at": datetime.datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "records_used": len(data),
    }
    save_metadata(metadata)

    return {"message": "Model trained successfully", "version": version, "accuracy": float(accuracy), "records_used": len(data)}


@app.post("/train")
async def train(data: List[StudentData]):
    if len(data) < 10:
        raise HTTPException(status_code=400, detail="Not enough data to train. Need at least 10 records.")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_sync_train, data))


def _sync_predict(data: StudentData):
    """Run blocking ML prediction in a worker thread."""
    metadata = load_metadata()
    current_version = metadata.get("current_version", "v1")
    model_info = metadata.get("models", {}).get(current_version)

    if not model_info or not os.path.exists(model_info["path"]):
        res = fallback_predict(data)
        res["version"] = "fallback"
        return res

    model = joblib.load(model_info["path"])

    df = pd.DataFrame([data.dict()])
    df["attendance_risk"] = (df["attendance_percentage"] < 75).astype(int)
    df["low_gpa"] = (df["gpa"] < 2.5).astype(int)

    X = df.drop(columns=["student_id", "subject_id", "exam_result"])

    risk_score = model.predict_proba(X)[0][1]  # prob of FAIL (class 1)

    if risk_score >= 0.7:
        risk_level = "HIGH"
    elif risk_score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    reasons = []
    if data.attendance_percentage < 75:
        reasons.append("Low attendance")
    if data.quiz_average < 50:
        reasons.append("Low quiz performance")
    if data.face_violation_count > 3:
        reasons.append("Multiple face violations")
    if data.average_assignment_score < 50:
        reasons.append("Low assignment scores")

    return {
        "student_id": data.student_id,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "reasons": reasons,
        "version": current_version,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: StudentData):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_sync_predict, data))


def fallback_predict(data: StudentData):
    score = 0.0
    reasons = []

    if data.attendance_percentage < 75:
        score += 0.4
        reasons.append("Low attendance")
    if data.quiz_average < 50:
        score += 0.2
        reasons.append("Low quiz performance")
    if data.average_assignment_score < 50:
        score += 0.2
        reasons.append("Low assignment scores")
    if data.face_violation_count > 3:
        score += 0.1
        reasons.append("Multiple face violations")
    if data.payment_delay_days > 15:
        score += 0.1
        reasons.append("Payment delays")

    score = min(score, 1.0)

    if score >= 0.7:
        risk_level = "HIGH"
    elif score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {"student_id": data.student_id, "risk_score": score, "risk_level": risk_level, "reasons": reasons}


# ----------------------------
# RAG: Schemas
# ----------------------------
class ChatRequest(BaseModel):
    courseId: str
    question: str
    top_k: Optional[int] = 5


class ChunkData(BaseModel):
    content: str
    metadata: Dict


def extract_text_from_file(file_content: bytes, filename: str) -> List[ChunkData]:
    chunks: List[ChunkData] = []
    ext = filename.split(".")[-1].lower()

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_content))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(ChunkData(content=text.strip(), metadata={"page": i + 1}))

    elif ext == "docx":
        doc = Document(io.BytesIO(file_content))
        current_text = ""
        for i, para in enumerate(doc.paragraphs):
            current_text += (para.text or "") + "\n"
            if len(current_text) > 1000:
                chunks.append(ChunkData(content=current_text.strip(), metadata={"paragraph_index": i}))
                current_text = ""
        if current_text.strip():
            chunks.append(ChunkData(content=current_text.strip(), metadata={"paragraph_index": len(doc.paragraphs)}))

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file_content))
        for i, slide in enumerate(prs.slides):
            text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += (shape.text or "") + " "
            if text.strip():
                chunks.append(ChunkData(content=text.strip(), metadata={"slide": i + 1}))

    else:
        # Generic text
        text = file_content.decode("utf-8", errors="ignore")
        for i in range(0, len(text), 1000):
            piece = text[i : i + 1000].strip()
            if piece:
                chunks.append(ChunkData(content=piece, metadata={"offset": i}))

    return chunks


def _sync_process_material(courseId: str, materialId: str, content: bytes, filename: str) -> dict:
    """Run blocking file parsing + embedding in a worker thread."""
    chunks = extract_text_from_file(content, filename)

    if not chunks:
        return {"message": "No text extracted", "chunks": []}

    texts = [c.content for c in chunks]
    embeddings = embed_model.encode(texts)

    dim = embeddings.shape[1]
    if courseId not in indices:
        indices[courseId] = {"index": faiss.IndexFlatL2(dim), "chunks": []}

    start_idx = len(indices[courseId]["chunks"])
    indices[courseId]["index"].add(embeddings.astype("float32"))

    processed_chunks = []
    for i, c in enumerate(chunks):
        chunk_info = {
            "id": f"{materialId}_{start_idx + i}",
            "materialId": materialId,
            "content": c.content,
            "metadata": c.metadata,
            "global_idx": start_idx + i,
        }
        indices[courseId]["chunks"].append(chunk_info)
        processed_chunks.append(chunk_info)

    return {"message": f"Processed {len(chunks)} chunks", "chunks": processed_chunks}


@app.post("/process-material")
async def process_material(courseId: str = Form(...), materialId: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_sync_process_material, courseId, materialId, content, file.filename)
    )


def build_prompt(question: str, chunks: list) -> str:
    sources = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        # Include metadata so Gemini can cite properly
        sources.append(f"[S{i}] meta={meta}\n{c['content']}")

    return f"""
You are a university lecture assistant.

RULES:
- Answer ONLY using the SOURCES.
- If the answer is not found, say exactly: "I can't find this in the uploaded lecture materials."
- Keep it student-friendly.
- Add citations like [S1], [S2] after each key statement.

QUESTION:
{question}

SOURCES:
{chr(10).join(sources)}
""".strip()


def _sync_chat(course_id: str, question: str, top_k: int) -> dict:
    """Run all blocking chat operations (embedding, FAISS, Gemini) in a worker thread."""
    if course_id not in indices or not indices[course_id]["chunks"]:
        return {
            "answer": "I don't have any materials for this course yet. Please upload some lecture notes first.",
            "citations": [],
        }

    # 1) Embed question
    q_emb = embed_model.encode([question])

    # 2) Search FAISS
    D, I = indices[course_id]["index"].search(q_emb.astype("float32"), top_k)

    retrieved_chunks = []
    for idx in I[0]:
        if idx != -1 and idx < len(indices[course_id]["chunks"]):
            retrieved_chunks.append(indices[course_id]["chunks"][idx])

    if not retrieved_chunks:
        return {"answer": "I can't find this in the uploaded lecture materials.", "citations": []}

    prompt = build_prompt(question, retrieved_chunks)

    # 3) Call Gemini (new SDK)
    if not gemini_client:
        answer = "Gemini is not configured. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env and restart."
    else:
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            answer = resp.text or "I couldn't generate an answer. Please try again."
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            answer = "I'm sorry, I'm having trouble connecting to Gemini right now."

    return {
        "answer": answer,
        "citations": [
            {
                "source": f"S{i+1}",
                "materialId": c["materialId"],
                "metadata": c["metadata"],
                "snippet": c["content"][:150],
            }
            for i, c in enumerate(retrieved_chunks)
        ],
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_sync_chat, request.courseId, request.question, request.top_k)
    )


# ----------------------------
# Proctoring: Object Detection
# ----------------------------
class ObjectDetectionRequest(BaseModel):
    image: str  # base64-encoded JPEG frame
    confidence_threshold: Optional[float] = 0.5


class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class ObjectDetectionResponse(BaseModel):
    suspicious_objects: List[DetectedObject]
    person_count: int
    is_violation: bool
    violation_details: Optional[str] = None


def _sync_detect_objects(image_b64: str, confidence_threshold: float) -> dict:
    """Run blocking YOLO inference in a worker thread."""
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = yolo_model(
        frame,
        device=DEVICE,
        conf=confidence_threshold,
        verbose=False,
        half=(DEVICE == "cuda"),
    )

    suspicious: List[DetectedObject] = []
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            if cls_id == 0:
                person_count += 1
            elif cls_id in SUSPICIOUS_OBJECTS:
                suspicious.append(DetectedObject(
                    label=SUSPICIOUS_OBJECTS[cls_id],
                    confidence=round(conf, 3),
                    bbox=bbox,
                ))

    is_violation = len(suspicious) > 0
    violation_details = None
    if suspicious:
        labels = [obj.label for obj in suspicious]
        violation_details = f"Detected: {', '.join(labels)}"

    return ObjectDetectionResponse(
        suspicious_objects=suspicious,
        person_count=person_count,
        is_violation=is_violation,
        violation_details=violation_details,
    )


@app.post("/api/proctor/detect-objects", response_model=ObjectDetectionResponse)
async def detect_objects(request: ObjectDetectionRequest):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(_sync_detect_objects, request.image, request.confidence_threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")


# ----------------------------
# Proctoring: Head Pose Detection (MediaPipe FaceMesh)
# ----------------------------
class HeadPoseRequest(BaseModel):
    image: str  # base64-encoded JPEG frame


class HeadPoseResponse(BaseModel):
    face_count: int
    head_pose: Optional[Dict] = None  # yaw, pitch, roll in degrees
    looking_away: bool
    looking_direction: str  # "center", "left", "right", "up", "down"
    eyes_closed: bool
    is_violation: bool
    violation_type: Optional[str] = None  # "NO_FACE", "MULTIPLE_FACES", "LOOKING_AWAY", "EYES_CLOSED"
    violation_details: Optional[str] = None


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> Dict:
    """Estimate head pose (yaw, pitch, roll) from MediaPipe FaceMesh landmarks."""
    # Key landmark indices for pose estimation
    # Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
    face_3d = []
    face_2d = []

    key_indices = [1, 33, 61, 199, 263, 291]

    for idx in key_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z * 3000])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix (approximate)
    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    yaw = angles[1] * 360    # left/right
    pitch = angles[0] * 360  # up/down
    roll = angles[2] * 360   # tilt

    return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}


def detect_eyes_closed(landmarks) -> bool:
    """Detect if eyes are closed using Eye Aspect Ratio (EAR)."""
    def ear(eye_indices):
        pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
        # Vertical distances
        v1 = math.dist(pts[1], pts[5])
        v2 = math.dist(pts[2], pts[4])
        # Horizontal distance
        h = math.dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    # MediaPipe FaceMesh eye landmark indices
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    left_ear = ear(left_eye)
    right_ear = ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear < 0.18  # threshold for closed eyes


def _sync_detect_head_pose(image_b64: str) -> HeadPoseResponse:
    """Run blocking MediaPipe inference in a worker thread."""
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame

    rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    face_results = face_detector.process(rgb)
    face_count = len(face_results.detections) if face_results.detections else 0

    if face_count == 0:
        return HeadPoseResponse(
            face_count=0, looking_away=False, looking_direction="none",
            eyes_closed=False, is_violation=True, violation_type="NO_FACE",
            violation_details="No face detected in frame",
        )

    if face_count > 1:
        return HeadPoseResponse(
            face_count=face_count, looking_away=False, looking_direction="center",
            eyes_closed=False, is_violation=True, violation_type="MULTIPLE_FACES",
            violation_details=f"{face_count} faces detected",
        )

    mesh_results = face_mesh.process(rgb)

    if not mesh_results.multi_face_landmarks:
        return HeadPoseResponse(
            face_count=1, looking_away=False, looking_direction="center",
            eyes_closed=False, is_violation=False,
        )

    landmarks = mesh_results.multi_face_landmarks[0].landmark
    head_pose = estimate_head_pose(landmarks, img_w, img_h)
    eyes_closed = detect_eyes_closed(landmarks)

    yaw = head_pose["yaw"]
    pitch = head_pose["pitch"]
    looking_direction = "center"
    looking_away = False

    if yaw < -15:
        looking_direction = "left"
        looking_away = True
    elif yaw > 15:
        looking_direction = "right"
        looking_away = True
    if pitch < -15:
        looking_direction = "down"
        looking_away = True
    elif pitch > 20:
        looking_direction = "up"
        looking_away = True

    is_violation = looking_away or eyes_closed
    violation_type = None
    violation_details = None

    if looking_away:
        violation_type = "LOOKING_AWAY"
        violation_details = f"Looking {looking_direction} (yaw={yaw:.0f}, pitch={pitch:.0f})"
    elif eyes_closed:
        violation_type = "EYES_CLOSED"
        violation_details = "Eyes appear to be closed"

    return HeadPoseResponse(
        face_count=face_count, head_pose=head_pose, looking_away=looking_away,
        looking_direction=looking_direction, eyes_closed=eyes_closed,
        is_violation=is_violation, violation_type=violation_type,
        violation_details=violation_details,
    )


@app.post("/api/proctor/head-pose", response_model=HeadPoseResponse)
async def detect_head_pose(request: HeadPoseRequest):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(_sync_detect_head_pose, request.image))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Head pose detection failed: {str(e)}")


# ----------------------------
# Proctoring: Combined Analysis (single endpoint for efficiency)
# ----------------------------
class FullProctorRequest(BaseModel):
    image: str  # base64-encoded JPEG frame
    run_object_detection: bool = True  # skip YOLO on some frames for perf
    confidence_threshold: Optional[float] = 0.6


class FullProctorResponse(BaseModel):
    # Face / Head Pose
    face_count: int
    head_pose: Optional[Dict] = None
    looking_away: bool
    looking_direction: str
    eyes_closed: bool
    # Object Detection
    suspicious_objects: List[DetectedObject]
    person_count: int
    # Violations
    violations: List[Dict]  # [{type, details, weight}]
    total_violation_weight: int


def _sync_full_proctor_analyze(image_b64: str, run_object_detection: bool, confidence_threshold: float) -> FullProctorResponse:
    """Run blocking combined proctoring analysis in a worker thread."""
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    violations: List[Dict] = []

    face_results = face_detector.process(rgb)
    face_count = len(face_results.detections) if face_results.detections else 0

    head_pose = None
    looking_away = False
    looking_direction = "none"
    eyes_closed = False

    if face_count == 0:
        violations.append({"type": "NO_FACE", "details": "No face detected", "weight": 1})
    elif face_count > 1:
        violations.append({"type": "MULTIPLE_FACES", "details": f"{face_count} faces detected", "weight": 2})
    else:
        mesh_results = face_mesh.process(rgb)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            head_pose = estimate_head_pose(landmarks, img_w, img_h)
            eyes_closed = detect_eyes_closed(landmarks)

            yaw = head_pose["yaw"]
            pitch = head_pose["pitch"]
            looking_direction = "center"

            if yaw < -15:
                looking_direction = "left"
                looking_away = True
            elif yaw > 15:
                looking_direction = "right"
                looking_away = True
            if pitch < -15:
                looking_direction = "down"
                looking_away = True
            elif pitch > 20:
                looking_direction = "up"
                looking_away = True

            if looking_away:
                violations.append({
                    "type": "LOOKING_AWAY",
                    "details": f"Looking {looking_direction} (yaw={yaw:.0f}, pitch={pitch:.0f})",
                    "weight": 1,
                })

    suspicious: List[DetectedObject] = []
    person_count = 0

    if run_object_detection:
        results = yolo_model(
            frame_bgr,
            device=DEVICE,
            conf=confidence_threshold,
            verbose=False,
            half=(DEVICE == "cuda"),
        )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if cls_id == 0:
                    person_count += 1
                elif cls_id in SUSPICIOUS_OBJECTS:
                    label = SUSPICIOUS_OBJECTS[cls_id]
                    suspicious.append(DetectedObject(
                        label=label,
                        confidence=round(conf, 3),
                        bbox=bbox,
                    ))
                    weight = 3 if label == "cell phone" else 2
                    violations.append({
                        "type": "CHEATING_OBJECT",
                        "details": f"Detected: {label} ({conf:.0%})",
                        "weight": weight,
                    })

    total_weight = sum(v["weight"] for v in violations)

    return FullProctorResponse(
        face_count=face_count,
        head_pose=head_pose,
        looking_away=looking_away,
        looking_direction=looking_direction,
        eyes_closed=eyes_closed,
        suspicious_objects=suspicious,
        person_count=person_count,
        violations=violations,
        total_violation_weight=total_weight,
    )


@app.post("/api/proctor/analyze", response_model=FullProctorResponse)
async def full_proctor_analyze(request: FullProctorRequest):
    """Combined proctoring analysis - face detection + head pose + optional object detection.
    Call this every frame for face/pose, but set run_object_detection=True only every 5th frame."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(_sync_full_proctor_analyze, request.image, request.run_object_detection, request.confidence_threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proctor analysis failed: {str(e)}")


@app.get("/api/proctor/health")
async def proctor_health():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "yolo_model": "yolov8m",
        "mediapipe": True,
    }


if __name__ == "__main__":
    import uvicorn
    import sys

    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    uvicorn.run(app, host="0.0.0.0", port=port)
