# ============================================================
# backend/app.py
# Real Sign-to-Speech backend for GVIT 5.0
# ============================================================

import os
import tempfile
import pickle
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import mediapipe as mp

from transformers import ViTImageProcessor, ViTModel

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFERENCE_PATH = os.path.join(
    BASE_DIR,
    "features_5.0",
    "features_5.0",
    "inference_package_5_0.pkl"
)


# ============================================================
# Constants from your demo.py
# ============================================================

VIT_DIM = 768
POSE_DIM = 99
HAND_DIM = 126
MOTION_DIM = 768
MASK_DIM = 2

INPUT_DIM = VIT_DIM + POSE_DIM + HAND_DIM + MOTION_DIM + MASK_DIM

MOTION_START = VIT_DIM + POSE_DIM + HAND_DIM
MOTION_END = MOTION_START + MOTION_DIM

START_THRESHOLD = 0.02
END_THRESHOLD = 0.015
MIN_SIGN_FRAMES = 12
MAX_SIGN_FRAMES = 48

WINDOW_SIZE = 16
WINDOW_STRIDE = 4
MIN_VOTES = 1
VOTE_THRESHOLD = 0.30

TOP_K = 5
CONFIDENCE_THRESHOLD = 0.02


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# Model architecture
# ============================================================

class SignTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        num_classes,
        max_len,
        dropout,
        dim_ff
    ):
        super().__init__()

        appearance_dim = input_dim - MOTION_DIM

        self.appearance_proj = nn.Sequential(
            nn.Linear(appearance_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        self.motion_proj = nn.Sequential(
            nn.Linear(MOTION_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        hidden = d_model * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        appearance = torch.cat([
            x[:, :, :MOTION_START],
            x[:, :, MOTION_END:]
        ], dim=-1)

        motion = x[:, :, MOTION_START:MOTION_END]

        fused = self.appearance_proj(appearance) + self.motion_proj(motion)
        fused = fused + self.pos_embedding[:, :fused.size(1), :]

        out = self.transformer(fused)

        pooled = out.mean(dim=1) + out.max(dim=1).values

        return self.classifier(pooled)


# ============================================================
# Load model package
# ============================================================

print("=" * 60)
print("Loading GVIT 5.0 inference package...")
print("Path:", INFERENCE_PATH)

if not os.path.exists(INFERENCE_PATH):
    raise FileNotFoundError(f"Inference package not found: {INFERENCE_PATH}")

with open(INFERENCE_PATH, "rb") as f:
    pkg = pickle.load(f)

idx_to_class = pkg["idx_to_class"]
mean = pkg["mean"].astype(np.float32)
std = pkg["std"].astype(np.float32)
cfg = pkg["model_config"]

model = SignTransformer(**cfg).to(device)
model.load_state_dict(pkg["model_state_dict"])
model.eval()

print("Classes:", len(idx_to_class))
print("Best val acc:", pkg.get("best_val_acc", "N/A"))
print("Best epoch:", pkg.get("best_epoch", "N/A"))
print("Device:", device)
print("GVIT model loaded OK")


# ============================================================
# Load ViT
# ============================================================

print("Loading ViT...")

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

vit = ViTModel.from_pretrained(
    "google/vit-base-patch16-224"
).to(device)

vit.eval()

for p in vit.parameters():
    p.requires_grad = False

print("ViT loaded OK")


# ============================================================
# Load MediaPipe
# ============================================================

print("Loading MediaPipe...")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0
)

print("MediaPipe loaded OK")
print("=" * 60)


# ============================================================
# Frame processing
# ============================================================

def process_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)

    pose_results = pose_model.process(rgb)

    if pose_results.pose_landmarks:
        pose_feat = np.array([
            [lm.x, lm.y, lm.z]
            for lm in pose_results.pose_landmarks.landmark
        ]).flatten().astype(np.float32)

        pose_detected = 1.0
    else:
        pose_feat = np.zeros(POSE_DIM, dtype=np.float32)
        pose_detected = 0.0

    hands_results = hands_model.process(rgb)

    hand_feats = []
    hand_landmarks = None

    if hands_results.multi_hand_landmarks:
        hand_landmarks = []

        for hl in hands_results.multi_hand_landmarks:
            for lm in hl.landmark:
                hand_feats.extend([lm.x, lm.y, lm.z])
                hand_landmarks.append([lm.x, lm.y, lm.z])

        hands_detected = 1.0
    else:
        hands_detected = 0.0

    while len(hand_feats) < HAND_DIM:
        hand_feats.extend([0.0, 0.0, 0.0])

    hand_feat = np.array(hand_feats[:HAND_DIM], dtype=np.float32)

    return (
        pose_feat,
        hand_feat,
        pose_detected,
        hands_detected,
        image_pil,
        hand_landmarks
    )


def compute_hand_velocity(prev_lm, curr_lm):
    if prev_lm is None or curr_lm is None:
        return 0.0

    prev = np.array(prev_lm)
    curr = np.array(curr_lm)

    n = min(len(prev), len(curr))

    if n == 0:
        return 0.0

    return float(np.mean(np.linalg.norm(curr[:n] - prev[:n], axis=1)))


# ============================================================
# Model prediction functions
# ============================================================

def run_window(window):
    images = [item[4] for item in window]

    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        out = vit(inputs["pixel_values"].to(device))

    vit_feats = out.last_hidden_state[:, 0, :].cpu().numpy()

    motion = np.diff(vit_feats, axis=0)

    if len(motion) == 0:
        motion = np.zeros_like(vit_feats)
    else:
        motion = np.concatenate([motion, motion[-1:]], axis=0)

    combined = []

    for i, (pf, hf, pd, hd, _) in enumerate(window):
        combined.append(np.concatenate([
            vit_feats[i],
            pf,
            hf,
            motion[i],
            [pd, hd]
        ]).astype(np.float32))

    seq = np.stack(combined)

    seq = (seq - mean[0]) / std[0]

    x = torch.tensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(probs))
    best_word = idx_to_class.get(best_idx, f"class_{best_idx}")

    return best_word, float(probs[best_idx]), probs


def sliding_window_vote(segment):
    n = len(segment)

    if n == 0:
        return None, 0.0, []

    if n < WINDOW_SIZE:
        segment = segment + [segment[-1]] * (WINDOW_SIZE - n)
        n = WINDOW_SIZE

    window_preds = []
    all_probs = []

    for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window = segment[start:start + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE:
            break

        word, conf, probs = run_window(window)

        window_preds.append((word, conf))
        all_probs.append(probs)

    if len(window_preds) < MIN_VOTES:
        return None, 0.0, []

    word_votes = Counter(w for w, _ in window_preds)
    total = len(window_preds)

    top_word, top_count = word_votes.most_common(1)[0]

    if top_count / total < VOTE_THRESHOLD:
        return None, 0.0, []

    avg_conf = float(np.mean([
        c for w, c in window_preds if w == top_word
    ]))

    if avg_conf < CONFIDENCE_THRESHOLD:
        return None, 0.0, []

    avg_probs = np.mean(all_probs, axis=0)

    top_k_idx = np.argsort(avg_probs)[::-1][:TOP_K]

    top_k = [
        {
            "word": idx_to_class.get(int(i), f"class_{i}"),
            "confidence": float(avg_probs[i])
        }
        for i in top_k_idx
    ]

    return top_word, avg_conf, top_k


def convert_to_english(words):
    if not words:
        return ""

    words = [w.lower() for w in words]

    deduped = [words[0]]

    for w in words[1:]:
        if w != deduped[-1]:
            deduped.append(w)

    pronoun_be = {
        "i": "am",
        "he": "is",
        "she": "is",
        "it": "is",
        "we": "are",
        "they": "are",
        "you": "are"
    }

    tokens = deduped[:]
    result = []
    i = 0

    while i < len(tokens):
        result.append(tokens[i])

        if (
            i + 1 < len(tokens)
            and tokens[i] in pronoun_be
            and tokens[i + 1] not in {
                "am", "is", "are", "was", "were",
                "have", "has", "do", "does",
                "will", "can", "could", "would", "not"
            }
        ):
            result.append(pronoun_be[tokens[i]])

        i += 1

    sentence = " ".join(result)

    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    q_words = {"what", "where", "when", "who", "why", "how", "which"}

    if deduped[0] in q_words or deduped[-1] in q_words:
        sentence = sentence.rstrip(".!?") + "?"
    elif sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence


# ============================================================
# Video prediction
# ============================================================

def predict_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    state = "IDLE"
    segment = []
    all_processed_frames = []
    prev_hand_lm = None

    final_word = None
    final_conf = 0.0
    final_top_k = []

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        (
            pose_feat,
            hand_feat,
            pose_detected,
            hands_detected,
            image_pil,
            hand_lm
        ) = process_frame(frame)

        item = (
            pose_feat,
            hand_feat,
            pose_detected,
            hands_detected,
            image_pil
        )

        all_processed_frames.append(item)

        velocity = compute_hand_velocity(prev_hand_lm, hand_lm)
        prev_hand_lm = hand_lm

        if state == "IDLE":
            if velocity > START_THRESHOLD:
                state = "SIGNING"
                segment = []

        if state == "SIGNING":
            segment.append(item)

            if (
                velocity < END_THRESHOLD and len(segment) >= MIN_SIGN_FRAMES
            ) or len(segment) >= MAX_SIGN_FRAMES:
                word, conf, top_k = sliding_window_vote(segment)

                if word is not None:
                    final_word = word
                    final_conf = conf
                    final_top_k = top_k

                segment = []
                state = "IDLE"

        if frame_count > 250:
            break

    cap.release()

    # If movement segmentation did not catch a valid sign,
    # use the whole video as fallback.
    if final_word is None:
        fallback_segment = all_processed_frames[:MAX_SIGN_FRAMES]

        if len(fallback_segment) == 0:
            raise RuntimeError("No readable frames found in video.")

        word, conf, top_k = sliding_window_vote(fallback_segment)

        if word is None:
            raise RuntimeError("No confident sign prediction found.")

        final_word = word
        final_conf = conf
        final_top_k = top_k

    word_upper = str(final_word).upper()

    english = convert_to_english([word_upper])

    return {
        "word": word_upper,
        "confidence": float(final_conf),
        "english": english,
        "top_predictions": final_top_k
    }


# ============================================================
# API routes
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Backend is running",
        "message": "Real GVIT 5.0 backend ready",
        "classes": len(idx_to_class),
        "device": str(device)
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    if not suffix:
        suffix = ".webm"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            content = await file.read()
            temp.write(content)
            temp_path = temp.name

        result = predict_video_file(temp_path)

        return result

    except Exception as error:
        return {
            "error": str(error)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)