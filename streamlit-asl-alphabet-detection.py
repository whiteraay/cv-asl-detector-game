import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import random
import os
from PIL import Image
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Learning Game",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (ORIGINAL UI) ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0f; color: #f0ede8; }
.stApp { background: #0d0d0f; }
.game-title { font-family: 'Bebas Neue', sans-serif; font-size: 3.6rem; letter-spacing: 0.12em; background: linear-gradient(135deg, #f5c842 0%, #ff6b35 60%, #e84393 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; line-height: 1; }
.game-subtitle { font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.25em; color: #555; text-transform: uppercase; margin-top: 4px; margin-bottom: 0; }
.target-card { background: linear-gradient(135deg, #1a1a1f 0%, #111115 100%); border: 1px solid #2a2a35; border-radius: 16px; padding: 28px 32px; text-align: center; position: relative; overflow: hidden; }
.target-card::before { content: ''; position: absolute; top: -1px; left: -1px; right: -1px; height: 3px; background: linear-gradient(90deg, #f5c842, #ff6b35, #e84393); border-radius: 16px 16px 0 0; }
.target-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.3em; color: #666; text-transform: uppercase; margin-bottom: 8px; }
.target-letter { font-family: 'Bebas Neue', sans-serif; font-size: 7rem; line-height: 1; background: linear-gradient(135deg, #f5c842, #ff6b35); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.stat-card { background: #111115; border: 1px solid #222228; border-radius: 12px; padding: 20px; text-align: center; }
.stat-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; color: #f5c842; line-height: 1; }
.stat-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 0.25em; color: #555; text-transform: uppercase; margin-top: 4px; }
.status-correct { background: rgba(0, 220, 100, 0.15); border: 1px solid rgba(0, 220, 100, 0.4); color: #00dc64; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em; padding: 6px 16px; border-radius: 20px; display: inline-block; text-transform: uppercase; }
.status-wrong { background: rgba(255, 60, 60, 0.12); border: 1px solid rgba(255, 60, 60, 0.3); color: #ff5555; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em; padding: 6px 16px; border-radius: 20px; display: inline-block; text-transform: uppercase; }
.status-waiting { background: rgba(150, 150, 180, 0.1); border: 1px solid rgba(150, 150, 180, 0.2); color: #888; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em; padding: 6px 16px; border-radius: 20px; display: inline-block; text-transform: uppercase; }
.progress-wrap { background: #1e1e26; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; margin-top: 8px; }
.progress-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #00dc64, #f5c842); transition: width 0.15s ease; }
[data-testid="stSidebar"] { background: #0a0a0d !important; border-right: 1px solid #1e1e26; }
.stButton > button { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; border-radius: 8px !important; border: 1px solid #333 !important; background: #1a1a1f !important; color: #f0ede8 !important; padding: 10px 24px !important; }
.detected-letter { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #f0ede8; }
.accuracy-number { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Logic ──────────────────────────────────────────────────────────────────
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def next_letter():
    st.session_state.target_letter = random.choice(LABELS)
    st.session_state.hold_counter = 0

if "target_letter" not in st.session_state:
    st.session_state.target_letter = random.choice(LABELS)
    st.session_state.hold_counter = 0
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.best_streak = 0
    st.session_state.running = False

# Camera resources stored in session state so they survive reruns
# This is the KEY fix — prevents camera restart when Next is clicked
if "cap" not in st.session_state:
    st.session_state.cap = None
if "detector" not in st.session_state:
    st.session_state.detector = None
if "classifier" not in st.session_state:
    st.session_state.classifier = None

OFFSET = 20
IMG_SIZE = 300

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_path   = st.text_input("Model path",    value="/Users/aknur/Desktop/CVision/asl-cv/converted_keras/keras_model.h5")
    labels_path  = st.text_input("Labels path",   value="/Users/aknur/Desktop/CVision/asl-cv/converted_keras/labels.txt")
    signs_dir    = st.text_input("Signs folder",  value="/Users/aknur/Desktop/CVision/asl-cv/signs-imgs")
    hold_frames_req = st.slider("Hold frames", 5, 40, 20)

# ─── Header ──────────────────────────────────────────────────────────────────
col_title, col_controls = st.columns([3, 1])
with col_title:
    st.markdown('<p class="game-title">ASL Learning Game</p>', unsafe_allow_html=True)
    st.markdown('<p class="game-subtitle">American Sign Language · Real-time Recognition</p>', unsafe_allow_html=True)

with col_controls:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶ Start" if not st.session_state.running else "⏹ Stop", use_container_width=True):
            st.session_state.running = not st.session_state.running
            if not st.session_state.running:
                # Release camera when stopped
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.session_state.detector = None
                st.session_state.classifier = None
            st.rerun()
    with btn_col2:
        # on_click only sets state — no camera restart needed since cap is in session_state
        st.button("⏭ Next", on_click=next_letter, use_container_width=True)

# ─── Layout ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")
with left_col:
    camera_placeholder = st.empty()

with right_col:
    target_placeholder = st.empty()
    s1, s2, s3 = st.columns(3)
    score_ph, streak_ph, best_ph = s1.empty(), s2.empty(), s3.empty()
    detection_ph, progress_ph, status_ph, hint_ph = st.empty(), st.empty(), st.empty(), st.empty()

# ─── UI Update Helper ────────────────────────────────────────────────────────
def refresh_ui(label, accuracy, is_correct):
    target = st.session_state.target_letter
    target_placeholder.markdown(
        f"<div class='target-card'><div class='target-label'>Target Letter</div>"
        f"<div class='target-letter'>{target}</div></div>",
        unsafe_allow_html=True
    )
    score_ph.markdown(
        f"<div class='stat-card'><div class='stat-value'>{st.session_state.score}</div>"
        f"<div class='stat-label'>Score</div></div>", unsafe_allow_html=True
    )
    streak_ph.markdown(
        f"<div class='stat-card'><div class='stat-value' style='color:#ff6b35'>{st.session_state.streak}</div>"
        f"<div class='stat-label'>Streak 🔥</div></div>", unsafe_allow_html=True
    )
    best_ph.markdown(
        f"<div class='stat-card'><div class='stat-value' style='color:#e84393'>{st.session_state.best_streak}</div>"
        f"<div class='stat-label'>Best</div></div>", unsafe_allow_html=True
    )

    if label:
        color = "#00dc64" if is_correct else "#ff5555"
        detection_ph.markdown(
            f"<div style='background:#111115; border:1px solid #222228; border-radius:12px; padding:20px; "
            f"display:flex; align-items:center; justify-content:space-between;'>"
            f"<div><div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#555; text-transform:uppercase;'>Detected</div>"
            f"<span class='detected-letter' style='color:{color}'>{label}</span></div>"
            f"<div style='text-align:right;'><div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#555; text-transform:uppercase;'>Accuracy</div>"
            f"<span class='accuracy-number' style='color:{color}'>{accuracy}%</span></div></div>",
            unsafe_allow_html=True
        )
    else:
        detection_ph.markdown(
            "<div style='background:#111115; border:1px solid #222228; border-radius:12px; "
            "padding:20px; text-align:center; color:#444;'>NO HAND DETECTED</div>",
            unsafe_allow_html=True
        )

    pct = int((st.session_state.hold_counter / hold_frames_req) * 100)
    progress_ph.markdown(
        f"<div style='margin-top:8px;'><div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#444;'>"
        f"HOLD PROGRESS — {pct}%</div><div class='progress-wrap'>"
        f"<div class='progress-fill' style='width:{pct}%'></div></div></div>",
        unsafe_allow_html=True
    )

    status_html = (
        "<span class='status-correct'>✓ Correct!</span>" if is_correct
        else ("<span class='status-wrong'>✗ Wrong</span>" if label
              else "<span class='status-waiting'>Waiting…</span>")
    )
    status_ph.markdown(f"<div style='margin-top:10px'>{status_html}</div>", unsafe_allow_html=True)

    # Hint image
    found = False
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(signs_dir, f"{target}{ext}")
        if os.path.exists(p):
            hint_ph.image(p, caption=f"How to sign: {target}", width=180)
            found = True
            break
    if not found:
        hint_ph.info(f"No hint image for {target}")


# ─── Draw CV Overlay on frame (mirrors original script) ─────────────────────
def draw_cv_overlay(imgOutput, hand_bbox, label, accuracy, is_correct, hold_counter, hold_frames_req):
    """Draws bounding box, label badge, and hold progress bar directly on the frame."""
    x, y, w, h = hand_bbox
    color = (0, 255, 0) if is_correct else (0, 0, 255)

    # Bounding box
    cv2.rectangle(
        imgOutput,
        (x - OFFSET, y - OFFSET),
        (x + w + OFFSET, y + h + OFFSET),
        color, 3
    )

    # Label badge above bounding box
    label_text = f"{label} ({accuracy}%)"
    badge_x1 = x - OFFSET
    badge_y1 = y - OFFSET - 40
    badge_x2 = x - OFFSET + max(200, len(label_text) * 14)
    badge_y2 = y - OFFSET
    cv2.rectangle(imgOutput, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
    cv2.putText(
        imgOutput, label_text,
        (badge_x1 + 6, badge_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2
    )

    # Hold progress bar below bounding box
    bar_x1 = x - OFFSET
    bar_x2 = x + w + OFFSET
    bar_y  = y + h + OFFSET + 12
    fill   = int((hold_counter / hold_frames_req) * (bar_x2 - bar_x1))
    cv2.rectangle(imgOutput, (bar_x1, bar_y), (bar_x2, bar_y + 8), (60, 60, 60), -1)
    if fill > 0:
        cv2.rectangle(imgOutput, (bar_x1, bar_y), (bar_x1 + fill, bar_y + 8), (0, 255, 0), -1)


# ─── Camera Loop ─────────────────────────────────────────────────────────────
if st.session_state.running:

    # ── Initialise resources only once (persist across reruns) ──────────────
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        st.session_state.cap = cap

    if st.session_state.detector is None:
        st.session_state.detector = HandDetector(maxHands=1)

    if st.session_state.classifier is None:
        st.session_state.classifier = Classifier(model_path, labels_path)

    cap        = st.session_state.cap
    detector   = st.session_state.detector
    classifier = st.session_state.classifier

    while st.session_state.running:
        success, img = cap.read()
        if not success:
            break

        img       = cv2.flip(img, 1)
        imgOutput = img.copy()

        results = detector.findHands(img, draw=False)
        hands   = results[0] if isinstance(results, tuple) else results

        label, acc, is_correct = "", 0, False

        if hands:
            hand       = hands[0]
            x, y, w, h = hand['bbox']
            y1 = max(0, y - OFFSET);         y2 = min(img.shape[0], y + h + OFFSET)
            x1 = max(0, x - OFFSET);         x2 = min(img.shape[1], x + w + OFFSET)
            imgCrop    = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                if h / w > 1:
                    k   = IMG_SIZE / h
                    wC  = math.ceil(k * w)
                    imgR = cv2.resize(imgCrop, (wC, IMG_SIZE))
                    gap  = math.ceil((IMG_SIZE - wC) / 2)
                    imgWhite[:, gap:wC + gap] = imgR
                else:
                    k   = IMG_SIZE / w
                    hC  = math.ceil(k * h)
                    imgR = cv2.resize(imgCrop, (IMG_SIZE, hC))
                    gap  = math.ceil((IMG_SIZE - hC) / 2)
                    imgWhite[gap:hC + gap, :] = imgR

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label      = LABELS[index]
                acc        = int(prediction[index] * 100)
                is_correct = (label == st.session_state.target_letter)

                if is_correct:
                    st.session_state.hold_counter += 1
                    if st.session_state.hold_counter >= hold_frames_req:
                        st.session_state.score      += 10
                        st.session_state.streak     += 1
                        st.session_state.best_streak = max(
                            st.session_state.best_streak, st.session_state.streak
                        )
                        next_letter()
                else:
                    st.session_state.hold_counter = 0
                    if st.session_state.streak > 0:
                        st.session_state.streak = 0

                # ── CV overlay: bounding box + label badge + progress bar ──
                draw_cv_overlay(
                    imgOutput,
                    (x, y, w, h),
                    label, acc, is_correct,
                    st.session_state.hold_counter,
                    hold_frames_req
                )

        camera_placeholder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), channels="RGB")
        refresh_ui(label, acc, is_correct)
        time.sleep(0.01)

else:
    # Release camera resources when not running
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap       = None
        st.session_state.detector  = None
        st.session_state.classifier = None

    camera_placeholder.info("Camera Stopped. Press Start.")
    refresh_ui("", 0, False)