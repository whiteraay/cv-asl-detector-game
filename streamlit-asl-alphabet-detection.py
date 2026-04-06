"""
ASL Learning Game — Professional Edition
Brutalist-terminal aesthetic · Real ML evaluation · Streamlit

Changes vs previous version:
  • START / NEXT buttons moved inside 01·GAME tab
  • Evaluation results saved to asl_eval_cache.pkl next to this script
  • On next open the cached result loads instantly — no re-run needed
  • "⟳ RE-RUN" overwrites cache; "🗑 CLEAR CACHE" wipes it
"""

import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import random
import os
import time
import pickle
import datetime
import h5py
import json

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL·ML",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS — Brutalist Terminal ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
:root{
  --bg:#080808; --bg1:#0e0e0e; --bg2:#141414; --bg3:#1c1c1c;
  --border:#2a2a2a; --border2:#3a3a3a;
  --green:#00ff88; --green-dim:#00cc6a; --green-xs:rgba(0,255,136,0.07);
  --amber:#ffb800; --red:#ff3355; --text:#e8e8e8;
  --text-dim:#666; --text-xs:#444;
  --mono:'IBM Plex Mono',monospace; --display:'Barlow Condensed',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:var(--mono);background-color:var(--bg);color:var(--text);}
.stApp{background:var(--bg);}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none;}
#MainMenu,footer,header{visibility:hidden;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}

/* masthead */
.masthead{display:flex;align-items:stretch;border-bottom:1px solid var(--border);background:var(--bg);}
.masthead-title{padding:18px 32px;border-right:1px solid var(--border);flex-shrink:0;}
.masthead-title h1{font-family:var(--display);font-weight:900;font-size:2.6rem;letter-spacing:.05em;
  color:var(--green);margin:0;line-height:1;text-shadow:0 0 30px rgba(0,255,136,.3);}
.masthead-title p{font-size:.6rem;letter-spacing:.35em;color:var(--text-xs);text-transform:uppercase;margin:3px 0 0;}
.masthead-meta{display:flex;gap:0;flex:1;}
.meta-cell{padding:10px 24px;border-right:1px solid var(--border);display:flex;flex-direction:column;justify-content:center;}
.meta-cell .k{font-size:.52rem;letter-spacing:.3em;color:var(--text-xs);text-transform:uppercase;}
.meta-cell .v{font-size:.85rem;font-weight:600;color:var(--text);margin-top:2px;}
.meta-cell .v.green{color:var(--green);}
.meta-cell .v.amber{color:var(--amber);}

/* panels */
.panel{background:var(--bg1);border:1px solid var(--border);border-radius:2px;overflow:hidden;}
.panel-header{padding:10px 18px;border-bottom:1px solid var(--border);font-size:.58rem;
  letter-spacing:.3em;text-transform:uppercase;color:var(--text-dim);display:flex;align-items:center;gap:8px;}
.panel-header .dot{width:6px;height:6px;border-radius:50%;background:var(--green);
  box-shadow:0 0 6px var(--green);flex-shrink:0;}
.panel-body{padding:18px;}

/* game target */
.target-wrap{text-align:center;padding:36px 20px;}
.target-letter-display{font-family:var(--display);font-size:11rem;font-weight:900;line-height:1;
  color:var(--green);text-shadow:0 0 60px rgba(0,255,136,.25),0 0 120px rgba(0,255,136,.1);letter-spacing:-.02em;}
.target-sub{font-size:.55rem;letter-spacing:.4em;color:var(--text-xs);text-transform:uppercase;margin-top:6px;}

/* kpi strip */
.kpi-row{display:flex;gap:1px;background:var(--border);border:1px solid var(--border);}
.kpi-cell{flex:1;background:var(--bg1);padding:16px 20px;text-align:center;}
.kpi-cell .val{font-family:var(--display);font-size:3.2rem;font-weight:700;line-height:1;color:var(--green);}
.kpi-cell .lbl{font-size:.52rem;letter-spacing:.3em;color:var(--text-xs);text-transform:uppercase;margin-top:4px;}

/* detection box */
.det-box{border:1px solid var(--border);padding:14px 18px;
  display:flex;align-items:center;justify-content:space-between;background:var(--bg2);}
.det-letter{font-family:var(--display);font-size:3.8rem;font-weight:900;line-height:1;}
.det-conf{text-align:right;}
.det-conf .pct{font-family:var(--display);font-size:2rem;font-weight:700;}
.det-conf .pct-lbl{font-size:.52rem;letter-spacing:.3em;color:var(--text-dim);}

/* hold bar */
.hold-track{height:3px;background:var(--bg3);margin-top:10px;overflow:hidden;}
.hold-fill{height:100%;background:var(--green);box-shadow:0 0 8px var(--green);transition:width .05s linear;}
.hold-label{font-size:.52rem;letter-spacing:.3em;color:var(--text-xs);text-transform:uppercase;
  margin-bottom:4px;display:flex;justify-content:space-between;}

/* pills */
.pill{display:inline-block;padding:4px 14px;font-size:.6rem;letter-spacing:.25em;
  text-transform:uppercase;border-radius:1px;}
.pill-ok{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.3);}
.pill-err{background:rgba(255,51,85,.1);color:var(--red);border:1px solid rgba(255,51,85,.25);}
.pill-wait{background:rgba(100,100,100,.1);color:var(--text-dim);border:1px solid var(--border);}

/* info grid */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);}
.info-cell{background:var(--bg1);padding:14px 18px;}
.info-cell .k{font-size:.52rem;letter-spacing:.28em;color:var(--text-xs);text-transform:uppercase;margin-bottom:5px;}
.info-cell .v{font-size:.82rem;color:var(--text);word-break:break-all;}
.info-cell .v.mono{font-family:var(--mono);font-size:.72rem;color:var(--green-dim);}

/* per-class bars */
.pc-row{display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid var(--border);}
.pc-row:last-child{border-bottom:none;}
.pc-lbl{font-size:.75rem;font-weight:600;width:22px;text-align:center;color:var(--text);}
.pc-track{flex:1;height:8px;background:var(--bg3);}
.pc-fill{height:100%;background:var(--green);}
.pc-fill.low{background:var(--red);}
.pc-fill.mid{background:var(--amber);}
.pc-pct{font-size:.65rem;width:44px;text-align:right;color:var(--text-dim);}
.pc-n{font-size:.55rem;width:36px;text-align:right;color:var(--text-xs);}

/* buttons */
.stButton>button{
  font-family:var(--mono)!important;font-size:.62rem!important;letter-spacing:.25em!important;
  text-transform:uppercase!important;border-radius:1px!important;border:1px solid var(--border2)!important;
  background:var(--bg2)!important;color:var(--text)!important;padding:9px 22px!important;transition:all .12s!important;
}
.stButton>button:hover{border-color:var(--green)!important;color:var(--green)!important;background:var(--green-xs)!important;}

/* tabs */
[data-testid="stTabs"] button{
  font-family:var(--mono)!important;font-size:.62rem!important;
  letter-spacing:.25em!important;text-transform:uppercase!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--green)!important;}

/* expander */
[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:1px!important;background:var(--bg1)!important;}

/* inputs */
div[data-testid="stTextInput"]>div>div>input{
  background:var(--bg2)!important;border:1px solid var(--border2)!important;border-radius:1px!important;
  color:var(--green-dim)!important;font-size:.72rem!important;font-family:var(--mono)!important;}
label{font-size:.62rem!important;letter-spacing:.2em!important;text-transform:uppercase!important;
  color:var(--text-dim)!important;font-family:var(--mono)!important;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
LABELS   = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OFFSET   = 20
IMG_SIZE = 300

# Cache file lives next to this script
EVAL_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_eval_cache.pkl")

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in dict(
    target_letter=random.choice(LABELS),
    hold_counter=0, score=0, streak=0, best_streak=0,
    running=False, cap=None, detector=None, classifier=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

def next_letter():
    st.session_state.target_letter = random.choice(LABELS)
    st.session_state.hold_counter  = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### ⚙ Paths")
    model_path      = st.text_input("Model (.h5)",   value="/Users/aknur/Desktop/CVision/asl-cv/converted_keras/keras_model.h5")
    labels_path     = st.text_input("Labels (.txt)", value="/Users/aknur/Desktop/CVision/asl-cv/converted_keras/labels.txt")
    signs_dir       = st.text_input("Signs folder (hint images)", value="/Users/aknur/Desktop/CVision/asl-cv/signs-imgs")
    eval_data_dir   = st.text_input("Eval dataset (sub-folders)", value="/Users/aknur/Desktop/CVision/asl-cv/data")
    hold_frames_req = st.slider("Hold frames", 5, 40, 20)

# ─── Masthead ─────────────────────────────────────────────────────────────────
run_color = "green" if st.session_state.running else "amber"
st.markdown(f"""
<div class="masthead">
  <div class="masthead-title"><h1>ASL·ML</h1><p>Sign language recognition system</p></div>
  <div class="masthead-meta">
    <div class="meta-cell"><div class="k">Status</div><div class="v {run_color}">{"LIVE" if st.session_state.running else "IDLE"}</div></div>
    <div class="meta-cell"><div class="k">Score</div><div class="v green">{st.session_state.score}</div></div>
    <div class="meta-cell"><div class="k">Streak</div><div class="v">{st.session_state.streak}</div></div>
    <div class="meta-cell"><div class="k">Best</div><div class="v">{st.session_state.best_streak}</div></div>
    <div class="meta-cell"><div class="k">Target</div><div class="v green">{st.session_state.target_letter}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_game, tab_model, tab_matrix = st.tabs(["01 · GAME", "02 · MODEL", "03 · EVALUATION"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — GAME
#  START / NEXT live inside this tab — they don't appear elsewhere
# ══════════════════════════════════════════════════════════════════════════════
with tab_game:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Control buttons (inside the tab) ─────────────────────────────────
    btn_c1, btn_c2, btn_spacer = st.columns([1, 1, 10])
    with btn_c1:
        if st.button(
            "▶ START" if not st.session_state.running else "⏹ STOP",
            use_container_width=True, key="game_startstop"
        ):
            st.session_state.running = not st.session_state.running
            if not st.session_state.running and st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = st.session_state.detector = st.session_state.classifier = None
            st.rerun()
    with btn_c2:
        st.button("⏭ NEXT", on_click=next_letter, use_container_width=True, key="game_next")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────
    g_left, g_right = st.columns([3, 2], gap="medium")
    with g_left:
        cam_ph = st.empty()
    with g_right:
        target_ph, det_ph, hold_ph, status_ph, hint_ph = (
            st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
        )

    # ── Right-panel renderer ──────────────────────────────────────────────
    def render_right(label, accuracy, is_correct):
        tgt = st.session_state.target_letter
        target_ph.markdown(f"""
        <div class="panel" style="margin-bottom:10px">
          <div class="panel-header"><span class="dot"></span>TARGET LETTER</div>
          <div class="target-wrap">
            <div class="target-letter-display">{tgt}</div>
            <div class="target-sub">Sign this letter</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if label:
            col = "#00ff88" if is_correct else "#ff3355"
            bdr = "rgba(0,255,136,.35)" if is_correct else "rgba(255,51,85,.3)"
            det_ph.markdown(f"""
            <div class="det-box" style="border-color:{bdr};margin-bottom:8px">
              <div>
                <div style="font-size:.52rem;letter-spacing:.3em;color:var(--text-xs);text-transform:uppercase;margin-bottom:4px">Detected</div>
                <div class="det-letter" style="color:{col}">{label}</div>
              </div>
              <div class="det-conf">
                <div class="pct-lbl" style="margin-bottom:3px">Confidence</div>
                <div class="pct" style="color:{col}">{accuracy}%</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            det_ph.markdown("""
            <div class="det-box" style="justify-content:center;margin-bottom:8px;color:var(--text-xs)">
              <span style="font-size:.62rem;letter-spacing:.3em">NO HAND DETECTED</span>
            </div>""", unsafe_allow_html=True)

        pct = int((st.session_state.hold_counter / hold_frames_req) * 100)
        hold_ph.markdown(f"""
        <div style="margin-bottom:10px">
          <div class="hold-label"><span>Hold Progress</span><span>{pct}%</span></div>
          <div class="hold-track"><div class="hold-fill" style="width:{pct}%"></div></div>
        </div>""", unsafe_allow_html=True)

        pill = ("<span class='pill pill-ok'>✓ Correct</span>" if is_correct
                else ("<span class='pill pill-err'>✗ Wrong</span>" if label
                      else "<span class='pill pill-wait'>Waiting…</span>"))
        status_ph.markdown(f"<div style='margin-bottom:12px'>{pill}</div>", unsafe_allow_html=True)

        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(signs_dir, f"{tgt}{ext}")
            if os.path.exists(p):
                hint_ph.image(p, caption=f"Reference: {tgt}", width=160)
                found = True; break
        if not found:
            hint_ph.markdown(
                f"<div style='font-size:.6rem;color:var(--text-xs);letter-spacing:.2em'>NO HINT IMAGE FOR {tgt}</div>",
                unsafe_allow_html=True)

    # ── CV overlay ────────────────────────────────────────────────────────
    def draw_overlay(frame, bbox, label, accuracy, is_correct, hold_c, hold_req):
        x, y, w, h = bbox
        clr = (0, 255, 136) if is_correct else (255, 51, 85)
        cv2.rectangle(frame, (x-OFFSET, y-OFFSET), (x+w+OFFSET, y+h+OFFSET), clr, 2)
        txt = f"{label}  {accuracy}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx1, by1 = x-OFFSET, y-OFFSET-th-14
        cv2.rectangle(frame, (bx1, by1), (bx1+tw+14, by1+th+12), clr, -1)
        cv2.putText(frame, txt, (bx1+7, by1+th+5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (8,8,8), 2)
        bx1b, bx2b = x-OFFSET, x+w+OFFSET
        by_b = y+h+OFFSET+10
        fill = int((hold_c / hold_req) * (bx2b - bx1b))
        cv2.rectangle(frame, (bx1b, by_b), (bx2b, by_b+5), (30,30,30), -1)
        if fill > 0:
            cv2.rectangle(frame, (bx1b, by_b), (bx1b+fill, by_b+5), clr, -1)

    # ── Camera loop ───────────────────────────────────────────────────────
    if st.session_state.running:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            cap = cv2.VideoCapture(0); cap.set(3, 640); cap.set(4, 480)
            st.session_state.cap = cap
        if st.session_state.detector   is None: st.session_state.detector   = HandDetector(maxHands=1)
        if st.session_state.classifier is None: st.session_state.classifier = Classifier(model_path, labels_path)

        cap, detector, classifier = (
            st.session_state.cap, st.session_state.detector, st.session_state.classifier
        )
        while st.session_state.running:
            ok, img = cap.read()
            if not ok: break
            img = cv2.flip(img, 1); output = img.copy()
            res   = detector.findHands(img, draw=False)
            hands = res[0] if isinstance(res, tuple) else res
            label, acc, is_correct = "", 0, False

            if hands:
                x, y, w, h = hands[0]['bbox']
                crop = img[max(0,y-OFFSET):min(img.shape[0],y+h+OFFSET),
                           max(0,x-OFFSET):min(img.shape[1],x+w+OFFSET)]
                if crop.size:
                    white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                    if h / w > 1:
                        k=IMG_SIZE/h; wc=math.ceil(k*w); r=cv2.resize(crop,(wc,IMG_SIZE))
                        g=math.ceil((IMG_SIZE-wc)/2); white[:,g:wc+g]=r
                    else:
                        k=IMG_SIZE/w; hc=math.ceil(k*h); r=cv2.resize(crop,(IMG_SIZE,hc))
                        g=math.ceil((IMG_SIZE-hc)/2); white[g:hc+g,:]=r
                    pred, idx = classifier.getPrediction(white, draw=False)
                    label = LABELS[idx]; acc = int(pred[idx]*100)
                    is_correct = (label == st.session_state.target_letter)
                    if is_correct:
                        st.session_state.hold_counter += 1
                        if st.session_state.hold_counter >= hold_frames_req:
                            st.session_state.score += 10; st.session_state.streak += 1
                            st.session_state.best_streak = max(st.session_state.best_streak, st.session_state.streak)
                            next_letter()
                    else:
                        st.session_state.hold_counter = 0; st.session_state.streak = 0
                    draw_overlay(output, (x,y,w,h), label, acc, is_correct,
                                 st.session_state.hold_counter, hold_frames_req)

            cam_ph.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            render_right(label, acc, is_correct)
            time.sleep(0.01)
    else:
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = st.session_state.detector = st.session_state.classifier = None
        cam_ph.markdown("""
        <div style="border:1px solid var(--border);background:var(--bg1);height:360px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px">
          <div style="font-size:.55rem;letter-spacing:.4em;color:var(--text-xs);text-transform:uppercase">Camera feed</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:4rem;font-weight:900;color:var(--text-xs)">IDLE</div>
          <div style="font-size:.52rem;letter-spacing:.3em;color:var(--text-xs)">Press ▶ START above</div>
        </div>""", unsafe_allow_html=True)
        render_right("", 0, False)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    def parse_labels(path):
        labs = []
        try:
            with open(path) as f:
                for line in f:
                    p = line.strip().split()
                    if p: labs.append(p[-1])
        except Exception: pass
        return labs

    def extract_model_meta(h5_path):
        out = {}
        if not os.path.exists(h5_path):
            return {"error": "File not found"}
        try:
            out["file_size_mb"] = f"{os.path.getsize(h5_path)/1_048_576:.2f} MB"
            with h5py.File(h5_path, "r") as f:
                for k in ("keras_version", "backend"):
                    v = f.attrs.get(k)
                    if v is not None:
                        out[k] = v.decode() if isinstance(v, bytes) else str(v)
                mc = f.attrs.get("model_config")
                if mc:
                    try:
                        cfg = json.loads(mc)
                        out["model_class"] = cfg.get("class_name", "—")
                        layer_cfgs = cfg.get("config", {}).get("layers", [])
                        out["num_declared_layers"] = str(len(layer_cfgs))
                        if layer_cfgs:
                            bs = layer_cfgs[0].get("config", {}).get("batch_input_shape")
                            if bs: out["input_shape"] = str(bs)
                    except Exception: pass
                layers, total_p = [], 0
                if "model_weights" in f:
                    mw = f["model_weights"]
                    for lname in mw.keys():
                        grp = mw[lname]; shapes = []
                        def _col(name, obj, _s=shapes):
                            if isinstance(obj, h5py.Dataset): _s.append(list(obj.shape))
                        grp.visititems(_col)
                        lp_val = sum(int(np.prod(s)) for s in shapes)
                        if shapes:
                            layers.append({"name": lname, "shapes": shapes, "params": lp_val})
                            total_p += lp_val
                    out["total_params"]  = f"{total_p:,}"
                    out["weight_layers"] = layers
                    out["num_w_layers"]  = str(len(layers))
        except Exception as e:
            out["error"] = str(e)
        return out

    meta = extract_model_meta(model_path)
    labs = parse_labels(labels_path)

    if "error" in meta:
        st.error(f"Model read error: {meta['error']}")
    else:
        st.markdown(f"""
        <div class="kpi-row" style="margin-bottom:20px">
          <div class="kpi-cell"><div class="val">{meta.get("total_params","—")}</div><div class="lbl">Total Parameters</div></div>
          <div class="kpi-cell"><div class="val">{meta.get("num_w_layers","—")}</div><div class="lbl">Weight Layers</div></div>
          <div class="kpi-cell"><div class="val">{len(labs) if labs else "—"}</div><div class="lbl">Classes</div></div>
          <div class="kpi-cell"><div class="val">{meta.get("file_size_mb","—")}</div><div class="lbl">File Size</div></div>
        </div>""", unsafe_allow_html=True)

        pairs = [
            ("Model Class",     meta.get("model_class","—")),
            ("Input Shape",     meta.get("input_shape","—")),
            ("Keras Version",   meta.get("keras_version","—")),
            ("Backend",         meta.get("backend","—")),
            ("Declared Layers", meta.get("num_declared_layers","—")),
            ("File Size",       meta.get("file_size_mb","—")),
            ("Model Path",      model_path),
            ("Labels Path",     labels_path),
        ]
        cells = "".join(
            f'<div class="info-cell"><div class="k">{k}</div><div class="v mono">{v}</div></div>'
            for k, v in pairs
        )
        st.markdown(f'<div class="info-grid" style="margin-bottom:16px">{cells}</div>', unsafe_allow_html=True)

        if labs:
            badges = "".join(
                f'<span style="display:inline-block;margin:3px 4px;padding:3px 10px;'
                f'border:1px solid var(--border2);font-size:.7rem;'
                f'font-family:var(--mono);color:var(--green-dim)">{l}</span>'
                for l in labs
            )
            st.markdown(f"""
            <div class="panel" style="margin-bottom:16px">
              <div class="panel-header"><span class="dot"></span>LABELS — {len(labs)} CLASSES</div>
              <div class="panel-body">{badges}</div>
            </div>""", unsafe_allow_html=True)

        if meta.get("weight_layers"):
            import pandas as pd
            with st.expander("▸ WEIGHT SHAPES PER LAYER", expanded=False):
                rows = [{"Layer": l["name"],
                         "Weight Shapes": "  |  ".join(str(s) for s in l["shapes"]),
                         "Parameters": f"{l['params']:,}"}
                        for l in meta["weight_layers"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EVALUATION
#
#  PERSISTENCE STRATEGY
#  ─────────────────────
#  Results are pickled to  asl_eval_cache.pkl  next to this .py file.
#  On every open, the cache is loaded instantly — no re-inference needed.
#  "⟳ RE-RUN" triggers fresh inference and overwrites the cache.
#  "🗑 CLEAR CACHE" deletes the file so the ready-state is shown again.
#
#  CORRECT EVALUATION APPROACH
#  ────────────────────────────
#  • Load Keras model via tf.keras (not cvzone — that's live-frame only).
#  • Read model.input_shape to get the real resize target (e.g. 224×224).
#  • Scan eval_data_dir for A/ B/ … Z/ sub-folders with ~800 images each.
#  • White-square preprocess every image to the model's input size.
#  • model.predict() in batches of 32; argmax → label via labels.txt map.
#  • sklearn confusion_matrix + balanced_accuracy + classification_report.
# ══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    from sklearn.metrics import (
        confusion_matrix as sk_cm,
        classification_report,
        balanced_accuracy_score,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    # ── helpers ──────────────────────────────────────────────────────────

    def eval_preprocess(img_bgr, size):
        """White-square resize to `size`×`size` — matches training pipeline."""
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0: return None
        white = np.ones((size, size, 3), np.uint8) * 255
        if h / w > 1:
            k=size/h; wc=math.ceil(k*w); rs=cv2.resize(img_bgr,(wc,size))
            g=math.ceil((size-wc)/2); white[:,g:min(g+wc,size)]=rs[:,:min(wc,size-g)]
        else:
            k=size/w; hc=math.ceil(k*h); rs=cv2.resize(img_bgr,(size,hc))
            g=math.ceil((size-hc)/2); white[g:min(g+hc,size),:]=rs[:min(hc,size-g),:]
        return white

    def scan_dataset(folder):
        """Walk folder expecting A/ B/ … sub-dirs. Falls back to flat layout."""
        if not os.path.isdir(folder): return [], "missing"
        subdirs = [d for d in os.listdir(folder)
                   if os.path.isdir(os.path.join(folder, d)) and d.upper() in LABELS]
        samples = []
        if subdirs:
            for sd in subdirs:
                lbl = sd.upper()
                for fname in sorted(os.listdir(os.path.join(folder, sd))):
                    if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                        samples.append((lbl, os.path.join(folder, sd, fname)))
            return sorted(samples), "subfolders"
        for fname in sorted(os.listdir(folder)):
            name, ext = os.path.splitext(fname)
            if ext.lower() in IMG_EXTS and name.upper() in LABELS:
                samples.append((name.upper(), os.path.join(folder, fname)))
        return sorted(samples), "flat"

    def read_label_map(labels_p):
        lmap = {}
        try:
            with open(labels_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:   lmap[int(parts[0])] = parts[1].upper()
                    elif len(parts) == 1: lmap[len(lmap)] = parts[0].upper()
        except Exception: pass
        return lmap

    def load_cache():
        try:
            if os.path.exists(EVAL_CACHE_PATH):
                with open(EVAL_CACHE_PATH, "rb") as f:
                    return pickle.load(f)
        except Exception: pass
        return None

    def save_cache(payload):
        try:
            with open(EVAL_CACHE_PATH, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            st.warning(f"Could not save cache: {e}")

    def run_fresh_eval(model_p, labels_p, data_dir):
        """Full inference pipeline. Returns (payload_dict, error_list)."""
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_p, compile=False)
        except Exception as e:
            return None, [f"Cannot load model: {e}"]

        # Read actual input size — never hardcode
        try:
            sh = model.input_shape
            model_h, model_w = int(sh[1]), int(sh[2])
        except Exception:
            model_h, model_w = 224, 224

        label_map = read_label_map(labels_p)
        if not label_map:
            return None, ["labels.txt unreadable or empty"]

        samples, mode = scan_dataset(data_dir)
        if not samples:
            return None, [
                f"No images found in: {data_dir}\n"
                "Expected: data/A/img1.jpg  data/B/img1.jpg …"
            ]

        y_true, y_pred, errors = [], [], []
        BATCH = 32
        b_imgs, b_lbls = [], []

        def flush(bi, bl):
            arr   = np.array(bi, dtype=np.float32) / 255.0
            preds = model.predict(arr, verbose=0)
            for p, tl in zip(preds, bl):
                y_true.append(tl)
                y_pred.append(label_map.get(int(np.argmax(p)), f"?{np.argmax(p)}"))

        for tl, path in samples:
            bgr = cv2.imread(path)
            if bgr is None: errors.append(f"Unreadable: {path}"); continue
            proc = eval_preprocess(bgr, model_h)
            if proc is None: errors.append(f"Zero crop: {path}"); continue
            b_imgs.append(proc); b_lbls.append(tl)
            if len(b_imgs) >= BATCH:
                flush(b_imgs, b_lbls); b_imgs, b_lbls = [], []
        if b_imgs: flush(b_imgs, b_lbls)

        if not y_true:
            return None, ["All images failed preprocessing"]

        present = sorted(set(y_true) | set(y_pred))
        payload = {
            "y_true":        y_true,
            "y_pred":        y_pred,
            "present":       present,
            "stats": {
                "total":      len(y_true),
                "correct":    sum(a == b for a, b in zip(y_true, y_pred)),
                "mode":       mode,
                "n_classes":  len(set(y_true)),
                "input_size": f"{model_h}×{model_w}",
            },
            "errors":        errors,
            "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "dataset_path":  data_dir,
            "model_path":    model_p,
        }
        return payload, []

    # ── render full results ───────────────────────────────────────────────
    def render_results(payload):
        y_true  = payload["y_true"]
        y_pred  = payload["y_pred"]
        present = payload["present"]
        stats   = payload["stats"]
        errors  = payload.get("errors", [])

        cm_arr  = sk_cm(y_true, y_pred, labels=present)
        oa      = stats["correct"] / stats["total"] * 100
        bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
        cr      = classification_report(y_true, y_pred, labels=present,
                                        output_dict=True, zero_division=0)

        # KPI strip
        st.markdown(f"""
        <div class="kpi-row" style="margin-bottom:20px">
          <div class="kpi-cell"><div class="val">{stats['total']:,}</div><div class="lbl">Images Tested</div></div>
          <div class="kpi-cell"><div class="val">{stats['correct']:,}</div><div class="lbl">Correct</div></div>
          <div class="kpi-cell">
            <div class="val" style="color:{'#00ff88' if oa>=85 else '#ffb800' if oa>=60 else '#ff3355'}">{oa:.1f}%</div>
            <div class="lbl">Overall Accuracy</div>
          </div>
          <div class="kpi-cell"><div class="val">{bal_acc:.1f}%</div><div class="lbl">Balanced Accuracy</div></div>
          <div class="kpi-cell"><div class="val">{stats['n_classes']}</div><div class="lbl">Classes</div></div>
          <div class="kpi-cell"><div class="val" style="font-size:1.6rem">{stats.get('input_size','—')}</div><div class="lbl">Input Size</div></div>
        </div>""", unsafe_allow_html=True)

        if stats.get("mode") == "flat":
            st.warning("⚠ Flat layout — 1 image per class. Use sub-folder layout for meaningful evaluation.")

        # Confusion matrix heatmap
        n   = len(present)
        fw  = max(12, n * 0.65)
        fh  = max(10, n * 0.58)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(row_sums == 0, 0.0, cm_arr / row_sums.astype(float))

        BG, BG1, BDR = "#080808", "#0e0e0e", "#2a2a2a"
        fig, ax = plt.subplots(figsize=(fw, fh))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG1)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "terminal", ["#0e0e0e","#0d2b1a","#0f5c2e","#00cc6a","#00ff88"]
        )
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        fs = max(5, 10 - n // 8)
        for i in range(n):
            for j in range(n):
                v = cm_arr[i, j]
                if v == 0: continue
                nv = cm_norm[i, j]
                tc = "#ffffff" if nv > 0.55 else ("#00cc6a" if nv > 0.15 else "#555")
                ax.text(j, i, str(v), ha="center", va="center", fontsize=fs,
                        color=tc, fontfamily="monospace",
                        fontweight="bold" if i == j else "normal")

        ax.set_xticks(range(n)); ax.set_xticklabels(present, fontsize=8, color="#888", fontfamily="monospace")
        ax.set_yticks(range(n)); ax.set_yticklabels(present, fontsize=8, color="#888", fontfamily="monospace")
        ax.tick_params(colors="#444", length=0)
        ax.set_xlabel("Predicted", fontsize=9, color="#555", labelpad=10, fontfamily="monospace")
        ax.set_ylabel("True",      fontsize=9, color="#555", labelpad=10, fontfamily="monospace")
        ax.set_title(
            f"CONFUSION MATRIX  ·  {stats['total']:,} samples  ·  OA {oa:.1f}%  ·  Balanced {bal_acc:.1f}%",
            fontsize=10, color="#00ff88", pad=14, fontfamily="monospace", fontweight="bold"
        )
        for i in range(n):
            ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor="#00ff88",lw=.9,alpha=.45))
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color=BDR, linewidth=0.35)
        ax.tick_params(which="minor", length=0)
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.ax.set_facecolor(BG)
        cbar.ax.yaxis.set_tick_params(color="#555", labelcolor="#555", labelsize=7)
        cbar.outline.set_edgecolor(BDR)
        cbar.set_label("Row-normalised proportion", fontsize=7, color="#555", fontfamily="monospace")
        for spine in ax.spines.values(): spine.set_edgecolor(BDR)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Per-class recall bars
        st.markdown("""
        <div style="font-size:.58rem;letter-spacing:.3em;color:var(--text-xs);
                    text-transform:uppercase;margin:20px 0 8px">Per-class recall</div>""",
                    unsafe_allow_html=True)
        pc_rows = ""; pc_data = []
        for lbl in present:
            row  = cr.get(lbl, {})
            rec  = row.get("recall",    0.0) * 100
            prec = row.get("precision", 0.0) * 100
            f1   = row.get("f1-score",  0.0) * 100
            sup  = int(row.get("support", 0))
            fc   = "low" if rec < 50 else ("mid" if rec < 80 else "")
            col  = "#00ff88" if rec >= 80 else ("#ffb800" if rec >= 50 else "#ff3355")
            pc_rows += f"""
            <div class="pc-row">
              <div class="pc-lbl">{lbl}</div>
              <div class="pc-track"><div class="pc-fill {fc}" style="width:{min(100,rec):.1f}%"></div></div>
              <div class="pc-pct" style="color:{col}">{rec:.0f}%</div>
              <div class="pc-n">{sup}img</div>
            </div>"""
            pc_data.append({"Letter":lbl,"Recall":f"{rec:.1f}%","Precision":f"{prec:.1f}%","F1":f"{f1:.1f}%","Support":sup})
        st.markdown(f'<div class="panel"><div class="panel-body" style="padding:8px 18px">{pc_rows}</div></div>',
                    unsafe_allow_html=True)

        with st.expander("▸ FULL CLASSIFICATION REPORT", expanded=False):
            st.dataframe(pd.DataFrame(pc_data), use_container_width=True, hide_index=True)

        # Top-10 confused pairs
        st.markdown("""
        <div style="font-size:.58rem;letter-spacing:.3em;color:var(--text-xs);
                    text-transform:uppercase;margin:20px 0 8px">Top-10 most confused pairs</div>""",
                    unsafe_allow_html=True)
        confused = sorted(
            [(present[i], present[j], int(cm_arr[i,j]))
             for i in range(n) for j in range(n) if i!=j and cm_arr[i,j]>0],
            key=lambda x: -x[2]
        )
        if confused:
            max_cnt = confused[0][2]
            ph = ""
            for tl, pl, cnt in confused[:10]:
                ph += f"""
                <div style="display:flex;align-items:center;gap:12px;padding:7px 0;
                            border-bottom:1px solid var(--border);font-size:.75rem">
                  <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                               font-weight:700;color:var(--text);width:24px;text-align:center">{tl}</span>
                  <span style="color:var(--text-xs);font-size:.6rem">→</span>
                  <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                               font-weight:700;color:var(--red);width:24px;text-align:center">{pl}</span>
                  <div style="flex:1;height:6px;background:var(--bg3)">
                    <div style="height:100%;background:var(--red);opacity:.7;
                                width:{min(100,cnt/max_cnt*100):.0f}%"></div>
                  </div>
                  <span style="color:var(--text-dim);font-size:.65rem;width:32px;text-align:right">{cnt}×</span>
                </div>"""
            st.markdown(f'<div class="panel"><div class="panel-body" style="padding:4px 18px">{ph}</div></div>',
                        unsafe_allow_html=True)

        if errors:
            with st.expander(f"▸ {len(errors)} SKIPPED IMAGE(S)", expanded=False):
                for e in errors: st.caption(e)

    # ── Load cache from disk ──────────────────────────────────────────────
    cached = load_cache()

    # ── Toolbar ───────────────────────────────────────────────────────────
    tb_info, tb_btns = st.columns([5, 2])

    with tb_info:
        if cached:
            st.markdown(f"""
            <div style="border-left:2px solid var(--green);padding:12px 20px;
                        background:var(--green-xs);font-size:.62rem;letter-spacing:.15em;
                        color:var(--text-dim);line-height:2">
              <span style="color:var(--green)">✓ CACHED EVALUATION</span><br>
              Saved: <span style="color:var(--text)">{cached.get('timestamp','—')}</span><br>
              Dataset: <span style="color:var(--text)">{cached.get('dataset_path','—')}</span>
              &nbsp;·&nbsp; {cached['stats']['total']:,} images
              &nbsp;·&nbsp; {cached['stats']['n_classes']} classes
              &nbsp;·&nbsp; input {cached['stats'].get('input_size','—')}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border-left:2px solid var(--amber);padding:12px 20px;
                        background:rgba(255,184,0,.06);font-size:.62rem;letter-spacing:.15em;
                        color:var(--text-dim);line-height:2">
              <span style="color:var(--amber)">NO SAVED EVALUATION</span><br>
              Dataset: <span style="color:var(--text)">{eval_data_dir}</span><br>
              Expects sub-folders A/ B/ … Z/ each with ~800 images
              &nbsp;·&nbsp; Input size auto-detected from model
            </div>""", unsafe_allow_html=True)

    with tb_btns:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        do_run = st.button(
            "⟳  RE-RUN EVALUATION" if cached else "▶  RUN EVALUATION",
            use_container_width=True, key="eval_run_btn"
        )
        if cached:
            if st.button("🗑  CLEAR CACHE", use_container_width=True, key="eval_clear_btn"):
                try: os.remove(EVAL_CACHE_PATH)
                except Exception: pass
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Fresh run triggered ───────────────────────────────────────────────
    if do_run:
        with st.spinner("Loading model · preprocessing ~20k images · running inference…"):
            payload, errs = run_fresh_eval(model_path, labels_path, eval_data_dir)
        if payload is None:
            for e in errs: st.error(e)
        else:
            save_cache(payload)
            cached = payload
            st.success(
                f"✓ Evaluation complete — {payload['stats']['total']:,} images · "
                f"Results saved to asl_eval_cache.pkl"
            )

    # ── Render from cache (or fresh result) ──────────────────────────────
    if cached and cached.get("y_true"):
        render_results(cached)
    elif not cached:
        st.markdown("""
        <div style="border:1px solid var(--border);background:var(--bg1);padding:60px 40px;
                    text-align:center">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:5rem;
                      font-weight:900;color:var(--text-xs)">READY</div>
          <div style="font-size:.58rem;letter-spacing:.35em;color:var(--text-xs);margin-top:8px">
            CLICK  ▶ RUN EVALUATION  TO BEGIN
          </div>
        </div>""", unsafe_allow_html=True)