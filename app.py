import asyncio
import io
import time
from pathlib import Path
import sys

# ─── BỘ VÁ LỖI ĐỘNG (MONKEY PATCH) CHO WINDOWS ──────────────────────────────
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        old_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        def _patched_call_connection_lost(self, exc):
            try:
                old_call_connection_lost(self, exc)
            except (ConnectionResetError, ConnectionAbortedError, OSError, AttributeError):
                pass 
        _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
    except Exception:
        pass

import edge_tts
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import food_history
import user_analytics
from food_info import get_food_info
from food_reasoner import generate_caption, answer_question
from food_similarity import get_top_similar_foods
from food_history import save_detection
from user_analytics import get_user_segment

import pandas as pd
import re

# Cấu hình hằng số hệ thống
MODEL_PATH  = "bestv8s.pt"
PAGE_TITLE  = "Food AI Assistant"
SEGMENT_CALO_BIAS = {
    "🥗 Người ăn lành mạnh": 0.2,
    "🌶️ Người thích ẩm thực đậm đà": 0.1,
    "🗺️ Khách khám phá đa vùng miền": 0.05,
}

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CẤU HÌNH CSS GIAO DIỆN TỐI ƯU ──────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    :root{
        --leaf:#1F3A2E; --leaf-2:#28483A; --paper:#FBF4E4; --paper-2:#F3E9D2;
        --chili:#C5432E; --turmeric:#E3A438; --herb:#4F7A52;
        --ink:#2A2118; --ink-soft:#6B5D4B; --line:#E4D9C0;
    }
    html, body, [class*="css"]  { font-family: 'Be Vietnam Pro', sans-serif; }

    /* Nền tổng thể */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: var(--leaf) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: var(--leaf-2) !important; border-right: 1px solid #3C5B4A; }
    [data-testid="stSidebar"] * { color: var(--paper) !important; }
    
    /* Đổi màu chữ mặc định trên nền dark */
    .stMarkdown, [data-testid="stMarkdownContainer"] p, label, .stCaption { color: var(--paper); }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--paper) !important; }

    /* Card bọc container */
    [data-testid="stVerticalBlockBorderWrapper"] { 
        background: var(--leaf-2) !important; 
        border: 1px solid #3C5B4A !important; 
        border-radius: 14px !important; 
        padding: 16px; 
    }
    [data-testid="stVerticalBlockBorderWrapper"] h3, [data-testid="stVerticalBlockBorderWrapper"] p { color: var(--paper) !important; }

    /* Tab */
    button[data-baseweb="tab"] { color: #C9D8CC !important; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--turmeric) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--turmeric) !important; }
    [data-baseweb="tab-border"] { background-color: #3C5B4A !important; }

    /* Định dạng thành phần Native của Streamlit */
    .stButton > button {
        background-color: var(--leaf-2); color: var(--paper) !important;
        border: 1px solid var(--herb); border-radius: 8px; transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: var(--herb); border-color: var(--turmeric); color: var(--paper) !important;
    }
    .stDownloadButton > button {
        background-color: var(--turmeric); color: var(--ink) !important; border: none; border-radius: 8px;
    }
    [data-testid="stAlert"] {
        background-color: var(--leaf-2) !important; border: 1px solid #3C5B4A !important;
        border-left: 5px solid var(--turmeric) !important; color: var(--paper) !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] svg { color: var(--paper) !important; fill: var(--paper) !important; }
    
    [data-testid="stProgress"] > div > div { background-color: var(--chili) !important; }
    [data-testid="stProgress"] { background-color: var(--leaf-2) !important; border-radius: 50px; }

    .stTextInput > div > div > input {
        background-color: var(--leaf-2); color: var(--paper) !important;
        border: 1px solid #3C5B4A; border-radius: 8px;
    }
    .stTextInput > div > div > input:focus { border-color: var(--turmeric); box-shadow: 0 0 0 1px var(--turmeric); }

    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.05); border: 2px dashed var(--herb); border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzone"] p, [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] svg { color: #C9D8CC !important; fill: #C9D8CC !important; }

    /* Bảng dữ liệu History (HTML table) */
    .history-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }
    .history-table th { background-color: var(--leaf-2); color: var(--turmeric); padding: 10px; text-align: left; border-bottom: 2px solid var(--herb); }
    .history-table td { background-color: var(--paper-2); color: var(--ink); padding: 10px; border-bottom: 1px solid var(--line); }
    .history-table tr:hover td { background-color: #E8F0E4; }

    /* Typography & Custom Classes */
    .main-title { font-size: 2.6rem; font-weight: 800; color: var(--paper); text-align: center; margin-bottom: 0.1rem; }
    .subtitle { text-align: center; color: #C9D8CC; font-size: 1.05rem; font-weight: 500; margin-bottom: 1rem; }
    .top-banner { background: var(--leaf-2); border: 1px solid #3C5B4A; border-radius: 14px; padding: 14px 22px; margin-bottom: 18px; font-weight: 700; color: var(--turmeric); font-size: 1.05rem; }

    .food-name { font-size: 2.1rem; font-weight: 800; color: var(--paper); margin: 0 0 8px 0; }
    .region-tag { display:inline-block; font-size: 0.78rem; font-weight:600; color: var(--herb); background:#E8F0E4; border:1px solid #CFE0CB; padding:3px 12px; border-radius:20px; margin-bottom:8px; }
    .conf-badge { background: var(--chili); color: white; padding: 6px 18px; border-radius: 50px; font-weight: 700; font-size: 0.9rem; display: inline-block; margin-bottom: 6px; margin-left:8px; }

    .calo-box { background: var(--leaf-2); border: none; border-radius: 14px; padding: 16px 22px; text-align: center; margin: 14px 0; }
    .calo-label { color: #C9D8CC; font-weight: 700; font-size: 0.85rem; margin: 0; }
    .calo-value { font-family:'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: var(--turmeric); margin: 4px 0 0 0; }

    .desc-box { background: var(--paper-2); padding: 16px 20px; border-radius: 12px; border-left: 5px solid var(--chili); color: var(--ink); margin: 8px 0 12px 0; font-size: 0.97rem; line-height: 1.6; }

    .macro-row { display: flex; gap: 10px; margin: 10px 0 14px 0; flex-wrap: wrap; }
    .macro-chip { background: var(--leaf-2); border: none; border-radius: 10px; padding: 8px 15px; font-weight: 600; color: var(--paper); font-size: 0.85rem; text-align:center; min-width:80px; }
    .macro-chip b { display:block; font-family:'JetBrains Mono', monospace; font-size:1.05rem; color: var(--turmeric); }

    .flavor-tags { display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 4px 0; }
    .ftag { font-size:0.8rem; font-weight:600; padding:5px 12px; border-radius:20px; border:1px solid var(--line); background:var(--paper-2); color:var(--ink); }

    .info-row { background: var(--paper-2); border-radius: 10px; padding: 9px 15px; margin: 5px 0; color: var(--ink); font-size: 0.91rem; line-height: 1.5; }

    .tts-section { background: var(--leaf-2); border: 1px solid #3C5B4A; border-radius: 14px; padding: 14px 18px; margin-top: 16px; }
    .tts-label { color: var(--paper); font-weight: 700; font-size: 0.88rem; margin-bottom: 8px; }

    .ticket { position:relative; background:var(--paper-2); border:1px solid var(--line); border-radius:6px; padding:14px 16px 16px; margin-bottom:10px; box-shadow:0 3px 8px rgba(0,0,0,0.2); }
    .ticket-rank { font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:var(--ink-soft); display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px dashed var(--line); padding-bottom:8px; }
    .ticket-stamp { background: var(--chili); color:#fff; font-weight:700; font-size:0.78rem; padding:2px 10px; border-radius:20px; font-family:'JetBrains Mono', monospace; }
    .ticket h4 { font-size:1.02rem; font-weight:700; margin:0 0 3px 0; color:var(--ink); }
    .ticket .sub { font-size:0.78rem; color:var(--ink-soft); font-family:'JetBrains Mono', monospace; }

    .suggest-chip-hint { font-size:0.78rem; color:#C9D8CC; margin-bottom:4px; }
    .bubble-ai { background: var(--paper-2); border:1px solid var(--line); border-radius:14px; border-bottom-left-radius:3px; padding:12px 16px; margin:8px 0; color:var(--ink); font-size:0.93rem; line-height:1.55; }
    .bubble-ai .src { display:inline-block; font-family:'JetBrains Mono', monospace; font-size:0.68rem; background:var(--herb); color:#fff; padding:2px 8px; border-radius:4px; margin-bottom:6px; }

    .segment-card { display:flex; align-items:flex-start; gap:14px; background: var(--leaf-2); color: var(--paper); border-radius:12px; padding:18px 20px; margin-bottom:18px; border: 1px solid #3C5B4A; }
    .segment-card .emoji { font-size:28px; }
    .segment-card h3 { font-size:1.05rem; margin:0 0 4px 0; color: var(--paper); }
    .segment-card p { font-size:0.85rem; color:#C9D8CC; line-height:1.5; margin:0; }
    .segment-card .why { margin-top:8px; font-size:0.8rem; color: var(--turmeric); font-weight:600; }
    .tag-container { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .tag-chip { background: var(--paper-2); padding: 6px 14px; border-radius: 50px; font-size: 0.85rem; font-weight: 600; color: var(--ink); border: 1px solid var(--line); }
    
    [data-testid="stImage"] img { border-radius: 12px; max-height: 360px; object-fit: cover; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE & BACKEND FUNCTIONS ───────────────────────────────────────
if "tts_audio_bytes" not in st.session_state: st.session_state.tts_audio_bytes = None
if "tts_food_key" not in st.session_state: st.session_state.tts_food_key = None

@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists(): return None
    return YOLO(model_path)

def get_user_taste_profile(session_id):
    history = food_history.get_session_history(session_id) 
    if not history: return ["Chưa có gu 🍽️"]
    flavor_map = {"Cay": "🌶️ Cay", "Chua": "🍋 Chua", "Ngọt": "🍬 Ngọt", "Mặn": "🧂 Mặn", "Đậm": "🥘 Đậm đà", "Béo": "🥥 Béo"}
    counts = {}
    for item in history:
        db_key = item[5] if len(item) > 5 else item[0]
        info = get_food_info(db_key) 
        flavors = info.get("vi_dac_trung", "") or ""
        for k, v in flavor_map.items():
            if k in flavors: counts[v] = counts.get(v, 0) + 1
    sorted_tags = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tags[:3]] if sorted_tags else ["Chưa có gu 🍽️"]

def run_detection(model, image: Image.Image, conf: float):
    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    detections = []
    if results:
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "class_name": result.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            })
    annotated = results[0].plot() if results else None
    return detections, annotated

def dedup_detections(detections: list) -> list:
    best = {}
    for d in detections:
        name = d["class_name"]
        if name not in best or d["confidence"] > best[name]["confidence"]:
            best[name] = d
    return sorted(best.values(), key=lambda x: x["confidence"], reverse=True)

def _normalize_tts(text: str) -> str:
    text = re.sub(r'(\d+)\s*g\b', r'\1 gam', text)
    return text

def build_tts_text(info: dict, conf: float) -> str:
    raw = " ".join([
        f"Món ăn được nhận diện là {info['ten_hien_thi']}, với độ tin cậy {conf*100:.0f} phần trăm.",
        f"Mô tả: {info['mo_ta']}",
        f"Lượng calo ước tính: {info.get('calo', 'không rõ')}.",
        f"Chất đạm: {info.get('protein', 'không rõ')}. Tinh bột: {info.get('carb', 'không rõ')}. Chất béo: {info.get('fat', 'không rõ')}.",
        f"Vùng miền: {info.get('vung_mien', 'không rõ')}.",
        f"Giá tham khảo: {info.get('gia_trung_binh', 'không rõ')}.",
        f"Khuyến nghị dinh dưỡng: {info.get('khuyen_nghi', 'không có')}.",
    ])
    return _normalize_tts(raw)

async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio": buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

def generate_tts(text: str, voice: str, rate: str = "-5%") -> bytes:
    try:
        return asyncio.run(_synthesize(text, voice, rate))
    except Exception:
        loop = asyncio.new_event_loop()
        try: return loop.run_until_complete(_synthesize(text, voice, rate))
        finally: loop.close()

def render_tts_section(top_det: dict, voice: str, lang: str):
    info = get_food_info(top_det["class_name"])
    conf = top_det["confidence"]
    tts_key = f"{top_det['class_name']}_{voice}_{lang}"
    tts_box_lbl = "🔊 Đọc to kết quả bằng Tiếng Anh — Highest Confidence Dish" if lang == "en" else "🔊 Đọc to kết quả — món có độ tin cậy cao nhất"
    tts_spin_lbl = "🎙️ Synthesizing English voice..." if lang == "en" else "🎙️ Đang tổng hợp giọng nói..."
    st.markdown(f"<div class='tts-section'><p class='tts-label'>{tts_box_lbl}</p></div>", unsafe_allow_html=True)
    if st.session_state.tts_food_key != tts_key:
        with st.spinner(tts_spin_lbl):
            try:
                text = build_tts_text(info, conf)
                if lang == "en":
                    from food_ai_service import ask_llm_expert
                    text = ask_llm_expert(f"Translate this comprehensive paragraph into fluent, natural spoken English for an audio tour presentation. Translate everything: {text}", top_det["class_name"], info, "en")
                audio_bytes = generate_tts(text, voice, "-5%")
                st.session_state.tts_audio_bytes = audio_bytes
                st.session_state.tts_food_key    = tts_key
            except Exception as e:
                st.warning(f"⚠️ Không thể tổng hợp giọng nói lúc này. Lỗi: {e}")
                return
    if st.session_state.tts_audio_bytes:
        st.audio(st.session_state.tts_audio_bytes, format="audio/mp3", autoplay=True)

def render_one_food(det: dict):
    info = get_food_info(det["class_name"])
    conf = det["confidence"]
    st.markdown(
        f"<p class='food-name'>{info['ten_hien_thi']}</p>"
        f"<span class='region-tag'>🗺️ {info.get('vung_mien', 'N/A')}</span>"
        f"<span class='conf-badge'>Độ tin cậy: {conf*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.progress(conf)
    flavor_icon = {"Cay": "🌶️", "Chua": "🍋", "Ngọt": "🍬", "Đậm": "🧂", "Thanh": "🍃", "Béo": "🥥"}
    raw_flavors = info.get("vi_dac_trung", "") or ""
    flavors = [f.strip() for f in re.split(r"[,/]", raw_flavors) if f.strip()]
    if flavors:
        chips = "".join(f"<span class='ftag'>{flavor_icon.get(f, '•')} {f}</span>" for f in flavors)
        st.markdown(f"<div class='flavor-tags'>{chips}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='calo-box'><p class='calo-label'>🔥 Lượng calo ước tính (1 khẩu phần)</p>"
        f"<p class='calo-value'>{info.get('calo', 'N/A')}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**📖 Mô tả chi tiết**")
    st.markdown(f"<div class='desc-box'>{info['mo_ta']}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='macro-row'>"
        f"<div class='macro-chip'><b>{info.get('protein','N/A')}</b>ĐẠM</div>"
        f"<div class='macro-chip'><b>{info.get('carb','N/A')}</b>TINH BỘT</div>"
        f"<div class='macro-chip'><b>{info.get('fat','N/A')}</b>CHẤT BÉO</div>"
        f"</div>", unsafe_allow_html=True,
    )
    rows = [
        ("📍", "Vùng miền", info.get("vung_mien", "N/A")),
        ("🍽️", "Khẩu phần", info.get("khau_phan", "N/A")),
        ("🧾", "Thành phần", info.get("thanh_phan", "N/A")),
        ("💰", "Giá tham khảo", info.get("gia_trung_binh", "N/A")),
        ("⏰", "Thời điểm", info.get("thoi_diem_phu_hop", "N/A")),
        ("❤️", "Sức khỏe", info.get("chi_so_suc_khoe", "N/A")),
        ("💡", "Khuyến nghị", info.get("khuyen_nghi", "N/A")),
    ]
    c1, c2 = st.columns(2)
    with c1:
        for icon, label, value in rows[:4]:
            st.markdown(f"<div class='info-row'>{icon} <strong>{label}:</strong> {value}</div>", unsafe_allow_html=True)
    with c2:
        for icon, label, value in rows[4:]:
            st.markdown(f"<div class='info-row'>{icon} <strong>{label}:</strong> {value}</div>", unsafe_allow_html=True)

def get_recommendations(food_key: str, session_id: str, top_k: int = 3):
    similar = get_top_similar_foods(food_key, top_k=5)
    segment = get_user_segment(session_id)
    current_food_info = get_food_info(food_key)
    current_region = current_food_info.get("vung_mien", "")
    scored = []
    for key, sim_score in similar:
        info = get_food_info(key)
        m = re.search(r"\d+", info.get("calo", "0") or "0")
        calo = int(m.group()) if m else 400
        A = 0.0
        if segment == "🥗 Người ăn lành mạnh": A = max(0.0, 1.0 - (calo / 800.0))
        elif segment == "🌶️ Người thích ẩm thực đậm đà":
            if "Cay" in info.get("vi_dac_trung", "") or "Đậm" in info.get("vi_dac_trung", ""): A = 1.0
        elif segment == "🗺️ Khách khám phá đa vùng miền":
            if info.get("vung_mien", "") != current_region: A = 1.0
        rec_score = 0.7 * sim_score + 0.3 * A
        scored.append((key, rec_score, sim_score, A))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ─── HÀM RENDER RIÊNG CHO TỪNG TAB ───────────────────────────────────────────
def render_explore_tab(detections: list, voice: str, source_img: Image.Image = None, lang: str = "vi"):
    if not detections:
        st.error("❌ Không nhận diện được món ăn nào. Hãy thử ảnh rõ hơn!")
        return None, None

    sorted_dets = dedup_detections(detections)
    top_det     = sorted_dets[0]
    top_info    = get_food_info(top_det["class_name"])
    session_id  = "demo_user" 

    if "history_saved_for" not in st.session_state:
        st.session_state.history_saved_for = set()
    if top_det["class_name"] not in st.session_state.history_saved_for:
        food_history.save_detection(session_id, top_det, top_info)
        st.session_state.history_saved_for.add(top_det["class_name"])
        st.rerun() 

    st.markdown(
        f"<div class='top-banner'>🎯 {top_info['ten_hien_thi']} — {top_det['confidence']*100:.1f}%</div>",
        unsafe_allow_html=True,
    )

    for i, det in enumerate(sorted_dets):
        current_info = get_food_info(det["class_name"])
        if i == 0: render_one_food(det)
        else:
            with st.expander(f"🍽️ {current_info['ten_hien_thi']} — {det['confidence']*100:.1f}%"):
                render_one_food(det)
    st.markdown("---")
    st.markdown("##### 🎯 Bạn có thể sẽ thích — xếp hạng theo RecScore")
    recs = get_recommendations(food_key=top_det["class_name"], session_id=session_id)
    rank_labels = ["#1 TOP SIMILAR", "#2 TOP SIMILAR", "#3 TOP SIMILAR"]
    if recs:
        ticket_cols = st.columns(len(recs), gap="medium")
        for idx, (col, (key, rec_score, sim, adj)) in enumerate(zip(ticket_cols, recs)):
            info = get_food_info(key)
            with col:
                st.markdown(f"""
                <div class='ticket'>
                    <div class='ticket-rank'>
                        <span>{rank_labels[idx] if idx < len(rank_labels) else f"#{idx+1}"}</span>
                        <span class='ticket-stamp'>{rec_score*100:.0f}đ</span>
                    </div>
                    <h4>{info['ten_hien_thi']}</h4>
                    <span class='sub'>Similarity {sim:.2f} · +Segment {adj:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("Chưa có đủ dữ liệu để gợi ý món tương tự.")
        
    return top_det, top_info

def render_history_tab(session_id: str):
    history = food_history.get_session_history(session_id)
    segment = get_user_segment(session_id)
    
    try:
        _, _, _, _, sil_score = user_analytics.cluster_own_history(session_id)
    except Exception:
        sil_score = 0
        
    bias = SEGMENT_CALO_BIAS.get(segment, 0)
    segment_emoji = "🥗" if "lành mạnh" in segment else ("🌶️" if "đậm đà" in segment else ("🗺️" if "vùng miền" in segment else "🍽️"))
    
    # ĐỔI CÁC DÒNG CHỮ THÀNH DỄ HIỂU HƠN
    if "Chưa đủ dữ liệu" in segment: 
        why_text = "Hãy quét thêm món ăn (tối thiểu 3 món) để AI có thể phân tích gu ẩm thực của bạn."
    elif "lành mạnh" in segment: 
        why_text = "→ Ảnh hưởng Tab 1: Hệ thống sẽ ưu tiên gợi ý những món ít calo, thanh đạm hơn."
    elif "đậm đà" in segment: 
        why_text = "→ Ảnh hưởng Tab 1: Hệ thống sẽ ưu tiên gợi ý những món có hương vị đậm đà, cay nồng hơn."
    else: 
        why_text = "→ Ảnh hưởng Tab 1: Hệ thống sẽ ưu tiên gợi ý những món thuộc vùng miền mới để bạn khám phá đa dạng hơn."

    st.markdown(f"""
    <div class='segment-card'>
        <div class='emoji'>{segment_emoji}</div>
        <div>
            <h3>{segment}</h3>
            <p>Phân khúc xác định bằng K-Means dựa trên lịch sử khám phá trong phiên này.</p>
            <div class='why'>{why_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    DAILY_CALO_GOAL = 2000
    total_calo = food_history.get_total_calo(session_id)
    pct = min(100, int(total_calo / DAILY_CALO_GOAL * 100)) if DAILY_CALO_GOAL else 0
    st.markdown(f"**Tổng calo phiên này:** {total_calo} / {DAILY_CALO_GOAL} kcal")
    st.progress(pct / 100)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Món đã quét", len(history))
    with c2: st.metric("Điều chỉnh Segment (bias)", f"{bias:+.2f}")
    with c3: 
        if sil_score > 0:
            st.metric("Silhouette Score (K-Means)", f"{sil_score:.2f}")
            st.caption("Điểm > 0 chứng minh K=3 phân cụm hợp lý")
        else:
            st.metric("Silhouette Score", "Chưa đủ data")

    st.markdown("**Gu ẩm thực của bạn:**")
    tags = get_user_taste_profile(session_id)
    st.markdown(f"<div class='tag-container'>{''.join([f'<span class=\"tag-chip\">{t}</span>' for t in tags])}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("##### 🍽️ Món đã khám phá trong phiên")
    if history:
        table_html = "<table class='history-table'><thead><tr><th>Món</th><th>Tin cậy</th><th>Calo</th><th>Vùng</th><th>Thời gian</th></tr></thead><tbody>"
        for h in history[:10]:
            clean_h = h[:5]
            row = f"<tr><td>{clean_h[0]}</td><td>{clean_h[1]}</td><td>{clean_h[2]}</td><td>{clean_h[3]}</td><td>{clean_h[4]}</td></tr>"
            table_html += row
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Chưa có lịch sử khám phá trong phiên này — hãy quét thêm món để mở khoá phân khúc cá nhân hoá.")

    with st.expander("💡 Tại sao AI gợi ý các món ở Tab 1 như vậy?"):
        st.markdown("`RecScore = 0.7 × SimilarityScore + 0.3 × SegmentAdjustment`")
        st.write(f"Hệ thống xác định bạn thuộc nhóm hành vi: **{segment}**.")
        # Bỏ cụm "→ Ảnh hưởng Tab 1: " đi để câu văn trong expander tự nhiên hơn
        st.write(why_text.replace("→ Ảnh hưởng Tab 1: ", ""))

# ─── HÀM MAIN ĐIỀU PHỐI CHÍNH ────────────────────────────────────────────────
def main():
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"❌ Không tìm thấy file model `{MODEL_PATH}`.")
        return

    with st.sidebar:
        lang_choice = st.selectbox("🌐 Giọng đọc & Hỏi đáp", ["Tiếng Việt", "English"], index=0)
        lang = "vi" if lang_choice == "Tiếng Việt" else "en"
        CONF_THRESHOLD = st.slider("🎯 Độ tin cậy tối thiểu", 0.1, 1.0, 0.35, 0.05)
        st.divider()
        
        voice_title = "**🔊 TTS Voice (English)**" if lang == "en" else "**🔊 Giọng đọc**"
        voice_options = ["en-US-AriaNeural (Female)", "en-US-GuyNeural (Male)"] if lang == "en" else ["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"]
        st.markdown(voice_title)
        voice_choice = st.radio("Voice Configuration", options=voice_options, index=0, label_visibility="collapsed")
        TTS_VOICE = "vi-VN-HoaiMyNeural" if "HoaiMy" in voice_choice else ("vi-VN-NamMinhNeural" if "NamMinh" in voice_choice else ("en-US-AriaNeural" if "Aria" in voice_choice else "en-US-GuyNeural"))

        st.divider()
        st.markdown("**✨ Tính năng**")
        st.markdown("• Nhận diện nhiều món\n• Tính calo & macro\n• Mô tả chi tiết\n• Vùng miền, giá, thời điểm\n• Gợi ý món tương tự\n• 🔊 Đọc to tự động\n• 💬 Hỏi đáp về món ăn")
        st.divider()
        st.caption("Demo MVP • Team AI Food Assistant © 2026")

    st.markdown("<div class='main-title'>🍜 Food AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Nhận diện · Tính calo · Gợi ý theo RecScore · Hỏi đáp về món ăn Việt Nam bằng AI</div>", unsafe_allow_html=True)
    st.divider()

    left, right = st.columns([1, 2.3], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### 📤 Đầu vào ảnh")
            tab_upload, tab_cam = st.tabs(["📁 Upload / Kéo thả", "📸 Camera trực tiếp"])
            source_img = None
            with tab_upload:
                uploaded = st.file_uploader("Chọn hoặc kéo thả ảnh món ăn vào đây", type=["jpg", "jpeg", "png"], key="upload")
                if uploaded: source_img = Image.open(uploaded).convert("RGB")
            with tab_cam:
                cam = st.camera_input("Chụp món ăn ngay", key="camera")
                if cam: source_img = Image.open(cam).convert("RGB") 
            if source_img:
                st.image(source_img, use_container_width=True)
                st.caption("Ảnh gốc")
            else:
                st.info("Chưa có ảnh — hãy upload hoặc bật camera để bắt đầu.")

    with right:
        st.markdown("### 📊 Kết quả nhận diện")
        
        qa_tab_title = "💬 Hỏi đáp AI (English Q&A)" if lang == "en" else "💬 Hỏi đáp AI"
        tab_explore, tab_qa, tab_history = st.tabs(["📊 Thông tin & Khám phá", qa_tab_title, "🗂️ Lịch sử Khám phá"])

        session_id = "demo_user"
        top_det = None
        top_info = None

        # TAB 3 LUÔN ĐƯỢC RENDER DÙ CHƯA CÓ ẢNH
        with tab_history:
            render_history_tab(session_id)

        # KIỂM TRA ẢNH ĐỂ RENDER TAB 1 VÀ 2
        if source_img is not None:
            with st.spinner("🤖 AI đang phân tích món ăn..."):
                t0 = time.time()
                detections, annotated = run_detection(model, source_img, CONF_THRESHOLD)
                elapsed = time.time() - t0
            
            with tab_explore:
                if not detections:
                    st.error("❌ Không nhận diện được món ăn nào. Hãy thử ảnh rõ hơn!")
                else:
                    if annotated is not None:
                        st.image(annotated, channels="BGR", use_container_width=True)
                        st.caption(f"✅ Xử lý xong trong {elapsed:.2f}s — phát hiện {len(detections)} đối tượng")
                    top_det, top_info = render_explore_tab(detections, TTS_VOICE, source_img, lang)

            with tab_qa:
                if not top_det:
                    st.info("❌ Chưa nhận diện được món ăn để hỏi đáp.")
                else:
                    render_tts_section(top_det, TTS_VOICE, lang)
                    caption = generate_caption(top_det["class_name"], top_info, top_det["confidence"], lang)
                    st.info(caption)
                    st.markdown("<p class='suggest-chip-hint'>Câu hỏi gợi ý:</p>", unsafe_allow_html=True)
                    suggested_questions = [f"{top_info['ten_hien_thi']} bao nhiêu calo?", "Ăn kèm rau gì cho đúng chuẩn?", "So sánh với món tương tự"]
                    chip_cols = st.columns(len(suggested_questions))
                    clicked_question = None
                    for col, q in zip(chip_cols, suggested_questions):
                        with col:
                            if st.button(q, key=f"chip_{q}", use_container_width=True):
                                clicked_question = q
                    if "chat_thread" not in st.session_state: st.session_state.chat_thread = []
                    user_question = st.text_input("Hoặc tự đặt câu hỏi khác", key="food_question")
                    final_question = clicked_question or (user_question if user_question else None)
                    if final_question:
                        with st.spinner("🤖 Hệ thống đang xử lý..."):
                            answer, source = answer_question(final_question, top_det["class_name"], top_info, top_det["confidence"], lang)
                            st.session_state.chat_thread.append((final_question, answer, source))
                    for q, a, source in st.session_state.chat_thread[-5:]:
                        st.markdown(f"**🧑 {q}**")
                        st.markdown(f"<div class='bubble-ai'><span class='src'>AI · {source}</span><br>{a}</div>", unsafe_allow_html=True)
        else:
            with tab_explore:
                st.info("⬅️ Hãy upload ảnh hoặc bật camera bên trái để bắt đầu nhận diện.")
            with tab_qa:
                st.info("⬅️ Hãy nhận diện món ăn để bắt đầu hỏi đáp.")

if __name__ == "__main__":
    main()