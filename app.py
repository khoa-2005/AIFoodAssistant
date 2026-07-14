import asyncio
import concurrent.futures
import io
import time
from pathlib import Path
import sys
from transformers import BlipProcessor, BlipForConditionalGeneration
# ─── BỘ VÁ LỖI ĐỘNG (MONKEY PATCH) CHO WINDOWS ──────────────────────────────
# Triệt tiêu im lặng lỗi WinError 10054 khi ngắt kết nối Sockets ngầm với server TTS
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
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO

# Trong phần imports của app.py
import food_history

# Import các hàm xử lý ẩm thực nội bộ
from food_info import get_food_info
from food_reasoner import generate_caption, answer_question
from food_blip import (
    is_available as blip_available,
    generate_caption_from_image,
    answer_question_from_image,
)

import pandas as pd
import re
from food_similarity import get_top_similar_foods
from user_analytics import get_user_segment
from food_history import save_detection # Thêm hàm lưu lịch sử

# Cấu hình hằng số hệ thống
MODEL_PATH  = "bestv8s.pt"
PAGE_TITLE  = "Food AI Assistant"
SEGMENT_CALO_BIAS = {
    "🥗 Người ăn lành mạnh": 0.2,
    "🌶️ Người thích ẩm thực đậm đà": 0.1,
    "🗺️ Khách khám phá đa vùng miền": 0.05,
}

# Cấu hình thiết lập trang Streamlit
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cấu hình phong cách CSS giao diện
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f8fafc 0%, #fefce8 100%); }
    .main-title { font-size: 3.4rem; font-weight: 900; background: linear-gradient(90deg, #f97316, #eab308); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; color: #475569; font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem; }
    .top-banner { background: linear-gradient(90deg, #dcfce7, #d1fae5); border: 2px solid #22c55e; border-radius: 14px; padding: 14px 22px; margin-bottom: 18px; font-weight: 700; color: #15803d; font-size: 1.05rem; }
    .food-name { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin: 0 0 8px 0; }
    .conf-badge { background: linear-gradient(90deg, #22c55e, #86efac); color: white; padding: 7px 20px; border-radius: 50px; font-weight: 700; font-size: 0.95rem; box-shadow: 0 4px 12px rgba(34,197,94,0.3); display: inline-block; margin-bottom: 6px; }
    .calo-box { background: linear-gradient(135deg, #fefce8, #fef3c7); border: 3px solid #fbbf24; border-radius: 18px; padding: 20px 24px; text-align: center; margin: 14px 0; }
    .calo-label { color: #b45309; font-weight: 700; font-size: 0.9rem; margin: 0; }
    .calo-value { font-size: 2.8rem; font-weight: 900; color: #c2410c; margin: 4px 0 0 0; }
    .desc-box { background: #f8fafc; padding: 16px 20px; border-radius: 14px; border-left: 6px solid #f97316; color: #334155; margin: 8px 0 12px 0; font-size: 0.97rem; line-height: 1.6; }
    .macro-row { display: flex; gap: 10px; margin: 10px 0 14px 0; flex-wrap: wrap; }
    .macro-chip { background: #fff7ed; border: 1.5px solid #fdba74; border-radius: 12px; padding: 7px 15px; font-weight: 600; color: #9a3412; font-size: 0.88rem; }
    .info-row { background: #f1f5f9; border-radius: 10px; padding: 9px 15px; margin: 5px 0; color: #334155; font-size: 0.91rem; line-height: 1.5; }
    .tts-section { background: linear-gradient(135deg, #fff7ed, #fef3c7); border: 2px solid #fb923c; border-radius: 16px; padding: 16px 20px; margin-top: 20px; }
    .tts-label { color: #9a3412; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; }
    .dashboard-card { background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .tag-container { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .tag-chip { background: #f1f5f9; padding: 6px 14px; border-radius: 50px; font-size: 0.85rem; font-weight: 600; color: #334155; border: 1px solid #cbd5e1; }
    .rec-card { background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #f97316; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #1e293b; }
</style>
""", unsafe_allow_html=True)

if "tts_audio_bytes" not in st.session_state:
    st.session_state.tts_audio_bytes = None
if "tts_food_key" not in st.session_state:
    st.session_state.tts_food_key = None


@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        return None
    return YOLO(model_path)

def get_user_taste_profile(session_id):
    # Dùng hàm chuẩn từ database
    history = food_history.get_session_history(session_id) 
    if not history: return ["Chưa có gu 🍽️"]
    
    flavor_map = {"Cay": "🌶️ Cay", "Chua": "🍋 Chua", "Ngọt": "🍬 Ngọt", "Mặn": "🧂 Mặn", "Đậm": "🥘 Đậm đà"}
    counts = {}
    
    for item in history:
        # item bây giờ là tuple: (ten_hien_thi, confidence, calo, vung, time)
        # item[0] là tên món ăn
        info = get_food_info(item[0]) 
        flavors = info.get("vi_dac_trung", "")
        for k, v in flavor_map.items():
            if k in flavors: counts[v] = counts.get(v, 0) + 1
            
    sorted_tags = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tags[:3]]

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


def crop_food_image(image: Image.Image, bbox: tuple) -> Image.Image:
    xmin, ymin, xmax, ymax = bbox
    w, h = image.size
    pad = 10
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(w, xmax + pad)
    ymax = min(h, ymax + pad)
    return image.crop((xmin, ymin, xmax, ymax))


def dedup_detections(detections: list) -> list:
    best = {}
    for d in detections:
        name = d["class_name"]
        if name not in best or d["confidence"] > best[name]["confidence"]:
            best[name] = d
    return sorted(best.values(), key=lambda x: x["confidence"], reverse=True)


def _normalize_tts(text: str) -> str:
    import re
    text = re.sub(r'(\d+)\s*g\b', r'\1 gam', text)
    return text


def build_tts_text(info: dict, conf: float) -> str:
    """Luôn sinh văn bản gốc tiếng Việt đầy đủ chỉ số để đảm bảo độ dài giọng đọc 32 giây chuẩn xác"""
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
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def generate_tts(text: str, voice: str, rate: str = "-5%") -> bytes:
    try:
        return asyncio.run(_synthesize(text, voice, rate))
    except Exception:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_synthesize(text, voice, rate))
        finally:
            loop.close()


def render_tts_section(top_det: dict, voice: str, lang: str):
    """Xử lý phát âm thanh. Tự động dịch ngầm kịch bản tiếng Việt sang tiếng Anh nếu bật English voice"""
    info = get_food_info(top_det["class_name"])
    conf = top_det["confidence"]
    tts_key = f"{top_det['class_name']}_{voice}_{lang}"

    # Nhãn hộp đọc dynamically đổi theo chế độ voice
    tts_box_lbl = "🔊 Đọc to kết quả bằng Tiếng Anh — Highest Confidence Dish" if lang == "en" else "🔊 Đọc to kết quả — món có độ tin cậy cao nhất"
    tts_spin_lbl = "🎙️ Synthesizing English voice..." if lang == "en" else "🎙️ Đang tổng hợp giọng nói..."

    st.markdown(
        f"<div class='tts-section'><p class='tts-label'>{tts_box_lbl}</p></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.tts_food_key != tts_key:
        with st.spinner(tts_spin_lbl):
            try:
                text = build_tts_text(info, conf)
                # Dịch ngầm kịch bản thuyết trình sang Tiếng Anh trước khi đẩy vào loa phát
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
    """Hiển thị bảng biểu chi tiết món ăn. Mọi dữ liệu và nhãn hoàn toàn là Tiếng Việt cố định"""
    info = get_food_info(det["class_name"])
    conf = det["confidence"]
    display_title = info['ten_hien_thi']

    st.markdown(
        f"<p class='food-name'>🥗 {display_title}</p>"
        f"<span class='conf-badge'>Độ tin cậy: {conf*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.progress(conf)

    st.markdown(
        f"<div class='calo-box'><p class='calo-label'>🔥 Lượng calo ước tính (1 khẩu phần)</p>"
        f"<p class='calo-value'>{info.get('calo', 'N/A')}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**📖 Mô tả chi tiết**")
    st.markdown(f"<div class='desc-box'>{info['mo_ta']}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='macro-row'>"
        f"<span class='macro-chip'>🥩 Chất đạm: {info.get('protein','N/A')}</span>"
        f"<span class='macro-chip'>🍚 Tinh bột: {info.get('carb','N/A')}</span>"
        f"<span class='macro-chip'>🥑 Chất béo: {info.get('fat','N/A')}</span>"
        f"</div>", unsafe_allow_html=True,
    )

    rows = [
        ("📍", "Vùng miền",         info.get("vung_mien",         "N/A")),
        ("🍽️", "Khẩu phần",         info.get("khau_phan",         "N/A")),
        ("🧾", "Thành phần chính",   info.get("thanh_phan",        "N/A")),
        ("💰", "Giá tham khảo",     info.get("gia_trung_binh",    "N/A")),
        ("⏰", "Thời điểm phù hợp", info.get("thoi_diem_phu_hop", "N/A")),
        ("❤️", "Chỉ số sức khỏe",   info.get("chi_so_suc_khoe",   "N/A")),
        ("💡", "Khuyến nghị",       info.get("khuyen_nghi",       "N/A")),
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
        if segment == "🥗 Người ăn lành mạnh":
            A = max(0.0, 1.0 - (calo / 800.0))
        elif segment == "🌶️ Người thích ẩm thực đậm đà":
            vi = info.get("vi_dac_trung", "")
            if "Cay" in vi or "Đậm" in vi:
                A = 1.0
        elif segment == "🗺️ Khách khám phá đa vùng miền":
            if info.get("vung_mien", "") != current_region:
                A = 1.0

        rec_score = 0.7 * sim_score + 0.3 * A
        scored.append((key, rec_score, sim_score, A))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def render_detections(detections: list, voice: str, source_img: Image.Image = None, use_blip: bool = False, lang: str = "vi"):
    if not detections:
        st.error("❌ Không nhận diện được món ăn nào. Hãy thử ảnh rõ hơn!")
        return

    sorted_dets = dedup_detections(detections)
    top_det     = sorted_dets[0]
    top_info    = get_food_info(top_det["class_name"])
    session_id  = "demo_user" 

    # --- LƯU LỊCH SỬ CHUẨN ---
    if "history_saved_for" not in st.session_state:
        st.session_state.history_saved_for = set()
        
    # Chỉ lưu khi món này là món mới so với phiên trước đó
    if top_det["class_name"] not in st.session_state.history_saved_for:
        food_history.save_detection(session_id, top_det, top_info)
        st.session_state.history_saved_for.add(top_det["class_name"])
        st.rerun() # <--- QUAN TRỌNG: Làm mới trang để Dashboard cập nhật ngay!
    # -------------------------

    st.markdown(
        f"<div class='top-banner'>🎯 {top_info['ten_hien_thi']} — {top_det['confidence']*100:.1f}%</div>",
        unsafe_allow_html=True,
    )

    qa_tab_title = "💬 Hỏi đáp (English Q&A)" if lang == "en" else "💬 Hỏi đáp"
    tab_info, tab_qa, tab_ai, tab_rec = st.tabs(["📊 Thông tin món ăn", qa_tab_title, "🧠 Phân tích ảnh AI", "💡 Gợi ý AI"])

    with tab_info:
        # Giữ nguyên code hiển thị thông tin cũ của bạn ở đây...
        for i, det in enumerate(sorted_dets):
            current_info = get_food_info(det["class_name"])
            if i == 0:
                render_one_food(det)
            else:
                with st.expander(f"🍽️ {current_info['ten_hien_thi']} — {det['confidence']*100:.1f}%"):
                    render_one_food(det)

    with tab_qa:
        render_tts_section(top_det, voice, lang)
        caption = generate_caption(top_det["class_name"], top_info, top_det["confidence"], lang)
        st.info(caption)
        
        user_question = st.text_input("Đặt câu hỏi về món ăn", key="food_question")
        if user_question:
            with st.spinner("🤖 Hệ thống đang xử lý..."):
                answer, source = answer_question(user_question, top_det["class_name"], top_info, top_det["confidence"], lang)
                st.success(f"🧠 **AI Expert:** {answer}")

    with tab_ai:
        if use_blip and source_img is not None and top_det.get("bbox"):
            with st.spinner("🧠 BLIP-1 đang mô tả ảnh..."):
                try:
                    cropped = crop_food_image(source_img, top_det["bbox"])
                    blip_caption = generate_caption_from_image(cropped)
                    st.write(f"**AI Vision:** {blip_caption}")
                except Exception as e:
                    st.warning(f"⚠️ BLIP-1 gặp lỗi: {e}")

    with tab_rec:
        # --- Đảm bảo hiển thị dữ liệu mới nhất từ session_state ---
        st.markdown("### 💠 Personal Food Dashboard")
        history = food_history.get_session_history(session_id)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Tổng Calo", f"{food_history.get_total_calo(session_id)} kcal")
        with c2: st.metric("Món đã quét", len(history))
        with c3: 
            st.markdown("**Phân khúc**")
            st.write(get_user_segment(session_id))
            
        # ... giữ nguyên phần còn lại của tab_rec ...
        recs = get_recommendations(food_key=top_det["class_name"], session_id=session_id)
        # ... render gợi ý ...

        # --- Taste Profile Tags ---
        st.markdown("**Gu ẩm thực của bạn:**")
        tags = get_user_taste_profile(session_id)
        st.markdown(f"<div class='tag-container'>{''.join([f'<span class=\"tag-chip\">{t}</span>' for t in tags])}</div>", unsafe_allow_html=True)
        
        st.divider()

        # --- Recommendations (Main Feature) ---
        st.markdown("### 🎯 Gợi ý AI hôm nay")
        recs = get_recommendations(food_key=top_det["class_name"], session_id=session_id)
        
        for key, rec_score, sim, adj in recs:
            info = get_food_info(key)
            st.markdown(f"""
            <div class='rec-card'>
                <div style='display:flex; justify-content:space-between;'>
                    <strong>{info['ten_hien_thi']}</strong>
                    <span style='color:#f97316; font-weight:bold;'>{rec_score*100:.0f}% Match</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(rec_score)

        # --- History (Hidden in Expander) ---
        with st.expander("📜 Lịch sử khám phá chi tiết"):
            history = food_history.get_session_history(session_id)
            if history:
                st.table(pd.DataFrame(history, columns=["Món", "Tin cậy", "Calo", "Vùng", "Thời gian"]))
            else:
                st.write("Chưa có lịch sử.")

        # --- Logic Explanation ---
        with st.expander("💡 Tại sao AI gợi ý các món này?"):
            bias = SEGMENT_CALO_BIAS.get(get_user_segment(session_id), 0)
            st.write(f"Hệ thống đang áp dụng trọng số **{bias}** vào thuật toán để tối ưu hóa gợi ý dựa trên phân khúc người dùng của bạn.")



def main():
    """Luồng điều khiển trung tâm chính của ứng dụng và xử lý gom nhóm scope sidebar"""
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"❌ Không tìm thấy file model `{MODEL_PATH}`.")
        return

    # ─── ĐỊNH NGHĨA TOÀN BỘ BIẾN SIDEBAR TRONG HÀM MAIN (SỬA TRIỆT ĐỂ LỖI PYLANCE) ───
    with st.sidebar:
        lang_choice = st.selectbox("🌐 Giọng đọc & Hỏi đáp", ["Tiếng Việt", "English"], index=0)
        lang = "vi" if lang_choice == "Tiếng Việt" else "en"
        
        CONF_THRESHOLD = st.slider("🎯 Độ tin cậy tối thiểu", 0.1, 1.0, 0.35, 0.05)
        st.divider()
        
        voice_title = "**🔊 TTS Voice (English)**" if lang == "en" else "**🔊 Giọng đọc**"
        voice_options = ["en-US-AriaNeural (Female)", "en-US-GuyNeural (Male)"] if lang == "en" else ["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"]
        
        st.markdown(voice_title)
        voice_choice = st.radio("Voice Configuration", options=voice_options, index=0, label_visibility="collapsed")
        
        if lang == "vi":
            TTS_VOICE = "vi-VN-HoaiMyNeural" if "HoaiMy" in voice_choice else "vi-VN-NamMinhNeural"
        else:
            TTS_VOICE = "en-US-AriaNeural" if "Aria" in voice_choice else "en-US-GuyNeural"
            
        st.divider()
        st.markdown("**🧠 Phân tích ảnh thật (BLIP-1)**")
        if blip_available():
            use_blip = st.toggle("Bật phân tích ảnh bằng AI", value=False)
        else:
            st.caption("⚠️ Chưa cài transformers")
            use_blip = False

        st.divider()
        st.markdown("**✨ Tính năng**")
        st.markdown("• Nhận diện nhiều món\n• Tính calo & macro\n• Mô tả chi tiết\n• Vùng miền, giá, thời điểm\n• Gợi ý món tương tự\n• 🔊 Đọc to tự động\n• 💬 Hỏi đáp về món ăn")
        st.divider()
        st.caption("Demo MVP • Team AI Food Assistant © 2026")

    # Vẽ bố cục tiêu đề đầu trang cố định tiếng Việt
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
    with col_title:
        st.markdown("<div class='main-title'>Food AI Assistant</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Nhận diện · Tính calo · Mô tả · Gợi ý · Hỏi đáp về món ăn Việt Nam bằng AI</div>", unsafe_allow_html=True)
    st.divider()

    # Phân tách bố cục cột Trái (Ảnh đầu vào) - Phải (Kết quả phân tích)
    left, right = st.columns([1, 1.12], gap="large")

    with left:
        st.markdown("### 📤 Đầu vào ảnh")
        tab_upload, tab_cam = st.tabs(["📁 Upload / Kéo thả", "📸 Camera trực tiếp"])
        source_img = None
        with tab_upload:
            uploaded = st.file_uploader("Chọn hoặc kéo thả ảnh món ăn vào đây", type=["jpg", "jpeg", "png"], key="upload")
            if uploaded:
                source_img = Image.open(uploaded).convert("RGB")
        with tab_cam:
            cam = st.camera_input("Chụp món ăn ngay", key="camera")
            if cam:
                source_img = Image.open(cam).convert("RGB") # Sửa hàm open chuẩn xác
        if source_img:
            st.image(source_img, width='stretch')
            st.caption("Ảnh gốc")

    with right:
        st.markdown("### 📊 Kết quả nhận diện")
        if source_img is not None:
            with st.spinner("🤖 AI đang phân tích món ăn..."):
                t0 = time.time()
                detections, annotated = run_detection(model, source_img, CONF_THRESHOLD)
                elapsed = time.time() - t0
            if annotated is not None:
                # annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated, channels="BGR", width='stretch')
                st.caption(f"✅ Xử lý xong trong {elapsed:.2f}s — phát hiện {len(detections)} đối tượng")
            render_detections(detections, TTS_VOICE, source_img, use_blip, lang)
        else:
            st.info("⬅️ Hãy upload ảnh hoặc bật camera bên trái để bắt đầu.")

if __name__ == "__main__":
    main()