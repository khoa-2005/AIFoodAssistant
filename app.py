import asyncio
import concurrent.futures
import io
import time
from pathlib import Path
import sys

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

# Import các hàm xử lý ẩm thực nội bộ
from food_info import get_food_info
from food_reasoner import generate_caption, answer_question
from food_blip import (
    is_available as blip_available,
    generate_caption_from_image,
    answer_question_from_image,
)

# Cấu hình hằng số hệ thống
MODEL_PATH  = "bestv8s.pt"
PAGE_TITLE  = "Food AI Assistant"

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


def render_detections(detections: list, voice: str, source_img: Image.Image = None, use_blip: bool = False, lang: str = "vi"):
    """Điều phối kết quả phân bổ vào hệ thống 3 Tabs chức năng cố định tiếng Việt"""
    if not detections:
        st.error("❌ Không nhận diện được món ăn nào. Hãy thử ảnh rõ hơn!")
        return

    sorted_dets = dedup_detections(detections)
    top_det     = sorted_dets[0]
    top_info    = get_food_info(top_det["class_name"])

    st.markdown(
        f"<div class='top-banner'>🎯 {top_info['ten_hien_thi']} — {top_det['confidence']*100:.1f}%</div>",
        unsafe_allow_html=True,
    )

    # Đặt tên tab cố định tiếng Việt, tab hỏi đáp có thêm ghi chú nhỏ nếu chọn tiếng Anh
    qa_tab_title = "💬 Hỏi đáp (English Q&A)" if lang == "en" else "💬 Hỏi đáp"
    tab_info, tab_qa, tab_ai = st.tabs(["📊 Thông tin món ăn", qa_tab_title, "🧠 Phân tích ảnh AI"])

    with tab_info:
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
        
        # Cấu hình nhãn tương tác chat linh hoạt theo ngôn ngữ
        ask_lbl = "Ask a question about this dish (Đặt câu hỏi bằng Tiếng Anh)" if lang == "en" else "Đặt câu hỏi về món ăn"
        spin_lbl = "🤖 AI Expert is processing your question..." if lang == "en" else "🤖 Hệ thống đang xử lý câu hỏi..."
        
        user_question = st.text_input(ask_lbl, key="food_question")
        if user_question:
            with st.spinner(spin_lbl):
                answer, source = answer_question(user_question, top_det["class_name"], top_info, top_det["confidence"], lang)
            
            if source == "AI":
                prefix = "🧠 **AI Expert Answer:** {}" if lang == "en" else "🧠 **Chuyên gia AI (Gemini 2.5) trả lời:** {}"
                st.success(prefix.format(answer))
            else:
                prefix = "📋 **System Data Answer:** {}" if lang == "en" else "📋 **Dữ liệu hệ thống (Local DB) trả lời:** {}"
                st.info(prefix.format(answer))

    with tab_ai:
        if use_blip and source_img is not None and top_det.get("bbox"):
            with st.spinner("🧠 BLIP-1 đang mô tả ảnh..."):
                try:
                    cropped = crop_food_image(source_img, top_det["bbox"])
                    blip_caption = generate_caption_from_image(cropped)
                    st.write(f"**AI Vision:** {blip_caption}")
                except Exception as e:
                    st.warning(f"⚠️ BLIP-1 gặp lỗi: {e}")


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