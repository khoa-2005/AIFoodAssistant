import asyncio
import concurrent.futures
import io
import time
from pathlib import Path

import edge_tts
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO

from food_info import get_food_info
from food_reasoner import generate_caption, answer_question
from food_blip import (
    is_available as blip_available,
    generate_caption_from_image,
    answer_question_from_image,
)


# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "bestv8s.pt"
PAGE_TITLE  = "Food AI Assistant"
TTS_VOICE   = "vi-VN-HoaiMyNeural"
TTS_RATE    = "-5%"
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .suggest-box { background: #ecfdf5; padding: 18px 22px; border-radius: 18px; border: 2px solid #34d399; margin-top: 14px; }
    .suggest-tag { background: #34d399; color: white; padding: 7px 16px; border-radius: 50px; margin: 4px 6px 4px 0; display: inline-block; font-weight: 600; font-size: 0.88rem; }
    .tts-section { background: linear-gradient(135deg, #fff7ed, #fef3c7); border: 2px solid #fb923c; border-radius: 16px; padding: 16px 20px; margin-top: 20px; }
    .tts-label { color: #9a3412; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

if "tts_audio_bytes" not in st.session_state:
    st.session_state.tts_audio_bytes = None
if "tts_food_key" not in st.session_state:
    st.session_state.tts_food_key = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046748.png", width=110)
    st.markdown("<h3 style='color:#f97316; text-align:center;'>Food AI</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>Nhận diện món ăn Việt Nam</p>", unsafe_allow_html=True)
    st.divider()
    CONF_THRESHOLD = st.slider("🎯 Độ tin cậy tối thiểu", 0.1, 1.0, 0.35, 0.05)
    st.divider()
    st.markdown("**🔊 Giọng đọc**")
    voice_choice = st.radio("Chọn giọng", options=["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"], index=0, label_visibility="collapsed")
    TTS_VOICE = "vi-VN-HoaiMyNeural" if "HoaiMy" in voice_choice else "vi-VN-NamMinhNeural"
    st.divider()
    st.markdown("**🧠 Phân tích ảnh thật (BLIP-1)**")
    if blip_available():
        use_blip = st.toggle(
            "Bật phân tích ảnh bằng AI",
            value=False,
            help="BLIP-1 nhìn vào ảnh thật để mô tả và trả lời câu hỏi. Lần đầu tải model ~900MB, mất 20-60 giây."
        )
    else:
        st.caption("⚠️ Chưa cài transformers\n`pip install transformers deep-translator`")
        use_blip = False

    st.divider()
    st.markdown("**✨ Tính năng**")
    st.markdown("• Nhận diện nhiều món\n• Tính calo & macro\n• Mô tả chi tiết\n• Vùng miền, giá, thời điểm\n• Gợi ý món tương tự\n• 🔊 Đọc to tự động (edge-tts)\n• 💬 Hỏi đáp về món ăn\n• 🧠 Phân tích ảnh thật (BLIP-1)")
    st.divider()
    st.caption("Demo MVP • Team AI Food Assistant © 2026")

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
with col_title:
    st.markdown("<div class='main-title'>Food AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Nhận diện · Tính calo · Mô tả · Gợi ý · Hỏi đáp về món ăn Việt Nam bằng AI</div>", unsafe_allow_html=True)
st.divider()


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
    """Crop ảnh theo bounding box từ YOLO để truyền vào BLIP."""
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
    def format_price_range(m):
        lo = m.group(1).replace(".", "")
        hi = m.group(2).replace(".", "")
        def to_words(n_str):
            n = int(n_str)
            if n >= 1_000_000:
                return f"{n // 1_000_000} triệu"
            elif n >= 1_000:
                return f"{n // 1_000} nghìn"
            return n_str
        return f"{to_words(lo)} đến {to_words(hi)} Việt Nam đồng"
    text = re.sub(r'([\d.]+)\s*-\s*([\d.]+)\s*(?:VNĐ|đ)\b', format_price_range, text)
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
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def generate_tts(text: str, voice: str, rate: str = "-5%") -> bytes:
    """Wrapper tương thích với thread của Streamlit (không có event loop sẵn)."""
    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_synthesize(text, voice, rate))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_in_new_loop)
        return future.result()


def render_tts_section(top_det: dict, voice: str):
    info    = get_food_info(top_det["class_name"])
    conf    = top_det["confidence"]
    tts_key = f"{top_det['class_name']}_{voice}"

    st.markdown(
        "<div class='tts-section'><p class='tts-label'>🔊 Đọc to kết quả — món có độ tin cậy cao nhất</p></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.tts_food_key != tts_key:
        with st.spinner("🎙️ Đang tổng hợp giọng nói..."):
            try:
                text = build_tts_text(info, conf)
                audio_bytes = generate_tts(text, voice, TTS_RATE)
                st.session_state.tts_audio_bytes = audio_bytes
                st.session_state.tts_food_key    = tts_key
            except Exception as e:
                st.warning(f"⚠️ Không thể tổng hợp giọng nói lúc này. Lỗi: {e}")
                return

    if st.session_state.tts_audio_bytes:
        st.audio(st.session_state.tts_audio_bytes, format="audio/mp3", autoplay=True)
        st.caption(f"🎙️ Giọng: {'HoaiMy (Nữ)' if 'HoaiMy' in voice else 'NamMinh (Nam)'} · Đang đọc: **{info['ten_hien_thi']}**")


def render_one_food(det: dict):
    info = get_food_info(det["class_name"])
    conf = det["confidence"]

    st.markdown(
        f"<p class='food-name'>🥗 {info['ten_hien_thi']}</p>"
        f"<span class='conf-badge'>Độ tin cậy: {conf*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.progress(conf)

    st.markdown(
        f"<div class='calo-box'><p class='calo-label'>🔥 Lượng calo ước tính (1 khẩu phần)</p>"
        f"<p class='calo-value'>{info.get('calo', 'N/A')}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**📖 Mô tả chi tiết**")
    st.markdown(f"<div class='desc-box'>{info['mo_ta']}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='macro-row'>"
        f"<span class='macro-chip'>🥩 Protein: {info.get('protein','N/A')}</span>"
        f"<span class='macro-chip'>🍚 Carb: {info.get('carb','N/A')}</span>"
        f"<span class='macro-chip'>🥑 Fat: {info.get('fat','N/A')}</span>"
        f"</div>", unsafe_allow_html=True,
    )

    rows = [
        ("📍", "Vùng miền",         info.get("vung_mien",         "N/A")),
        ("🍽️", "Khẩu phần",         info.get("khau_phan",         "Không rõ")),
        ("🧾", "Thành phần chính",   info.get("thanh_phan",        "Không rõ")),
        ("💰", "Giá tham khảo",     info.get("gia_trung_binh",    "Không rõ")),
        ("⏰", "Thời điểm phù hợp", info.get("thoi_diem_phu_hop", "Không rõ")),
        ("❤️", "Chỉ số sức khỏe",   info.get("chi_so_suc_khoe",   "Không rõ")),
        ("💡", "Khuyến nghị",       info.get("khuyen_nghi",       "Không có.")),
    ]
    
    # Gộp info-row thành lưới 2 cột thay vì xếp hàng dọc
    c1, c2 = st.columns(2)
    left_rows  = rows[:4]   # vùng miền, khẩu phần, thành phần, giá
    right_rows = rows[4:]   # thời điểm, sức khỏe, khuyến nghị

    with c1:
        for icon, label, value in left_rows:
            st.markdown(f"<div class='info-row'>{icon} <strong>{label}:</strong> {value}</div>", unsafe_allow_html=True)
    with c2:
        for icon, label, value in right_rows:
            st.markdown(f"<div class='info-row'>{icon} <strong>{label}:</strong> {value}</div>", unsafe_allow_html=True)

    if info.get("goi_y"):
        tags_html = "".join(f"<span class='suggest-tag'>🍽️ {mon}</span>" for mon in info["goi_y"])
        st.markdown(
            f"<div class='suggest-box'><strong>💡 Gợi ý món ăn tương tự:</strong><br><br>{tags_html}</div>",
            unsafe_allow_html=True,
        )


def render_detections(detections: list, voice: str, source_img: Image.Image = None, use_blip: bool = False):
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

    # Chia Tabs tách nội dung hiển thị tinh gọn
    tab_info, tab_qa, tab_ai = st.tabs(["📊 Thông tin món ăn", "💬 Hỏi đáp", "🧠 Phân tích ảnh AI"])

    with tab_info:
        for i, det in enumerate(sorted_dets):
            if i == 0:
                render_one_food(det)          # Món chính: Hiện đầy đủ
            else:
                # Món phụ: Gấp gọn lại mặc định trong expander
                with st.expander(f"🍽️ {get_food_info(det['class_name'])['ten_hien_thi']} — {det['confidence']*100:.1f}%"):
                    render_one_food(det)

    with tab_qa:
        render_tts_section(top_det, voice)
        caption = generate_caption(top_det["class_name"], top_info, top_det["confidence"])
        st.info(caption)
        
        user_question = st.text_input("Đặt câu hỏi về món ăn", key="food_question")
        if user_question:
            answer = answer_question(user_question, top_det["class_name"], top_info, top_det["confidence"])
            st.success(f"📋 **Từ cơ sở dữ liệu:** {answer}")
            
            if use_blip and source_img is not None and top_det.get("bbox"):
                with st.spinner("🧠 BLIP-1 đang phân tích..."):
                    try:
                        cropped = crop_food_image(source_img, top_det["bbox"])
                        blip_answer = answer_question_from_image(user_question, cropped)
                        st.info(f"🧠 **Từ ảnh thật:** {blip_answer}")
                    except Exception as e:
                        st.warning(f"⚠️ BLIP-1 gặp lỗi: {e}")

    with tab_ai:
        if use_blip and source_img is not None and top_det.get("bbox"):
            with st.spinner("🧠 BLIP-1 đang mô tả ảnh..."):
                try:
                    cropped = crop_food_image(source_img, top_det["bbox"])
                    blip_caption = generate_caption_from_image(cropped)
                    st.write(f"**AI nhìn thấy:** {blip_caption}")
                    st.caption("BLIP-1 mô tả nội dung thật trong ảnh, không tra bảng dữ liệu.")
                except Exception as e:
                    st.warning(f"⚠️ BLIP-1 gặp lỗi khi phân tích: {e}")
        else:
            st.info("Bật '🧠 Phân tích ảnh thật (BLIP-1)' ở sidebar để dùng tính năng này.")


def main():
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"❌ Không tìm thấy file model `{MODEL_PATH}`.")
        return

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
                source_img = Image.open(cam).convert("RGB")
        if source_img:
            st.image(source_img, caption="Ảnh gốc", use_container_width=True)

    with right:
        st.markdown("### 📊 Kết quả nhận diện")
        if source_img is not None:
            with st.spinner("🤖 AI đang phân tích món ăn..."):
                t0 = time.time()
                detections, annotated = run_detection(model, source_img, CONF_THRESHOLD)
                elapsed = time.time() - t0
            if annotated is not None:
                st.image(annotated, channels="BGR",
                         caption=f"✅ Xử lý xong trong {elapsed:.2f}s — phát hiện {len(detections)} đối tượng",
                         use_container_width=True)
            render_detections(detections, TTS_VOICE, source_img, use_blip)
        else:
            st.info("⬅️ Hãy upload ảnh hoặc bật camera bên trái để bắt đầu.")


if __name__ == "__main__":
    main()
