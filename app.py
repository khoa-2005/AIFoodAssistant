import asyncio
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


# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "best.pt"
PAGE_TITLE  = "Food AI Assistant"
TTS_VOICE   = "vi-VN-HoaiMyNeural"   # hoặc "vi-VN-NamMinhNeural"
TTS_RATE    = "-5%"                   # âm = chậm hơn, dương = nhanh hơn
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f8fafc 0%, #fefce8 100%); }

    .main-title {
        font-size: 3.4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f97316, #eab308);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .top-banner {
        background: linear-gradient(90deg, #dcfce7, #d1fae5);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 18px;
        font-weight: 700;
        color: #15803d;
        font-size: 1.05rem;
    }
    .food-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0 0 8px 0;
    }
    .conf-badge {
        background: linear-gradient(90deg, #22c55e, #86efac);
        color: white;
        padding: 7px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(34,197,94,0.3);
        display: inline-block;
        margin-bottom: 6px;
    }
    .calo-box {
        background: linear-gradient(135deg, #fefce8, #fef3c7);
        border: 3px solid #fbbf24;
        border-radius: 18px;
        padding: 20px 24px;
        text-align: center;
        margin: 14px 0;
    }
    .calo-label { color: #b45309; font-weight: 700; font-size: 0.9rem; margin: 0; }
    .calo-value { font-size: 2.8rem; font-weight: 900; color: #c2410c; margin: 4px 0 0 0; }
    .desc-box {
        background: #f8fafc;
        padding: 16px 20px;
        border-radius: 14px;
        border-left: 6px solid #f97316;
        color: #334155;
        margin: 8px 0 12px 0;
        font-size: 0.97rem;
        line-height: 1.6;
    }
    .macro-row {
        display: flex;
        gap: 10px;
        margin: 10px 0 14px 0;
        flex-wrap: wrap;
    }
    .macro-chip {
        background: #fff7ed;
        border: 1.5px solid #fdba74;
        border-radius: 12px;
        padding: 7px 15px;
        font-weight: 600;
        color: #9a3412;
        font-size: 0.88rem;
    }
    .info-row {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 9px 15px;
        margin: 5px 0;
        color: #334155;
        font-size: 0.91rem;
        line-height: 1.5;
    }
    .suggest-box {
        background: #ecfdf5;
        padding: 18px 22px;
        border-radius: 18px;
        border: 2px solid #34d399;
        margin-top: 14px;
    }
    .suggest-tag {
        background: #34d399;
        color: white;
        padding: 7px 16px;
        border-radius: 50px;
        margin: 4px 6px 4px 0;
        display: inline-block;
        font-weight: 600;
        font-size: 0.88rem;
    }
    /* TTS section */
    .tts-section {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border: 2px solid #fb923c;
        border-radius: 16px;
        padding: 16px 20px;
        margin-top: 20px;
    }
    .tts-label {
        color: #9a3412;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
if "tts_audio_bytes" not in st.session_state:
    st.session_state.tts_audio_bytes = None   # bytes MP3 của lần detect gần nhất
if "tts_food_key" not in st.session_state:
    st.session_state.tts_food_key = None      # key để biết có cần re-generate không


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046748.png", width=110)
    st.markdown("<h3 style='color:#f97316; text-align:center;'>Food AI</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>Nhận diện món ăn Việt Nam</p>", unsafe_allow_html=True)

    st.divider()
    CONF_THRESHOLD = st.slider("🎯 Độ tin cậy tối thiểu", 0.1, 1.0, 0.35, 0.05)

    st.divider()
    st.markdown("**🔊 Giọng đọc**")
    voice_choice = st.radio(
        "Chọn giọng",
        options=["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"],
        index=0,
        label_visibility="collapsed",
    )
    TTS_VOICE = (
        "vi-VN-HoaiMyNeural"
        if "HoaiMy" in voice_choice
        else "vi-VN-NamMinhNeural"
    )

    st.divider()
    st.markdown("**✨ Tính năng**")
    st.markdown(
        "• Nhận diện nhiều món\n"
        "• Tính calo & macro\n"
        "• Mô tả chi tiết\n"
        "• Vùng miền, giá, thời điểm\n"
        "• Gợi ý món tương tự\n"
        "• 🔊 Đọc to tự động (edge-tts)"
    )

    st.divider()
    st.caption("Demo MVP • Team AI Food Assistant © 2026")


# ─── HEADER ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
with col_title:
    st.markdown("<div class='main-title'>Food AI Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Nhận diện · Tính calo · Mô tả · Gợi ý món ăn Việt Nam bằng AI</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ─── MODEL ────────────────────────────────────────────────────────────────────
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
            detections.append({
                "class_name": result.names[cls_id],
                "confidence": float(box.conf[0]),
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


# ─── TTS (edge-tts) ───────────────────────────────────────────────────────────
def _normalize_tts(text: str) -> str:
    """Chuẩn hóa chuỗi cho TTS:
    - '6g' → '6 gam'
    - '15.000 - 30.000 VNĐ' → '15 nghìn đến 30 nghìn Việt Nam đồng'
    """
    import re

    # số + g → số + gam (chỉ khi g đứng ngay sau số, không dính chữ khác)
    text = re.sub(r'(\d+)\s*g\b', r'\1 gam', text)

    # Giá tiền dạng: "15.000 - 30.000 VNĐ" hoặc "15.000 - 30.000 đ"
    def format_price_range(m):
        lo  = m.group(1).replace(".", "")
        hi  = m.group(2).replace(".", "")

        def to_words(n_str):
            n = int(n_str)
            if n >= 1_000_000:
                return f"{n // 1_000_000} triệu"
            elif n >= 1_000:
                return f"{n // 1_000} nghìn"
            return n_str

        return f"{to_words(lo)} đến {to_words(hi)} Việt Nam đồng"

    text = re.sub(
        r'([\d.]+)\s*-\s*([\d.]+)\s*(?:VNĐ|đ)\b',
        format_price_range,
        text,
    )

    return text


def build_tts_text(info: dict, conf: float) -> str:
    """Ghép nội dung món ăn thành chuỗi tiếng Việt tự nhiên để đọc."""
    raw = " ".join([
        f"Món ăn được nhận diện là {info['ten_hien_thi']}, "
        f"với độ tin cậy {conf*100:.0f} phần trăm.",
        f"Mô tả: {info['mo_ta']}",
        f"Lượng calo ước tính: {info.get('calo', 'không rõ')}.",
        f"Chất đạm: {info.get('protein', 'không rõ')}. "
        f"Tinh bột: {info.get('carb', 'không rõ')}. "
        f"Chất béo: {info.get('fat', 'không rõ')}.",
        f"Vùng miền: {info.get('vung_mien', 'không rõ')}.",
        f"Thành phần chính: {info.get('thanh_phan', 'không rõ')}.",
        f"Giá tham khảo: {info.get('gia_trung_binh', 'không rõ')}.",
        f"Khuyến nghị dinh dưỡng: {info.get('khuyen_nghi', 'không có')}.",
    ])
    return _normalize_tts(raw)


async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    """Gọi edge-tts async, trả về bytes MP3 trong RAM."""
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def generate_tts(text: str, voice: str, rate: str = "-5%") -> bytes:
    """Wrapper đồng bộ để gọi từ Streamlit."""
    return asyncio.run(_synthesize(text, voice, rate))


def render_tts_section(top_det: dict, voice: str):
    """
    Hiển thị section TTS cho món confidence cao nhất.
    - Tự động generate audio khi detect xong (lần đầu hoặc món mới).
    - Cache trong session_state, không re-generate khi Streamlit re-run.
    - Có nút để người dùng bấm phát lại.
    """
    info    = get_food_info(top_det["class_name"])
    conf    = top_det["confidence"]
    tts_key = f"{top_det['class_name']}_{voice}"   # key cache

    st.markdown(
        "<div class='tts-section'>"
        "<p class='tts-label'>🔊 Đọc to kết quả — món có độ tin cậy cao nhất</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Generate nếu chưa có hoặc món/giọng thay đổi
    if st.session_state.tts_food_key != tts_key:
        with st.spinner("🎙️ Đang tổng hợp giọng nói..."):
            try:
                text = build_tts_text(info, conf)
                audio_bytes = generate_tts(text, voice, TTS_RATE)
                st.session_state.tts_audio_bytes = audio_bytes
                st.session_state.tts_food_key    = tts_key
            except Exception as e:
                st.error(f"❌ Lỗi TTS: {e}")
                return

    # Phát audio — autoplay=True để tự động phát sau detect
    if st.session_state.tts_audio_bytes:
        st.audio(
            st.session_state.tts_audio_bytes,
            format="audio/mp3",
            autoplay=True,
        )
        st.caption(
            f"🎙️ Giọng: {'HoaiMy (Nữ)' if 'HoaiMy' in voice else 'NamMinh (Nam)'} · "
            f"Đang đọc: **{info['ten_hien_thi']}**"
        )


# ─── RENDER KẾT QUẢ ───────────────────────────────────────────────────────────
def render_one_food(det: dict):
    info = get_food_info(det["class_name"])
    conf = det["confidence"]

    # Tên + badge
    st.markdown(
        f"<p class='food-name'>🥗 {info['ten_hien_thi']}</p>"
        f"<span class='conf-badge'>Độ tin cậy: {conf*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.progress(conf)

    # Calo box
    st.markdown(
        f"<div class='calo-box'>"
        f"<p class='calo-label'>🔥 Lượng calo ước tính (1 khẩu phần)</p>"
        f"<p class='calo-value'>{info.get('calo', 'N/A')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Mô tả
    st.markdown("**📖 Mô tả chi tiết**")
    st.markdown(
        f"<div class='desc-box'>{info['mo_ta']}</div>",
        unsafe_allow_html=True,
    )

    # Macro chips
    st.markdown(
        f"<div class='macro-row'>"
        f"<span class='macro-chip'>🥩 Protein: {info.get('protein','N/A')}</span>"
        f"<span class='macro-chip'>🍚 Carb: {info.get('carb','N/A')}</span>"
        f"<span class='macro-chip'>🥑 Fat: {info.get('fat','N/A')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Info rows
    rows = [
        ("📍", "Vùng miền",         info.get("vung_mien",         "N/A")),
        ("🍽️", "Khẩu phần",         info.get("khau_phan",         "Không rõ")),
        ("🧾", "Thành phần chính",  info.get("thanh_phan",        "Không rõ")),
        ("💰", "Giá tham khảo",     info.get("gia_trung_binh",    "Không rõ")),
        ("⏰", "Thời điểm phù hợp", info.get("thoi_diem_phu_hop", "Không rõ")),
        ("❤️", "Chỉ số sức khỏe",   info.get("chi_so_suc_khoe",   "Không rõ")),
        ("💡", "Khuyến nghị",       info.get("khuyen_nghi",       "Không có.")),
    ]
    for icon, label, value in rows:
        st.markdown(
            f"<div class='info-row'>{icon} <strong>{label}:</strong> {value}</div>",
            unsafe_allow_html=True,
        )

    # Gợi ý món tương tự (nếu có trong food_info)
    if info.get("goi_y"):
        tags_html = "".join(
            f"<span class='suggest-tag'>🍽️ {mon}</span>"
            for mon in info["goi_y"]
        )
        st.markdown(
            f"<div class='suggest-box'>"
            f"<strong>💡 Gợi ý món ăn tương tự:</strong><br><br>{tags_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_detections(detections: list, voice: str):
    if not detections:
        st.error("❌ Không nhận diện được món ăn nào. Hãy thử ảnh rõ hơn!")
        return

    sorted_dets = dedup_detections(detections)
    top_det     = sorted_dets[0]
    top_info    = get_food_info(top_det["class_name"])

    # Banner tóm tắt
    st.markdown(
        f"<div class='top-banner'>"
        f"🎯 AI dự đoán món chính: "
        f"<span style='color:#166534;'>{top_info['ten_hien_thi']}</span>"
        f" — độ tin cậy {top_det['confidence']*100:.1f}%"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── TTS tự động cho món top 1 ──
    render_tts_section(top_det, voice)

    st.divider()

    # ── Chi tiết từng món ──
    for det in sorted_dets:
        with st.container(border=True):
            render_one_food(det)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(
            f"❌ Không tìm thấy file model `{MODEL_PATH}`. "
            "Hãy train model trước để tạo file best.pt."
        )
        return

    left, right = st.columns([1, 1.12], gap="large")

    # ── CỘT TRÁI: đầu vào ──
    with left:
        st.markdown("### 📤 Đầu vào ảnh")
        tab_upload, tab_cam = st.tabs(["📁 Upload / Kéo thả", "📸 Camera trực tiếp"])

        source_img = None
        with tab_upload:
            uploaded = st.file_uploader(
                "Chọn hoặc kéo thả ảnh món ăn vào đây",
                type=["jpg", "jpeg", "png"],
                key="upload",
            )
            if uploaded:
                source_img = Image.open(uploaded).convert("RGB")

        with tab_cam:
            cam = st.camera_input("Chụp món ăn ngay", key="camera")
            if cam:
                source_img = Image.open(cam).convert("RGB")

        if source_img:
            st.image(source_img, caption="Ảnh gốc", use_container_width=True)

    # ── CỘT PHẢI: kết quả ──
    with right:
        st.markdown("### 📊 Kết quả nhận diện")

        if source_img is not None:
            with st.spinner("🤖 AI đang phân tích món ăn..."):
                t0 = time.time()
                detections, annotated = run_detection(model, source_img, CONF_THRESHOLD)
                elapsed = time.time() - t0

            if annotated is not None:
                st.image(
                    annotated,
                    channels="BGR",
                    caption=f"✅ Xử lý xong trong {elapsed:.2f}s — phát hiện {len(detections)} đối tượng",
                    use_container_width=True,
                )

            render_detections(detections, TTS_VOICE)

        else:
            st.info("⬅️ Hãy upload ảnh hoặc bật camera bên trái để bắt đầu.")


if __name__ == "__main__":
    main()
