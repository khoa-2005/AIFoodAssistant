import streamlit as st
from PIL import Image
import time

from food_info import get_food_info

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Food AI Assistant",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== BRIGHT FRESH THEME ======================
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
    }
    
    .res-card {
        background: white;
        padding: 32px;
        border-radius: 22px;
        box-shadow: 0 15px 40px rgba(249, 115, 22, 0.15);
        border: 1px solid #fed7aa;
        transition: all 0.3s ease;
    }
    .res-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(249, 115, 22, 0.2);
    }
    
    .food-name {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
    }
    
    .conf-badge {
        background: linear-gradient(90deg, #22c55e, #86efac);
        color: white;
        padding: 8px 22px;
        border-radius: 50px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(34,197,94,0.3);
    }
    
    .calo-box {
        background: linear-gradient(135deg, #fefce8, #fef3c7);
        border: 3px solid #fbbf24;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        margin: 24px 0;
    }
    .calo-value { font-size: 3.2rem; font-weight: 900; color: #c2410c; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046748.png", width=110)
    st.markdown("<h3 style='color:#f97316; text-align:center;'>Food AI</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>Nhận diện món ăn Việt Nam</p>", unsafe_allow_html=True)
    
    st.divider()
    CONF_THRESHOLD = st.slider("🎯 Độ tin cậy tối thiểu", 0.1, 1.0, 0.35, 0.05)
    
    st.divider()
    st.markdown("**✨ Tính năng**")
    st.markdown("• Nhận diện nhanh\n• Tính calo chính xác\n• Mô tả chi tiết\n• Gợi ý món tương tự")
    
    st.divider()
    st.caption("Demo MVP • Team AI Food Assistant © 2026")

# ====================== HEADER (Đẹp hơn) ======================
col1, col2 = st.columns([1, 5])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)  # Logo khác (bát phở)

with col2:
    st.markdown("<div class='main-title'>Food AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Nhận diện - Tính calo - Mô tả - Gợi ý món ăn Việt Nam bằng AI</div>", unsafe_allow_html=True)

st.divider()

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO("best.pt")
    except:
        st.error("❌ Không tìm thấy file best.pt")
        return None

model = load_yolo_model()

# ====================== LAYOUT CHÍNH ======================
left, right = st.columns([1, 1.12], gap="large")

with left:
    st.markdown("### 📤 Đầu vào ảnh")
    tab1, tab2 = st.tabs(["📁 Upload / Kéo thả", "📸 Camera trực tiếp"])
    
    source_img = None
    with tab1:
        uploaded = st.file_uploader("Chọn hoặc kéo thả ảnh món ăn vào đây", 
                                  type=["jpg","jpeg","png"], key="upload")
        if uploaded:
            source_img = Image.open(uploaded)
    
    with tab2:
        cam = st.camera_input("Chụp món ăn ngay")
        if cam:
            source_img = Image.open(cam)
    
    if source_img:
        st.image(source_img, caption="Ảnh gốc", use_container_width=True)

with right:
    st.markdown("### 📊 Kết quả nhận diện")
    
    if source_img is not None and model is not None:
        with st.spinner("🤖 AI đang phân tích món ăn..."):
            t0 = time.time()
            results = model.predict(source_img, conf=CONF_THRESHOLD)
            elapsed = time.time() - t0

        st.image(results[0].plot(), 
                caption=f"✅ Xử lý xong trong {elapsed:.2f} giây", 
                use_container_width=True)

        boxes = results[0].boxes
        if len(boxes) > 0:
            best_idx = boxes.conf.argmax().item()
            best_cls = int(boxes.cls[best_idx])
            best_name = model.names[best_cls]
            best_conf = boxes.conf[best_idx].item() * 100

            food = get_food_info(best_name)

            if food:
                st.markdown(f"""
                    <div class="res-card">
                        <p class="food-name">🥗 {food['ten_hien_thi']}</p>
                        <span class="conf-badge">Độ tin cậy: {best_conf:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="calo-box">
                        <p style="color:#b45309; font-weight:700;">🔥 Lượng calo ước tính (1 khẩu phần)</p>
                        <p class="calo-value">{food.get('calo', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("**📝 Mô tả chi tiết**")
                st.markdown(f"<div style='background:#f8fafc; padding:22px; border-radius:14px; border-left:6px solid #f97316;'>{food['mo_ta']}</div>", unsafe_allow_html=True)

                if food.get('goi_y'):
                    tags = "".join(f"<span style='background:#34d399; color:white; padding:8px 18px; border-radius:50px; margin:4px 6px 4px 0; display:inline-block; font-weight:600;'>🍽️ {mon}</span>" for mon in food['goi_y'])
                    st.markdown(f"""
                        <div style="background:#ecfdf5; padding:22px; border-radius:18px; border:2px solid #34d399; margin-top:20px;">
                            <strong>💡 Gợi ý món ăn tương tự:</strong><br><br>
                            {tags}
                        </div>
                    """, unsafe_allow_html=True)

                if st.button("🔊 Đọc to kết quả", type="primary", use_container_width=True):
                    st.success("🔊 Đang đọc kết quả... (TTS sẽ được tích hợp)")
            else:
                st.warning(f"Nhận diện: **{best_name}** - Chưa có dữ liệu dinh dưỡng.")
        else:
            st.error("❌ Không nhận diện được món ăn nào. Hãy thử chụp rõ hơn!")
    else:
        st.info("⬅️ Hãy upload ảnh hoặc bật camera bên trái để bắt đầu")
