"""
app.py - Food AI Assistant
Streamlit app nhận diện món ăn Việt Nam bằng mô hình YOLOv8 đã huấn luyện.
Hỗ trợ: upload ảnh từ máy HOẶC chụp ảnh trực tiếp từ webcam.

Chạy:
streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from food_info import get_food_info


# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.25
PAGE_TITLE = "🍜 Food AI Assistant - Nhận diện món ăn Việt Nam"
# ──────────────────────────────────────────────────────────────────────────────


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🍜",
    layout="wide"
)


@st.cache_resource
def load_model(model_path: str):
    """Nạp model YOLOv8 một lần duy nhất."""
    if not Path(model_path).exists():
        return None
    return YOLO(model_path)


def run_detection(model: YOLO, image: Image.Image, conf: float):
    """Chạy nhận diện món ăn."""

    results = model.predict(
        source=np.array(image),
        conf=conf,
        verbose=False
    )

    detections = []

    if results and len(results) > 0:

        result = results[0]
        names = result.names

        for box in result.boxes:

            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            confidence = float(box.conf[0])

            detections.append({
                "class_name": class_name,
                "confidence": confidence
            })

    annotated = results[0].plot() if results else None

    return detections, annotated


def render_detections(detections: list):
    """Hiển thị kết quả nhận diện kèm thông tin dinh dưỡng."""

    if not detections:
        st.warning(
            "Không phát hiện được món ăn nào trong ảnh. "
            "Hãy thử ảnh khác hoặc chụp rõ hơn."
        )
        return

    # Loại bỏ món trùng lặp
    best_per_class = {}

    for d in detections:
        name = d["class_name"]

        if (
            name not in best_per_class
            or d["confidence"] > best_per_class[name]["confidence"]
        ):
            best_per_class[name] = d

    sorted_dets = sorted(
        best_per_class.values(),
        key=lambda x: x["confidence"],
        reverse=True
    )

    # Món có độ tin cậy cao nhất
    top_food = sorted_dets[0]
    top_info = get_food_info(top_food["class_name"])

    st.success(
        f"🎯 AI dự đoán món ăn chính: "
        f"{top_info['ten_hien_thi']} "
        f"({top_food['confidence'] * 100:.1f}%)"
    )

    st.divider()

    for det in sorted_dets:

        info = get_food_info(det["class_name"])

        with st.container(border=True):

            col1, col2 = st.columns([1, 3])

            with col1:

                st.metric(
                    "Độ tự tin",
                    f"{det['confidence'] * 100:.1f}%"
                )

                st.progress(float(det["confidence"]))

            with col2:

                st.subheader(info["ten_hien_thi"])

                st.write("📖 Mô tả:")
                st.write(info["mo_ta"])

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "📍 Vùng miền",
                        info["vung_mien"]
                    )

                with c2:
                    st.metric(
                        "🔥 Calories",
                        info["calo"]
                    )

                st.write("🍽️ Khẩu phần:")
                st.write(
                    info.get(
                        "khau_phan",
                        "Không rõ"
                    )
                )

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric(
                        "🥩 Protein",
                        info.get(
                            "protein",
                            "N/A"
                        )
                    )

                with col_b:
                    st.metric(
                        "🍚 Carb",
                        info.get(
                            "carb",
                            "N/A"
                        )
                    )

                with col_c:
                    st.metric(
                        "🥑 Fat",
                        info.get(
                            "fat",
                            "N/A"
                        )
                    )

                st.write("🧾 Thành phần chính:")
                st.write(
                    info.get(
                        "thanh_phan",
                        "Không rõ"
                    )
                )

                st.write("💰 Giá tham khảo:")
                st.write(
                    info.get(
                        "gia_trung_binh",
                        "Không rõ"
                    )
                )

                st.write("⏰ Thời điểm phù hợp:")
                st.write(
                    info.get(
                        "thoi_diem_phu_hop",
                        "Không rõ"
                    )
                )

                st.write("❤️ Chỉ số sức khỏe:")
                st.write(
                    info.get(
                        "chi_so_suc_khoe",
                        "Không rõ"
                    )
                )

                st.write("💡 Khuyến nghị dinh dưỡng:")
                st.info(
                    info.get(
                        "khuyen_nghi",
                        "Không có khuyến nghị."
                    )
                )


def main():

    st.title(PAGE_TITLE)

    st.markdown(
    """
    ### 🇻🇳 Hệ thống nhận diện món ăn Việt Nam bằng AI
    Upload ảnh hoặc chụp trực tiếp từ webcam để:
    - Nhận diện món ăn bằng YOLOv8
    - Hiển thị thông tin dinh dưỡng
    - Xem calories, protein, carb, fat
    - Tra cứu nguồn gốc món ăn
    """
    )

    st.caption(
        "Tải ảnh hoặc chụp ảnh món ăn để AI nhận diện và xem thông tin dinh dưỡng."
    )

    model = load_model(MODEL_PATH)

    if model is None:
        st.error(
            f"Không tìm thấy file model '{MODEL_PATH}'. "
            "Hãy chạy train model trước để tạo file best.pt."
        )
        return

    tab_upload, tab_camera = st.tabs(
        [
            "📁 Upload ảnh",
            "📷 Chụp ảnh (Webcam)"
        ]
    )

    image = None

    # Upload ảnh
    with tab_upload:

        uploaded_file = st.file_uploader(
            "Chọn ảnh món ăn",
            type=["jpg", "jpeg", "png"],
            key="uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    # Chụp ảnh từ webcam
    with tab_camera:

        camera_file = st.camera_input(
            "Chụp ảnh món ăn",
            key="camera"
        )

        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is not None:

        with st.spinner("Đang nhận diện món ăn..."):

            detections, annotated = run_detection(
                model,
                image,
                CONF_THRESHOLD
            )

        col_img, col_result = st.columns(2)

        with col_img:

            st.subheader("Ảnh đã nhận diện")

            if annotated is not None:
                st.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )
            else:
                st.image(
                    image,
                    use_container_width=True
                )

        with col_result:

            st.subheader("Kết quả nhận diện")

            render_detections(detections)

    else:

        st.info(
            "👆 Hãy upload ảnh hoặc chụp ảnh món ăn để bắt đầu."
        )


if __name__ == "__main__":
    main()
