"""
user_analytics.py
------------------
K-Means phân tệp user + áp dụng ngược vào Recommendation (feedback loop).
Đúng interface trong FOOD_AI_CONTEXT.md mục 6.4:
    get_all_users_data()
    cluster_users(n_clusters=3)
    get_user_segment(session_id)
    SEGMENT_NAMES, SEGMENT_CALO_BIAS

Feature dùng cho K-Means — ĐÃ SỬA đúng theo ghi chú trong context file:
    [AVG(calo_so), COUNT(*), COUNT(DISTINCT vung_mien)]
    KHÔNG dùng AVG(confidence) vì đó là chỉ số kỹ thuật của model YOLO,
    không phản ánh hành vi người dùng.

2 FIX quan trọng so với bản pseudocode gốc (ghi rõ để đưa vào báo cáo P3
mục "Chuẩn bị trả lời phản biện"):

FIX 1 — Thứ tự nhãn cụm KMeans không cố định:
    sklearn.KMeans không đảm bảo cluster label 0 luôn là "ăn lành mạnh".
    Mỗi lần fit lại, cụm "calo thấp nhất" có thể đổi từ label 0 sang label 1.
    Nếu map cứng {0: "lành mạnh", 1: "đậm đà", 2: "khám phá"} như bản gốc,
    tên phân khúc hiển thị cho user sẽ SAI/ĐỔI CHỖ giữa các lần chạy.
    -> Giải pháp: sắp xếp lại nhãn cụm theo AVG(calo_so) của centroid,
    tăng dần, rồi mới gán vào SEGMENT_NAMES. Đảm bảo cụm calo thấp nhất
    luôn là "Người ăn lành mạnh" bất kể sklearn gán label gì.

FIX 2 — Chặn phân segment khi session mới chưa đủ lịch sử:
    Bản gốc: get_user_segment() gán segment cho BẤT KỲ session nào miễn
    tổng số session trong DB >= 3, kể cả session mới có 1 lượt khám phá.
    Điều này sai với checklist kiểm thử P5 ("session mới -> chưa đủ dữ liệu",
    "session >= 3 lần khám phá -> phân khúc đúng").
    -> Giải pháp: kiểm tra get_session_detection_count(session_id) >= 3
    trước khi gọi cluster_users().
"""

import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict

from food_history import DB_PATH, init_db, get_session_detection_count

MIN_DETECTIONS_FOR_SEGMENT = 3  # khớp checklist "≥ 3 lần khám phá"

SEGMENT_NAMES = {
    "low_calo": "🥗 Người ăn lành mạnh",
    "mid_calo": "🌶️ Người thích ẩm thực đậm đà",
    "high_variety": "🗺️ Khách khám phá đa vùng miền",
}

SEGMENT_CALO_BIAS = {
    "🥗 Người ăn lành mạnh": -1,       # ưu tiên món ít calo hơn
    "🌶️ Người thích ẩm thực đậm đà": 0,
    "🗺️ Khách khám phá đa vùng miền": 1,  # ưu tiên món khác vùng miền đã ăn
}


def get_all_users_data(db_path: str = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT session_id, AVG(calo_so), COUNT(*), COUNT(DISTINCT vung_mien)
        FROM history GROUP BY session_id
        """
    ).fetchall()
    conn.close()
    return rows


def _assign_segment_names_by_centroid(kmeans: KMeans, scaler: StandardScaler) -> Dict[int, str]:
    """
    FIX 1 (Bản nâng cấp):
    - Nhóm 'Khám phá' = Cụm có số vùng miền đa dạng nhất (index 2).
    - 2 nhóm còn lại phân định 'Lành mạnh' (calo thấp) và 'Đậm đà' (calo cao) dựa vào AVG Calo (index 0).
    """
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)

    # 1. Tìm nhãn của cụm có số vùng miền đa dạng nhất (Khám phá)
    explorer_label = int(np.argmax(centroids_unscaled[:, 2]))

    # 2. Lọc ra 2 nhãn còn lại
    remaining_labels = [i for i in range(3) if i != explorer_label]

    # 3. So sánh calo (index 0) của 2 cụm còn lại để gán Lành mạnh / Đậm đà
    label_a, label_b = remaining_labels
    if centroids_unscaled[label_a, 0] < centroids_unscaled[label_b, 0]:
        healthy_label, hearty_label = label_a, label_b
    else:
        healthy_label, hearty_label = label_b, label_a

    label_to_name = {
        healthy_label: SEGMENT_NAMES["low_calo"],
        hearty_label: SEGMENT_NAMES["mid_calo"],
        explorer_label: SEGMENT_NAMES["high_variety"]
    }
    return label_to_name


def cluster_users(n_clusters: int = 3, db_path: str = DB_PATH) -> Dict[str, str]:
    data = get_all_users_data(db_path)
    if len(data) < n_clusters:
        return {}

    # Feature: [calo trung bình, số lần khám phá, số vùng miền đã thử]
    X = np.array([[r[1] or 0, r[2] or 0, r[3] or 0] for r in data], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    label_to_name = _assign_segment_names_by_centroid(kmeans, scaler)

    return {
        data[i][0]: label_to_name.get(int(labels[i]), "Người dùng thông thường")
        for i in range(len(data))
    }


def get_user_segment(session_id: str, db_path: str = DB_PATH) -> str:
    """
    FIX 2: chỉ gán segment nếu session này đã có >= MIN_DETECTIONS_FOR_SEGMENT
    lượt khám phá. Nếu chưa đủ -> trả về thông báo rõ ràng cho Tab 3 UI,
    RecScore ở nơi gọi nên fallback SegmentAdjustment = 0 trong trường hợp này.
    """
    if get_session_detection_count(session_id, db_path) < MIN_DETECTIONS_FOR_SEGMENT:
        return "Chưa đủ dữ liệu để phân tích"

    segments = cluster_users(db_path=db_path)
    return segments.get(session_id, "Chưa đủ dữ liệu để phân tích")


if __name__ == "__main__":
    # Data giả lập để test K-Means ra 3 cụm hợp lý (P3, deliverable Ngày 1)
    from food_history import save_detection

    demo_sessions = {
        "healthy_user": [
            ("Goi_Cuon", 120, "Miền Nam"),
            ("Canh_Chua", 150, "Miền Nam"),
            ("Rau_Muong_Xao", 100, "Miền Nam"),
        ],
        "hearty_user": [
            ("Bun_Bo_Hue", 450, "Miền Trung"),
            ("Com_Tam_Suon", 600, "Miền Nam"),
            ("Mi_Quang", 480, "Miền Trung"),
        ],
        "explorer_user": [
            ("Pho_Bo", 350, "Miền Bắc"),
            ("Bun_Bo_Hue", 450, "Miền Trung"),
            ("Hu_Tieu", 400, "Miền Nam"),
        ],
    }

    for session_id, dishes in demo_sessions.items():
        for name, calo, region in dishes:
            fake_det = {"class_name": name, "confidence": 0.9}
            fake_info = {"ten_hien_thi": name, "calo": f"{calo} kcal", "vung_mien": region}
            save_detection(session_id, fake_det, fake_info)

    for session_id in demo_sessions:
        seg = get_user_segment(session_id)
        bias = SEGMENT_CALO_BIAS.get(seg, 0)
        print(f"{session_id} -> {seg} (bias={bias})")
