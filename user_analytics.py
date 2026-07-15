"""
user_analytics.py
------------------
K-Means phân tích HÀNH VI ĂN UỐNG của CHÍNH 1 người dùng + áp dụng ngược vào
Recommendation (feedback loop).

Interface public (app.py chỉ import get_user_segment, giữ nguyên chữ ký):
    get_user_segment(session_id)
    SEGMENT_NAMES, SEGMENT_CALO_BIAS

Feature dùng cho K-Means:
    Mỗi ĐIỂM DỮ LIỆU = 1 MÓN đã khám phá (không phải 1 user).
    [calo_so, is_mien_bac, is_mien_trung, is_mien_nam] — one-hot vùng miền.
    KHÔNG dùng AVG(confidence) vì đó là chỉ số kỹ thuật của model YOLO,
    không phản ánh hành vi người dùng.

FIX QUAN TRỌNG (app cá nhân — 1 user duy nhất, session_id="demo_user" cố định
trong app.py) so với bản pseudocode gốc trong FOOD_AI_CONTEXT.md mục 6.4:

    Bản gốc dùng `GROUP BY session_id` để coi MỖI SESSION là MỘT USER rồi
    phân cụm giữa các "user" đó. Vì app.py hardcode session_id = "demo_user"
    cho MỌI lượt dùng, GROUP BY session_id luôn trả về ĐÚNG 1 DÒNG dữ liệu
    bất kể user đã quét bao nhiêu món -> len(data) < n_clusters luôn đúng
    -> cluster_users() luôn trả về {} -> get_user_segment() luôn fallback
    "Chưa đủ dữ liệu để phân tích", KỂ CẢ KHI user đã quét >100 món.
    Đây là lỗi logic thật đang tồn tại trong code, không phải giả định.

    -> Giải pháp: phân cụm trên CÁC MÓN đã khám phá của CHÍNH session đó
    (WHERE session_id = ?, không GROUP BY session_id). Mỗi món là 1 điểm
    dữ liệu, nên chỉ cần user quét >= 3 món là đã đủ chạy K-Means, và dữ
    liệu không bao giờ lẫn giữa các user khác (nếu sau này app được mở rộng
    nhiều session thật thì mỗi session chỉ nhìn thấy dữ liệu của chính nó).

    Xu hướng ("segment") hiện tại của user được suy ra từ cụm chiếm đa số
    trong N món GẦN NHẤT (mặc định 5) — để phản ánh khẩu vị GẦN ĐÂY, không
    bị pha loãng bởi các món ăn từ rất lâu.

FIX phụ — thứ tự nhãn cụm KMeans không cố định:
    sklearn.KMeans không đảm bảo cluster label 0 luôn là "ăn lành mạnh".
    -> Giải pháp: gán tên cụm theo AVG(calo_so) của centroid (thấp nhất =
    lành mạnh, cao nhất = đậm đà, còn lại = khám phá đa vùng miền), không
    hardcode theo index.

FIX phụ — chặn phân segment khi chưa đủ lịch sử:
    Dùng get_session_detection_count(session_id) >= MIN_DETECTIONS_FOR_SEGMENT
    trước khi chạy K-Means, khớp checklist kiểm thử P5.
"""

import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from food_history import DB_PATH, init_db, get_session_detection_count

MIN_DETECTIONS_FOR_SEGMENT = 3   # khớp checklist "≥ 3 lần khám phá"
RECENT_WINDOW = 5                # số món gần nhất dùng để xác định segment hiện tại

REGION_ORDER = ["Miền Bắc", "Miền Trung", "Miền Nam"]

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


def get_own_dish_records(session_id: str, db_path: str = DB_PATH) -> List[Tuple[int, str]]:
    """
    CHỈ lấy các món thuộc về CHÍNH session_id này (WHERE, không GROUP BY),
    theo đúng thứ tự thời gian khám phá — dữ liệu của user khác (nếu có)
    không bao giờ lọt vào đây.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT calo_so, vung_mien FROM history
        WHERE session_id = ? ORDER BY timestamp ASC
        """,
        (session_id,),
    ).fetchall()
    conn.close()
    return rows


def _region_onehot(vung_mien: str) -> List[int]:
    return [1 if vung_mien == r else 0 for r in REGION_ORDER]


def _assign_segment_names_by_centroid(centroids_unscaled: np.ndarray) -> Dict[int, str]:
    """Gán tên cụm dựa trên Ý NGHĨA THẬT của centroid, không suy diễn mò:
    - 'Khám phá đa vùng miền' = cụm mà thành viên trải đều nhiều vùng miền
      nhất (đo bằng centroid vùng miền KHÔNG lệch hẳn về 1 giá trị onehot —
      max onehot thấp nghĩa là món trong cụm này đến từ nhiều vùng khác nhau).
    - Trong 2 cụm còn lại (vốn tập trung vào 1 vùng miền rõ rệt): calo thấp
      hơn = 'lành mạnh', calo cao hơn = 'đậm đà'.
    Không hardcode theo index, vì sklearn không đảm bảo thứ tự nhãn cố định
    giữa các lần fit."""
    n = centroids_unscaled.shape[0]
    region_cols = centroids_unscaled[:, 1:]          # 3 cột one-hot vùng miền
    region_spread = 1.0 - region_cols.max(axis=1)     # càng cao = càng đa vùng miền

    if n < 3:
        order_by_calo = np.argsort(centroids_unscaled[:, 0])
        label_to_name = {int(order_by_calo[0]): SEGMENT_NAMES["low_calo"]}
        if n == 2:
            label_to_name[int(order_by_calo[-1])] = SEGMENT_NAMES["mid_calo"]
        return label_to_name

    explorer_label = int(np.argmax(region_spread))
    remaining = [i for i in range(n) if i != explorer_label]
    remaining.sort(key=lambda i: centroids_unscaled[i, 0])  # tăng dần theo calo
    healthy_label, hearty_label = remaining[0], remaining[-1]

    return {
        healthy_label: SEGMENT_NAMES["low_calo"],
        hearty_label: SEGMENT_NAMES["mid_calo"],
        explorer_label: SEGMENT_NAMES["high_variety"],
    }


from sklearn.metrics import silhouette_score

def cluster_own_history(session_id: str, n_clusters: int = 3, db_path: str = DB_PATH):
    records = get_own_dish_records(session_id, db_path)
    if len(records) < n_clusters:
        return None, None, {}, records, 0

    X = np.array(
        [[calo or 0] + _region_onehot(vm) for calo, vm in records], dtype=float
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- THÊM ĐOẠN TÍNH SILHOUETTE SCORE ĐỂ CHỨNG MINH K=3 LÀ TỐI ƯU ---
    best_k = 2
    best_score = -1
    # Chỉ thử K từ 2 đến số lượng mẫu (tối đa 4 để tránh chạy lâu)
    max_k_to_test = min(4, len(records) - 1) 
    for k_test in range(2, max_k_to_test + 1):
        try:
            kmeans_test = KMeans(n_clusters=k_test, n_init=10, random_state=42).fit(X_scaled)
            score = silhouette_score(X_scaled, kmeans_test.labels_)
            if score > best_score:
                best_score = score
                best_k = k_test
        except Exception:
            continue
    
    # Dùng best_k tìm được để fit model chính thức
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    label_to_name = _assign_segment_names_by_centroid(centroids_unscaled)
    
    # Trả về thêm best_score
    return kmeans, scaler, label_to_name, records, best_score


def get_user_segment(session_id: str, db_path: str = DB_PATH) -> str:
    """
    Chỉ gán segment nếu CHÍNH session này đã có >= MIN_DETECTIONS_FOR_SEGMENT
    lượt khám phá.
    """
    if get_session_detection_count(session_id, db_path) < MIN_DETECTIONS_FOR_SEGMENT:
        return "Chưa đủ dữ liệu để phân tích"

    # SỬA Ở ĐÂY: Thêm biến _ (hoặc sil_score) để nhận giá trị thứ 5
    kmeans, scaler, label_to_name, records, _ = cluster_own_history(session_id, db_path=db_path)
    if kmeans is None:
        return "Chưa đủ dữ liệu để phân tích"

    recent = records[-RECENT_WINDOW:] if len(records) >= RECENT_WINDOW else records
    recent_X = np.array(
        [[calo or 0] + _region_onehot(vm) for calo, vm in recent], dtype=float
    )
    avg_vec = recent_X.mean(axis=0, keepdims=True)
    avg_scaled = scaler.transform(avg_vec)
    nearest_label = int(kmeans.predict(avg_scaled)[0])
    return label_to_name.get(nearest_label, "Người dùng thông thường")


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