"""
food_similarity.py
-------------------
Food Similarity Engine — thay thế dict tay "so_sanh"/"goi_y".
Đúng interface đã thống nhất trong FOOD_AI_CONTEXT.md mục 6.1:
    build_feature_vector(food_key) -> np.ndarray
    cosine_similarity(v1, v2) -> float
    get_top_similar_foods(food_key, top_k=3) -> List[Tuple[str, float]]

Giả định về food_info.FOOD_DB (theo đúng format hiện có trong project):
    FOOD_DB = {
        "Pho_Bo": {
            "vung_mien": "Miền Bắc",              # "Miền Bắc" / "Miền Trung" / "Miền Nam"
            "vi_dac_trung": "Đậm, Thanh",          # chuỗi, kiểm tra bằng "Cay" in vi ...
            "calo": "350 kcal / tô",               # chuỗi, số lấy ra bằng regex
            ...
        },
        ...
    }

Đã xử lý 2 edge case P2 phụ trách Ngày 2:
    1. Món không tồn tại trong FOOD_DB -> báo lỗi rõ ràng thay vì crash mơ hồ.
    2. Hai món có vector giống hệt nhau -> similarity = 1.0 là kết quả ĐÚNG
       (toán học), không phải bug — được test riêng ở cuối file.
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict

from food_info import FOOD_DB

VUNG_MIEN_MAP = {
    "Miền Bắc": [1, 0, 0],
    "Miền Trung": [0, 1, 0],
    "Miền Nam": [0, 0, 1],
}

# Trọng số cho từng chiều vector: [vùng miền x3, cay, chua, ngọt, đậm, calo_norm]
# Mặc định bằng nhau — P2 có thể tinh chỉnh nếu Top-3 similarity Ngày 1 chưa hợp lý
# (ví dụ tăng trọng số vùng miền nếu muốn ưu tiên món cùng miền hơn món cùng vị).
DEFAULT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

CALO_MAX_NORM = 800  # ước lượng calo tối đa để chuẩn hóa, khớp context file


def build_feature_vector(food_key: str, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if food_key not in FOOD_DB:
        raise KeyError(
            f"'{food_key}' không có trong FOOD_DB — kiểm tra lại tên món hoặc "
            f"food_info.py đã bàn giao đủ 35 món chưa."
        )
    info = FOOD_DB[food_key]

    vung = VUNG_MIEN_MAP.get(info.get("vung_mien", "Miền Nam"), [0, 0, 1])

    vi = info.get("vi_dac_trung", "") or ""
    cay = 1.0 if "Cay" in vi else 0.0
    chua = 1.0 if "Chua" in vi else 0.0
    ngot = 1.0 if "Ngọt" in vi else 0.0
    dam = 1.0 if "Đậm" in vi else 0.0
    beo = 1.0 if "Béo" in vi else 0.0  # THÊM DÒNG NÀY

    calo_match = re.search(r"\d+", info.get("calo", "0") or "0")
    calo_norm = (int(calo_match.group()) if calo_match else 0) / CALO_MAX_NORM

    vec = np.array(vung + [cay, chua, ngot, dam, beo, calo_norm])

    w = weights if weights is not None else DEFAULT_WEIGHTS
    return vec * w


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))


def get_top_similar_foods(
    food_key: str, top_k: int = 3, weights: Optional[np.ndarray] = None
) -> List[Tuple[str, float]]:
    """Trả về Top-K món giống nhất, dùng cho Tab 1 'Gợi ý món tương tự' và Tab 2 'So sánh'."""
    target_vec = build_feature_vector(food_key, weights)

    scores = []
    for key in FOOD_DB:
        if key == food_key:
            continue
        vec = build_feature_vector(key, weights)
        sim = cosine_similarity(target_vec, vec)
        scores.append((key, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    # Test case thủ công cho báo cáo (P2, mục 6.1 deliverable):
    # in Top-3 similar cho vài món, tự đánh giá bằng mắt có hợp lý không.
    sample_keys = list(FOOD_DB.keys())[:3]
    for k in sample_keys:
        print(f"Top-3 similar to {k}: {get_top_similar_foods(k, top_k=3)}")

    # Edge case: vector giống hệt nhau -> similarity phải = 1.0 (không phải bug)
    v = build_feature_vector(sample_keys[0])
    print("Self-similarity (phải = 1.0):", cosine_similarity(v, v))
