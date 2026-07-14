"""
app_integration_snippet.py
---------------------------
KHÔNG phải module độc lập — đây là đoạn code P4 dán trực tiếp vào app.py,
đúng vị trí mô tả trong FOOD_AI_CONTEXT.md mục 6.2 (nối K-Means vào RecScore).

Cách dùng trong app.py:
    from food_similarity import get_top_similar_foods
    from user_analytics import get_user_segment, SEGMENT_CALO_BIAS
    from food_info import get_food_info
    # copy hàm get_recommendations() bên dưới vào app.py

Rồi gọi trong render_one_food():
    recs = get_recommendations(food_key=det["class_name"], session_id=session_id)
    # recs = [("Bun_Rieu", 0.81), ("Hu_Tieu", 0.74), ("Com_Tam", 0.69)]
"""

from food_similarity import get_top_similar_foods
from user_analytics import get_user_segment, SEGMENT_CALO_BIAS
from food_info import get_food_info


def get_recommendations(food_key: str, session_id: str, top_k: int = 3):
    """
    RecScore = 0.7 * SimilarityScore + 0.3 * SegmentAdjustment

    SegmentAdjustment ở đây được tính từ calo: nếu segment là "ăn lành mạnh"
    (bias = -1), món càng ít calo càng được cộng điểm; nếu "khám phá đa vùng
    miền" (bias = +1), món càng nhiều calo (thường = món cầu kỳ hơn) được
    cộng điểm nhẹ. bias = 0 ("đậm đà") thì SegmentAdjustment = 0, RecScore
    lúc đó chỉ còn phụ thuộc Similarity — đúng hành vi fallback khi
    get_user_segment() trả "Chưa đủ dữ liệu để phân tích".
    """
    similar = get_top_similar_foods(food_key, top_k=5)  # lấy dư ra rồi rerank
    segment = get_user_segment(session_id)
    bias = SEGMENT_CALO_BIAS.get(segment, 0)

    scored = []
    for key, sim_score in similar:
        info = get_food_info(key)
        calo = info.get("calo_so")
        if calo is None:
            # food_info.py hiện lưu calo dạng chuỗi "350 kcal / tô" — nếu chưa có
            # sẵn field calo_so (số), parse tạm ở đây để không phụ thuộc P1.
            import re
            m = re.search(r"\d+", info.get("calo", "0") or "0")
            calo = int(m.group()) if m else 400

        calo_factor = calo / 1000
        calo_adj = calo_factor * bias  # bias âm -> ưu tiên calo thấp
        rec_score = 0.7 * sim_score + 0.3 * calo_adj
        scored.append((key, rec_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # Ví dụ chạy thử — cần food_info.FOOD_DB và food_history.db đã có dữ liệu
    print(get_recommendations(food_key="Pho_Bo", session_id="explorer_user"))
