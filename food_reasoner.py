"""
food_reasoner.py - Food AI Assistant
Module điều phối ngôn ngữ lý luận nội bộ và kích hoạt kịch bản dự phòng đa ngữ.
Bản đầy đủ chi tiết các điều kiện lọc từ khóa (Rule-based) khi offline cho cả VI và EN.
"""

from food_ai_service import ask_llm_expert


def generate_caption(class_name: str, info: dict, confidence: float = None, lang: str = "vi") -> str:
    """Sinh caption ngắn hiển thị ngay sau khi scan ảnh."""
    ten = info.get("ten_hien_thi", class_name)
    calo = info.get("calo", "không rõ")
    vung_mien = info.get("vung_mien", "không rõ")

    if lang == "en":
        caption = f"This is {ten}, a traditional dish from {vung_mien}. Estimated: {calo}."
        if confidence is not None:
            caption += f" Confidence rate: {confidence*100:.1f}%."
    else:
        caption = f"Đây là {ten}, món ăn thuộc {vung_mien}. Ước tính khoảng {calo}."
        if confidence is not None:
            caption += f" Độ tin cậy nhận diện: {confidence*100:.1f}%."

    return caption


def answer_question(question: str, class_name: str, info: dict, confidence: float = None, lang: str = "vi") -> tuple:
    """
    Trả lời câu hỏi tự do, hỗ trợ Fallback chi tiết khi offline.
    Trả về bộ tuple: (nội_dung_câu_trả_lời, nguồn_gốc)
    """
    
    # ─── BƯỚC 1: ƯU TIÊN GỌI AI ĐA NGỮ (GEMINI 2.5) ──────────────────
    ai_response = ask_llm_expert(question, class_name, info, lang)
    if ai_response and not ai_response.startswith("[Lỗi"):
        return ai_response, "AI"


    # ─── BƯỚC 2: BỘ LUẬT CỨNG DỰ PHÒNG KHI OFFLINE (FALLBACK) ─────────────────────
    q = question.lower()
    ten = info.get("ten_hien_thi", class_name)

    # --- PHÂN LOẠI CÂU HỎI THEO NGÔN NGỮ TIẾNG ANH ---
    if lang == "en":
        asks_calo = any(kw in q for kw in ["calo", "kcal", "calorie", "calories", "fat", "energy"])
        asks_price = any(kw in q for kw in ["price", "cost", "money", "how much", "vnd"])
        asks_region = any(kw in q for kw in ["region", "where", "origin", "from", "location"])
        asks_health = any(kw in q for kw in ["health", "healthy", "diet", "advice", "diabetes", "weight"])
        asks_ingredient = any(kw in q for kw in ["ingredient", "ingredients", "made of", "contain"])
        asks_time = any(kw in q for kw in ["when", "time", "breakfast", "lunch", "dinner", "morning", "night"])
        asks_what = any(kw in q for kw in ["what is", "description", "details"])

        if asks_calo:
            return (f"{ten} contains about {info.get('calo', 'unknown')} per {info.get('khau_phan', 'serving')}. "
                    f"Protein: {info.get('protein', 'unknown')}, "
                    f"Carbs: {info.get('carb', 'unknown')}, "
                    f"Fat: {info.get('fat', 'unknown')}."), "DATABASE"

        if asks_price:
            return f"The reference price for {ten} is around {info.get('gia_trung_binh', 'unknown')}.", "DATABASE"

        if asks_region:
            return f"{ten} is a specialty dish from {info.get('vung_mien', 'unknown')}.", "DATABASE"

        if asks_health:
            return (f"Health index for {ten}: {info.get('chi_so_suc_khoe', 'not evaluated')}. "
                    f"Advice: {info.get('khuyen_nghi', 'No dynamic advice available.')}"), "DATABASE"

        if asks_ingredient:
            return f"Main ingredients of {ten} include: {info.get('thanh_phan', 'unknown')}.", "DATABASE"

        if asks_time:
            return f"{ten} is typically enjoyed during {info.get('thoi_diem_phu_hop', 'unknown')}.", "DATABASE"

        if asks_what:
            base = generate_caption(class_name, info, confidence, lang)
            return base + f" Description: {info.get('mo_ta', '')}", "DATABASE"

        return (f"About {ten}: {info.get('mo_ta', 'No description available')}. "
                f"Calories: {info.get('calo', 'unknown')}, reference price: {info.get('gia_trung_binh', 'unknown')}."), "DATABASE"

    # --- PHÂN LOẠI CÂU HỎI THEO NGÔN NGỮ TIẾNG VIỆT ---
    else:
        asks_calo = any(kw in q for kw in ["calo", "kcal", "năng lượng", "nang luong", "béo", "beo phi"])
        asks_price = any(kw in q for kw in ["giá", "gia", "bao nhiêu tiền", "bao nhieu tien", "tiền", "tien", "vnđ", "price"])
        asks_region = any(kw in q for kw in ["vùng miền", "vung mien", "ở đâu", "o dau", "xuất xứ", "xuat xu", "miền nào", "mien nao", "region"])
        asks_health = any(kw in q for kw in ["tốt cho sức khỏe", "tot cho suc khoe", "tiểu đường", "tieu duong", "giảm cân", "giam can", "có nên ăn", "co nen an", "khuyến nghị", "khuyen nghi", "health", "healthy"])
        asks_ingredient = any(kw in q for kw in ["thành phần", "thanh phan", "nguyên liệu", "nguyen lieu", "làm từ gì", "lam tu gi", "ingredient"])
        asks_time = any(kw in q for kw in ["ăn khi nào", "an khi nao", "bữa nào", "bua nao", "thời điểm", "thoi diem", "buổi", "buoi"])
        asks_what = any(kw in q for kw in ["món gì", "mon gi", "đây là gì", "day la gi", "là món", "la mon", "what is"])

        if asks_calo:
            return (f"{ten} có khoảng {info.get('calo', 'không rõ')} cho {info.get('khau_phan', 'một khẩu phần')}. "
                    f"Protein {info.get('protein', 'không rõ')}, "
                    f"carb {info.get('carb', 'không rõ')}, "
                    f"fat {info.get('fat', 'không rõ')}.", "DATABASE")

        if asks_price:
            return f"{ten} có giá tham khảo khoảng {info.get('gia_trung_binh', 'không rõ')}.", "DATABASE"

        if asks_region:
            return f"{ten} là món đặc trưng của {info.get('vung_mien', 'không rõ')}.", "DATABASE"

        if asks_health:
            return (f"Chỉ số sức khỏe của {ten}: {info.get('chi_so_suc_khoe', 'chưa đánh giá')}. "
                    f"{info.get('khuyen_nghi', 'Không có khuyến nghị thêm.')}", "DATABASE")

        if asks_ingredient:
            return f"{ten} gồm các thành phần chính: {info.get('thanh_phan', 'không rõ')}.", "DATABASE"

        if asks_time:
            return f"{ten} thường được ăn vào {info.get('thoi_diem_phu_hop', 'không rõ')}.", "DATABASE"

        if asks_what:
            base = generate_caption(class_name, info, confidence, lang)
            return base + f" Mô tả: {info.get('mo_ta', '')}", "DATABASE"

        return (f"Về {ten}: {info.get('mo_ta', 'chưa có mô tả')}. "
                f"Calo khoảng {info.get('calo', 'không rõ')}, "
                f"giá tham khảo {info.get('gia_trung_binh', 'không rõ')}.", "DATABASE")