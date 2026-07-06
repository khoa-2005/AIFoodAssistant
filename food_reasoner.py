"""
food_reasoner.py - Food AI Assistant
Module reasoning rule-based cho phần caption và VQA.
Không cần train, không cần model AI ngoài YOLO đã có.
Nhận class_name + info (từ food_info.py) để sinh caption hoặc trả lời câu hỏi.
"""


def generate_caption(class_name: str, info: dict, confidence: float = None) -> str:
    """
    Sinh 1 câu mô tả ngắn, tự động, không cần người dùng hỏi gì.
    Dùng để hiện ngay sau khi detect xong.
    """
    ten = info.get("ten_hien_thi", class_name)
    calo = info.get("calo", "không rõ")
    vung_mien = info.get("vung_mien", "không rõ")

    caption = f"Đây là {ten}, món ăn thuộc {vung_mien}. Ước tính khoảng {calo}."

    if confidence is not None:
        caption += f" Độ tin cậy nhận diện: {confidence*100:.1f}%."

    return caption


def answer_question(question: str, class_name: str, info: dict, confidence: float = None) -> str:
    """
    Trả lời câu hỏi tự do của người dùng về món ăn đang xét.
    Phân loại câu hỏi theo từ khóa, hỏi gì đáp nấy, không nhồi thông tin thừa.
    """
    q = question.lower()
    ten = info.get("ten_hien_thi", class_name)

    asks_calo = any(kw in q for kw in [
        "calo", "kcal", "năng lượng", "nang luong", "béo", "beo phi"
    ])
    asks_price = any(kw in q for kw in [
        "giá", "gia", "bao nhiêu tiền", "bao nhieu tien", "tiền", "tien", "vnđ", "price"
    ])
    asks_region = any(kw in q for kw in [
        "vùng miền", "vung mien", "ở đâu", "o dau", "xuất xứ", "xuat xu", "miền nào", "mien nao", "region"
    ])
    asks_health = any(kw in q for kw in [
        "tốt cho sức khỏe", "tot cho suc khoe", "tiểu đường", "tieu duong", "giảm cân", "giam can",
        "có nên ăn", "co nen an", "khuyến nghị", "khuyen nghi", "health", "healthy"
    ])
    asks_ingredient = any(kw in q for kw in [
        "thành phần", "thanh phan", "nguyên liệu", "nguyen lieu", "làm từ gì", "lam tu gi", "ingredient"
    ])
    asks_time = any(kw in q for kw in [
        "ăn khi nào", "an khi nao", "bữa nào", "bua nao", "thời điểm", "thoi diem", "buổi", "buoi"
    ])
    asks_what = any(kw in q for kw in [
        "món gì", "mon gi", "đây là gì", "day la gi", "là món", "la mon", "what is"
    ])

    if asks_calo:
        return (f"{ten} có khoảng {info.get('calo', 'không rõ')} cho {info.get('khau_phan', 'một khẩu phần')}. "
                f"Protein {info.get('protein', 'không rõ')}, "
                f"carb {info.get('carb', 'không rõ')}, "
                f"fat {info.get('fat', 'không rõ')}.")

    if asks_price:
        return f"{ten} có giá tham khảo khoảng {info.get('gia_trung_binh', 'không rõ')}."

    if asks_region:
        return f"{ten} là món đặc trưng của {info.get('vung_mien', 'không rõ')}."

    if asks_health:
        return (f"Chỉ số sức khỏe của {ten}: {info.get('chi_so_suc_khoe', 'chưa đánh giá')}. "
                f"{info.get('khuyen_nghi', 'Không có khuyến nghị thêm.')}")

    if asks_ingredient:
        return f"{ten} gồm các thành phần chính: {info.get('thanh_phan', 'không rõ')}."

    if asks_time:
        return f"{ten} thường được ăn vào {info.get('thoi_diem_phu_hop', 'không rõ')}."

    if asks_what:
        base = generate_caption(class_name, info, confidence)
        return base + f" Mô tả: {info.get('mo_ta', '')}"

    # Câu hỏi không khớp nhóm nào, trả lời tổng quát
    return (f"Về {ten}: {info.get('mo_ta', 'chưa có mô tả')}. "
            f"Calo khoảng {info.get('calo', 'không rõ')}, "
            f"giá tham khảo {info.get('gia_trung_binh', 'không rõ')}.")