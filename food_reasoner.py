"""
food_reasoner.py - Decision Engine & Intent Router for Food AI Assistant

Module điều phối thông minh đóng vai trò là lõi lý luận hệ thống (Decision Engine).
Tự động phân loại ý định câu hỏi (Intent Classification) để quyết định luồng xử lý:
1. Tra cứu trực tiếp từ Cơ sở dữ liệu nội bộ (Local Database) với chi phí bằng 0 và độ trễ cực thấp.
2. Định hướng sang Trí tuệ nhân tạo (Google Gemini 2.5 Flash) đối với các câu hỏi mở, suy luận sâu.
3. Kích hoạt tầng dự phòng (Hybrid Fallback) tự động phản hồi bằng DB cục bộ khi AI gặp sự cố.
"""

import logging
from food_ai_service import ask_llm_expert

# Cấu hình logging để dễ dàng kiểm thử và theo dõi luồng dữ liệu (QA-Friendly)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_caption(class_name: str, info: dict, confidence: float = None, lang: str = "vi") -> str:
    """
    Sinh dòng mô tả ngắn (caption) hiển thị ngay sau khi quét và nhận diện món ăn.
    """
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


def is_database_question(question: str, lang: str = "vi") -> bool:
    """
    [Ý tưởng 1 & 3: Intent Classification & Lọc Từ Khóa]
    Kiểm tra xem câu hỏi có thể giải quyết trực tiếp bằng Cơ sở dữ liệu cục bộ hay không.
    Trả về True nếu thuộc nhóm dữ liệu tĩnh có cấu trúc, ngược lại trả về False.
    """
    if not question or not isinstance(question, str):
        return False
        
    q = question.lower().strip()
    
    if lang == "en":
        db_keywords = [
            "calo", "kcal", "calorie", "calories", "fat", "energy", "carb", "carbs", "protein",
            "price", "cost", "money", "how much", "vnd",
            "region", "where", "origin", "from", "location",
            "health", "healthy", "diet", "advice", "diabetes", "weight",
            "ingredient", "ingredients", "made of", "contain", "contains",
            "when", "time", "breakfast", "lunch", "dinner", "morning", "night",
            "what is", "description", "details"
        ]
    else:
        db_keywords = [
            "calo", "kcal", "năng lượng", "nang luong", "chất béo", "chat beo", "carb", "protein", "dinh dưỡng",
            "giá", "gia", "bao nhiêu tiền", "bao nhieu tien", "tiền", "tien", "vnđ", "vnd", "price",
            "vùng miền", "vung mien", "ở đâu", "o dau", "xuất xứ", "xuat xu", "miền nào", "mien nao", "gốc",
            "thành phần", "thanh phan", "nguyên liệu", "nguyen lieu", "làm từ gì", "lam tu gi", "chứa gì", "chua gi",
            "ăn khi nào", "an khi nao", "bữa nào", "bua nao", "thời điểm", "thoi diem", "buổi", "buoi",
            "tốt cho sức khỏe", "tot cho suc khoe", "tiểu đường", "tieu duong", "giảm cân", "giam can",
            "có nên ăn", "co nen an", "khuyến nghị", "khuyen nghi", "chỉ số", "chi so",
            "món gì", "mon gi", "đây là gì", "day la gi", "là món", "la mon", "mô tả", "mo ta"
        ]
        
    return any(kw in q for kw in db_keywords)


def query_local_database(question: str, class_name: str, info: dict, confidence: float = None, lang: str = "vi") -> str:
    """
    [Local Database Engine]
    Truy xuất chính xác trường dữ liệu có cấu trúc từ dictionary nội bộ dựa trên ý định câu hỏi.
    """
    q = question.lower().strip() if question else ""
    ten = info.get("ten_hien_thi", class_name)

    # ──── TRUY XUẤT DỮ LIỆU TIẾNG ANH ────
    if lang == "en":
        asks_calo = any(kw in q for kw in ["calo", "kcal", "calorie", "calories", "fat", "energy", "carb", "protein"])
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
                    f"Fat: {info.get('fat', 'unknown')}.")

        if asks_price:
            return f"The reference price for {ten} is around {info.get('gia_trung_binh', 'unknown')}."

        if asks_region:
            return f"{ten} is a specialty dish from {info.get('vung_mien', 'unknown')}."

        if asks_health:
            return (f"Health index for {ten}: {info.get('chi_so_suc_khoe', 'not evaluated')}. "
                    f"Advice: {info.get('khuyen_nghi', 'No dynamic advice available.')}")

        if asks_ingredient:
            return f"Main ingredients of {ten} include: {info.get('thanh_phan', 'unknown')}."

        if asks_time:
            return f"{ten} is typically enjoyed during {info.get('thoi_diem_phu_hop', 'unknown')}."

        if asks_what:
            base = generate_caption(class_name, info, confidence, lang)
            return base + f" Description: {info.get('mo_ta', '')}"

        return (f"About {ten}: {info.get('mo_ta', 'No description available')}. "
                f"Calories: {info.get('calo', 'unknown')}, reference price: {info.get('gia_trung_binh', 'unknown')}.")

    # ──── TRUY XUẤT DỮ LIỆU TIẾNG VIỆT ────
    else:
        asks_calo = any(kw in q for kw in ["calo", "kcal", "năng lượng", "nang luong", "béo", "carb", "protein", "dinh dưỡng"])
        asks_price = any(kw in q for kw in ["giá", "gia", "bao nhiêu tiền", "bao nhieu tien", "tiền", "vnđ", "price"])
        asks_region = any(kw in q for kw in ["vùng miền", "vung mien", "ở đâu", "xuất xứ", "miền nào", "gốc"])
        asks_health = any(kw in q for kw in ["sức khỏe", "suc khoe", "tiểu đường", "tieu duong", "giảm cân", "có nên ăn", "khuyến nghị"])
        asks_ingredient = any(kw in q for kw in ["thành phần", "thanh phan", "nguyên liệu", "nguyen lieu", "làm từ gì", "chứa gì"])
        asks_time = any(kw in q for kw in ["ăn khi nào", "an khi nao", "bữa nào", "thời điểm", "buổi"])
        asks_what = any(kw in q for kw in ["món gì", "đây là gì", "là món", "mô tả"])

        if asks_calo:
            return (f"{ten} có hàm lượng khoảng {info.get('calo', 'không rõ')} cho {info.get('khau_phan', 'một khẩu phần')}. "
                    f"Thông tin dinh dưỡng chi tiết: Protein đạt {info.get('protein', 'không rõ')}, "
                    f"Carb (Carbohydrate) khoảng {info.get('carb', 'không rõ')}, "
                    f"và Chất béo (Fat) khoảng {info.get('fat', 'không rõ')}.")

        if asks_price:
            return f"{ten} hiện có giá bán tham khảo trên thị trường khoảng {info.get('gia_trung_binh', 'không rõ')}."

        if asks_region:
            return f"{ten} là món ăn đặc sản nổi tiếng mang đậm hương vị của {info.get('vung_mien', 'không rõ')}."

        if asks_health:
            return (f"Chỉ số sức khỏe của {ten}: {info.get('chi_so_suc_khoe', 'chưa được đánh giá')}. "
                    f"Khuyến nghị tiêu dùng: {info.get('khuyen_nghi', 'Không có khuyến nghị đặc biệt thêm.')}")

        if asks_ingredient:
            return f"{ten} được chế biến từ các thành phần nguyên liệu chính bao gồm: {info.get('thanh_phan', 'không rõ')}."

        if asks_time:
            return f"{ten} thường được người dùng ưu chuộng thưởng thức vào {info.get('thoi_diem_phu_hop', 'không rõ')}."

        if asks_what:
            base = generate_caption(class_name, info, confidence, lang)
            return base + f" Mô tả chi tiết: {info.get('mo_ta', 'Chưa có bài viết mô tả cụ thể.')}"

        return (f"Thông tin về {ten}: {info.get('mo_ta', 'Chưa có mô tả cụ thể')}. "
                f"Năng lượng ước tính: {info.get('calo', 'không rõ')}, Giá thị trường tham khảo: {info.get('gia_trung_binh', 'không rõ')}.")


def answer_question(question: str, class_name: str, info: dict, confidence: float = None, lang: str = "vi") -> tuple:
    """
    [Ý tưởng 6: Decision Engine Router]
    Hàm điều phối trung tâm quản lý luồng xử lý câu hỏi dựa trên Intent Classification.
    
    Trả về bộ tuple: (nội_dung_câu_trả_lời, nguồn_gốc_hiển_thị)
    - Nhóm 1: Câu hỏi Database cục bộ -> Trả về kết quả tức thì từ local DB.
    - Nhóm 2: Câu hỏi suy luận mở rộng -> Chuyển tiếp và tái cấu trúc câu hỏi cho Gemini AI.
    - Nhóm 3: Lỗi kết nối/Hệ thống -> Tự động Fallback sang Local DB bảo vệ hệ thống không bị crash.
    """
    
    # ─── BƯỚC 1: PHÂN LOẠI Ý ĐỊNH (INTENT CLASSIFICATION) ──────────────────
    is_db_query = is_database_question(question, lang)
    
    # ─── BƯỚC 2: Ý TƯỞNG 5 - ĐỊNH NGHĨA STRINGS HIỂN THỊ NGUỒN UI ──────────────
    source_db = "📋 Nguồn:\nFood Database"
    source_ai = "🧠 Nguồn:\nGoogle Gemini 2.5 Flash"
    source_fallback = "⚠️ Nguồn:\nFood Database (Dự phòng do AI mất kết nối)"

    # ─── BƯỚC 3: ĐIỀU PHỐI VÀ ROUTING CÂU HỎI ──────────────────────────────
    if is_db_query:
        logging.info(f"[Decision Engine] Chuyển hướng thành công câu hỏi '{question}' sang LOCAL DATABASE.")
        db_response = query_local_database(question, class_name, info, confidence, lang)
        return db_response, source_db
    
    else:
        logging.info(f"[Decision Engine] Chuyển hướng câu hỏi phức tạp '{question}' sang GEMINI AI.")
        
        # [Ý tưởng 4: Xử lý triệt để tính nhút nhát của Gemini thông qua Prompt Injection]
        # Kỹ thuật bọc câu hỏi nâng cao để ép AI sử dụng kiến thức mở rộng của nó khi Context thiếu thông tin
        system_injection = (
            "\n\n[YÊU CẦU HỆ THỐNG DÀNH CHO AI]:\n"
            "1. Hãy ưu tiên sử dụng thông tin món ăn được cung cấp trong Context để trả lời.\n"
            "2. Nếu Context không đủ dữ liệu để trả lời câu hỏi mở rộng này (như lịch sử, nguồn gốc, cách nấu, so sánh...), "
            "TUYỆT ĐỐI không được nói câu 'Thông tin không đề cập'. Hãy sử dụng kho tri thức mở rộng của bạn để giải thích chi tiết.\n"
            "3. Khi sử dụng kiến thức mở rộng ngoài Context, bạn BẮT BUỘC phải tự động thêm dòng thông báo sau ở cuối câu trả lời:\n"
            "'(Đây là kiến thức bổ sung từ AI, không có trong cơ sở dữ liệu nội bộ).'"
        )
        
        enhanced_question = question + system_injection
        
        try:
            # Gọi API thực thi
            ai_response = ask_llm_expert(enhanced_question, class_name, info, lang)
            
            # Kiểm tra xem kết quả từ API có hợp lệ không (loại bỏ trường hợp API trả chuỗi báo lỗi kết nối mạng)
            if ai_response and not ai_response.strip().startswith("[Lỗi"):
                return ai_response, source_ai
                
        except Exception as e:
            logging.error(f"[Decision Engine] Lỗi khi kết nối với LLM Expert API: {e}")
        
        # ─── BƯỚC 4: DỰ PHÒNG KHẨN CẤP (HYBRID FALLBACK) ───────────────────
        # Nếu AI gặp sự cố (mất mạng, hết quota token...), lập tức lấy dữ liệu cấu trúc local cứu nguy UI
        logging.warning("[Decision Engine] Kích hoạt kịch bản dự phòng khẩn cấp do AI không phản hồi.")
        fallback_response = query_local_database(question, class_name, info, confidence, lang)
        return fallback_response, source_fallback