"""
food_ai_service.py - Kết nối API với Google Gemini sử dụng SDK google-genai mới.
Hỗ trợ phản hồi đa ngôn ngữ (Việt / Anh) cho luồng Hỏi đáp và biên dịch giọng nói.
"""

from google import genai
from google.genai import types
import food_config

_client = None

def _init_gemini():
    global _client
    if _client is not None:
        return True
        
    try:
        # Sử dụng getattr chống crash ứng dụng nếu biến cấu hình bị xóa trống
        api_key = getattr(food_config, "GEMINI_API_KEY", None)
    except Exception:
        api_key = None
        
    if not api_key or api_key == "AIzaSy...":
        print("[AI Service] CẢNH BÁO: GEMINI_API_KEY bị trống hoặc đã bị xóa khỏi file food_config.py!")
        return False
        
    try:
        _client = genai.Client(api_key=api_key)
        return True
    except Exception as e:
        print(f"[AI Service] Lỗi cấu hình Gemini: {e}")
        return False

def ask_llm_expert(question: str, class_name: str, info: dict, lang: str = "vi") -> str:
    """Trả lời câu hỏi tự do hoặc xử lý biên dịch kịch bản hội thoại dựa trên ngôn ngữ chỉ định"""
    if not _init_gemini():
        return "[Lỗi Hệ Thống] Trình điều khiển AI chưa được cấu hình API Key chính xác."

    ten_mon = info.get("ten_hien_thi", class_name)
    
    if lang == "en":
        system_instruction = (
            "You are a knowledgeable and friendly Vietnamese culinary expert and nutritionist.\n"
            "Your task is to answer the user's question or translate text about a specific dish based on the provided data.\n"
            "Please respond naturally, coherently, and concisely in ENGLISH (about 2-4 sentences).\n"
            "Do NOT use complex markdown formats, titles, or unnecessary headers."
        )
    else:
        system_instruction = (
            "Bạn là một chuyên gia ẩm thực và bác sĩ tư vấn dinh dưỡng Việt Nam thông thái, thân thiện.\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng về một món ăn cụ thể dựa trên thông tin được cung cấp.\n"
            "Hãy trả lời một cách tự nhiên, mạch lạc, dễ hiểu bằng tiếng Việt ngắn gọn (khoảng 2-4 câu). "
            "Tuyệt đối không sử dụng các định dạng markdown phức tạp hay tiêu đề không cần thiết."
        )

    context = (
        f"--- THÔNG TIN MÓN ĂN HIỆN TẠI ---\n"
        f"Tên món ăn: {ten_mon}\n"
        f"Mô tả gốc: {info.get('mo_ta', '')}\n"
        f"Vùng miền: {info.get('vung_mien', '')}\n"
        f"Năng lượng (Calo): {info.get('calo', '')} cho {info.get('khau_phan', '')}\n"
        f"Thành phần dinh dưỡng chính: Protein {info.get('protein', '')}, Carb {info.get('carb', '')}, Fat {info.get('fat', '')}\n"
        f"Nguyên liệu/Thành phần chính: {info.get('thanh_phan', '')}\n"
        f"Giá tham khảo: {info.get('gia_trung_binh', '')}\n"
        f"Thời điểm ăn phù hợp: {info.get('thoi_diem_phu_hop', '')}\n"
        f"Khuyến nghị sức khỏe: {info.get('khuyen_nghi', '')}\n"
        f"----------------------------------\n"
    )

    full_prompt = f"{context}\nUser request: {question}\nProcess based on the info above:"

    try:
        response = _client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"[AI Service Exception] Lỗi API: {e}")
        return f"[Lỗi Kết Nối AI]: Không thể trò chuyện với AI. Lỗi chi tiết: {str(e)}"