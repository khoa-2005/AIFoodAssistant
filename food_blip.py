"""
food_blip.py - Food AI Assistant
Module dùng BLIP-1 (nhẹ hơn BLIP-2 ~6 lần) để:
  - Sinh caption mô tả những gì thấy trong ảnh crop món ăn
  - Trả lời câu hỏi VQA dựa trên nội dung ảnh thật

Khác với food_reasoner.py (tra bảng food_info), module này
thực sự "nhìn" vào ảnh và mô tả/trả lời dựa trên nội dung hình ảnh.

Cần cài:
    pip install transformers deep-translator

Model tải lần đầu ~900MB, lưu cache tại ~/.cache/huggingface/
Chạy được trên CPU 8GB RAM, tự động dùng GPU nếu có NVIDIA.
"""

import torch
from PIL import Image

# ── Lazy import để không crash khi chưa cài ──────────────────────────────────
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import BlipForQuestionAnswering
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_OK = True
except ImportError:
    _TRANSLATOR_OK = False

# ── Global cache (load 1 lần, dùng nhiều lần) ────────────────────────────────
_caption_processor = None
_caption_model     = None
_vqa_processor     = None
_vqa_model         = None
_device            = None

CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"   # ~900MB
VQA_MODEL_ID     = "Salesforce/blip-vqa-base"               # ~900MB


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def is_available() -> bool:
    """Kiểm tra xem BLIP có thể dùng không (đã cài đủ thư viện chưa)."""
    return _TRANSFORMERS_OK


def load_caption_model():
    """Load BLIP-1 captioning model, cache lại sau lần đầu."""
    global _caption_processor, _caption_model

    if not _TRANSFORMERS_OK:
        raise ImportError(
            "Chưa cài transformers. Chạy: pip install transformers"
        )

    if _caption_model is not None:
        return _caption_processor, _caption_model

    device = _get_device()
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    _caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    _caption_model     = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_ID, torch_dtype=dtype
    ).to(device)
    _caption_model.eval()

    return _caption_processor, _caption_model


def load_vqa_model():
    """Load BLIP-1 VQA model, cache lại sau lần đầu."""
    global _vqa_processor, _vqa_model

    if not _TRANSFORMERS_OK:
        raise ImportError(
            "Chưa cài transformers. Chạy: pip install transformers"
        )

    if _vqa_model is not None:
        return _vqa_processor, _vqa_model

    device = _get_device()
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    _vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL_ID)
    _vqa_model     = BlipForQuestionAnswering.from_pretrained(
        VQA_MODEL_ID, torch_dtype=dtype
    ).to(device)
    _vqa_model.eval()

    return _vqa_processor, _vqa_model


def _translate_to_vi(text: str) -> str:
    """
    Dịch text tiếng Anh sang tiếng Việt.
    Dùng deep-translator (GoogleTranslator).
    Nếu chưa cài hoặc lỗi mạng, trả về nguyên text gốc.
    """
    if not _TRANSLATOR_OK or not text:
        return text
    try:
        translated = GoogleTranslator(source="en", target="vi").translate(text)
        return translated if translated else text
    except Exception:
        return text   # fallback: trả về tiếng Anh nếu dịch lỗi


def generate_caption_from_image(
    cropped_img: Image.Image,
    translate: bool = True,
    max_new_tokens: int = 50,
) -> str:
    """
    Sinh caption mô tả những gì BLIP thấy trong ảnh crop.

    Args:
        cropped_img: ảnh PIL đã crop theo bbox từ YOLO
        translate: True = dịch sang tiếng Việt, False = giữ tiếng Anh
        max_new_tokens: độ dài caption tối đa

    Returns:
        Chuỗi mô tả ảnh (tiếng Việt nếu translate=True)
    """
    processor, model = load_caption_model()
    device = _get_device()

    # BLIP cần ảnh RGB
    if cropped_img.mode != "RGB":
        cropped_img = cropped_img.convert("RGB")

    inputs = processor(
        images=cropped_img,
        return_tensors="pt"
    ).to(device, model.dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

    caption_en = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    return _translate_to_vi(caption_en) if translate else caption_en


def answer_question_from_image(
    question: str,
    cropped_img: Image.Image,
    translate: bool = True,
) -> str:
    """
    Trả lời câu hỏi dựa trên nội dung ảnh thật (VQA).

    Args:
        question: câu hỏi tiếng Việt của người dùng
        cropped_img: ảnh PIL đã crop theo bbox từ YOLO
        translate: True = dịch câu trả lời sang tiếng Việt

    Returns:
        Câu trả lời (tiếng Việt nếu translate=True)
    """
    processor, model = load_vqa_model()
    device = _get_device()

    if cropped_img.mode != "RGB":
        cropped_img = cropped_img.convert("RGB")

    # BLIP-1 VQA nhận câu hỏi tiếng Anh tốt hơn
    # → dịch câu hỏi tiếng Việt sang tiếng Anh trước
    if _TRANSLATOR_OK and question:
        try:
            question_en = GoogleTranslator(
                source="vi", target="en"
            ).translate(question)
        except Exception:
            question_en = question
    else:
        question_en = question

    inputs = processor(
        images=cropped_img,
        text=question_en,
        return_tensors="pt"
    ).to(device, model.dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
        )

    answer_en = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    return _translate_to_vi(answer_en) if translate else answer_en


def unload_models():
    """
    Giải phóng bộ nhớ GPU/RAM khi không còn dùng.
    Gọi khi cần tiết kiệm tài nguyên.
    """
    global _caption_processor, _caption_model, _vqa_processor, _vqa_model

    _caption_model     = None
    _caption_processor = None
    _vqa_model         = None
    _vqa_processor     = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()