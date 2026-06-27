# food_info.py
FOOD_INFO = {
    "banh_beo": {
        "ten_hien_thi": "Bánh Bèo",
        "mo_ta": "Bánh bèo chén nhỏ nhắn, tôm chấy thơm lừng, da heo giòn tan, ăn kèm nước mắm chua ngọt.",
        "vung_mien": "Huế",
        "calo": "180-250 kcal / phần (4-5 chén)",
        "similar_foods": ["Bánh bột lọc", "Bánh căn", "Bánh giò"]
    },
    "banh_bot_loc": {
        "ten_hien_thi": "Bánh Bột Lọc",
        "mo_ta": "Vỏ bánh trong suốt dai ngon, nhân tôm thịt đậm đà, thường gói lá chuối hoặc để trần.",
        "vung_mien": "Huế",
        "calo": "220-300 kcal / phần",
        "similar_foods": ["Bánh bèo", "Bánh giò", "Bánh căn"]
    },
    "banh_can": {
        "ten_hien_thi": "Bánh Căn",
        "mo_ta": "Bánh nướng giòn từ bột gạo trong khuôn đất nung, nhân tôm/trứng, chấm mắm nêm.",
        "vung_mien": "Nha Trang",
        "calo": "230-280 kcal / phần",
        "similar_foods": ["Bánh bột lọc", "Bánh bèo", "Bánh khọt"]
    },
    "banh_gio": {
        "ten_hien_thi": "Bánh Giò",
        "mo_ta": "Bánh bột gạo mềm, nhân thịt băm mộc nhĩ, gói lá chuối thơm phức.",
        "vung_mien": "Miền Bắc",
        "calo": "250-320 kcal / cái",
        "similar_foods": ["Bánh bột lọc", "Bánh bèo", "Bánh chưng"]
    },
    "banh_chung": {
        "ten_hien_thi": "Bánh Chưng",
        "mo_ta": "Bánh vuông gói lá dong, nhân gạo nếp, đậu xanh, thịt mỡ - món ăn Tết truyền thống.",
        "vung_mien": "Miền Bắc",
        "calo": "350-450 kcal / phần",
        "similar_foods": ["Bánh giò", "Bánh tét", "Xôi"]
    },
    # Bạn có thể thêm dần các món khác sau
}

def get_food_info(class_name: str):
    """Tìm kiếm linh hoạt, bỏ qua hoa thường và dấu cách"""
    if not class_name:
        return None
    
    key = class_name.lower().strip().replace(" ", "_").replace("-", "_")
    
    # Thử tìm trực tiếp
    info = FOOD_INFO.get(key)
    if info:
        return info
    
    # Thử tìm gần giống
    for k in FOOD_INFO.keys():
        if k.lower() in key or key in k.lower():
            return FOOD_INFO[k]
    
    return None