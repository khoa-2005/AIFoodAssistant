"""
food_info.py - Food AI Assistant
Dữ liệu thông tin chi tiết cho 30 món ăn Việt Nam.
Key phải khớp CHÍNH XÁC với tên class trong dataset.yaml (names).
"""

FOOD_INFO = {
    "Banh Flan": {
"ten_hien_thi": "Bánh flan",
"mo_ta": "Bánh trứng sữa mềm mịn phủ caramel, món tráng miệng phổ biến tại Việt Nam.",
"vung_mien": "Toàn quốc",
"calo": "180 kcal / phần",
"khau_phan": "1 phần (~100g)",
"protein": "4g",
"carb": "22g",
"fat": "8g",
"thanh_phan": "Trứng gà, sữa tươi, sữa đặc, đường caramel",
"gia_trung_binh": "10.000 - 25.000 VNĐ",
"thoi_diem_phu_hop": "Tráng miệng",
"chi_so_suc_khoe": "⭐⭐⭐☆☆",
"khuyen_nghi": "Hạn chế nếu đang kiểm soát đường huyết."
},

"Banh beo": {
"ten_hien_thi": "Bánh bèo",
"mo_ta": "Bánh hấp từ bột gạo ăn kèm tôm chấy và nước mắm.",
"vung_mien": "Huế",
"calo": "80 kcal / chén",
"khau_phan": "1 chén",
"protein": "2g",
"carb": "15g",
"fat": "1g",
"thanh_phan": "Bột gạo, tôm chấy, hành phi",
"gia_trung_binh": "5.000 - 10.000 VNĐ",
"thoi_diem_phu_hop": "Ăn vặt",
"chi_so_suc_khoe": "⭐⭐⭐⭐☆",
"khuyen_nghi": "Ít chất béo, phù hợp ăn nhẹ."
},

"Banh bot loc": {
"ten_hien_thi": "Bánh bột lọc",
"mo_ta": "Bánh trong suốt nhân tôm thịt đặc trưng miền Trung.",
"vung_mien": "Huế",
"calo": "55 kcal / cái",
"khau_phan": "1 cái",
"protein": "2g",
"carb": "10g",
"fat": "1g",
"thanh_phan": "Bột năng, tôm, thịt heo",
"gia_trung_binh": "3.000 - 7.000 VNĐ",
"thoi_diem_phu_hop": "Ăn vặt",
"chi_so_suc_khoe": "⭐⭐⭐⭐☆",
"khuyen_nghi": "Ăn kèm rau để tăng chất xơ."
},

"Banh can": {
"ten_hien_thi": "Bánh căn",
"mo_ta": "Bánh đổ khuôn đất nung, thường ăn với trứng hoặc hải sản.",
"vung_mien": "Ninh Thuận - Bình Thuận",
"calo": "70 kcal / cái",
"khau_phan": "1 cái",
"protein": "3g",
"carb": "8g",
"fat": "3g",
"thanh_phan": "Bột gạo, trứng, hành lá",
"gia_trung_binh": "5.000 - 10.000 VNĐ",
"thoi_diem_phu_hop": "Ăn sáng",
"chi_so_suc_khoe": "⭐⭐⭐⭐☆",
"khuyen_nghi": "Ăn kèm rau sống và nước mắm."
},

"Banh canh": {
"ten_hien_thi": "Bánh canh",
"mo_ta": "Món nước với sợi bánh canh dai và nước dùng đậm đà.",
"vung_mien": "Toàn quốc",
"calo": "420 kcal / tô",
"khau_phan": "1 tô",
"protein": "20g",
"carb": "55g",
"fat": "12g",
"thanh_phan": "Bánh canh, giò heo hoặc hải sản",
"gia_trung_binh": "35.000 - 60.000 VNĐ",
"thoi_diem_phu_hop": "Bữa chính",
"chi_so_suc_khoe": "⭐⭐⭐⭐☆",
"khuyen_nghi": "Bổ sung rau xanh để cân bằng dinh dưỡng."
},

"Banh chung": {
"ten_hien_thi": "Bánh chưng",
"mo_ta": "Món bánh truyền thống ngày Tết của người Việt.",
"vung_mien": "Miền Bắc",
"calo": "250 kcal / 100g",
"khau_phan": "100g",
"protein": "6g",
"carb": "40g",
"fat": "7g",
"thanh_phan": "Gạo nếp, đậu xanh, thịt heo",
"gia_trung_binh": "50.000 - 120.000 VNĐ",
"thoi_diem_phu_hop": "Bữa chính",
"chi_so_suc_khoe": "⭐⭐⭐☆☆",
"khuyen_nghi": "Không nên ăn quá nhiều vì nhiều tinh bột."
},

"Banh gio": {
"ten_hien_thi": "Bánh giò",
"mo_ta": "Bánh hấp hình chóp, nhân thịt băm và mộc nhĩ.",
"vung_mien": "Miền Bắc",
"calo": "320 kcal / cái",
"khau_phan": "1 cái",
"protein": "10g",
"carb": "42g",
"fat": "12g",
"thanh_phan": "Bột gạo, thịt heo, mộc nhĩ",
"gia_trung_binh": "15.000 - 30.000 VNĐ",
"thoi_diem_phu_hop": "Ăn sáng",
"chi_so_suc_khoe": "⭐⭐⭐☆☆",
"khuyen_nghi": "Ăn kèm dưa góp để đỡ ngấy."
},

"Banh khot": {
"ten_hien_thi": "Bánh khọt",
"mo_ta": "Bánh chiên nhỏ giòn với nhân tôm đặc trưng Vũng Tàu.",
"vung_mien": "Vũng Tàu",
"calo": "400 kcal / phần",
"khau_phan": "8 bánh",
"protein": "15g",
"carb": "35g",
"fat": "20g",
"thanh_phan": "Bột gạo, tôm, nước cốt dừa",
"gia_trung_binh": "30.000 - 60.000 VNĐ",
"thoi_diem_phu_hop": "Ăn vặt",
"chi_so_suc_khoe": "⭐⭐⭐☆☆",
"khuyen_nghi": "Ăn cùng rau sống để giảm cảm giác ngấy."
},

"Banh mi": {
"ten_hien_thi": "Bánh mì",
"mo_ta": "Món ăn đường phố nổi tiếng của Việt Nam.",
"vung_mien": "Toàn quốc",
"calo": "400 kcal / ổ",
"khau_phan": "1 ổ",
"protein": "15g",
"carb": "45g",
"fat": "15g",
"thanh_phan": "Bánh mì, pate, thịt, rau",
"gia_trung_binh": "20.000 - 50.000 VNĐ",
"thoi_diem_phu_hop": "Ăn sáng",
"chi_so_suc_khoe": "⭐⭐⭐☆☆",
"khuyen_nghi": "Hạn chế nếu đang giảm cân."
},

"Banh pia": {
    "ten_hien_thi": "Bánh pía",
    "mo_ta": "Đặc sản Sóc Trăng với lớp vỏ nhiều tầng, nhân đậu xanh, sầu riêng và trứng muối.",
    "vung_mien": "Sóc Trăng",
    "calo": "420 kcal / cái",
    "khau_phan": "1 cái (~120g)",
    "protein": "7g",
    "carb": "60g",
    "fat": "17g",
    "thanh_phan": "Bột mì, đậu xanh, sầu riêng, trứng muối",
    "gia_trung_binh": "15.000 - 40.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn vặt, tráng miệng",
    "chi_so_suc_khoe": "⭐⭐☆☆☆",
    "khuyen_nghi": "Nhiều đường và chất béo, nên ăn vừa phải."
},

"Banh tet": {
    "ten_hien_thi": "Bánh tét",
    "mo_ta": "Bánh nếp hình trụ truyền thống của miền Nam, thường xuất hiện dịp Tết.",
    "vung_mien": "Miền Nam",
    "calo": "290 kcal / 100g",
    "khau_phan": "100g",
    "protein": "7g",
    "carb": "45g",
    "fat": "9g",
    "thanh_phan": "Gạo nếp, đậu xanh, thịt heo",
    "gia_trung_binh": "50.000 - 150.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa chính",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Không nên ăn quá nhiều vào buổi tối."
},

"Banh trang nuong": {
    "ten_hien_thi": "Bánh tráng nướng",
    "mo_ta": "Món ăn vặt nổi tiếng Đà Lạt, được ví như pizza Việt Nam.",
    "vung_mien": "Đà Lạt",
    "calo": "380 kcal / cái",
    "khau_phan": "1 cái",
    "protein": "14g",
    "carb": "40g",
    "fat": "18g",
    "thanh_phan": "Bánh tráng, trứng, hành lá, khô bò",
    "gia_trung_binh": "15.000 - 35.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn vặt",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Hạn chế ăn thường xuyên do nhiều chất béo."
},

"Banh uot": {
    "ten_hien_thi": "Bánh ướt",
    "mo_ta": "Bánh gạo hấp mỏng ăn kèm chả lụa và nước mắm.",
    "vung_mien": "Toàn quốc",
    "calo": "320 kcal / phần",
    "khau_phan": "1 đĩa",
    "protein": "12g",
    "carb": "55g",
    "fat": "6g",
    "thanh_phan": "Bột gạo, chả lụa, hành phi",
    "gia_trung_binh": "20.000 - 40.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn sáng",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Ít dầu mỡ, phù hợp ăn sáng."
},

"Banh xeo": {
    "ten_hien_thi": "Bánh xèo",
    "mo_ta": "Bánh chiên giòn với nhân tôm, thịt và giá đỗ.",
    "vung_mien": "Miền Trung - Nam",
    "calo": "450 kcal / cái",
    "khau_phan": "1 cái",
    "protein": "18g",
    "carb": "40g",
    "fat": "24g",
    "thanh_phan": "Bột gạo, tôm, thịt heo, giá đỗ",
    "gia_trung_binh": "25.000 - 60.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa chính",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Ăn kèm nhiều rau xanh."
},

"Banh_Duc": {
    "ten_hien_thi": "Bánh đúc",
    "mo_ta": "Món ăn dân dã làm từ bột gạo hoặc bột năng.",
    "vung_mien": "Toàn quốc",
    "calo": "180 kcal / phần",
    "khau_phan": "1 phần",
    "protein": "4g",
    "carb": "35g",
    "fat": "2g",
    "thanh_phan": "Bột gạo, thịt băm, mộc nhĩ",
    "gia_trung_binh": "10.000 - 25.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn nhẹ",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Ít chất béo, dễ tiêu hóa."
},

"Bun mam": {
    "ten_hien_thi": "Bún mắm",
    "mo_ta": "Đặc sản miền Tây với nước lèo nấu từ mắm cá đậm đà.",
    "vung_mien": "Miền Tây",
    "calo": "550 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "30g",
    "carb": "60g",
    "fat": "18g",
    "thanh_phan": "Bún, mắm cá, tôm, mực, thịt quay",
    "gia_trung_binh": "40.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa, tối",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Hàm lượng natri khá cao."
},

"Bun_Bo_Hue": {
    "ten_hien_thi": "Bún bò Huế",
    "mo_ta": "Đặc sản Huế với nước dùng cay nồng từ sả và ruốc.",
    "vung_mien": "Huế",
    "calo": "500 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "28g",
    "carb": "55g",
    "fat": "16g",
    "thanh_phan": "Bún, thịt bò, giò heo, chả cua",
    "gia_trung_binh": "40.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng, trưa",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Giàu protein và năng lượng."
},

"Bun_Dau_mam_tom": {
    "ten_hien_thi": "Bún đậu mắm tôm",
    "mo_ta": "Món ăn nổi tiếng Hà Nội gồm bún, đậu phụ chiên và mắm tôm.",
    "vung_mien": "Hà Nội",
    "calo": "600 kcal / phần",
    "khau_phan": "1 mẹt",
    "protein": "25g",
    "carb": "50g",
    "fat": "28g",
    "thanh_phan": "Bún lá, đậu phụ, thịt luộc, chả cốm",
    "gia_trung_binh": "35.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Nhiều chất béo từ đồ chiên."
},"Bun_Rieu": {
    "ten_hien_thi": "Bún riêu",
    "mo_ta": "Món bún nước truyền thống với riêu cua, cà chua và đậu phụ.",
    "vung_mien": "Miền Bắc",
    "calo": "450 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "22g",
    "carb": "55g",
    "fat": "12g",
    "thanh_phan": "Bún, riêu cua, cà chua, đậu phụ",
    "gia_trung_binh": "30.000 - 60.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng, trưa",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Giàu đạm và khoáng chất từ cua đồng."
},

"Bun_Thit_Nuong": {
    "ten_hien_thi": "Bún thịt nướng",
    "mo_ta": "Bún tươi ăn kèm thịt nướng, rau sống và nước mắm chua ngọt.",
    "vung_mien": "Miền Nam",
    "calo": "500 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "25g",
    "carb": "60g",
    "fat": "15g",
    "thanh_phan": "Bún, thịt heo nướng, rau sống",
    "gia_trung_binh": "35.000 - 70.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa, tối",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Cân bằng giữa tinh bột và protein."
},

"Bun_ngan": {
    "ten_hien_thi": "Bún ngan",
    "mo_ta": "Món bún nước với thịt ngan mềm và nước dùng đậm vị.",
    "vung_mien": "Miền Bắc",
    "calo": "480 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "30g",
    "carb": "45g",
    "fat": "18g",
    "thanh_phan": "Bún, thịt ngan, nước dùng xương",
    "gia_trung_binh": "40.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa chính",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Giàu protein, phù hợp người vận động."
},

"Ca_Kho": {
    "ten_hien_thi": "Cá kho tộ",
    "mo_ta": "Món cá kho truyền thống với nước màu, tiêu và gia vị đậm đà.",
    "vung_mien": "Toàn quốc",
    "calo": "250 kcal / phần",
    "khau_phan": "100g",
    "protein": "25g",
    "carb": "5g",
    "fat": "12g",
    "thanh_phan": "Cá, nước mắm, tiêu, đường",
    "gia_trung_binh": "30.000 - 70.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa, tối",
    "chi_so_suc_khoe": "⭐⭐⭐⭐⭐",
    "khuyen_nghi": "Giàu Omega-3 và protein."
},

"Canh_chua": {
    "ten_hien_thi": "Canh chua",
    "mo_ta": "Món canh đặc trưng miền Nam với vị chua thanh từ me.",
    "vung_mien": "Miền Nam",
    "calo": "120 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "10g",
    "carb": "8g",
    "fat": "4g",
    "thanh_phan": "Cá, me, bạc hà, đậu bắp",
    "gia_trung_binh": "20.000 - 50.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa, tối",
    "chi_so_suc_khoe": "⭐⭐⭐⭐⭐",
    "khuyen_nghi": "Ít calo, giàu vitamin và chất xơ."
},

"Cao_lau": {
    "ten_hien_thi": "Cao lầu",
    "mo_ta": "Đặc sản Hội An với sợi mì dai đặc trưng và thịt xá xíu.",
    "vung_mien": "Hội An",
    "calo": "450 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "20g",
    "carb": "55g",
    "fat": "14g",
    "thanh_phan": "Mì cao lầu, thịt xá xíu, rau sống",
    "gia_trung_binh": "35.000 - 70.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa trưa",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Ăn kèm nhiều rau để tăng chất xơ."
},

"Cha gio": {
    "ten_hien_thi": "Chả giò",
    "mo_ta": "Món cuốn chiên giòn với nhân thịt, miến và rau củ.",
    "vung_mien": "Toàn quốc",
    "calo": "120 kcal / cuốn",
    "khau_phan": "1 cuốn",
    "protein": "4g",
    "carb": "10g",
    "fat": "7g",
    "thanh_phan": "Bánh tráng, thịt heo, miến",
    "gia_trung_binh": "5.000 - 15.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn khai vị",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Hạn chế ăn quá nhiều do chiên dầu."
},

"Chao long": {
    "ten_hien_thi": "Cháo lòng",
    "mo_ta": "Món cháo nấu từ gạo và nội tạng heo phổ biến tại Việt Nam.",
    "vung_mien": "Toàn quốc",
    "calo": "350 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "18g",
    "carb": "35g",
    "fat": "12g",
    "thanh_phan": "Gạo, lòng heo, huyết heo",
    "gia_trung_binh": "25.000 - 50.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Người có cholesterol cao nên hạn chế."
},

"Com chien": {
    "ten_hien_thi": "Cơm chiên",
    "mo_ta": "Món cơm rang với trứng, thịt và rau củ.",
    "vung_mien": "Toàn quốc",
    "calo": "550 kcal / phần",
    "khau_phan": "1 đĩa",
    "protein": "18g",
    "carb": "65g",
    "fat": "20g",
    "thanh_phan": "Cơm, trứng, thịt, rau củ",
    "gia_trung_binh": "30.000 - 60.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa chính",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Nhiều năng lượng, phù hợp người hoạt động nhiều."
},"Com tam": {
    "ten_hien_thi": "Cơm tấm",
    "mo_ta": "Đặc sản Sài Gòn với cơm tấm ăn kèm sườn nướng, bì và chả trứng.",
    "vung_mien": "TP. Hồ Chí Minh",
    "calo": "650 kcal / phần",
    "khau_phan": "1 đĩa",
    "protein": "32g",
    "carb": "68g",
    "fat": "25g",
    "thanh_phan": "Cơm tấm, sườn nướng, bì, chả trứng",
    "gia_trung_binh": "35.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng, trưa",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Khá giàu năng lượng, phù hợp người lao động hoặc vận động nhiều."
},

"Goi cuon": {
    "ten_hien_thi": "Gỏi cuốn",
    "mo_ta": "Món cuốn tươi với tôm, thịt, rau sống và bún.",
    "vung_mien": "Miền Nam",
    "calo": "90 kcal / cuốn",
    "khau_phan": "1 cuốn",
    "protein": "6g",
    "carb": "10g",
    "fat": "2g",
    "thanh_phan": "Bánh tráng, tôm, thịt, bún, rau sống",
    "gia_trung_binh": "5.000 - 15.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn nhẹ, khai vị",
    "chi_so_suc_khoe": "⭐⭐⭐⭐⭐",
    "khuyen_nghi": "Ít dầu mỡ, phù hợp chế độ ăn lành mạnh."
},

"Hu tieu": {
    "ten_hien_thi": "Hủ tiếu",
    "mo_ta": "Món nước phổ biến miền Nam với nước dùng thanh ngọt từ xương.",
    "vung_mien": "Miền Nam",
    "calo": "430 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "22g",
    "carb": "52g",
    "fat": "10g",
    "thanh_phan": "Hủ tiếu, thịt heo, tôm, xương hầm",
    "gia_trung_binh": "35.000 - 70.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng, trưa",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Khá cân bằng giữa đạm và tinh bột."
},

"Mi quang": {
    "ten_hien_thi": "Mì Quảng",
    "mo_ta": "Đặc sản Quảng Nam với sợi mì vàng, ít nước dùng và nhiều topping.",
    "vung_mien": "Quảng Nam",
    "calo": "470 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "24g",
    "carb": "55g",
    "fat": "14g",
    "thanh_phan": "Mì Quảng, tôm, thịt, trứng cút",
    "gia_trung_binh": "35.000 - 70.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng, trưa",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Giàu đạm và khoáng chất."
},

"Nem chua": {
    "ten_hien_thi": "Nem chua",
    "mo_ta": "Đặc sản Thanh Hóa làm từ thịt heo lên men tự nhiên.",
    "vung_mien": "Thanh Hóa",
    "calo": "70 kcal / cái",
    "khau_phan": "1 cái",
    "protein": "5g",
    "carb": "2g",
    "fat": "4g",
    "thanh_phan": "Thịt heo, bì heo, tỏi, ớt",
    "gia_trung_binh": "3.000 - 8.000 VNĐ",
    "thoi_diem_phu_hop": "Ăn vặt",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Nên sử dụng sản phẩm đảm bảo vệ sinh thực phẩm."
},

"Pho": {
    "ten_hien_thi": "Phở",
    "mo_ta": "Món ăn truyền thống nổi tiếng của Việt Nam với nước dùng ninh từ xương.",
    "vung_mien": "Hà Nội",
    "calo": "450 kcal / tô",
    "khau_phan": "1 tô",
    "protein": "25g",
    "carb": "50g",
    "fat": "12g",
    "thanh_phan": "Bánh phở, thịt bò hoặc gà, nước dùng",
    "gia_trung_binh": "35.000 - 80.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng",
    "chi_so_suc_khoe": "⭐⭐⭐⭐⭐",
    "khuyen_nghi": "Cân bằng dinh dưỡng và dễ tiêu hóa."
},

"Sup_cua": {
    "ten_hien_thi": "Súp cua",
    "mo_ta": "Món súp giàu dinh dưỡng từ cua, trứng và nấm.",
    "vung_mien": "Toàn quốc",
    "calo": "180 kcal / chén",
    "khau_phan": "1 chén",
    "protein": "12g",
    "carb": "15g",
    "fat": "6g",
    "thanh_phan": "Thịt cua, trứng, nấm, bột năng",
    "gia_trung_binh": "20.000 - 45.000 VNĐ",
    "thoi_diem_phu_hop": "Khai vị, ăn nhẹ",
    "chi_so_suc_khoe": "⭐⭐⭐⭐☆",
    "khuyen_nghi": "Giàu protein và khoáng chất."
},

"Xoi xeo": {
    "ten_hien_thi": "Xôi xéo",
    "mo_ta": "Món xôi truyền thống Hà Nội với đậu xanh và hành phi.",
    "vung_mien": "Hà Nội",
    "calo": "420 kcal / phần",
    "khau_phan": "1 gói",
    "protein": "10g",
    "carb": "75g",
    "fat": "8g",
    "thanh_phan": "Gạo nếp, đậu xanh, hành phi",
    "gia_trung_binh": "15.000 - 35.000 VNĐ",
    "thoi_diem_phu_hop": "Bữa sáng",
    "chi_so_suc_khoe": "⭐⭐⭐☆☆",
    "khuyen_nghi": "Nhiều tinh bột, tạo cảm giác no lâu."
},

}


def get_food_info(class_name: str) -> dict:
    return FOOD_INFO.get(
        class_name,
        {
            "ten_hien_thi": class_name,
            "mo_ta": "Chưa có thông tin chi tiết cho món này.",
            "vung_mien": "Không rõ",

            "calo": "Không rõ",
            "khau_phan": "Không rõ",

            "protein": "Không rõ",
            "carb": "Không rõ",
            "fat": "Không rõ",

            "thanh_phan": "Không rõ",

            "gia_trung_binh": "Không rõ",
            "thoi_diem_phu_hop": "Không rõ",
            "chi_so_suc_khoe": "Chưa đánh giá",

            "khuyen_nghi": "Không có khuyến nghị."
        },
    )
