# 🍜 Food AI Assistant

**Ứng dụng nhận diện món ăn Việt Nam bằng AI**  
Sử dụng YOLOv8 + Streamlit

![Food AI](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00A3E0?style=for-the-badge&logo=ultralytics&logoColor=white)

---

## ✨ Tính năng nổi bật

- Nhận diện **5 món ăn Việt Nam tiêu biểu**: Bánh Căn, Bánh Chưng, Bánh Bèo, Bánh Bột Lọc, Bánh Giò
- Tính toán **lượng calo** tự động
- Hiển thị **mô tả chi tiết** món ăn
- Gợi ý **món ăn tương tự** có mức calo gần nhau
- Hỗ trợ **chụp camera trực tiếp** hoặc upload ảnh
- Giao diện hiện đại, thân thiện

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1: Clone dự án
```bash
git clone https://github.com/TenBanCuaBan/AIFoodAssistant.git
cd AIFoodAssistant


Bước 2: Cài đặt thư viện
pip install -r requirements.txt


Bước 3: Tải mô hình AI (best.pt) đã có sẵn trên file gốc


Bước 4: Chạy ứng dụng
Bashstreamlit run app.py


📁 Cấu trúc thư mục
textAIFoodAssistant/
├── app.py
├── food_info.py
├── best.pt                
├── requirements.txt
├── README.md
├── .gitignore
└── dataset.yaml



🛠 Công nghệ sử dụng

Backend: Ultralytics YOLOv8
Frontend: Streamlit
Ngôn ngữ: Python



👥 Thành viên dự án
Team AI Food Assistant



