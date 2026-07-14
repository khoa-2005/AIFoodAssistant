"""
food_history.py
----------------
Discovery History — SQLite. KHÔNG phải AI, chỉ là hạ tầng lưu trữ cần thiết
để user_analytics.py (K-Means) có dữ liệu đầu vào. Đúng interface đã thống
nhất trong FOOD_AI_CONTEXT.md mục 6.3.

Interface:
    init_db()
    save_detection(session_id, det, info)
    get_session_history(session_id)
    get_total_calo(session_id)
    get_session_detection_count(session_id)   # thêm mới — xem ghi chú bên dưới

`det` là dict kết quả 1 detection từ YOLOv8s, tối thiểu cần key "class_name"
và "confidence" (đúng field app.py hiện đang dùng).
`info` là dict trả về từ food_info.get_food_info(), tối thiểu cần
"ten_hien_thi", "calo", "vung_mien".

GHI CHÚ FIX theo checklist kiểm thử (P5, cuối PHAN_CONG_3_NGAY.md):
    "Tab 3: với session mới (chưa có lịch sử) -> hiển thị đúng thông báo
    'chưa đủ dữ liệu'" và "session có >= 3 lần khám phá -> phân khúc hiển
    thị đúng". Bản pseudocode gốc trong context file (mục 6.4) không có
    hàm đếm số lượt của RIÊNG 1 session, nên user_analytics.get_user_segment()
    không thể phân biệt "session mới có 1 lượt" với "session đã dùng nhiều
    lần" — nó sẽ gán segment cho bất kỳ session nào miễn tổng số session
    trong DB >= 3. Hàm get_session_detection_count() dưới đây được thêm để
    user_analytics.py dùng làm điều kiện chặn trước khi gán segment.
"""

import re
import sqlite3
import time
from typing import Dict, List, Tuple

DB_PATH = "food_history.db"


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            mon_an TEXT,
            ten_hien_thi TEXT,
            confidence REAL,
            calo_so INTEGER,
            vung_mien TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_detection(session_id: str, det: Dict, info: Dict, db_path: str = DB_PATH) -> None:
    """Gọi ngay sau khi YOLOv8s nhận diện + tra food_info xong 1 món."""
    init_db(db_path)

    calo_str = info.get("calo", "0") or "0"
    calo_match = re.search(r"\d+", calo_str)
    calo_so = int(calo_match.group()) if calo_match else 0

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO history
        (session_id, mon_an, ten_hien_thi, confidence, calo_so, vung_mien, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            det["class_name"],
            info.get("ten_hien_thi", det["class_name"]),
            det.get("confidence", 0.0),
            calo_so,
            info.get("vung_mien", "N/A"),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def get_session_history(session_id: str, db_path: str = DB_PATH) -> List[Tuple]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ten_hien_thi, confidence, calo_so, vung_mien, timestamp, mon_an "
        "FROM history WHERE session_id = ? ORDER BY timestamp DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return rows


def get_total_calo(session_id: str, db_path: str = DB_PATH) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    result = conn.execute(
        "SELECT SUM(calo_so) FROM history WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return result[0] or 0


def get_session_detection_count(session_id: str, db_path: str = DB_PATH) -> int:
    """Số lượt khám phá của RIÊNG session này — dùng để chặn phân segment quá sớm."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    result = conn.execute(
        "SELECT COUNT(*) FROM history WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return result[0] or 0


if __name__ == "__main__":
    # Test tạo DB, insert giả lập, query lại đúng (P3, deliverable Ngày 1)
    init_db()
    fake_det = {"class_name": "Pho_Bo", "confidence": 0.91}
    fake_info = {"ten_hien_thi": "Phở Bò", "calo": "350 kcal / tô", "vung_mien": "Miền Bắc"}
    save_detection("test_session", fake_det, fake_info)

    fake_det2 = {"class_name": "Bun_Bo_Hue", "confidence": 0.87}
    fake_info2 = {"ten_hien_thi": "Bún Bò Huế", "calo": "400 kcal / tô", "vung_mien": "Miền Trung"}
    save_detection("test_session", fake_det2, fake_info2)

    print("History:", get_session_history("test_session"))
    print("Total calo:", get_total_calo("test_session"))
    print("Detection count:", get_session_detection_count("test_session"))
