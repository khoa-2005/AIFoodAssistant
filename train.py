"""
train.py - Food AI Assistant
Huấn luyện mô hình YOLOv8 nhận diện món ăn Việt Nam
Giai đoạn 2: Huấn luyện Mô hình YOLO
"""

import os
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO


# ─── CẤU HÌNH HUẤN LUYỆN ──────────────────────────────────────────────────────

# Đường dẫn tới file dataset.yaml (xuất từ Roboflow theo định dạng YOLOv8)
DATASET_YAML = "dataset.yaml"

# Kiến trúc mô hình: yolov8n.pt (Nano) - phù hợp máy cá nhân & Google Colab
# Các lựa chọn khác: yolov8s.pt (Small), yolov8m.pt (Medium)
MODEL_WEIGHTS = "yolov8n.pt"

# Số epochs - khuyến nghị 50 cho tập dữ liệu nhỏ (5-10 món, 30-50 ảnh/món)
EPOCHS = 50

# Kích thước ảnh đầu vào (pixels)
IMG_SIZE = 640

# Batch size - giảm xuống 8 hoặc 4 nếu bị lỗi Out of Memory
BATCH_SIZE = 16

# Tên thư mục lưu kết quả trong runs/detect/
PROJECT_NAME = "food_ai"
RUN_NAME = "vietnamese_food_v1"

# Thiết bị: "0" = GPU CUDA, "cpu" = CPU
# Tự động phát hiện GPU nếu có
import torch

DEVICE = "0" if torch.cuda.is_available() else "cpu"
# Độ tự tin tối thiểu khi đánh giá
CONF_THRESHOLD = 0.25

# ──────────────────────────────────────────────────────────────────────────────


def check_dataset(yaml_path: str) -> bool:
    """Kiểm tra file dataset.yaml tồn tại và hợp lệ."""
    if not Path(yaml_path).exists():
        print(f"[LỖI] Không tìm thấy file dataset: {yaml_path}")
        print("  → Xuất dataset từ Roboflow theo định dạng YOLOv8")
        print("  → Đặt file dataset.yaml vào cùng thư mục với train.py")
        return False
    print(f"[OK] Tìm thấy dataset: {yaml_path}")
    return True


def train(
    dataset_yaml: str = DATASET_YAML,
    model_weights: str = MODEL_WEIGHTS,
    epochs: int = EPOCHS,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
    project: str = PROJECT_NAME,
    name: str = RUN_NAME,
) -> str | None:
    """
    Chạy huấn luyện YOLOv8.

    Returns:
        Đường dẫn tới file best.pt nếu thành công, None nếu thất bại.
    """
    # 1. Kiểm tra dataset
    if not check_dataset(dataset_yaml):
        return None

    # 2. Nạp mô hình pre-trained
    print(f"\n[INFO] Nạp mô hình: {model_weights}")
    model = YOLO(model_weights)

    # 3. Bắt đầu huấn luyện
    print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device}")
    print(f"       Epochs: {epochs} | Img size: {img_size} | Batch: {batch_size}\n")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        # Augmentation nhẹ phù hợp với tập dữ liệu nhỏ
        hsv_h=0.015,      # biến thiên màu sắc (Hue)
        hsv_s=0.7,        # biến thiên độ bão hòa (Saturation)
        hsv_v=0.4,        # biến thiên độ sáng (Value)
        flipud=0.0,        # không lật dọc (ảnh thức ăn thường có hướng cố định)
        fliplr=0.5,        # lật ngang 50%
        mosaic=1.0,        # ghép ảnh mosaic để tăng ngữ cảnh
        # Lưu checkpoints
        save=True,
        save_period=10,    # lưu checkpoint mỗi 10 epochs
        # Đánh giá trên val set
        val=True,
        conf=CONF_THRESHOLD,
        # Verbose
        verbose=True,
    )

    # 4. Trích xuất đường dẫn best.pt
    weights_dir = Path(project) / name / "weights"
    best_pt = weights_dir / "best.pt"

    if best_pt.exists():
        print(f"\n[✓] Huấn luyện hoàn tất!")
        print(f"    Trọng số tốt nhất: {best_pt}")

        # Sao chép best.pt ra thư mục gốc để tiện sử dụng trong app.py
        dest = Path("best.pt")
        shutil.copy(best_pt, dest)
        print(f"    Đã sao chép → {dest.resolve()}")
        return str(dest)
    else:
        print(f"\n[LỖI] Không tìm thấy best.pt tại {weights_dir}")
        return None


def evaluate(model_path: str = "best.pt", dataset_yaml: str = DATASET_YAML):
    """
    Đánh giá mô hình đã huấn luyện trên tập validation.
    Chạy sau khi train() hoàn tất để xem mAP, Precision, Recall.
    """
    if not Path(model_path).exists():
        print(f"[LỖI] Không tìm thấy model: {model_path}")
        return

    print(f"\n[INFO] Đánh giá mô hình: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, conf=CONF_THRESHOLD)

    print("\n── KẾT QUẢ ĐÁNH GIÁ ──────────────────────")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Food AI - YOLOv8 Trainer")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="train",
        help="Chế độ chạy: train (huấn luyện), eval (đánh giá), both (cả hai)",
    )
    parser.add_argument("--data",    default=DATASET_YAML,  help="Đường dẫn dataset.yaml")
    parser.add_argument("--weights", default=MODEL_WEIGHTS, help="Mô hình khởi tạo (vd: yolov8n.pt)")
    parser.add_argument("--epochs",  type=int, default=EPOCHS,    help="Số epochs")
    parser.add_argument("--imgsz",   type=int, default=IMG_SIZE,  help="Kích thước ảnh")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--device",  default=DEVICE, help="Thiết bị: 0 (GPU) hoặc cpu")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        best_model = train(
            dataset_yaml=args.data,
            model_weights=args.weights,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
        )

    if args.mode in ("eval", "both"):
        evaluate(dataset_yaml=args.data)