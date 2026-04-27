from ultralytics import YOLO

# 1. Trỏ tới file model tốt nhất vừa train xong bằng PyTorch
model_path = "runs/train/yolo11n_custom/weights/best.pt"

print("🔄 Đang load mô hình PyTorch gốc...")
model = YOLO(model_path)

# 2. Xuất mô hình sang định dạng TensorFlow Lite
print("⚙️ Đang chuyển đổi sang định dạng TensorFlow Lite...")
# Lệnh này sẽ tự động gọi các thư viện chuyển đổi và lưu file mới
model.export(format="tflite")

print("✅ Hoàn tất! Hãy kiểm tra trong thư mục weights, bạn sẽ thấy file dạng .tflite")