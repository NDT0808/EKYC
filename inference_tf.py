import tensorflow as tf
from ultralytics import YOLO
import cv2

# ==========================================
# PHẦN 1: KHỞI TẠO HỆ THỐNG
# ==========================================
print("🚀 Khởi động hệ thống nhận diện eKYC...")
print(f"📦 Đang sử dụng TensorFlow Backend phiên bản: {tf.__version__}")

# ĐƯỜNG DẪN ĐẾN MODEL TFLITE
# (Lưu ý: Hãy kiểm tra lại đúng đường dẫn file .tflite trong thư mục máy của bạn nhé)
tflite_model_path = "runs/train/yolo11n_custom/weights/best_saved_model/best_float32.tflite"

print("🧠 Đang load model TensorFlow Lite vào bộ nhớ...")
tf_model = YOLO(tflite_model_path)


# ==========================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
# ==========================================
# ĐƯỜNG DẪN ẢNH CẦN NHẬN DIỆN
# (Bạn có thể đổi tên file ảnh khác trong thư mục test để demo nhiều trường hợp)
test_image = "datasets/bao.jpg"

print(f"🔍 Đang tiến hành nhận diện CCCD trên ảnh: {test_image}")


# ==========================================
# PHẦN 3: XỬ LÝ AI (INFERENCE)
# ==========================================
# Chạy dự đoán với ngưỡng độ tự tin (Confidence) là 50%
results = tf_model.predict(
    source=test_image,
    conf=0.2
)


# ==========================================
# PHẦN 4: HIỂN THỊ KẾT QUẢ DEMO TRỰC QUAN
# ==========================================
annotated_frame = results[0].plot()

window_name = "Ket Qua Nhan Dien eKYC (TFLite)"

# --- THÊM 2 DÒNG NÀY ĐỂ FIX LỖI ẢNH TO TRÀN MÀN HÌNH ---
# Khởi tạo cửa sổ cho phép co giãn (WINDOW_NORMAL)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
# Ép kích thước cửa sổ về 800x600 cho vừa đẹp trên màn hình
cv2.resizeWindow(window_name, 800, 600) 

# Hiển thị ảnh
cv2.imshow(window_name, annotated_frame)

print("✅ Nhận diện hoàn tất! Cửa sổ kết quả đang được hiển thị.")
cv2.waitKey(0)
cv2.destroyAllWindows()