# --- CÁC THƯ VIỆN CẦN THIẾT ---
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from deepface import DeepFace
from collections import deque
from ultralytics import YOLO
import imutils
import threading
import time
import os

# --- CẤU HÌNH ---
REFERENCE_IMAGE_PATH = "datasets/trong.png"
PERSON_NAME = os.path.splitext(os.path.basename(REFERENCE_IMAGE_PATH))[0].upper()

pt_model_path = "runs/train/yolo11n_custom/weights/best.pt"
tflite_model_path = "runs/train/yolo11n_custom/weights/best_saved_model/best_float32.tflite" 
WEBCAM_ID = 1
 
# Cấu hình chung và các model
DETECTOR_BACKEND = "opencv"
LIVENESS_MODEL_NAME = "nguyenkhoa/vit_Liveness_detection_v1.0"
DEEPFACE_MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.6

# Cấu hình tối ưu hóa hiệu năng
FRAME_WIDTH = 640
FRAME_SKIP = 5 

# Cấu hình logic làm mượt
DEQUE_SIZE = 10
STABLE_THRESHOLD = 0.7 

# Cấu hình văn bản hiển thị
UI_TEXT_MATCH = f"DA XAC THUC: {PERSON_NAME}"
UI_TEXT_NO_MATCH = "KHONG TRUNG KHOP"
UI_TEXT_PROCESSING = "DANG PHAN TICH..."
UI_TEXT_PROMPT = "DUA KHUON MAT VAO KHUNG HINH"

# Biến toàn cục cho threading
latest_frame = None
lock = threading.Lock()
stop_thread = False

# Hàm chạy trong luồng đọc webcam
def webcam_reader_thread(cap_id):
    global latest_frame, stop_thread
    cap = cv2.VideoCapture(cap_id)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam trong thread.")
        return
        
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            latest_frame = frame.copy()
    cap.release()
    print("[INFO] Webcam thread đã dừng.")

def run_live_face_matching():
    global latest_frame, stop_thread

    # --- Bước 1: Tải các model và xử lý ảnh tham chiếu ---
    try:
        print("[INFO] Đang tải các model...")
        liveness_processor = AutoImageProcessor.from_pretrained(LIVENESS_MODEL_NAME)
        liveness_model = AutoModelForImageClassification.from_pretrained(LIVENESS_MODEL_NAME)
        
        print("[INFO] Đang tải YOLO PyTorch và TFLite...")
        yolo_model_pt = YOLO(pt_model_path)
        yolo_model_tflite = YOLO(tflite_model_path)
        
        print(f"[INFO] Đang xử lý ảnh tham chiếu: {REFERENCE_IMAGE_PATH}")
        reference_img_full = cv2.imread(REFERENCE_IMAGE_PATH)
        if reference_img_full is None: raise ValueError(f"Không thể đọc ảnh: {REFERENCE_IMAGE_PATH}")

        # Warm-up (Mồi)
        yolo_model_pt.predict(reference_img_full, verbose=False)
        yolo_model_tflite.predict(reference_img_full, verbose=False)

        # Đo tốc độ PyTorch
        start_pt = time.time()
        _ = yolo_model_pt.predict(reference_img_full, verbose=False)
        time_pt = (time.time() - start_pt) * 1000

        # Đo tốc độ TFLite
        start_tflite = time.time()
        results = yolo_model_tflite.predict(reference_img_full, verbose=False)
        time_tflite = (time.time() - start_tflite) * 1000

        fps_pt = 1000 / time_pt if time_pt > 0 else 0
        fps_tflite = 1000 / time_tflite if time_tflite > 0 else 0

        print("\n" + "="*50)
        print("📊 SO SÁNH TỐC ĐỘ CẮT ẢNH THỰC TẾ (INFERENCE)")
        print("="*50)
        print(f"1. PyTorch (.pt):     {time_pt:.2f} ms | ~{fps_pt:.1f} FPS")
        print(f"2. TFLite (.tflite):  {time_tflite:.2f} ms | ~{fps_tflite:.1f} FPS")
        print("="*50 + "\n")

        reference_face_crop = None
        for r in results:
            for box in r.boxes:
                if yolo_model_tflite.names[int(box.cls[0])] == 'image_person':
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    face_crop_original = reference_img_full[y1:y2, x1:x2]
                    reference_face_crop = imutils.rotate_bound(face_crop_original, -90)
                    cv2.imwrite("image_person_rotated_reference_face.jpg", reference_face_crop)
                    break
            if reference_face_crop is not None: break
        
        if reference_face_crop is None: raise ValueError("Không tìm thấy 'image_person' trong ảnh.")

        _ = DeepFace.represent(img_path=reference_face_crop, model_name=DEEPFACE_MODEL_NAME, enforce_detection=False)
        print("[SUCCESS] Tải model và xử lý ảnh tham chiếu thành công!")

    except Exception as e:
        print(f"[ERROR] Lỗi nghiêm trọng khi khởi tạo: {e}")
        return

    # --- Bước 2: Khởi tạo luồng Webcam và các biến ---
    results_window = deque(maxlen=DEQUE_SIZE)
    frame_number = 0
    stable_result = "ANALYZING"
    last_known_location = None
    
    # Khởi động luồng đọc webcam
    reader_thread = threading.Thread(target=webcam_reader_thread, args=(WEBCAM_ID,))
    reader_thread.start()
    
    print("\n[INFO] Bắt đầu xử lý... QUAN SÁT TERMINAL ĐỂ XEM LỖI.")
    time.sleep(2.0) 

    # --- Bước 3: Vòng lặp xử lý chính ---
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        frame_display = frame.copy()
        frame_number += 1
        
        # Chỉ xử lý AI trên các khung hình then chốt
        if frame_number % FRAME_SKIP == 0:
            try:
                print(f"\n--- Frame {frame_number} ---") # Dòng phân cách log
                
                # 1. LIVENESS CHECK
                liveness_label = "fake"
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = liveness_processor(images=image_pil, return_tensors="pt")
                with torch.no_grad():
                    outputs = liveness_model(**inputs)
                liveness_label = liveness_model.config.id2label[outputs.logits.argmax(-1).item()]
                
                # [DEBUG] In kết quả Liveness
                print(f"[DEBUG] Liveness Check: {liveness_label.upper()}")

                current_frame_result = "SKIPPED_AS_FAKE"
                face_location = None

                if liveness_label.lower() == 'live':
                    # 2. FACE MATCHING
                    verification_result = DeepFace.verify(
                        img1_path=frame, img2_path=reference_face_crop,
                        model_name=DEEPFACE_MODEL_NAME, detector_backend=DETECTOR_BACKEND,
                        distance_metric='cosine', enforce_detection=False, threshold=DISTANCE_THRESHOLD
                    )
                    
                    # [DEBUG] In khoảng cách khuôn mặt
                    dist = verification_result.get('distance', 'N/A')
                    is_verified = verification_result['verified']
                    print(f"[DEBUG] Khoảng cách Face Match: {dist} (Ngưỡng: {DISTANCE_THRESHOLD})")
                    print(f"[DEBUG] Kết quả so khớp: {'TRÙNG KHỚP' if is_verified else 'KHÔNG TRÙNG'}")

                    if 'facial_areas' in verification_result and 'img1' in verification_result['facial_areas']:
                        face_data = verification_result['facial_areas']['img1']
                        face_location = (face_data['x'], face_data['y'], face_data['w'], face_data['h'])
                    
                    current_frame_result = "MATCH" if is_verified else "NO_MATCH"
                else:
                    print(f"[DEBUG] >>> BỊ CHẶN: Model nghĩ đây là ảnh GIẢ (Fake/Spoof).")
                
                last_known_location = face_location
                results_window.append(current_frame_result)

            except Exception as e:
                print(f"[ERROR] Lỗi trong vòng lặp AI: {e}")
                last_known_location = None
                results_window.append("NO_FACE_DETECTED")
        
        # Cập nhật kết quả ổn định
        if len(results_window) >= DEQUE_SIZE:
            match_count = results_window.count("MATCH")
            if (match_count / len(results_window)) >= STABLE_THRESHOLD:
                stable_result = "STABLE MATCH"
            else:
                stable_result = "NOT MATCH"

        # Logic hiển thị
        display_text = ""
        color = (255, 255, 255) 

        if last_known_location: 
            if stable_result == "STABLE MATCH":
                display_text = UI_TEXT_MATCH
                color = (0, 255, 0) 
            elif stable_result == "NOT MATCH":
                display_text = UI_TEXT_NO_MATCH
                color = (0, 0, 255) 
            else: 
                display_text = UI_TEXT_PROCESSING
                color = (0, 255, 255) 
        else: 
            display_text = UI_TEXT_PROMPT

        if last_known_location:
            x, y, w, h = last_known_location
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_display, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (frame_display.shape[1] - text_width) // 2
            text_y = (frame_display.shape[0] + text_height) // 2
            cv2.putText(frame_display, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live Face Matching - DEBUG MODE", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_thread = True
    reader_thread.join()
    cv2.destroyAllWindows()
    
    print("\n\n-----------------------------------------")
    print("🏆 KẾT LUẬN CUỐI CÙNG")
    print("-----------------------------------------")
    
    if stable_result == "STABLE MATCH":
        print(f"✅ Kết luận: XÁC THỰC THÀNH CÔNG.")
        print(f"👤 Đã nhận diện: {PERSON_NAME}")
    else:
        print(f"❌ Kết luận: XÁC THỰC THẤT BẠI.")

if __name__ == "__main__":
    run_live_face_matching()
















