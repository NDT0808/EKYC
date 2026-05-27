import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from deepface import DeepFace
import imutils
import os
from ultralytics import YOLO

# --- CẤU HÌNH ---
REFERENCE_IMAGE_PATH = "datasets/trong.png"
YOLO_MODEL_PATH = "best.pt"
PERSON_NAME = os.path.splitext(os.path.basename(REFERENCE_IMAGE_PATH))[0].upper()

DETECTOR_BACKEND = "opencv"
LIVENESS_MODEL_NAME = "nguyenkhoa/vit_Liveness_detection_v1.0"
DEEPFACE_MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.55

FRAME_WIDTH = 640
FRAME_SKIP = 3  # Giảm xuống 3 để web app phản hồi nhanh hơn
DEQUE_SIZE = 10
STABLE_THRESHOLD = 0.7 

# --- TẢI MODEL TRƯỚC KHI CHẠY WEB ---
print("[INFO] Đang tải các model...")
liveness_processor = AutoImageProcessor.from_pretrained(LIVENESS_MODEL_NAME)
liveness_model = AutoModelForImageClassification.from_pretrained(LIVENESS_MODEL_NAME)
yolo_model = YOLO(YOLO_MODEL_PATH)

print(f"[INFO] Đang xử lý ảnh tham chiếu: {REFERENCE_IMAGE_PATH}")
reference_img_full = cv2.imread(REFERENCE_IMAGE_PATH)
if reference_img_full is None:
    print(f"[WARNING] Không thể đọc ảnh tham chiếu: {REFERENCE_IMAGE_PATH}")
    reference_face_crop = None
else:
    results = yolo_model(reference_img_full)
    reference_face_crop = None
    for r in results:
        for box in r.boxes:
            if yolo_model.names[int(box.cls[0])] == 'image_person':
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                face_crop_original = reference_img_full[y1:y2, x1:x2]
                reference_face_crop = imutils.rotate_bound(face_crop_original, -90)
                break
        if reference_face_crop is not None: break

    if reference_face_crop is not None:
        _ = DeepFace.represent(img_path=reference_face_crop, model_name=DEEPFACE_MODEL_NAME, enforce_detection=False)
        print("[SUCCESS] Khởi tạo xong!")

# UI Text
UI_TEXT_MATCH = f"DA XAC THUC: {PERSON_NAME}"
UI_TEXT_NO_MATCH = "KHONG TRUNG KHOP"
UI_TEXT_PROCESSING = "DANG PHAN TICH..."
UI_TEXT_PROMPT = "DUA KHUON MAT VAO KHUNG HINH"

def process_frame(frame, frame_number, results_window_list, last_known_location, stable_result):
    if frame is None or reference_face_crop is None:
        return frame, frame_number, results_window_list, last_known_location, stable_result
        
    # frame từ Gradio có dạng RGB numpy array
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = imutils.resize(frame_bgr, width=FRAME_WIDTH)
    frame_display = frame_bgr.copy()
    
    frame_number += 1
    
    # Chỉ gọi AI mỗi vài frame để tránh lag trình duyệt
    if frame_number % FRAME_SKIP == 0:
        try:
            # 1. LIVENESS CHECK
            image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            inputs = liveness_processor(images=image_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = liveness_model(**inputs)
            liveness_label = liveness_model.config.id2label[outputs.logits.argmax(-1).item()]
            
            current_frame_result = "SKIPPED_AS_FAKE"
            face_location = None
            
            if liveness_label.lower() == 'live':
                # 2. FACE MATCHING
                verification_result = DeepFace.verify(
                    img1_path=frame_bgr, img2_path=reference_face_crop,
                    model_name=DEEPFACE_MODEL_NAME, detector_backend=DETECTOR_BACKEND,
                    distance_metric='cosine', enforce_detection=False, threshold=DISTANCE_THRESHOLD
                )
                
                is_verified = verification_result['verified']
                if 'facial_areas' in verification_result and 'img1' in verification_result['facial_areas']:
                    face_data = verification_result['facial_areas']['img1']
                    face_location = (face_data['x'], face_data['y'], face_data['w'], face_data['h'])
                
                current_frame_result = "MATCH" if is_verified else "NO_MATCH"
            
            last_known_location = face_location
            results_window_list.append(current_frame_result)
            if len(results_window_list) > DEQUE_SIZE:
                results_window_list.pop(0)
                
        except Exception as e:
            last_known_location = None
            results_window_list.append("NO_FACE_DETECTED")
            if len(results_window_list) > DEQUE_SIZE:
                results_window_list.pop(0)

    # Cập nhật kết quả ổn định
    if len(results_window_list) >= DEQUE_SIZE:
        match_count = results_window_list.count("MATCH")
        if (match_count / len(results_window_list)) >= STABLE_THRESHOLD:
            stable_result = "STABLE MATCH"
        else:
            stable_result = "NOT MATCH"

    # Vẽ UI lên frame
    color = (255, 255, 255)
    display_text = UI_TEXT_PROMPT
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

    if last_known_location:
        x, y, w, h = last_known_location
        cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_display, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = (frame_display.shape[1] - text_width) // 2
        text_y = (frame_display.shape[0] + text_height) // 2
        cv2.putText(frame_display, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert lại RGB để trả về cho Gradio
    return cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), frame_number, results_window_list, last_known_location, stable_result

# --- TẠO GIAO DIỆN WEB VỚI GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Hệ Thống EKYC Nhận Diện Khuôn Mặt (Webcam Real-time)")
    gr.Markdown("Đưa khuôn mặt vào camera để xác thực.")
    
    with gr.Row():
        # Input camera
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Camera của bạn")
        # Output xử lý
        image_output = gr.Image(label="Kết quả xử lý AI")
    
    # State variables để lưu trữ dữ liệu giữa các frame liên tiếp
    state_frame_number = gr.State(0)
    state_results_window = gr.State([])
    state_last_known_location = gr.State(None)
    state_stable_result = gr.State("ANALYZING")
    
    # Hàm stream sẽ chạy liên tục khi có frame từ webcam
    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, state_frame_number, state_results_window, state_last_known_location, state_stable_result],
        outputs=[image_output, state_frame_number, state_results_window, state_last_known_location, state_stable_result],
        time_limit=30, # Thời gian stream tối đa mỗi lần (có thể tự connect lại)
        stream_every=0.1 # Delay nhẹ để tránh quá tải
    )

if __name__ == "__main__":
    # share=True sẽ tạo một đường link public (ví dụ: https://xyz.gradio.live)
    # giúp bạn có thể truy cập bằng điện thoại hoặc máy tính từ xa và mở webcam
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
