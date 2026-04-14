# ==============================================================================
# HỆ THỐNG EKYC & TESSERACT OCR (V12 - FIX PATH & CHECK DATA)
# ==============================================================================

import cv2
import torch
import numpy as np
import threading
import time
import os
import re
import difflib
from collections import deque
from PIL import Image

# --- THƯ VIỆN AI ---
from ultralytics import YOLO
from deepface import DeepFace
from transformers import AutoImageProcessor, AutoModelForImageClassification
import imutils

# --- THƯ VIỆN TESSERACT ---
import pytesseract

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CHUẨN HÓA)
# ==========================================
# Dùng đường dẫn chuẩn, không dấu ngoặc kép
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_PATH  = r'C:\Program Files\Tesseract-OCR\tessdata'

# Cấu hình ngay lập tức
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

# Model khác
REFERENCE_IMAGE_PATH = "datasets/Trong.jpg"
YOLO_MODEL_PATH = "best.pt"
LIVENESS_MODEL_NAME = "nguyenkhoa/vit_Liveness_detection_v1.0"
DEEPFACE_MODEL_NAME = "ArcFace"

WEBCAM_ID = 1
FRAME_WIDTH = 640
FRAME_SKIP = 5
DEQUE_SIZE = 10
STABLE_THRESHOLD = 0.7
DISTANCE_THRESHOLD = 0.55
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================================
# 2. BIẾN TOÀN CỤC
# ==========================================
shared_frame = None
video_lock = threading.Lock()
stop_event = threading.Event()

ui_state = {
    "status": "WAITING",
    "box": None,
    "ocr_data": {},
    "ocr_status": ""
}

# ==========================================
# 3. CLASS TESSERACT WORKER
# ==========================================
class TesseractWorker:
    def __init__(self):
        print("⏳ [OCR] Đang kiểm tra hệ thống Tesseract...")
        
        # 1. Kiểm tra file .exe
        if not os.path.exists(TESSERACT_PATH):
            print(f"❌ [LỖI] Không tìm thấy file chạy tại: {TESSERACT_PATH}")
            return
            
        # 2. Kiểm tra file ngôn ngữ vie.traineddata
        vie_path = os.path.join(TESSDATA_PATH, 'vie.traineddata')
        if not os.path.exists(vie_path):
            print(f"❌ [LỖI] Không tìm thấy file ngôn ngữ tại: {vie_path}")
            print("👉 Vui lòng tải 'vie.traineddata' bỏ vào thư mục 'tessdata'.")
            return
        
        # 3. Kiểm tra dung lượng file (để tránh trường hợp tải nhầm file lỗi 0KB)
        file_size_kb = os.path.getsize(vie_path) / 1024
        print(f"   -> Tìm thấy 'vie.traineddata': {file_size_kb:.1f} KB")
        if file_size_kb < 100:
            print("⚠️ [CẢNH BÁO] File ngôn ngữ quá nhẹ (<100KB). Có thể bạn tải lỗi!")
        
        print("✅ [OCR] Tesseract Sẵn Sàng!")
        self.is_running = False

    def preprocess_image(self, img):
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. Resize x2.5
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        # 3. Adaptive Threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        return binary

    def clean_text(self, text):
        return text.strip().replace("\n\n", "\n")

    def fuzzy_check(self, keyword, text, threshold=0.6):
        s = difflib.SequenceMatcher(None, keyword.lower(), text.lower())
        return s.ratio() > threshold

    def parse_text(self, full_text):
        info = {}
        lines = full_text.split('\n')
        lines = [l.strip() for l in lines if len(l.strip()) > 2]
        
        print("\n--- RAW TESSERACT ---")
        for l in lines: print(f"  {l}")
        print("---------------------")

        # Regex tìm dữ liệu
        id_match = re.search(r'\d{12}', full_text)
        if id_match: info['So CCCD'] = id_match.group(0)

        dates = re.findall(r'\d{2}/\d{2}/\d{4}', full_text)
        if len(dates) >= 1: info['Ngay sinh'] = dates[0]
        if len(dates) >= 2: info['Co gia tri den'] = dates[-1]

        if "Nam" in full_text: info['Gioi tinh'] = "Nam"
        elif "Nữ" in full_text: info['Gioi tinh'] = "Nữ"

        for i, line in enumerate(lines):
            # HỌ TÊN
            if self.fuzzy_check("Họ và tên", line) or "Full name" in line:
                if i + 1 < len(lines):
                    pot_name = lines[i+1]
                    if pot_name.isupper() and len(pot_name) > 3 and not any(c.isdigit() for c in pot_name):
                        info['Ho va ten'] = pot_name
            
            # QUÊ QUÁN
            if self.fuzzy_check("Quê quán", line) or "origin" in line:
                hometown = []
                for k in range(1, 3):
                    if i + k < len(lines):
                        next_l = lines[i+k]
                        if self.fuzzy_check("Nơi thường trú", next_l): break
                        hometown.append(next_l)
                if hometown: info['Que quan'] = ", ".join(hometown)

            # THƯỜNG TRÚ
            if self.fuzzy_check("Nơi thường trú", line) or "residence" in line:
                addr = []
                for k in range(1, 4):
                    if i + k < len(lines):
                        next_l = lines[i+k]
                        if self.fuzzy_check("Có giá trị", next_l): break
                        addr.append(next_l)
                if addr: info['Noi thuong tru'] = ", ".join(addr)

        if 'Ho va ten' not in info:
            for line in lines:
                if line.isupper() and len(line) > 10 and "CỘNG HÒA" not in line and "ĐỘC LẬP" not in line:
                     info['Ho va ten'] = line
                     break
        return info

    def scan(self, frame):
        if self.is_running: return
        self.is_running = True
        ui_state["ocr_status"] = "Dang quet..."
        ui_state["ocr_data"] = {}

        try:
            print("\n>>> BẮT ĐẦU QUÉT TESSERACT...")
            proc_img = self.preprocess_image(frame)
            
            # QUAN TRỌNG: Chỉ dùng biến môi trường, không truyền --tessdata-dir để tránh lỗi ngoặc kép
            custom_config = r'--psm 6'
            
            full_text = pytesseract.image_to_string(proc_img, lang='vie', config=custom_config)
            
            final_data = self.parse_text(full_text)
            ui_state["ocr_data"] = final_data
            
            if final_data:
                ui_state["ocr_status"] = "XONG!"
                print("\n✅ KẾT QUẢ:")
                for k, v in final_data.items():
                    print(f"   🔹 {k}: {v}")
            else:
                ui_state["ocr_status"] = "KHONG THAY TT"

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            ui_state["ocr_status"] = "Loi Runtime"
        finally:
            self.is_running = False

# ==========================================
# 4. HÀM CHÍNH
# ==========================================
def run_system():
    global shared_frame

    # Load EKYC
    try:
        print("⏳ [SYSTEM] Đang tải EKYC Models...")
        liveness_proc = AutoImageProcessor.from_pretrained(LIVENESS_MODEL_NAME)
        liveness_model = AutoModelForImageClassification.from_pretrained(LIVENESS_MODEL_NAME)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        if not os.path.exists(REFERENCE_IMAGE_PATH): raise ValueError("Thiếu ảnh gốc!")
        ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
        
        # Crop Face logic
        res = yolo_model(ref_img, verbose=False)
        ref_crop = ref_img
        for r in res:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                ref_crop = ref_img[y1:y2, x1:x2]
                try: ref_crop = imutils.rotate_bound(ref_crop, -90)
                except: pass
                break
            break
        
        DeepFace.represent(ref_crop, model_name=DEEPFACE_MODEL_NAME, enforce_detection=False)
        print("✅ [SYSTEM] EKYC Models OK!")
    except Exception as e:
        print(f"❌ {e}")
        return

    ocr_worker = TesseractWorker()
    person_name = os.path.splitext(os.path.basename(REFERENCE_IMAGE_PATH))[0].upper()

    cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(WEBCAM_ID)

    print("\n🚀 HỆ THỐNG ĐANG CHẠY...")
    print("👉 Bấm 'C' để quét")
    print("👉 Bấm 'Q' để thoát\n")

    frame_cnt = 0
    res_queue = deque(maxlen=DEQUE_SIZE)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        frame_small = imutils.resize(frame, width=FRAME_WIDTH)
        frame_display = frame_small.copy()
        frame_cnt += 1

        if frame_cnt % FRAME_SKIP == 0:
            try:
                img_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
                inputs = liveness_proc(images=img_pil, return_tensors="pt")
                with torch.no_grad():
                    out = liveness_model(**inputs)
                lbl = liveness_model.config.id2label[out.logits.argmax(-1).item()]
                
                stt = "FAKE"
                box = None
                
                if lbl.lower() == 'live':
                    ver = DeepFace.verify(frame_small, ref_crop, model_name=DEEPFACE_MODEL_NAME, 
                                        detector_backend="opencv", distance_metric='cosine', 
                                        enforce_detection=False, threshold=DISTANCE_THRESHOLD)
                    if ver['verified']: stt = "MATCH"
                    else: stt = "NO_MATCH"
                    if 'facial_areas' in ver:
                        fa = ver['facial_areas']['img1']
                        box = (fa['x'], fa['y'], fa['w'], fa['h'])
                
                res_queue.append(stt)
                ui_state["box"] = box
            except: pass

        if len(res_queue) >= DEQUE_SIZE:
            if res_queue.count("MATCH") / len(res_queue) >= STABLE_THRESHOLD: ui_state["status"] = "MATCH"
            elif res_queue.count("FAKE") / len(res_queue) >= STABLE_THRESHOLD: ui_state["status"] = "FAKE"
            else: ui_state["status"] = "NO_MATCH"

        status = ui_state["status"]
        color, txt = (0,0,255), "Unknown"
        if status == "MATCH": color, txt = (0,255,0), f"HELLO: {person_name}"
        elif status == "NO_MATCH": color, txt = (0,255,255), "NO MATCH"

        if ui_state["box"]:
            x,y,w,h = ui_state["box"]
            cv2.rectangle(frame_display, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame_display, txt, (x, y-10), FONT, 0.8, color, 2)
        
        if ui_state["ocr_status"]:
            cv2.putText(frame_display, f"OCR: {ui_state['ocr_status']}", (10, frame_display.shape[0]-10), FONT, 0.6, (255,255,255), 1)
        
        y_off = 50
        for k,v in ui_state["ocr_data"].items():
            cv2.putText(frame_display, f"{k}: {v[:25]}", (frame_display.shape[1]-300, y_off), FONT, 0.5, (0,255,0), 1)
            y_off += 20

        cv2.imshow("EKYC & TESSERACT V12", frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            threading.Thread(target=ocr_worker.scan, args=(frame.copy(),)).start()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()