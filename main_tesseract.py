# ==============================================================================
# HỆ THỐNG TESSERACT OCR (CHỈ ĐỌC THÔNG TIN TỪ ẢNH)
# ==============================================================================

import cv2
import os
import re
import difflib
import imutils
import pytesseract

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CHUẨN HÓA)
# ==========================================
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_PATH  = r'C:\Program Files\Tesseract-OCR\tessdata'

# Cấu hình ngay lập tức
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

# ==========================================
# 2. CLASS TESSERACT WORKER
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

        def clean_noise(text, is_name=False):
            # Các ký tự tiếng Việt hợp lệ
            vi_chars = 'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỮỰỲỴÝỶỸữựỳỵỷỹ'
            
            if is_name:
                # Nếu là tên: CHỈ giữ lại chữ cái và khoảng trắng (bỏ dấu chấm, phẩy, số)
                cleaned = re.sub(rf'[^a-zA-Z{vi_chars}\s]', '', text)
            else:
                # Nếu là địa chỉ: giữ lại thêm số, dấu phẩy, dấu chấm, gạch ngang
                cleaned = re.sub(rf'[^a-zA-Z0-9{vi_chars}\s\,\.\-]', '', text)
                
            # Xóa khoảng trắng thừa và xóa các dấu câu bị thừa ở đầu/cuối chuỗi
            return re.sub(r'\s+', ' ', cleaned).strip(" .,-")

        for i, line in enumerate(lines):
            # HỌ TÊN
            if self.fuzzy_check("Họ và tên", line) or "Full name" in line:
                if i + 1 < len(lines):
                    pot_name = lines[i+1]
                    clean_name = clean_noise(pot_name, is_name=True)
                    # Nếu là chữ in hoa (hoặc phần lớn là chữ in hoa) và độ dài hợp lý
                    if len(clean_name) > 3:
                        info['Ho va ten'] = clean_name
            
            # QUÊ QUÁN
            if self.fuzzy_check("Quê quán", line) or "origin" in line:
                hometown = []
                for k in range(1, 3):
                    if i + k < len(lines):
                        next_l = lines[i+k]
                        if self.fuzzy_check("Nơi thường trú", next_l): break
                        cleaned_line = clean_noise(next_l, is_name=False)
                        if len(cleaned_line) > 2: hometown.append(cleaned_line)
                if hometown: info['Que quan'] = ", ".join(hometown)

            # THƯỜNG TRÚ
            if self.fuzzy_check("Nơi thường trú", line) or "residence" in line:
                addr = []
                for k in range(1, 4):
                    if i + k < len(lines):
                        next_l = lines[i+k]
                        if self.fuzzy_check("Có giá trị", next_l): break
                        cleaned_line = clean_noise(next_l, is_name=False)
                        if len(cleaned_line) > 2: addr.append(cleaned_line)
                if addr: info['Noi thuong tru'] = ", ".join(addr)

        if 'Ho va ten' not in info:
            for line in lines:
                clean_name = clean_noise(line, is_name=True)
                if clean_name.isupper() and len(clean_name) > 10 and "CỘNG HÒA" not in clean_name and "ĐỘC LẬP" not in clean_name:
                     info['Ho va ten'] = clean_name
                     break
        return info

    def scan(self, frame):
        try:
            print("\n>>> BẮT ĐẦU QUÉT TESSERACT...")
            proc_img = self.preprocess_image(frame)
            
            custom_config = r'--psm 6'
            
            full_text = pytesseract.image_to_string(proc_img, lang='vie', config=custom_config)
            
            final_data = self.parse_text(full_text)
            
            if final_data:
                print("\n✅ KẾT QUẢ TRÍCH XUẤT:")
                for k, v in final_data.items():
                    print(f"   🔹 {k}: {v}")
            else:
                print("\n⚠️ KHÔNG TÌM THẤY THÔNG TIN.")
                
            return final_data

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None

# ==========================================
# 3. HÀM CHÍNH (CHỈ ĐỌC ẢNH)
# ==========================================
def run_system():
    # Đường dẫn tới ảnh CCCD cần đọc
    test_image_path = r"idCCCD-1\train\images\0-0_jpg.rf.868289615ce0c9901b141fd721265cd5.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Không tìm thấy ảnh test tại: {test_image_path}")
        return
        
    print(f"🔍 Đang tiến hành đọc OCR trên ảnh: {test_image_path}")
    
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(test_image_path)
    if img is None:
        print("❌ Lỗi khi đọc ảnh (có thể đường dẫn chứa ký tự tiếng Việt không hỗ trợ).")
        return
        
    # Khởi tạo Worker và chạy OCR
    ocr_worker = TesseractWorker()
    ocr_worker.scan(img)
    
    # Hiển thị ảnh gốc
    cv2.imshow("Input Image", imutils.resize(img, width=600))
    print("\n👉 Bấm phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
