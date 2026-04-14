# 🛡️ eKYC System (Electronic Know Your Customer)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow?style=for-the-badge)
![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-blueviolet?style=for-the-badge)

An automated identity verification pipeline integrating robust Face Liveness Detection and Optical Character Recognition (OCR). Designed to ensure secure, spoof-proof, and efficient user authentication processes.

## 🌟 Key Features

* **Liveness Detection:** Identifies whether the face presented to the camera is a real, live person or a spoof attack (e.g., printed photo, screen replay) using Vision Transformer (ViT) / CNN models.
* **Face Verification:** Extracts and compares facial features using **ArcFace** to match the user's live face with the ID card portrait.
* **Information Extraction (OCR):** Accurately extracts text from identity documents using **Tesseract OCR**.
* **Object Detection:** Utilizes **YOLO** for detecting ID cards and face cropping within the frame.

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Computer Vision:** OpenCV
* **Deep Learning Frameworks:** PyTorch, Ultralytics (YOLO)
* **OCR Engine:** Tesseract
* **Face Recognition:** DeepFace (ArcFace model)

## 📁 Project Structure

```text
EKYC/
├── datasets/                 # Sample images for testing (Not tracked by Git)
├── idCCCD-1/                 # ID card dataset/processing 
├── runs/                     # YOLO training logs and weights
├── Liveness_detection.py     # Core liveness detection module
├── main_tesseract.py         # Main pipeline integrating OCR and Liveness
├── test_train.py             # Script for model evaluation
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


⚙️ Installation & Setup
1. Clone the repository

Bash
git clone [https://github.com/NDT0808/EKYC.git](https://github.com/NDT0808/EKYC.git)
cd EKYC
2. Set up a virtual environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
3. Install dependencies

Bash
pip install -r requirements.txt
4. Install Tesseract OCR

Windows: Download the installer from UB-Mannheim/tesseract and install it. Update the TESSERACT_PATH in main_tesseract.py to your installation directory (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).

Linux (Ubuntu): sudo apt install tesseract-ocr

5. Download Pre-trained Models
Due to size constraints, model weights are not hosted on this repository. Please download them from the following links and place them in the project root directory:

Download best.pt (YOLO Model) here (<- Replace with your Google Drive link)

Download Liveness Model here (<- Replace with your Google Drive link)

🚀 Usage
To run the main eKYC pipeline with webcam feed:

Bash
python main_tesseract.py
Make sure your webcam is connected and the pre-trained models are correctly placed before executing the script.

🤝 Contact & Contribution
Created by NDT0808.
Feel free to open an issue or submit a pull request if you have suggestions for improving this pipeline!


**Một vài chỗ bạn cần tùy chỉnh lại sau khi dán:**
1.  **Chỗ `[Download best.pt... here](#)`:** Bạn tải file `best.pt` của bạn lên Google Drive, lấy link chia sẻ công khai và thay thế vào dấu `#`.
2.  Kiểm tra lại phần **Project Structure** xem đã mô tả đúng ý nghĩa các thư mục
