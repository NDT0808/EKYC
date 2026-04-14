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
* [Download Liveness Model here](#)
