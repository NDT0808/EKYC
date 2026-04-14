## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/NDT0808/EKYC.git](https://github.com/NDT0808/EKYC.git)
cd EKYC

2. Set up a virtual environment (Recommended)

Bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate  

# On Windows:
venv\Scripts\activate     
3. Install dependencies

Bash
pip install -r requirements.txt
4. Install Tesseract OCR

Windows: Download the installer from UB-Mannheim/tesseract and install it. Update the TESSERACT_PATH in main_tesseract.py to your installation directory (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).

Linux (Ubuntu): ```bash
sudo apt install tesseract-ocr


5. Download Pre-trained Models
Due to size constraints, model weights are not hosted on this repository. Please download them from the following links and place them in the project root directory:

Download best.pt (YOLO Model) here * Download yolov8n.pt here * Download Liveness Model here https://www.google.com/search?q=%23https://www.google.com/search?q=%23 🚀 Usage

To run the main eKYC pipeline with webcam feed:

Bash
python main_tesseract.py
Note: Make sure your webcam is connected and the pre-trained models are correctly placed before executing the script.

🤝 Contact & Contribution
Created by NDT0808.

Feel free to open an issue or submit a pull request if you have suggestions for improving this pipeline!
