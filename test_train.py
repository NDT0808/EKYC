from ultralytics import YOLO
from roboflow import Roboflow
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CẤU HÌNH ---
# Cảnh báo: Batch 128 rất lớn cho RTX 3050 (VRAM 4GB/6GB). 
# Nếu bị lỗi "Out of Memory", hãy giảm xuống 16 hoặc 32.
BATCH_SIZE = 16  
WORKERS = 0      # Trên Windows nên để 0 để tránh lỗi đa luồng

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"✅ Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Cảnh báo: PyTorch không tìm thấy GPU. Sẽ chạy bằng CPU.")

# --- HÀM VẼ BIỂU ĐỒ TÙY CHỈNH ---
def plot_custom_results(save_dir):
    """
    Đọc file results.csv và vẽ biểu đồ mAP50 (Độ chính xác)
    """
    csv_path = os.path.join(save_dir, 'results.csv')
    
    if os.path.exists(csv_path):
        # Đọc dữ liệu
        df = pd.read_csv(csv_path)
        # Xóa khoảng trắng trong tên cột
        df.columns = [x.strip() for x in df.columns]
        
        # Lấy dữ liệu
        epochs = df['epoch']
        map50 = df['metrics/mAP50(B)'] # Độ chính xác (mAP@0.5)
        loss = df['train/box_loss']    # Loss
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        
        # Biểu đồ mAP50
        plt.subplot(1, 2, 1)
        plt.plot(epochs, map50, marker='o', color='b', label='mAP50 (Accuracy)')
        plt.title('Độ chính xác qua từng Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP50')
        plt.grid(True)
        plt.legend()
        
        # Biểu đồ Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, marker='x', color='r', label='Box Loss')
        plt.title('Mức độ lỗi (Loss) qua từng Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Lưu ảnh
        plot_path = os.path.join(save_dir, 'custom_accuracy_chart.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"✅ Đã vẽ xong biểu đồ tùy chỉnh tại: {plot_path}")
        plt.show() # Hiển thị lên màn hình nếu chạy local
    else:
        print("❌ Không tìm thấy file results.csv để vẽ biểu đồ.")

# --- CALLBACK ĐỂ IN ACCURACY RA MÀN HÌNH ---
def on_train_epoch_end(trainer):
    """Hàm này chạy mỗi khi xong 1 epoch"""
    metrics = trainer.metrics
    epoch = trainer.epoch + 1
    # Lấy mAP50 (Độ chính xác thông dụng nhất)
    map50 = metrics.get("metrics/mAP50(B)", 0)
    print(f"📊 [Epoch {epoch}] Accuracy (mAP50): {map50:.4f}")

if __name__ == '__main__':
    # 1. Tải dataset
    rf = Roboflow(api_key="85uNlO01ymcOS0rvA6r6") # Hãy thay bằng API Key thật của bạn
    project = rf.workspace("tris-nexkg").project("idcccd-1lgnb-mqob5")
    version = project.version(1)
    dataset = version.download("yolov8")

    # 2. Load model
    model = YOLO('yolo11n.pt')
# Đăng ký callback để in Accuracy mỗi epoch
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 3. Train model
    print("🚀 Bắt đầu train...")
    
    results = model.train(
        data=f"{dataset.location}/data.yaml", 
        
        # Tham số training
        epochs=1,            
        imgsz=640,             
        batch=BATCH_SIZE,      
        optimizer='AdamW',     
        lr0=0.0001,            
        
        # Cấu hình hệ thống
        device=0,              
        workers=WORKERS,             
        project='runs/train', 
        name='yolo11n_custom' 
    )
    
    # 4. Vẽ biểu đồ sau khi train xong
    print("🎨 Đang vẽ biểu đồ Accuracy...")
    plot_custom_results(results.save_dir)