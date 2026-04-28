"""
Cấu hình chung cho toàn bộ pipeline eKYC CCCD.
Chỉnh sửa file này trước khi chạy bất kỳ script nào.
"""

import os

# ── Classes ──────────────────────────────────────────────────────────────────
CLASSES     = ["bottom_left", "bottom_right", "image_person", "top_left", "top_right"]
NUM_CLASSES = len(CLASSES)

# ── Model ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 320          # input resolution (320 hoặc 416)
BATCH_SIZE      = 8            # giảm xuống 4 nếu OOM
EPOCHS          = 100
LR              = 1e-3         # learning rate ban đầu
BACKBONE_PRESET = "yolo_v8_s_backbone_coco"  # xs/s/m/l

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_IMG_DIR   = os.path.join(DATA_DIR, "images", "train")
TRAIN_LBL_DIR   = os.path.join(DATA_DIR, "labels", "train")
VAL_IMG_DIR     = os.path.join(DATA_DIR, "images", "val")
VAL_LBL_DIR     = os.path.join(DATA_DIR, "labels", "val")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
TFLITE_DIR      = os.path.join(BASE_DIR, "tflite")

BEST_MODEL_PATH = os.path.join(CKPT_DIR, "best_model.keras")
