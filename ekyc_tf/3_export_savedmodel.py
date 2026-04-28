"""
Bước 3 — Export model đã train → SavedModel

model.export() tích hợp NMS decoder vào SavedModel — không cần
TF OD API exporter_main_v2.py (đã bị hỏng từ TF 2.16+).

Chạy:
    python 3_export_savedmodel.py
"""

import os

import keras
import tensorflow as tf

from config import BEST_MODEL_PATH, SAVED_MODEL_DIR


def main() -> None:
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model tại {BEST_MODEL_PATH}\n"
            "Hãy chạy 2_train.py trước."
        )

    print(f"Đang load model: {BEST_MODEL_PATH} ...")
    model = keras.models.load_model(BEST_MODEL_PATH)

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    print(f"Đang export SavedModel → {SAVED_MODEL_DIR} ...")

    # export() tích hợp prediction decoder (NMS) vào SavedModel
    model.export(SAVED_MODEL_DIR)

    total_bytes = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(SAVED_MODEL_DIR)
        for f in files
    )
    print(f"Kích thước SavedModel: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"\nBước tiếp theo: python 4_convert_tflite.py")


if __name__ == "__main__":
    main()
