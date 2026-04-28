"""
Bước 2 — Huấn luyện YOLOv8 với KerasCV

Pipeline thuần TensorFlow/Keras — không dùng TF OD API.
TF OD API bị hỏng từ TF 2.16+ do tf.estimator bị xóa.

Các fix so với notebook gốc:
  1. jit_compile=False  — XLA không hỗ trợ RaggedTensor
  2. Bỏ ReduceLROnPlateau — xung đột với CosineDecay schedule
  3. ragged_batch thay vì batch — giữ nguyên số box khác nhau mỗi ảnh

Chạy:
    python 2_train.py

Theo dõi với TensorBoard:
    tensorboard --logdir checkpoints/
"""

import glob
import os

import keras
import keras_cv
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

from config import (
    BACKBONE_PRESET, BATCH_SIZE, BEST_MODEL_PATH, CKPT_DIR,
    CLASSES, EPOCHS, IMAGE_SIZE, LR, NUM_CLASSES,
    TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR,
)

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


# ── tf.data pipeline ─────────────────────────────────────────────────────────

def _load_sample(img_path: str, lbl_dir: str) -> tuple:
    """Đọc 1 ảnh + nhãn YOLO → (image_arr, boxes_arr, classes_arr)."""
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, stem + ".txt")

    img = PILImage.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_arr = np.array(img, dtype=np.float32) / 255.0

    boxes, classes = [], []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                # YOLO (cx,cy,w,h) → rel_xyxy (x1,y1,x2,y2)
                x1 = max(0.0, cx - bw / 2)
                y1 = max(0.0, cy - bh / 2)
                x2 = min(1.0, cx + bw / 2)
                y2 = min(1.0, cy + bh / 2)
                boxes.append([x1, y1, x2, y2])
                classes.append(float(cls_id))

    if boxes:
        return img_arr, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.float32)
    return img_arr, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def build_dataset(img_dir: str, lbl_dir: str, shuffle: bool) -> tuple[tf.data.Dataset, int]:
    """Tạo tf.data.Dataset với ragged_batch cho KerasCV YOLOV8Detector."""
    img_paths = sorted(
        p for ext in IMG_EXTS for p in glob.glob(os.path.join(img_dir, ext))
    )
    if not img_paths:
        raise ValueError(f"Không có ảnh trong {img_dir}")

    def py_load(path_tensor):
        img, boxes, classes = _load_sample(
            path_tensor.numpy().decode("utf-8"), lbl_dir
        )
        return img, boxes, classes

    def tf_load(path):
        img, boxes, classes = tf.py_function(
            py_load, [path], [tf.float32, tf.float32, tf.float32]
        )
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        boxes.set_shape([None, 4])
        classes.set_shape([None])
        return img, {"boxes": boxes, "classes": classes}

    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    if shuffle:
        ds = ds.shuffle(len(img_paths), reshuffle_each_iteration=True)
    ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    # ragged_batch: giữ nguyên số box khác nhau mỗi ảnh, không cần padding
    ds = ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(img_paths)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> keras_cv.models.YOLOV8Detector:
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(BACKBONE_PRESET)
    model = keras_cv.models.YOLOV8Detector(
        num_classes         = NUM_CLASSES,
        bounding_box_format = "rel_xyxy",
        backbone            = backbone,
        fpn_depth           = 1,
    )

    total_steps = EPOCHS * 1  # will be multiplied after dataset size is known
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = LR,
        decay_steps           = EPOCHS * 50,  # placeholder, overridden in main()
        alpha                 = LR * 0.01,
    )

    # jit_compile=False — XLA không hỗ trợ RaggedTensor (fix #1)
    # ReduceLROnPlateau bị bỏ — xung đột với CosineDecay schedule (fix #2)
    model.compile(
        optimizer           = keras.optimizers.SGD(lr_schedule, momentum=0.9),
        box_loss            = "ciou",
        classification_loss = "binary_crossentropy",
        jit_compile         = False,
    )
    return model, lr_schedule


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"TensorFlow : {tf.__version__}")
    print(f"Keras      : {keras.__version__}")
    print(f"KerasCV    : {keras_cv.__version__}")
    print(f"GPU        : {tf.config.list_physical_devices('GPU')}\n")

    for d in [CKPT_DIR]:
        os.makedirs(d, exist_ok=True)

    print("Đang xây dựng dataset...")
    train_ds, n_train = build_dataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, shuffle=True)
    val_ds,   n_val   = build_dataset(VAL_IMG_DIR,   VAL_LBL_DIR,   shuffle=False)
    steps_per_epoch   = max(1, n_train // BATCH_SIZE)
    print(f"  Train : {n_train} ảnh | {steps_per_epoch} steps/epoch")
    print(f"  Val   : {n_val} ảnh\n")

    print(f"Đang tải backbone: {BACKBONE_PRESET} ...")
    model, lr_schedule = build_model()

    # Cập nhật decay_steps thực tế sau khi biết kích thước dataset
    lr_schedule.decay_steps = EPOCHS * steps_per_epoch

    # Khởi tạo weights
    for imgs, bboxes in train_ds.take(1):
        _ = model(imgs, training=False)

    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Tham số trainable: {trainable:,}\n")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath       = BEST_MODEL_PATH,
            monitor        = "val_loss",
            save_best_only = True,
            verbose        = 1,
        ),
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 15,
            restore_best_weights = True,
            verbose              = 1,
        ),
        keras.callbacks.TensorBoard(log_dir=CKPT_DIR, histogram_freq=0),
    ]

    print(f"Bắt đầu train {EPOCHS} epochs...")
    print(f"Best model sẽ lưu tại: {BEST_MODEL_PATH}\n")

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks,
        verbose         = 1,
    )

    best_val = min(history.history["val_loss"])
    best_ep  = history.history["val_loss"].index(best_val) + 1
    print(f"\nBest val_loss: {best_val:.4f} (epoch {best_ep})")
    print(f"Model đã lưu: {BEST_MODEL_PATH}")
    print("\nBước tiếp theo: python 3_export_savedmodel.py")


if __name__ == "__main__":
    main()
