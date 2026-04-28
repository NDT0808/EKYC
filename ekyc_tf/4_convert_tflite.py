"""
Bước 4 — Chuyển đổi SavedModel → TFLite (fp32 / fp16 / int8)

Tại sao cần SELECT_TF_OPS:
    KerasCV YOLOV8Detector dùng tf.image.combined_non_max_suppression (NMS).
    Op này không có trong TFLITE_BUILTINS chuẩn → phải bật Flex Delegate.
    Trên Android cần thêm: tensorflow-lite-select-tf-ops

Fix so với notebook gốc:
    - Phát hiện signature input key tự động (tên tensor do Keras tự sinh,
      thay đổi theo phiên bản — không được hard-code).

Kết quả:
    tflite/model_fp32.tflite  ~48 MB  baseline
    tflite/model_fp16.tflite  ~24 MB  khuyến nghị (cân bằng tốc độ/độ chính xác)
    tflite/model_int8.tflite  ~12 MB  nhỏ nhất, dùng cho thiết bị low-end

Chạy:
    python 4_convert_tflite.py
"""

import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

from config import IMAGE_SIZE, SAVED_MODEL_DIR, TFLITE_DIR, TRAIN_IMG_DIR

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")
CALIB_MAX = 100


def _get_concrete_function(saved_model_dir: str):
    """Load SavedModel và wrap thành concrete function với fixed input shape."""
    loaded = tf.saved_model.load(saved_model_dir)
    serving_fn = loaded.signatures["serve"]

    # Fix: phát hiện tên input key tự động thay vì hard-code
    input_key = list(serving_fn.structured_input_signature[1].keys())[0]
    print(f"  Signature input key: {input_key}")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32, name="input_tensor")
    ])
    def fixed_fn(input_tensor):
        return serving_fn(**{input_key: input_tensor})

    return fixed_fn.get_concrete_function()


def _representative_dataset():
    paths = sorted(
        p for ext in IMG_EXTS for p in glob.glob(os.path.join(TRAIN_IMG_DIR, ext))
    )[:CALIB_MAX]
    for path in paths:
        img = PILImage.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32)[np.newaxis] / 255.0
        yield [arr]


def convert_fp32(concrete_fn, output_path: str) -> None:
    print("\n[1/3] Converting → fp32 ...")
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    _save(conv.convert(), output_path)


def convert_fp16(concrete_fn, output_path: str) -> None:
    print("\n[2/3] Converting → fp16 ...")
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    _save(conv.convert(), output_path)


def convert_int8(concrete_fn, output_path: str) -> None:
    print(f"\n[3/3] Converting → int8 (hiệu chỉnh với {CALIB_MAX} ảnh) ...")
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = _representative_dataset
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    _save(conv.convert(), output_path)


def _save(model_bytes: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(model_bytes)
    print(f"  Saved: {path}  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def main() -> None:
    if not os.path.isdir(SAVED_MODEL_DIR):
        raise FileNotFoundError(
            f"SavedModel không tìm thấy tại {SAVED_MODEL_DIR}\n"
            "Hãy chạy 3_export_savedmodel.py trước."
        )

    os.makedirs(TFLITE_DIR, exist_ok=True)

    print("Đang load SavedModel và tạo concrete function ...")
    concrete_fn = _get_concrete_function(SAVED_MODEL_DIR)

    fp32_path = os.path.join(TFLITE_DIR, "model_fp32.tflite")
    fp16_path = os.path.join(TFLITE_DIR, "model_fp16.tflite")
    int8_path = os.path.join(TFLITE_DIR, "model_int8.tflite")

    convert_fp32(concrete_fn, fp32_path)
    convert_fp16(concrete_fn, fp16_path)
    convert_int8(concrete_fn, int8_path)

    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Kích thước':>12}")
    print("-" * 50)
    for label, path in [("fp32 (baseline)", fp32_path), ("fp16 (khuyến nghị)", fp16_path), ("int8 (nhỏ nhất)", int8_path)]:
        size = os.path.getsize(path) / 1024 / 1024
        print(f"{label:<20} {size:>10.1f} MB")
    print("=" * 50)
    print(f"\nBước tiếp theo: python 5_inference_tflite.py --image <ảnh> --model {fp16_path}")


if __name__ == "__main__":
    main()
