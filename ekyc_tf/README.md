# eKYC CCCD — YOLOv8 Detection với KerasCV + TensorFlow

Huấn luyện mô hình nhận diện 5 vùng đặc trưng trên **Căn Cước Công Dân (CCCD)** bằng **KerasCV YOLOv8**, xuất ra **TFLite** để triển khai mobile (Android/iOS) trong hệ thống eKYC.

> Pipeline này thay thế hoàn toàn TF Object Detection API — `tf.estimator` đã bị xóa khỏi TF 2.16+, khiến TF OD API không còn hoạt động.

---

## Lớp nhận diện

| ID | Tên | Mô tả |
|:--:|-----|-------|
| 0 | `bottom_left` | Góc dưới trái CCCD |
| 1 | `bottom_right` | Góc dưới phải CCCD |
| 2 | `image_person` | Vùng ảnh chân dung |
| 3 | `top_left` | Góc trên trái CCCD |
| 4 | `top_right` | Góc trên phải CCCD |

---

## Yêu Cầu

- Python 3.10+
- TensorFlow ≥ 2.16
- GPU khuyến nghị (Google Colab T4 hoặc tốt hơn)

```bash
pip install -r requirements.txt
```

---

## Cấu Trúc Thư Mục

```
ekyc_tf/
├── config.py                    # Cấu hình chung — chỉnh tại đây
├── 1_verify_dataset.py          # Kiểm tra dataset
├── 2_train.py                   # Huấn luyện mô hình
├── 3_export_savedmodel.py       # Export → SavedModel
├── 4_convert_tflite.py          # Chuyển đổi → TFLite
├── 5_inference_tflite.py        # Inference từ file TFLite
├── requirements.txt
├── BAO_CAO_EKYC_CCCD.md         # Báo cáo kỹ thuật chi tiết
│
├── data/                        # Dataset (tạo thủ công)
│   ├── images/
│   │   ├── train/  *.jpg|png
│   │   └── val/    *.jpg|png
│   └── labels/
│       ├── train/  *.txt
│       └── val/    *.txt
│
├── checkpoints/                 # Sinh ra sau khi train
│   └── best_model.keras
├── saved_model/                 # Sinh ra sau bước 3
└── tflite/                      # Sinh ra sau bước 4
    ├── model_fp32.tflite
    ├── model_fp16.tflite
    └── model_int8.tflite
```

### Định dạng nhãn YOLO (mỗi dòng trong file .txt)

```
<class_id> <cx> <cy> <width> <height>
```

Tất cả giá trị tọa độ đã normalized về `[0, 1]`. Ví dụ:

```
3 0.1797 0.1461 0.0516 0.0750
4 0.8555 0.1406 0.0719 0.0625
```

---

## Hướng Dẫn Sử Dụng

### Bước 0 — Cấu hình

Mở [config.py](config.py) và điều chỉnh theo nhu cầu:

```python
IMAGE_SIZE      = 320          # 320 hoặc 416
BATCH_SIZE      = 8            # giảm xuống 4 nếu OOM
EPOCHS          = 100
BACKBONE_PRESET = "yolo_v8_s_backbone_coco"  # xs / s / m / l
```

### Bước 1 — Kiểm tra Dataset

```bash
python 1_verify_dataset.py
```

Hiển thị số ảnh, số nhãn, phân phối lớp, và mẫu nhãn để xác nhận dataset hợp lệ.

### Bước 2 — Huấn luyện

```bash
python 2_train.py
```

Theo dõi quá trình train bằng TensorBoard:

```bash
tensorboard --logdir checkpoints/
```

Model tốt nhất được lưu tại `checkpoints/best_model.keras`.

### Bước 3 — Export SavedModel

```bash
python 3_export_savedmodel.py
```

### Bước 4 — Chuyển đổi TFLite

```bash
python 4_convert_tflite.py
```

Sinh ra 3 file trong `tflite/`:

| File | Kích thước | Ghi chú |
|------|:---:|---|
| `model_fp32.tflite` | ~48 MB | Baseline |
| `model_fp16.tflite` | ~24 MB | **Khuyến nghị** |
| `model_int8.tflite` | ~12 MB | Thiết bị low-end |

### Bước 5 — Inference

```bash
python 5_inference_tflite.py --image path/to/cccd.jpg
python 5_inference_tflite.py --image path/to/cccd.jpg --model tflite/model_int8.tflite --threshold 0.3
```

Kết quả lưu tại `result_detection.jpg`.

---

## Chạy trên Google Colab

Mở notebook [eKYC_CCCD_KerasCV_Training_fix.ipynb](eKYC_CCCD_KerasCV_Training_fix.ipynb) trên Colab:

1. **Runtime** → Change runtime type → **GPU (T4)**
2. Chạy từng cell theo thứ tự
3. Upload dataset `.zip` khi được yêu cầu
4. Sau khi train xong, TFLite files được tự động lưu về Google Drive

---

## Triển Khai Android

Thêm Flex Delegate vào `build.gradle`:

```gradle
implementation 'org.tensorflow:tensorflow-lite:+'
implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:+'
```

Khởi tạo interpreter:

```java
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(new FlexDelegate());
Interpreter interpreter = new Interpreter(
    FileUtil.loadMappedFile(context, "model_fp16.tflite"), options
);
```

---

## Tại Sao Không Dùng TF OD API?

| | TF OD API | KerasCV (pipeline này) |
|---|:---:|:---:|
| Hoạt động TF 2.16+ | Không | **Có** |
| Phụ thuộc tf.estimator | Có (đã bị xóa) | Không |
| Cần TFRecord | Có | Không |
| Cần protobuf config | Có | Không |
| Tương thích Keras 3 | Không | **Có** |

---

## Tài Liệu

- [BAO_CAO_EKYC_CCCD.md](BAO_CAO_EKYC_CCCD.md) — Báo cáo kỹ thuật đầy đủ, giải thích từng fix
- [KerasCV YOLOV8Detector](https://keras.io/api/keras_cv/models/tasks/yolo_v8_detector/)
- [TFLite Flex Delegate](https://www.tensorflow.org/lite/guide/ops_select)
