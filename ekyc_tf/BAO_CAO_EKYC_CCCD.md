# Báo Cáo: Huấn Luyện Mô Hình eKYC Nhận Diện CCCD
## Pipeline KerasCV YOLOv8 + TensorFlow → TFLite

---

## 1. Tổng Quan

Dự án xây dựng pipeline **hoàn toàn thuần TensorFlow/Keras** để huấn luyện mô hình nhận diện 5 vùng đặc trưng trên Căn Cước Công Dân (CCCD), sau đó xuất ra file **TFLite** để triển khai trên thiết bị di động Android/iOS phục vụ hệ thống eKYC.

### Lớp nhận diện

| Class ID | Tên | Mô tả |
|:---:|---|---|
| 0 | `bottom_left` | Góc dưới bên trái CCCD |
| 1 | `bottom_right` | Góc dưới bên phải CCCD |
| 2 | `image_person` | Vùng ảnh chân dung |
| 3 | `top_left` | Góc trên bên trái CCCD |
| 4 | `top_right` | Góc trên bên phải CCCD |

### Môi trường thực thi

| Thành phần | Phiên bản |
|---|---|
| TensorFlow | 2.19.0 |
| Keras | 3.13.2 |
| KerasCV | 0.9.0 |
| GPU | NVIDIA T4 (Google Colab) |
| Python | 3.12 |

---

## 2. Vấn Đề Gốc: TF OD API Không Còn Hoạt Động

Pipeline ban đầu được thiết kế sử dụng **TensorFlow Object Detection API** (TF OD API) với mô hình **SSD MobileNet V2 FPN-Lite 320×320**. Pipeline cũ bao gồm:

```
1_create_label_map.py        ← tạo label_map.pbtxt (1-indexed)
2_convert_to_tfrecord.py     ← chuyển YOLO → TFRecord
3_download_base_model.py     ← tải SSD MobileNet checkpoint
4_generate_pipeline_config.py ← sinh pipeline.config protobuf
5_train.py                   ← gọi model_main_tf2.py
6_export_savedmodel.py       ← gọi exporter_main_v2.py
7_convert_to_tflite.py       ← chuyển sang TFLite
8_inference_tflite.py        ← chạy kết quả
```

**Nguyên nhân hỏng:** Kể từ **TF 2.16+**, module `tf.estimator` bị **xóa hoàn toàn** khỏi TensorFlow. TF OD API phụ thuộc trực tiếp vào `tf.estimator` thông qua `model_main_tf2.py` — không có cách patch, không có workaround.

```
AttributeError: module 'tensorflow' has no attribute 'estimator'
```

Toàn bộ 8 file cũ đã được xóa và thay thế bằng pipeline mới dùng **KerasCV**.

---

## 3. Giải Pháp: KerasCV YOLOV8Detector

Thay thế SSD MobileNet (TF OD API) bằng **`keras_cv.models.YOLOV8Detector`** — hoạt động ổn định trên Keras 3 + TF 2.16+, không phụ thuộc `tf.estimator`.

### So sánh pipeline cũ vs mới

| | Pipeline cũ (TF OD API) | Pipeline mới (KerasCV) |
|---|---|---|
| Framework | TF OD API + tf.estimator | keras-cv + Keras 3 |
| Model | SSD MobileNet V2 FPN-Lite | YOLOv8-S |
| Config | pipeline.config (protobuf) | Python dict (`config.py`) |
| Data format | TFRecord + label_map.pbtxt | YOLO .txt trực tiếp |
| Hoạt động TF 2.16+ | **Không** | **Có** |
| Số file pipeline | 8 file | 5 file + 1 config |

---

## 4. Các Fix Kỹ Thuật Đã Thực Hiện

### Fix 1 — Tắt JIT Compilation (`jit_compile=False`)

```python
model.compile(
    optimizer           = keras.optimizers.SGD(...),
    box_loss            = "ciou",
    classification_loss = "binary_crossentropy",
    jit_compile         = False,   # FIX
)
```

**Nguyên nhân lỗi:** `jit_compile=True` (mặc định) kích hoạt XLA compiler. XLA không hỗ trợ `RaggedTensor` — kiểu dữ liệu bắt buộc phải dùng khi mỗi ảnh có số lượng bounding box khác nhau.

---

### Fix 2 — Bỏ `ReduceLROnPlateau`

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(...),
    keras.callbacks.EarlyStopping(...),
    # keras.callbacks.ReduceLROnPlateau(...)  ← ĐÃ XÓA
    keras.callbacks.TensorBoard(...),
]
```

**Nguyên nhân lỗi:** Model dùng `CosineDecay` schedule — một `LearningRateSchedule` object. Keras không cho phép `ReduceLROnPlateau` override learning rate khi optimizer đang dùng schedule object → RuntimeError khi gọi `optimizer.lr.assign()`.

---

### Fix 3 — Phát hiện động Signature Key cho TFLite

```python
# SAI: hard-code tên tensor
kwargs = {"keras_tensor_123": input_tensor}

# ĐÚNG: phát hiện tự động
input_key = list(serving_fn.structured_input_signature[1].keys())[0]
kwargs = {input_key: input_tensor}
```

**Nguyên nhân lỗi:** Tên tensor nội bộ (`keras_tensor_417`) do Keras tự sinh khi build graph, thay đổi theo phiên bản Keras. Hard-code dẫn đến `KeyError` khi nâng cấp Keras.

---

### Fix 4 — Manual DFL Decode + NMS cho Inference

```python
def decode_dfl(raw_dfl, image_size=320, reg_max=16):
    """Giải mã YOLOv8 DFL tensor → tọa độ normalized xyxy."""
    ...

def nms(boxes, scores, iou_thr=0.45):
    """Non-Maximum Suppression thuần NumPy."""
    ...
```

**Nguyên nhân cần fix:** `model.export()` tích hợp NMS qua `tf.image.combined_non_max_suppression` — op này yêu cầu **Flex Delegate** trên thiết bị di động. Script inference mới tự decode raw DFL output và tự chạy NMS bằng NumPy → không phụ thuộc Flex Delegate, chạy được trên mọi runtime.

---

## 5. TensorFlow Được Sử Dụng Để Làm Gì

### 5.1 `tf.data` — Pipeline Nạp Dữ Liệu (file: `2_train.py`)

```python
ds = tf.data.Dataset.from_tensor_slices(img_paths)
ds = ds.shuffle(len(img_paths), reshuffle_each_iteration=True)
ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.ragged_batch(BATCH_SIZE)   # ← quan trọng
ds = ds.prefetch(tf.data.AUTOTUNE)
```

| API | Vai trò |
|---|---|
| `from_tensor_slices` | Tạo dataset từ danh sách đường dẫn ảnh |
| `shuffle` | Trộn ngẫu nhiên mỗi epoch, tránh overfit theo thứ tự |
| `map` + `AUTOTUNE` | Song song hóa đọc ảnh + decode YOLO label |
| `ragged_batch` | Gom batch với số bounding box khác nhau — không cần zero-padding |
| `prefetch` | GPU train batch N trong khi CPU đang chuẩn bị batch N+1 |

**Chuyển đổi tọa độ YOLO → KerasCV:**
```
YOLO:    (class, cx, cy, w, h)   — normalized [0,1]
KerasCV: (x1, y1, x2, y2)        — rel_xyxy, normalized [0,1]

x1 = cx - w/2,  x2 = cx + w/2
y1 = cy - h/2,  y2 = cy + h/2
```

### 5.2 `tf.py_function` — Kết Nối Python với TF Graph

```python
img, boxes, classes = tf.py_function(
    py_load, [path], [tf.float32, tf.float32, tf.float32]
)
```

Cho phép gọi code Python thuần (PIL, NumPy, file I/O) bên trong `tf.data` pipeline mà không cần viết lại bằng TF ops.

### 5.3 `keras_cv.models.YOLOV8Detector` — Kiến Trúc Mô Hình

```python
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
model = keras_cv.models.YOLOV8Detector(
    num_classes         = 5,
    bounding_box_format = "rel_xyxy",
    backbone            = backbone,
    fpn_depth           = 1,
)
```

| Thành phần | Vai trò |
|---|---|
| YOLOv8-S Backbone | Trích xuất đặc trưng, pretrained ImageNet+COCO → hội tụ nhanh trên dataset nhỏ |
| FPN (Feature Pyramid) | Phát hiện vật thể nhiều tỉ lệ → bắt được góc nhỏ của CCCD |
| DFL Head | Dự đoán phân phối khoảng cách → box chính xác hơn anchor-based |
| Anchor-free | Không cần điều chỉnh anchor — phù hợp dataset chuyên biệt |

**Tham số:** 12,777,455 trainable / 12,799,023 tổng

### 5.4 `keras.optimizers.schedules.CosineDecay` — Học Tốc Độ

```python
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 1e-3,
    decay_steps           = EPOCHS * steps_per_epoch,
    alpha                 = 1e-3 * 0.01,   # LR cuối = 1% LR ban đầu
)
optimizer = keras.optimizers.SGD(lr_schedule, momentum=0.9)
```

LR giảm từ `1e-3` xuống `1e-5` theo đường cong cosine — tránh dao động cuối quá trình train, hội tụ mượt hơn.

**Loss function:**
- `box_loss = "ciou"` — Complete IoU, phạt cả kích thước + tỉ lệ + khoảng cách tâm box
- `classification_loss = "binary_crossentropy"` — phân loại đa nhãn độc lập

### 5.5 `keras.callbacks` — Giám Sát Quá Trình Train

| Callback | Cấu hình | Mục đích |
|---|---|---|
| `ModelCheckpoint` | `monitor="val_loss"`, `save_best_only=True` | Lưu model tốt nhất |
| `EarlyStopping` | `patience=15`, `restore_best_weights=True` | Dừng sớm, tránh overfit |
| `TensorBoard` | `log_dir=checkpoints/` | Theo dõi loss/metrics real-time |

### 5.6 `model.export()` → SavedModel

```python
model.export(SAVED_MODEL_DIR)
```

Xuất model + NMS decoder tích hợp sang **TF SavedModel**. Không cần `exporter_main_v2.py` (đã broken).

### 5.7 `tf.lite.TFLiteConverter` — Chuyển Đổi TFLite

Ba phiên bản với mức quantization khác nhau:

| File | Kích thước | Kỹ thuật | Trường hợp dùng |
|---|---:|---|---|
| `model_fp32.tflite` | ~48.8 MB | Không quantize | Baseline, debug |
| `model_fp16.tflite` | ~24.5 MB | Weights → float16 | **Khuyến nghị** |
| `model_int8.tflite` | ~12.6 MB | Full integer quantize | Thiết bị low-end |

```python
# fp16 — kỹ thuật quantization được dùng
conv.optimizations             = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,   # bắt buộc cho NMS ops
]
```

---

## 6. Dataset & Kết Quả

| | Train | Validation |
|---|---:|---:|
| Số ảnh | 371 | 58 |
| Số nhãn | 371 | 58 |

| Thông số Training | Giá trị |
|---|---|
| Epochs | 100 |
| Batch size | 8 |
| Steps/epoch | 46 |
| Thời gian | ~1 giờ (GPU T4) |
| Image size | 320×320 |

---

## 7. Cấu Trúc File Pipeline Mới

```
ekyc_tf/
├── config.py                    ← cấu hình chung (classes, paths, hyperparams)
├── 1_verify_dataset.py          ← kiểm tra dataset YOLO
├── 2_train.py                   ← tf.data + KerasCV YOLOV8 training
├── 3_export_savedmodel.py       ← export model → SavedModel
├── 4_convert_tflite.py          ← SavedModel → fp32/fp16/int8 TFLite
├── 5_inference_tflite.py        ← inference với manual DFL decode + NMS
├── requirements.txt             ← dependencies (keras-cv, tensorflow≥2.16)
└── eKYC_CCCD_KerasCV_Training_fix.ipynb  ← notebook Colab đã fix
```

---

## 8. Pipeline Tổng Thể

```
Dataset YOLO
(images/ + labels/)
       │
       ▼  [2_train.py — tf.data + ragged_batch]
tf.data.Dataset (RaggedTensor bounding boxes)
       │
       ▼  [keras_cv.YOLOV8Detector.fit() — 100 epochs]
best_model.keras
       │
       ▼  [3_export_savedmodel.py — model.export()]
saved_model/  (weights + NMS decoder)
       │
   ┌───┴───┬──────────┐
   ▼       ▼          ▼
fp32    fp16        int8
.tflite .tflite     .tflite
       │
       ▼  [5_inference_tflite.py — decode DFL + NMS NumPy]
Kết quả detection trên ảnh CCCD
```

---

## 9. Triển Khai Android

```gradle
// build.gradle
implementation 'org.tensorflow:tensorflow-lite:+'
implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:+'
```

> **Lưu ý:** `SELECT_TF_OPS` (Flex Delegate) bắt buộc khi dùng file TFLite có tích hợp NMS. Nếu muốn tránh phụ thuộc này, dùng `5_inference_tflite.py` làm tham chiếu để decode DFL + NMS trong Java/Kotlin trực tiếp.
