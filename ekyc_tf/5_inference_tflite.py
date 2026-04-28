"""
Bước 5 — Inference với TFLite (không cần TF OD API, không cần GPU)

Tại sao phải decode thủ công:
    KerasCV YOLOV8Detector export NMS qua tf.image.combined_non_max_suppression.
    Op này không có trong TFLITE_BUILTINS → khi chạy trên device Flex-delegate
    có thể không khả dụng. Script này tự decode DFL head và tự chạy NMS bằng
    NumPy thuần, không phụ thuộc bất kỳ framework nào ngoài tensorflow-lite.

Chạy:
    python 5_inference_tflite.py --image path/to/cccd.jpg
    python 5_inference_tflite.py --image path/to/cccd.jpg --model tflite/model_int8.tflite
    python 5_inference_tflite.py --image path/to/cccd.jpg --threshold 0.3 --output result.jpg
"""

import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from config import CLASSES, IMAGE_SIZE, TFLITE_DIR

COLORS = [
    (0,  255,   0),   # bottom_left  — xanh lá
    (255,  68,  68),  # bottom_right — đỏ
    (68, 136, 255),   # image_person — xanh dương
    (255, 136,   0),  # top_left     — cam
    (170,  51, 255),  # top_right    — tím
]
DEFAULT_MODEL    = os.path.join(TFLITE_DIR, "model_fp16.tflite")
DEFAULT_THRESH   = 0.35
REG_MAX          = 16   # YOLOv8 DFL bins


# ── Decode DFL + NMS ──────────────────────────────────────────────────────────

def _build_anchors(image_size: int) -> np.ndarray:
    """Tạo anchor grid cho YOLOv8 (3 stride: 8, 16, 32)."""
    strides = [8, 16, 32]
    anchors = []
    for stride in strides:
        g = image_size // stride
        for y in range(g):
            for x in range(g):
                anchors.append([x + 0.5, y + 0.5, stride])
    return np.array(anchors, dtype=np.float32)


_ANCHORS = _build_anchors(IMAGE_SIZE)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def decode_dfl(raw_dfl: np.ndarray) -> np.ndarray:
    """
    Giải mã DFL tensor → tọa độ normalized rel_xyxy.
    raw_dfl shape: [N, 4 * REG_MAX]
    """
    n = raw_dfl.shape[0]
    dfl = raw_dfl.reshape(n, 4, REG_MAX)
    dfl = _softmax(dfl)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = (dfl * bins).sum(axis=-1)   # [N, 4]: (lt_x, lt_y, rb_x, rb_y)

    ax = _ANCHORS[:n, 0]
    ay = _ANCHORS[:n, 1]
    stride = _ANCHORS[:n, 2]

    x1 = np.clip((ax - dist[:, 0]) * stride / IMAGE_SIZE, 0, 1)
    y1 = np.clip((ay - dist[:, 1]) * stride / IMAGE_SIZE, 0, 1)
    x2 = np.clip((ax + dist[:, 2]) * stride / IMAGE_SIZE, 0, 1)
    y2 = np.clip((ay + dist[:, 3]) * stride / IMAGE_SIZE, 0, 1)

    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """Non-Maximum Suppression thuần NumPy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model_path: str, image_bgr: np.ndarray, score_thresh: float) -> dict:
    interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()

    in_det  = interp.get_input_details()[0]
    out_dets = interp.get_output_details()

    # Tiền xử lý
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    arr     = resized.astype(np.float32)[np.newaxis] / 255.0

    interp.set_tensor(in_det["index"], arr)
    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    # Map output tensors theo shape: [N, 64] = DFL, [N, num_classes] = class scores
    outputs_by_shape = {}
    for d in out_dets:
        tensor = interp.get_tensor(d["index"])[0]   # bỏ batch dim
        outputs_by_shape[tensor.shape[-1]] = tensor

    num_cls = len(CLASSES)
    raw_dfl = outputs_by_shape.get(REG_MAX * 4)   # [N, 64]
    raw_cls = outputs_by_shape.get(num_cls)        # [N, num_classes]

    if raw_dfl is None or raw_cls is None:
        raise RuntimeError(
            f"Không tìm thấy output tensor đúng shape.\n"
            f"Shapes có: {list(outputs_by_shape.keys())}\n"
            f"Cần: {REG_MAX * 4} (DFL) và {num_cls} (classes)"
        )

    scores_all  = raw_cls.max(axis=1)
    classes_all = raw_cls.argmax(axis=1)
    boxes_all   = decode_dfl(raw_dfl)

    # Threshold filter
    mask    = scores_all >= score_thresh
    boxes   = boxes_all[mask]
    scores  = scores_all[mask]
    classes = classes_all[mask]

    # NMS
    keep    = nms(boxes, scores)
    return {
        "boxes":      boxes[keep],
        "scores":     scores[keep],
        "classes":    classes[keep],
        "latency_ms": latency_ms,
    }


# ── Vẽ kết quả ────────────────────────────────────────────────────────────────

def draw_results(image_bgr: np.ndarray, results: dict, score_thresh: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out  = image_bgr.copy()

    for box, score, cls_id in zip(results["boxes"], results["scores"], results["classes"]):
        if score < score_thresh:
            continue
        x1 = int(box[0] * w);  y1 = int(box[1] * h)
        x2 = int(box[2] * w);  y2 = int(box[3] * h)
        color = COLORS[int(cls_id) % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[int(cls_id)]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.putText(out, f"{results['latency_ms']:.1f} ms", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="eKYC CCCD TFLite inference")
    ap.add_argument("--image",     required=True,             help="Đường dẫn ảnh đầu vào")
    ap.add_argument("--model",     default=DEFAULT_MODEL,     help="Đường dẫn file .tflite")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESH, help="Score threshold")
    ap.add_argument("--output",    default="result_detection.jpg", help="Ảnh kết quả")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model không tìm thấy: {args.model}\nChạy 4_convert_tflite.py trước.")

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {args.image}")

    print(f"Model  : {args.model}")
    print(f"Ảnh    : {args.image}  ({image.shape[1]}×{image.shape[0]})")

    results = run_inference(args.model, image, args.threshold)

    print(f"\nLatency: {results['latency_ms']:.1f} ms")
    print(f"Phát hiện (score > {args.threshold}):")
    if len(results["boxes"]) == 0:
        print("  (không có detection nào)")
    for box, score, cls_id in zip(results["boxes"], results["scores"], results["classes"]):
        print(f"  {CLASSES[int(cls_id)]:15s}  score={score:.3f}  "
              f"box=[{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}]")

    out_img = draw_results(image, results, args.threshold)
    cv2.imwrite(args.output, out_img)
    print(f"\nKết quả đã lưu → {args.output}")


if __name__ == "__main__":
    main()
