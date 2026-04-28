"""
Bước 1 — Kiểm tra cấu trúc dataset YOLO

Cấu trúc cần có:
    data/
    ├── images/train/*.jpg|png
    ├── images/val/*.jpg|png
    ├── labels/train/*.txt
    └── labels/val/*.txt

Mỗi file .txt gồm các dòng:
    class_id  cx  cy  w  h   (tất cả normalized [0,1])

Chạy:
    python 1_verify_dataset.py
"""

import glob
import os

from config import CLASSES, TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def collect_images(directory: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def verify_split(img_dir: str, lbl_dir: str, split: str) -> bool:
    imgs = collect_images(img_dir)
    lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))

    print(f"\n[{split.upper()}]")
    print(f"  Ảnh  : {len(imgs)}  ({img_dir})")
    print(f"  Nhãn : {len(lbls)}  ({lbl_dir})")

    if not imgs:
        print("  LỖI: Không tìm thấy ảnh!")
        return False

    missing_labels, class_counts = 0, {}
    sample_shown = False

    for img_path in imgs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            missing_labels += 1
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

                if not sample_shown:
                    cx, cy, w, h = map(float, parts[1:])
                    print(f"\n  Mẫu nhãn ({os.path.basename(lbl_path)}):")
                    print(f"    class={cls_id}({CLASSES[cls_id]})  cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
                    sample_shown = True

    if missing_labels:
        print(f"  CẢNH BÁO: {missing_labels} ảnh thiếu file nhãn")

    print("\n  Phân phối lớp:")
    total_boxes = sum(class_counts.values())
    for cls_id, count in sorted(class_counts.items()):
        bar = "█" * int(count / max(class_counts.values()) * 20)
        print(f"    {cls_id} {CLASSES[cls_id]:15s} {count:4d} {bar}")
    print(f"    Tổng bounding boxes: {total_boxes}")

    return len(imgs) > 0 and len(lbls) > 0


def main() -> None:
    print("=" * 55)
    print("  Kiểm tra Dataset eKYC CCCD")
    print("=" * 55)

    ok_train = verify_split(TRAIN_IMG_DIR, TRAIN_LBL_DIR, "train")
    ok_val   = verify_split(VAL_IMG_DIR,   VAL_LBL_DIR,   "val")

    print("\n" + "=" * 55)
    if ok_train and ok_val:
        print("  Dataset hợp lệ — có thể chạy 2_train.py")
    else:
        print("  Dataset CHƯA hợp lệ — kiểm tra lại cấu trúc thư mục")
    print("=" * 55)


if __name__ == "__main__":
    main()
