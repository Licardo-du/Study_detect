"""训练图片预处理模块。

本文件用于把爬取或人工收集到的原始图片整理成更适合训练的数据：
校验坏图、去重、统一尺寸、划分 train/val。它属于数据准备流程，
不会直接影响实时检测，但会影响后续模型训练质量。
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def _collect_images(input_dir):
    """递归收集目录中的图片文件，并跳过内部处理目录。"""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    images = []
    for f in sorted(input_dir.rglob("*")):
        # Skip underscore-prefixed directories (e.g., _duplicates, _rejected)
        if any(p.name.startswith("_") for p in f.parents if p != input_dir):
            continue
        if f.suffix.lower() in IMG_EXTS:
            images.append(f)
    return images


def cmd_validate(input_dir):
    """检查目录内图片是否能被 OpenCV 正常读取，并生成校验报告。"""
    images = _collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    valid = 0
    corrupt = []
    sizes = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None or frame.size == 0:
            corrupt.append(str(img_path))
        else:
            valid += 1
            h, w = frame.shape[:2]
            sizes.append((w, h))

    print(f"Total files scanned: {len(images)}")
    print(f"Valid images:      {valid}")
    print(f"Corrupt files:     {len(corrupt)}")
    if sizes:
        widths, heights = zip(*sizes)
        print(f"Min size:          {min(widths)}x{min(heights)}")
        print(f"Max size:          {max(widths)}x{max(heights)}")
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        print(f"Average size:      {int(avg_w)}x{int(avg_h)}")
    if corrupt:
        print(f"\nCorrupt files:")
        for path in corrupt[:20]:
            print(f"  {path}")
        if len(corrupt) > 20:
            print(f"  ... and {len(corrupt) - 20} more")

    report_path = Path(input_dir) / "_validate_report.json"
    report = {
        "total": len(images),
        "valid": valid,
        "corrupt_count": len(corrupt),
        "corrupt_files": corrupt,
        "size_stats": {
            "min": f"{min(widths)}x{min(heights)}" if sizes else None,
            "max": f"{max(widths)}x{max(heights)}" if sizes else None,
            "average": f"{int(avg_w)}x{int(avg_h)}" if sizes else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {report_path}")


def cmd_deduplicate(input_dir):
    """按 MD5 哈希查找重复图片，并移动到 _duplicates 目录。"""
    input_dir = Path(input_dir)
    total = 0
    total_dup = 0
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        images = _collect_images(sub)
        if not images:
            continue
        seen = {}
        dupes = []
        for img_path in images:
            md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
            if md5 in seen:
                dupes.append(img_path)
            else:
                seen[md5] = img_path
        dupes_dir = sub / "_duplicates"
        dupes_dir.mkdir(exist_ok=True)
        for dup in dupes:
            dest = dupes_dir / dup.name
            if dest.exists():
                dest = dupes_dir / f"{hashlib.md5(str(dup).encode()).hexdigest()[:6]}_{dup.name}"
            dup.rename(dest)
        print(f"  {sub.name}: {len(images)} scanned, {len(seen)} unique, {len(dupes)} duplicates")
        total += len(images)
        total_dup += len(dupes)
    print(f"Total: {total} scanned, {total - total_dup} unique, {total_dup} duplicates")


def _scale_to_max(img, target_size):
    """按最长边等比缩放图片，保持原始宽高比例。"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cmd_resize(input_dir, output_dir, size=640):
    """批量缩放图片并转换为 JPG，统一训练输入尺寸。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    count = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        resized = _scale_to_max(frame, size)
        out_name = img_path.stem + ".jpg"
        cv2.imwrite(str(output_dir / out_name), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1

    print(f"Resized {count} images → {output_dir}")


def cmd_split(input_dir, output_base, ratio=0.8):
    """Split images into train/val following YOLO directory structure."""
    input_dir = Path(input_dir)
    output_base = Path(output_base)
    images = _collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    random.shuffle(images)
    split_idx = int(len(images) * ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = output_base / subset / "images"
        lbl_dir = output_base / subset / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            shutil.copy2(img_path, img_dir / img_path.name)
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)

    print(f"Total images: {len(images)}")
    print(f"Train:        {len(train_imgs)} → {output_base / 'train'}")
    print(f"Val:          {len(val_imgs)} → {output_base / 'val'}")


def cmd_pipeline(input_dir, output_dir, size=640, ratio=0.8):
    """完整预处理流水线：校验、去重、缩放、划分训练集和验证集。"""
    print("=" * 50)
    print("Step 1/4: Validate")
    print("=" * 50)
    cmd_validate(input_dir)

    print("\n" + "=" * 50)
    print("Step 2/4: Deduplicate")
    print("=" * 50)
    cmd_deduplicate(input_dir)

    processed_dir = Path(output_dir).parent / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    print("\n" + "=" * 50)
    print("Step 3/4: Resize")
    print("=" * 50)
    cmd_resize(input_dir, processed_dir, size)

    dataset_dir = Path(output_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    print("\n" + "=" * 50)
    print("Step 4/4: Split")
    print("=" * 50)
    cmd_split(processed_dir, dataset_dir, ratio)

    print(f"\nPipeline complete. Dataset ready at: {dataset_dir.resolve()}")


def cmd_rename(input_dir):
    """Rename images to label_0001.jpg format based on parent directory name."""
    input_dir = Path(input_dir)
    label = input_dir.name
    # Collect all files that are valid images (ignore extension)
    images = []
    for f in sorted(input_dir.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        frame = cv2.imread(str(f))
        if frame is not None and frame.size > 0:
            images.append(f)
    if not images:
        print(f"No valid images found in {input_dir}")
        return
    # Two-step rename to avoid collisions
    temp_names = []
    for i, img_path in enumerate(images, 1):
        tmp_name = f"_tmp_{i:04d}{img_path.suffix.lower() or '.jpg'}"
        img_path.rename(input_dir / tmp_name)
        temp_names.append((input_dir / tmp_name, f"{label}_{i:04d}.jpg"))
    for tmp_path, final_name in temp_names:
        tmp_path.rename(input_dir / final_name)
    print(f"Renamed {len(images)} images → {label}_0001 ... {label}_{len(images):04d}")


def cmd_autolabel(input_dir, conf=0.5):
    """Auto-label images using the existing YOLOv8 model."""
    from ai_core import StudyBehaviorDetector

    input_dir = Path(input_dir)
    images = sorted(_collect_images(input_dir))
    if not images:
        print(f"No images found in {input_dir}")
        return

    detector = StudyBehaviorDetector()
    detector.load_model()
    label_dir = input_dir
    label_dir.mkdir(parents=True, exist_ok=True)

    labeled = 0
    skipped = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        _, events, _ = detector.predict_frame(frame)
        lines = []
        dir_label = input_dir.name
        label_map = {"phone": 0, "sleep": 1, "eat": 2}
        default_class = label_map.get(dir_label, -1)

        for ev in events:
            if ev["confidence"] < conf:
                continue
            x1, y1, x2, y2 = ev["bbox"]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            # Trust directory name for class_id, model only provides bbox
            class_id = default_class if default_class >= 0 else label_map.get(ev["label"], -1)
            if class_id < 0:
                continue
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_path = label_dir / (img_path.stem + ".txt")
        if lines:
            label_path.write_text("\n".join(lines))
            labeled += 1
        else:
            skipped += 1

    print(f"Images scanned: {len(images)}")
    print(f"Labeled:       {labeled}")
    print(f"Skipped:       {skipped} (no detection above conf={conf})")
    print(f"Labels saved to: {label_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vp = subparsers.add_parser("validate", help="Validate images in a directory")
    vp.add_argument("--input", required=True, help="Input directory")

    dp = subparsers.add_parser("deduplicate", help="Remove duplicate images (MD5)")
    dp.add_argument("--input", required=True, help="Input directory")

    rp = subparsers.add_parser("resize", help="Resize images to target size")
    rp.add_argument("--input", required=True, help="Input directory")
    rp.add_argument("--output", required=True, help="Output directory")
    rp.add_argument("--size", type=int, default=640, help="Target size (default: 640)")

    sp = subparsers.add_parser("split", help="Split into train/val")
    sp.add_argument("--input", required=True, help="Input directory with images")
    sp.add_argument("--output", required=True, help="Output base directory")
    sp.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")

    pp = subparsers.add_parser("pipeline", help="Run full pipeline")
    pp.add_argument("--input", required=True, help="Input directory with raw images")
    pp.add_argument("--output", required=True, help="Output dataset directory")
    pp.add_argument("--size", type=int, default=640, help="Target size (default: 640)")
    pp.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")

    rnp = subparsers.add_parser("rename", help="Rename images to label_0001.jpg format")
    rnp.add_argument("--input", required=True, help="Input directory")

    alp = subparsers.add_parser("autolabel", help="Auto-label images using YOLO model")
    alp.add_argument("--input", required=True, help="Input directory with images")
    alp.add_argument("--conf", type=float, default=0.5,
                     help="Confidence threshold (default: 0.5)")

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args.input)
    elif args.command == "deduplicate":
        cmd_deduplicate(args.input)
    elif args.command == "resize":
        cmd_resize(args.input, args.output, args.size)
    elif args.command == "split":
        cmd_split(args.input, args.output, args.ratio)
    elif args.command == "pipeline":
        cmd_pipeline(args.input, args.output, args.size, args.ratio)
    elif args.command == "rename":
        cmd_rename(args.input)
    elif args.command == "autolabel":
        cmd_autolabel(args.input, args.conf)
    else:
        parser.print_help()
