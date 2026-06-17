"""YOLO 数据集校验与预览模块。

本文件用于检查训练数据集是否符合 YOLO 格式：图片和标签是否配对、
标签数值是否在 0-1 范围、class_id 是否超过 data.yaml 中的类别数。
同时可以生成带框预览图，方便在报告中展示数据质量。
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import yaml

from path_utils import runtime_path

BASE_DIR = runtime_path(".")
COLORS = [(0, 255, 0), (255, 200, 0), (0, 165, 255)]


def _load_yaml(yaml_path):
    """读取 data.yaml，获得类别数量、类别名称和数据集路径配置。"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _read_label(label_path):
    """读取单个 YOLO 标签文件，并校验每行 class/x/y/w/h 格式。"""
    if not label_path.exists():
        return [], None
    lines = []
    with open(label_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                return None, f"Invalid line format: {ln}"
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                return None, f"Non-numeric values in line: {ln}"
            for v in vals:
                if v < 0 or v > 1:
                    return None, f"Value out of range [0,1]: {ln}"
            lines.append({"class_id": int(vals[0]), "x": vals[1], "y": vals[2],
                          "w": vals[3], "h": vals[4]})
    return lines, None


def validate_dataset(dataset_dir, yaml_path):
    """校验 YOLO 数据集结构，并返回包含统计信息和问题列表的字典。"""
    dataset_dir = Path(dataset_dir)
    cfg = _load_yaml(yaml_path)
    expected_nc = cfg.get("nc", 3)
    expected_names = cfg.get("names", [])
    issues = []
    stats = {"total_images": 0, "total_labels": 0, "empty_labels": 0,
             "missing_labels": 0, "missing_images": 0, "invalid_labels": 0,
             "corrupt_images": 0, "class_ids_seen": set()}

    for subset in ["train", "val"]:
        img_dir = dataset_dir / subset / "images"
        lbl_dir = dataset_dir / subset / "labels"
        if not img_dir.is_dir():
            issues.append(f"[{subset}] Missing directory: {img_dir}")
            continue

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue
            stats["total_images"] += 1
            label_file = lbl_dir / (img_file.stem + ".txt")

            if not label_file.exists():
                stats["missing_labels"] += 1
                issues.append(f"[{subset}] Missing label: {img_file.name}")
                continue

            labels, error = _read_label(label_file)
            if error:
                stats["invalid_labels"] += 1
                issues.append(f"[{subset}] Invalid label {label_file.name}: {error}")
                continue

            if len(labels) == 0:
                stats["empty_labels"] += 1

            stats["total_labels"] += len(labels)
            for lb in labels:
                if lb["class_id"] >= expected_nc:
                    issues.append(
                        f"[{subset}] {label_file.name}: "
                        f"class_id {lb['class_id']} >= nc={expected_nc}"
                    )
                stats["class_ids_seen"].add(lb["class_id"])

    for subset in ["train", "val"]:
        lbl_dir = dataset_dir / subset / "labels"
        img_dir = dataset_dir / subset / "images"
        if not lbl_dir.is_dir():
            continue
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            expected_img = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                candidate = img_dir / (lbl_file.stem + ext)
                if candidate.exists():
                    expected_img = candidate
                    break
            if expected_img is None:
                stats["missing_images"] += 1
                issues.append(f"[{subset}] Orphan label: {lbl_file.name}")

    for img_file in (dataset_dir / "train" / "images").glob("*"):
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            frame = cv2.imread(str(img_file))
            if frame is None:
                stats["corrupt_images"] += 1
                issues.append(f"[train] Corrupt image: {img_file.name}")
    for img_file in (dataset_dir / "val" / "images").glob("*"):
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            frame = cv2.imread(str(img_file))
            if frame is None:
                stats["corrupt_images"] += 1
                issues.append(f"[val] Corrupt image: {img_file.name}")

    yaml_issues = []
    actual_class_count = len(stats["class_ids_seen"])
    if actual_class_count > expected_nc:
        yaml_issues.append(f"nc={expected_nc} but found class_ids up to {max(stats['class_ids_seen'])}")
    max_id = max(stats["class_ids_seen"]) if stats["class_ids_seen"] else -1
    if max_id >= len(expected_names):
        yaml_issues.append(f"names list has {len(expected_names)} entries but class_id {max_id} is out of range")
    for subset in ["train", "val"]:
        yaml_path_subset = cfg.get(subset, "")
        if yaml_path_subset and not (dataset_dir / subset).exists():
            yaml_issues.append(f"data.yaml {subset}: {yaml_path_subset} not found")

    stats["class_ids_seen"] = sorted(stats["class_ids_seen"])
    report = {
        "dataset": str(dataset_dir),
        "yaml": str(yaml_path),
        "statistics": {k: v for k, v in stats.items()},
        "yaml_issues": yaml_issues,
        "issues": issues,
    }
    return report


def cmd_validate(dataset_dir, yaml_path, report_path=None):
    """命令行入口：输出校验摘要，并把完整报告保存为 JSON。"""
    report = validate_dataset(dataset_dir, yaml_path)
    stats = report["statistics"]

    print(f"Dataset:     {report['dataset']}")
    print(f"YAML:        {report['yaml']}")
    print("-" * 50)
    print(f"Total images:       {stats['total_images']}")
    print(f"Total labels:       {stats['total_labels']}")
    print(f"Empty labels:       {stats['empty_labels']}")
    print(f"Missing labels:     {stats['missing_labels']}")
    print(f"Missing images:     {stats['missing_images']}")
    print(f"Invalid labels:     {stats['invalid_labels']}")
    print(f"Corrupt images:     {stats['corrupt_images']}")
    print(f"Class IDs seen:     {stats['class_ids_seen']}")

    # Count images per class from filename prefix
    from collections import Counter
    class_counts = Counter()
    for subset in ["train", "val"]:
        img_dir = Path(dataset_dir) / subset / "images"
        if img_dir.is_dir():
            for f in img_dir.iterdir():
                prefix = f.stem.rsplit("_", 1)[0] if "_" in f.stem else "unknown"
                class_counts[prefix] += 1
    if class_counts:
        print(f"\nPer-class breakdown:")
        for cls in sorted(class_counts):
            print(f"  {cls}: {class_counts[cls]} images")

    if report["yaml_issues"]:
        print(f"\nYAML issues:")
        for issue in report["yaml_issues"]:
            print(f"  - {issue}")

    issues = report["issues"]
    if issues:
        print(f"\nIssues ({len(issues)}):")
        for issue in issues[:30]:
            print(f"  - {issue}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")

    output = Path(report_path or str(Path(dataset_dir) / "_validation_report.json"))
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {output}")


def generate_preview(dataset_dir, output_path, grid_size=3):
    """随机抽取样本绘制标签框，生成九宫格预览图。"""
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    if not img_dir.is_dir():
        img_dir = dataset_dir / "val" / "images"
        lbl_dir = dataset_dir / "val" / "labels"
    if not img_dir.is_dir():
        print(f"No images directory found in {dataset_dir}")
        return

    images = [f for f in sorted(img_dir.iterdir())
              if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        print(f"No images found in {img_dir}")
        return

    sample = random.sample(images, min(grid_size * grid_size, len(images)))
    cells = []
    target_w, target_h = 640, 480

    for img_path in sample:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (target_w, target_h))
        label_file = lbl_dir / (img_path.stem + ".txt")
        labels, _ = _read_label(label_file)
        if labels:
            for lb in labels:
                x1 = int((lb["x"] - lb["w"] / 2) * target_w)
                y1 = int((lb["y"] - lb["h"] / 2) * target_h)
                x2 = int((lb["x"] + lb["w"] / 2) * target_w)
                y2 = int((lb["y"] + lb["h"] / 2) * target_h)
                color = COLORS[lb["class_id"] % len(COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cells.append(frame)

    while len(cells) < grid_size * grid_size:
        cells.append(cv2.resize(cv2.imread(str(sample[0])), (target_w, target_h)) if sample
                     else (0, 0, 0))

    rows = []
    for r in range(grid_size):
        rows.append(cv2.hconcat([cells[r * grid_size + c] for c in range(grid_size)]))
    grid_img = cv2.vconcat(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid_img)
    print(f"Preview saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO dataset validator")
    parser.add_argument("--dataset", required=True, help="Dataset root directory")
    parser.add_argument("--yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--report", type=str, default=None, help="Report output path")
    parser.add_argument("--preview", action="store_true",
                        help="Generate a preview grid with bounding boxes")
    parser.add_argument("--preview-output", type=str, default="preview_grid.jpg",
                        help="Preview image output path (default: preview_grid.jpg)")

    args = parser.parse_args()
    cmd_validate(args.dataset, args.yaml, args.report)
    if args.preview:
        print("\nGenerating preview...")
        generate_preview(args.dataset, args.preview_output)
