"""统计图表导出模块。

本文件只负责把结构化统计数据转换成 PNG 图表。UI 层负责选择数据来源
和保存位置，visualization.py 负责绘制柱状图、饼图、折线图、模型量化
对比图和模型性能对比图。
"""

import json
from collections import Counter
from pathlib import Path


def count_alert_labels(records):
    """从 detection_records 或行为明细转换记录中统计各告警类别次数。"""
    counter = Counter()
    for record in records:
        alerts_text = record.get("alerts_json") or "[]"
        try:
            alerts = json.loads(alerts_text)
        except json.JSONDecodeError:
            alerts = []
        for label in alerts:
            counter[label] += 1
    return counter


def export_alert_chart(records, output_path, chart_type="bar"):
    """导出告警统计图，chart_type 支持 bar、pie、line。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for chart export.") from exc

    counts = count_alert_labels(records)
    labels = ["phone", "sleep", "eat"]
    values = [counts.get(label, 0) for label in labels]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    if chart_type == "pie":
        if sum(values) == 0:
            values = [1, 1, 1]
            labels = ["phone(0)", "sleep(0)", "eat(0)"]
        plt.pie(values, labels=labels, autopct="%1.1f%%", colors=colors)
        plt.title("Alert Category Share")
    elif chart_type == "line":
        plt.plot(labels, values, marker="o", color="#2563eb", linewidth=2)
        plt.title("Alert Count Trend by Category")
        plt.xlabel("Behavior")
        plt.ylabel("Alert Count")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
    else:
        plt.bar(labels, values, color=colors)
        plt.title("Study Behavior Alert Statistics")
        plt.xlabel("Behavior")
        plt.ylabel("Alert Count")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def export_quantization_chart(metrics, output_path):
    """导出 FP32/FP16/INT8 模型体积、存储和加载耗时对比图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for chart export.") from exc

    if not metrics:
        raise RuntimeError("No quantization metrics are available.")

    labels = [item["precision"] for item in metrics]
    file_sizes = [item.get("file_size_mb", 0) for item in metrics]
    tensor_sizes = [item.get("tensor_storage_mb", 0) for item in metrics]
    load_times = [item.get("load_ms", 0) for item in metrics]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 三个子图分别展示文件体积、权重存储量和加载耗时，便于放入实验报告。
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2563eb", "#16a34a", "#f59e0b"]

    axes[0].bar(labels, file_sizes, color=colors)
    axes[0].set_title("File Size")
    axes[0].set_ylabel("MB")

    axes[1].bar(labels, tensor_sizes, color=colors)
    axes[1].set_title("Tensor Storage")
    axes[1].set_ylabel("MB")

    axes[2].bar(labels, load_times, color=colors)
    axes[2].set_title("Load Time")
    axes[2].set_ylabel("ms")

    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("FP32 / FP16 / INT8 Quantization Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def export_model_benchmark_chart(metrics, output_path):
    """导出不同模型精度的推理耗时、检测框数量和平均置信度对比图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for chart export.") from exc

    runnable = [item for item in metrics if item.get("status") == "ok"]
    if not runnable:
        raise RuntimeError("No runnable benchmark metrics are available.")

    labels = [item["precision"] for item in runnable]
    avg_times = [item.get("avg_ms", 0) for item in runnable]
    min_times = [item.get("min_ms", 0) for item in runnable]
    max_times = [item.get("max_ms", 0) for item in runnable]
    box_counts = [item.get("box_count", 0) for item in runnable]
    avg_conf = [item.get("avg_confidence", 0) for item in runnable]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 两行图表分别展示推理耗时和检测结果，便于课程报告直接引用。
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ["#2563eb", "#16a34a", "#f59e0b"]

    axes[0][0].bar(labels, avg_times, color=colors)
    axes[0][0].set_title("Average Inference Time")
    axes[0][0].set_ylabel("ms")

    axes[0][1].plot(labels, min_times, marker="o", label="Fastest")
    axes[0][1].plot(labels, max_times, marker="o", label="Slowest")
    axes[0][1].set_title("Fastest / Slowest Time")
    axes[0][1].set_ylabel("ms")
    axes[0][1].legend()

    axes[1][0].bar(labels, box_counts, color=colors)
    axes[1][0].set_title("Detected Box Count")
    axes[1][0].set_ylabel("count")

    axes[1][1].bar(labels, avg_conf, color=colors)
    axes[1][1].set_title("Average Confidence")
    axes[1][1].set_ylabel("confidence")
    axes[1][1].set_ylim(0, 1)

    for row in axes:
        for axis in row:
            axis.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Model Inference Benchmark Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
