"""AI 检测核心模块。

本文件封装 YOLOv8 推理和课堂行为后处理规则。UI 层只调用
StudyBehaviorDetector 的 predict_image、predict_frame、run_video 等方法，
不直接接触模型细节。这里同时实现了 phone、sleep、eat 三类行为的阈值、
姿态比例、滑动窗口和运动量规则。
"""

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# 摄像头断流后短暂等待再重连，避免循环过快占满 CPU。
import time

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class DetectionEvent:
    """单个检测框转换后的业务事件，便于写入数据库和报告。"""

    label: str
    confidence: float
    bbox: tuple
    alert: bool
    reason: str

    def to_dict(self):
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["confidence"] = round(self.confidence, 4)
        return data


@dataclass
class SmartDetectConfig:
    """检测阈值配置集中放在这里，便于调参和报告说明。"""

    model_path: Path = BASE_DIR / "models" / "best.pt"
    image_size: int = 640
    base_conf: float = 0.25
    conf_phone: float = 0.40
    conf_sleep: float = 0.85
    conf_eat: float = 0.70
    sleep_ratio_threshold: float = 1.2
    sleep_min_frames: int = 90
    alpha: float = 1.3
    beta: int = 15
    eat_window_size: int = 60
    eat_alert_ratio: float = 0.55
    eat_motion_low: float = 0.01
    eat_motion_high: float = 0.35


class StudyBehaviorDetector:
    """YOLOv8 检测器，负责模型加载、逐帧推理和行为规则判断。"""

    def __init__(self, config=None):
        self.config = config or SmartDetectConfig()
        self.model = None
        self.prev_gray = None
        self.eat_status_window = deque(
            [0] * self.config.eat_window_size, maxlen=self.config.eat_window_size
        )
        self.eat_motion_history = deque(maxlen=30)
        self.sleep_duration_counter = 0

    def load_model(self):
        """懒加载 YOLO 模型，第一次推理时才真正读取 best.pt。"""
        self._ensure_vision_packages()
        if self.model is not None:
            return self.model
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.config.model_path}. "
                "Put the trained weight file at models/best.pt."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Please install ultralytics before running detection.") from exc
        self.model = YOLO(str(self.config.model_path))
        return self.model

    def reset_state(self):
        """重置跨帧状态，避免上一张图片或上一段视频影响下一次检测。"""
        self.prev_gray = None
        self.eat_status_window = deque(
            [0] * self.config.eat_window_size, maxlen=self.config.eat_window_size
        )
        self.eat_motion_history.clear()
        self.sleep_duration_counter = 0

    def predict_frame(self, frame):
        """对单帧图像进行推理，并应用 phone/sleep/eat 的业务规则。"""
        self._ensure_vision_packages()
        model = self.load_model()
        cfg = self.config

        processed = cv2.convertScaleAbs(frame, alpha=cfg.alpha, beta=cfg.beta)
        results = model.predict(
            processed, imgsz=cfg.image_size, conf=cfg.base_conf, verbose=False
        )

        current_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

        events = []
        found_phone = False
        this_frame_is_eating = 0
        this_frame_is_sleeping = False

        for result in results:
            for box in result.boxes:
                label, conf, bbox = self._parse_box(model, box)
                x1, y1, x2, y2 = bbox

                if label == "phone" and conf >= cfg.conf_phone:
                    found_phone = True
                    event = DetectionEvent(
                        label, conf, bbox, True, "Phone has the highest alert priority."
                    )
                    events.append(event)
                    self._draw_box(processed, bbox, (0, 255, 0), f"PHONE {conf:.2f}")
                    continue

                if label == "sleep" and conf >= cfg.conf_sleep:
                    width = max(1, x2 - x1)
                    height = max(1, y2 - y1)
                    ratio = width / height
                    if ratio > cfg.sleep_ratio_threshold:
                        this_frame_is_sleeping = True
                        alert = self.sleep_duration_counter + 1 >= cfg.sleep_min_frames
                        reason = (
                            f"Horizontal posture ratio {ratio:.2f}; "
                            f"duration {self.sleep_duration_counter + 1} frames."
                        )
                        events.append(DetectionEvent(label, conf, bbox, alert, reason))
                        color = (0, 0, 255) if alert else (255, 200, 0)
                        self._draw_box(processed, bbox, color, f"SLEEP {conf:.2f}")
                    else:
                        events.append(
                            DetectionEvent(
                                label,
                                conf,
                                bbox,
                                False,
                                f"Rejected as low-head posture, ratio {ratio:.2f}.",
                            )
                        )
                    continue

                if label == "eat" and conf >= cfg.conf_eat and not found_phone:
                    motion = self._roi_motion(current_gray, bbox)
                    self.eat_motion_history.append(motion)
                    avg_motion = (
                        sum(self.eat_motion_history) / len(self.eat_motion_history)
                        if self.eat_motion_history
                        else 0.0
                    )
                    if cfg.eat_motion_low < avg_motion < cfg.eat_motion_high:
                        this_frame_is_eating = 1
                    events.append(
                        DetectionEvent(
                            label,
                            conf,
                            bbox,
                            False,
                            f"Motion score {avg_motion:.4f}; waiting for window vote.",
                        )
                    )
                    self._draw_box(processed, bbox, (0, 165, 255), f"EAT {conf:.2f}")

        if this_frame_is_sleeping:
            self.sleep_duration_counter += 1
        else:
            self.sleep_duration_counter = 0

        self.eat_status_window.append(this_frame_is_eating)
        eat_ratio = sum(self.eat_status_window) / cfg.eat_window_size
        if eat_ratio > cfg.eat_alert_ratio:
            for index, event in enumerate(events):
                if event.label == "eat":
                    events[index] = DetectionEvent(
                        event.label,
                        event.confidence,
                        event.bbox,
                        True,
                        f"Eating detected by sliding window ratio {eat_ratio:.2%}.",
                    )
            cv2.putText(
                processed,
                f"EATING ALERT {int(eat_ratio * 100)}%",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 100, 255),
                3,
            )

        alert_labels = sorted({event.label for event in events if event.alert})
        summary = {
            "event_count": len(events),
            "alert_labels": alert_labels,
            "eat_ratio": round(eat_ratio, 4),
            "sleep_frames": self.sleep_duration_counter,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.prev_gray = current_gray
        return processed, [event.to_dict() for event in events], summary

    def predict_image(self, image_path, output_dir=None):
        """检测单张图片，输出带框图片、事件列表和汇总信息。"""
        self._ensure_vision_packages()
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read image: {image_path}")

        self.reset_state()
        annotated, events, summary = self.predict_frame(frame)

        output_dir = Path(output_dir or BASE_DIR / "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), annotated)
        return output_path, events, summary

    def run_camera(self, source=0, stop_event=None, reconnect=True, reconnect_delay=2):
        """旧式 OpenCV 弹窗摄像头检测入口；新版 UI 主要使用内嵌预览。"""
        self._ensure_vision_packages()
        self.reset_state()
        title = "Study Behavior Monitor - press q to quit"
        try:
            while not self._should_stop(stop_event):
                # 每轮重新打开视频源，支持网络流短暂断开后的自动恢复。
                cap = self._open_capture(source)
                if not cap.isOpened():
                    raise RuntimeError(f"Unable to open camera/video source: {source}")

                try:
                    while cap.isOpened() and not self._should_stop(stop_event):
                        ok, frame = cap.read()
                        if not ok:
                            # 文件结束或网络流断开时，根据 reconnect 决定退出或重连。
                            if reconnect and not self._should_stop(stop_event):
                                break
                            return
                        annotated, _, summary = self.predict_frame(frame)
                        cv2.imshow(title, annotated)
                        if summary["alert_labels"]:
                            print("Alert:", ", ".join(summary["alert_labels"]))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
                finally:
                    cap.release()

                if not reconnect:
                    break
                time.sleep(reconnect_delay)
        finally:
            cv2.destroyAllWindows()

    def run_video(self, input_path, output_path=None):
        """逐帧检测视频文件，输出带框视频和整体告警汇总。"""
        self._ensure_vision_packages()
        self.reset_state()
        input_path = Path(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        output_path = Path(output_path or BASE_DIR / "outputs" / f"{input_path.stem}_smart.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        all_alerts = set()
        frame_count = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, _, summary = self.predict_frame(frame)
                writer.write(annotated)
                all_alerts.update(summary["alert_labels"])
                frame_count += 1
        finally:
            cap.release()
            writer.release()

        return output_path, {
            "frames": frame_count,
            "alert_labels": sorted(all_alerts),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _parse_box(self, model, box):
        class_id = int(box.cls[0])
        label = model.names[class_id]
        conf = float(box.conf[0])
        bbox = tuple(map(int, box.xyxy[0]))
        return label, conf, bbox

    @staticmethod
    def _should_stop(stop_event):
        """Return True when the tkinter thread has requested camera shutdown."""
        return stop_event is not None and stop_event.is_set()

    @staticmethod
    def _open_capture(source):
        """Open an OpenCV capture with timeouts so bad streams fail gracefully."""
        cap = cv2.VideoCapture()
        # 不同 OpenCV 构建支持的属性不同，设置前先判断可用性。
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.open(source)
        return cap

    @staticmethod
    def _ensure_vision_packages():
        missing = []
        if cv2 is None:
            missing.append("opencv-python")
        if np is None:
            missing.append("numpy")
        if missing:
            raise RuntimeError(
                "Missing vision dependencies. Please run: pip install "
                + " ".join(missing)
            )

    def _roi_motion(self, current_gray, bbox):
        if self.prev_gray is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(current_gray.shape[1], x2)
        y2 = min(current_gray.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi_curr = current_gray[y1:y2, x1:x2]
        roi_prev = self.prev_gray[y1:y2, x1:x2]
        if roi_curr.shape != roi_prev.shape or roi_curr.size == 0:
            return 0.0

        diff = cv2.absdiff(roi_curr, roi_prev)
        _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return float(np.sum(threshold) / (threshold.size * 255))

    @staticmethod
    def _draw_box(image, bbox, color, text):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
