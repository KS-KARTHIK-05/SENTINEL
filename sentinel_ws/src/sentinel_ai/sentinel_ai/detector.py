"""Model-agnostic thermal detector with per-stage latency measurement.

Any Ultralytics-loadable artefact works (YOLOv8/YOLO11/YOLO26 ``.pt``,
``.onnx`` or TensorRT ``.engine``), so a new model is swapped in through the
``model_path`` parameter only. Preprocessing reproduces SENTINEL v1 (min-max
normalisation + CLAHE) but without the inversion, because the Gazebo thermal
camera is configured white-hot like HIT-UAV.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    u: float            # box centre, pixels
    v: float
    w: float
    h: float
    conf: float


class ThermalDetector:
    def __init__(self, model_path: str, imgsz: int = 640, conf_floor: float = 0.25,
                 device: str = '', preprocess: str = 'clahe', invert: bool = False,
                 window: int = 300):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='detect')
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_floor = conf_floor
        self.device = device or None
        self.preprocess_mode = preprocess
        self.invert = invert
        self._clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        self.t_pre = deque(maxlen=window)
        self.t_inf = deque(maxlen=window)
        self.t_tot = deque(maxlen=window)
        self.frames = 0

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.invert:
            img = cv2.bitwise_not(img)
        if self.preprocess_mode == 'clahe':
            img = self._clahe.apply(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def __call__(self, img: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        x = self.preprocess(img)
        t1 = time.perf_counter()
        res = self.model.predict(x, imgsz=self.imgsz, conf=self.conf_floor, device=self.device,
                                 verbose=False)[0]
        t2 = time.perf_counter()
        out = []
        if res.boxes is not None and len(res.boxes):
            xywh = res.boxes.xywh.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            out = [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                   for b, c in zip(xywh, conf)]
        t3 = time.perf_counter()
        self.t_pre.append((t1 - t0) * 1e3)
        self.t_inf.append((t2 - t1) * 1e3)
        self.t_tot.append((t3 - t0) * 1e3)
        self.frames += 1
        return out

    def stats(self) -> dict:
        def q(d, p):
            return float(np.percentile(d, p)) if d else float('nan')
        return {'preprocess_ms_p50': q(self.t_pre, 50), 'inference_ms_p50': q(self.t_inf, 50),
                'inference_ms_p95': q(self.t_inf, 95), 'total_ms_p50': q(self.t_tot, 50),
                'total_ms_p95': q(self.t_tot, 95), 'frames': self.frames}
