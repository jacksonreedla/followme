"""Person detection + multi-object tracking in one call (ultralytics YOLOv8n + ByteTrack).

Returns Track objects with a stable id across frames. CPU only; ~30-60 ms at 640px.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from followme.control.follow import BBox

MODELS = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class Track:
    track_id: int
    box: BBox
    conf: float


class PersonTracker:
    def __init__(self, weights: str | Path = MODELS / "yolov8n.pt", conf: float = 0.4, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def update(self, frame_bgr: np.ndarray) -> list[Track]:
        res = self.model.track(frame_bgr, persist=True, classes=[0], conf=self.conf,
                               imgsz=self.imgsz, tracker="bytetrack.yaml", verbose=False)[0]
        out: list[Track] = []
        if res.boxes is None or res.boxes.id is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
            out.append(Track(int(tid), BBox(float(x1), float(y1), float(x2), float(y2)), float(c)))
        return out

    def reset(self) -> None:
        # ultralytics keeps tracker state on the predictor; dropping it restarts ids
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = []
