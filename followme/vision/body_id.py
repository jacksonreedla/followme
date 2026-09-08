"""Body-appearance embedding with OSNet-x0.25 (MSMT17 weights) via onnxruntime CPU.

Input: a person crop (BGR). Output: L2-normalised 512-d vector. Same person in the
same clothes ≈ 0.6-0.9 cosine; different clothing ≈ 0.2-0.4. ~5 ms per crop.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from followme.control.follow import BBox

MODELS = Path(__file__).resolve().parents[2] / "models"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BodyEmbedder:
    def __init__(self, onnx_path: str | Path = MODELS / "osnet_x0_25_msmt17.onnx"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def crop(frame_bgr: np.ndarray, box: BBox, pad: float = 0.05) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        px, py = box.w * pad, box.h * pad
        x1, y1 = max(0, int(box.x1 - px)), max(0, int(box.y1 - py))
        x2, y2 = min(w, int(box.x2 + px)), min(h, int(box.y2 + py))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = np.stack([self._prep(c) for c in crops])
        feats = self.sess.run(None, {self.input_name: batch})[0].astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-6)

    def embed(self, frame_bgr: np.ndarray, boxes: dict[int, BBox]) -> dict[int, np.ndarray]:
        ids, crops = [], []
        for tid, b in boxes.items():
            c = self.crop(frame_bgr, b)
            if c is not None:
                ids.append(tid); crops.append(c)
        feats = self.embed_crops(crops)
        return dict(zip(ids, feats))

    @staticmethod
    def _prep(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return rgb.transpose(2, 0, 1)
