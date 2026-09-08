"""Face detection + ArcFace embedding via insightface `buffalo_s` (SCRFD-500m + ArcFace-MBF, ONNX, CPU).

`embed_in_box` looks for a face inside a person box so the caller can attach face
evidence to a specific tracker id. Embeddings are L2-normalised 512-d vectors;
cosine similarity of the same person is typically > 0.5, strangers < 0.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from followme.control.follow import BBox


@dataclass(frozen=True)
class Face:
    box: BBox
    emb: np.ndarray
    det_score: float


class FaceEmbedder:
    def __init__(self, det_size: tuple[int, int] = (320, 320), min_face_px: int = 24):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"],
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.min_face_px = min_face_px

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        faces = []
        for f in self.app.get(frame_bgr):
            x1, y1, x2, y2 = map(float, f.bbox)
            if min(x2 - x1, y2 - y1) < self.min_face_px:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            faces.append(Face(BBox(x1, y1, x2, y2), emb, float(f.det_score)))
        return faces

    @staticmethod
    def assign_to_boxes(faces: list[Face], boxes: dict[int, BBox]) -> dict[int, Face]:
        """Map each person box id → the largest face whose centre lies inside it."""
        out: dict[int, Face] = {}
        for tid, b in boxes.items():
            inside = [f for f in faces if b.x1 <= f.box.cx <= b.x2 and b.y1 <= f.box.cy <= b.y2]
            if inside:
                out[tid] = max(inside, key=lambda f: f.box.w * f.box.h)
        return out
