"""HUD drawing for the live view and the demo video."""
from __future__ import annotations

import cv2
import numpy as np

from followme.control.follow import BBox
from followme.control.safety import State
from followme.drone.base import VelocityCmd
from followme.vision.detector import Track

GREY, WHITE, GREEN, RED = (90, 90, 90), (240, 240, 240), (80, 200, 80), (60, 60, 230)


def draw(frame: np.ndarray, tracks: list[Track], scores: dict[int, float], subject_id: int | None,
         state: State, cmd: VelocityCmd, standoff_m: float | None, fps: float,
         faces: dict[int, BBox] | None = None) -> np.ndarray:
    img = frame.copy()
    for t in tracks:
        is_subj = t.track_id == subject_id
        col = GREEN if is_subj else GREY
        b = t.box
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, 2 if is_subj else 1)
        label = f"#{t.track_id} {scores.get(t.track_id, 0):.2f}" + (" SUBJECT" if is_subj else "")
        cv2.putText(img, label, (int(b.x1), max(12, int(b.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if faces and t.track_id in faces:
            f = faces[t.track_id]
            cv2.rectangle(img, (int(f.x1), int(f.y1)), (int(f.x2), int(f.y2)), WHITE, 1)
    h, w = img.shape[:2]
    cv2.line(img, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), WHITE, 1)
    cv2.line(img, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), WHITE, 1)
    col = RED if state in (State.LANDING, State.KILLED) else WHITE
    lines = [f"{state.value.upper()}  {fps:4.1f} fps",
             f"vx {cmd.vx:+.2f}  vz {cmd.vz:+.2f}  yaw {cmd.yaw_rate:+5.1f}",
             f"standoff {standoff_m:.1f} m" if standoff_m not in (None, float('inf')) else "standoff --"]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return img
