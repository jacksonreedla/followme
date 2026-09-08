"""Frame source for perception development: a webcam index or a video file.

It is a Drone only in the sense that it satisfies the protocol so the app loop
does not care. Commands are recorded and ignored. NOT a flight simulator.
"""
from __future__ import annotations

import time
from typing import Iterator

import cv2

from followme.drone.base import Drone, Frame, VelocityCmd


class ReplayDrone(Drone):
    def __init__(self, source: str | int = 0, realtime: bool = True):
        self.source = int(source) if str(source).isdigit() else source
        self.realtime = realtime
        self.cap: cv2.VideoCapture | None = None
        self.last_cmd = VelocityCmd.zero()
        self.cmds: list[tuple[float, VelocityCmd]] = []

    def connect(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.source!r}")

    def takeoff(self, altitude_m: float) -> None: pass
    def land(self) -> None: pass
    def hold(self) -> None: self.send_velocity(VelocityCmd.zero())
    def kill(self) -> None: pass

    def send_velocity(self, cmd: VelocityCmd) -> None:
        self.last_cmd = cmd
        self.cmds.append((time.monotonic(), cmd))

    def frames(self) -> Iterator[Frame]:
        assert self.cap is not None, "connect() first"
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        seq = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            yield Frame(image=img, t=time.monotonic(), seq=seq)
            seq += 1
            if self.realtime and isinstance(self.source, str):
                time.sleep(1.0 / fps)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
