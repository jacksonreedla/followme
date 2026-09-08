"""Drone adapter interface. Everything above this layer is hardware-agnostic.

Velocity commands are in the BODY frame, NED-style as PX4 uses them:
  vx  forward (m/s), vy right (m/s), vz DOWN (m/s, positive descends),
  yaw_rate clockwise-positive (deg/s).
Adapters convert to whatever their SDK wants (Tello uses cm/s and up-positive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class VelocityCmd:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    @staticmethod
    def zero() -> "VelocityCmd":
        return VelocityCmd()


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8, HxWx3
    t: float           # monotonic seconds when the frame was captured/received
    seq: int


class Drone(Protocol):
    """One adapter per airframe / SDK. Implementations: sitl, tello, mavlink, replay."""

    def connect(self) -> None: ...
    def takeoff(self, altitude_m: float) -> None: ...
    def land(self) -> None: ...
    def hold(self) -> None:
        """Zero velocity, hover in place. Must be safe to call repeatedly."""
        ...
    def send_velocity(self, cmd: VelocityCmd) -> None: ...
    def frames(self) -> Iterator[Frame]:
        """Yield frames as they arrive; never block forever (yield None-free stream, raise on link loss)."""
        ...
    def kill(self) -> None:
        """Emergency: cut offboard control and land/disarm as the platform allows."""
        ...
    def close(self) -> None: ...
