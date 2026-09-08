"""Map the subject's bounding box to a body-frame velocity command.

  horizontal offset of box centre  → yaw_rate   (turn to keep subject centred)
  vertical offset of box centre    → vz         (climb/descend to keep eye-level)
  standoff error (want - estimate) → vx         (close in / back off)

Standoff distance is estimated from the box height with a pinhole model:
    distance = f_px * person_height_m / box_height_px
It is crude (a crouching subject looks far away) but monotonic, which is all
the P-controller needs. Ground truth in sim is used to measure its error.
"""
from __future__ import annotations

from dataclasses import dataclass

from followme.control.pid import PID
from followme.drone.base import VelocityCmd


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def w(self) -> float:
        return self.x2 - self.x1


@dataclass(frozen=True)
class FollowConfig:
    standoff_m: float = 4.0
    person_height_m: float = 1.7
    focal_px: float = 554.0          # x500_mono_cam: 640px wide, 1.047 rad HFOV → ~554 px
    frame_w: int = 640
    frame_h: int = 480
    # gains: errors are normalised to [-1, 1] before the PIDs
    yaw_kp: float = 60.0             # deg/s per unit normalised x error
    yaw_deadband: float = 0.04
    vz_kp: float = 0.8               # m/s per unit normalised y error
    vz_deadband: float = 0.08
    vx_kp: float = 0.5               # m/s per metre of standoff error
    vx_deadband_m: float = 0.3
    max_speed_mps: float = 1.5
    max_vz_mps: float = 0.8
    max_yaw_rate_dps: float = 45.0


class FollowController:
    def __init__(self, cfg: FollowConfig | None = None):
        self.cfg = cfg or FollowConfig()
        c = self.cfg
        self.yaw = PID(kp=c.yaw_kp, deadband=c.yaw_deadband,
                       out_min=-c.max_yaw_rate_dps, out_max=c.max_yaw_rate_dps)
        self.vz = PID(kp=c.vz_kp, deadband=c.vz_deadband,
                      out_min=-c.max_vz_mps, out_max=c.max_vz_mps)
        self.vx = PID(kp=c.vx_kp, deadband=c.vx_deadband_m,
                      out_min=-c.max_speed_mps, out_max=c.max_speed_mps)

    def estimate_standoff(self, box: BBox) -> float:
        if box.h <= 0:
            return float("inf")
        return self.cfg.focal_px * self.cfg.person_height_m / box.h

    def update(self, box: BBox, dt: float) -> tuple[VelocityCmd, float]:
        """Return (command, estimated standoff in metres)."""
        c = self.cfg
        ex = (box.cx - c.frame_w / 2) / (c.frame_w / 2)   # +1 = right edge
        ey = (box.cy - c.frame_h / 2) / (c.frame_h / 2)   # +1 = bottom edge
        standoff = self.estimate_standoff(box)
        e_dist = standoff - c.standoff_m                   # + = too far → go forward
        yaw_rate = self.yaw.update(ex, dt)                 # subject right → yaw right (+)
        vz = self.vz.update(ey, dt)                        # subject low in frame → descend (+vz is down)
        vx = self.vx.update(e_dist, dt)
        return VelocityCmd(vx=vx, vy=0.0, vz=vz, yaw_rate=yaw_rate), standoff

    def reset(self) -> None:
        self.yaw.reset(); self.vz.reset(); self.vx.reset()
