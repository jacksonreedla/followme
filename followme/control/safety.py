"""Flight-state machine that sits between the follow controller and the drone.

States:
  ENROLLING  on the ground / hovering, waiting for the operator to confirm a subject
  FOLLOW     subject visible; controller output passes through (clamped)
  HOLD       subject lost < land_after_s; zero velocity, hover
  LANDING    subject lost >= land_after_s, geofence breached, or operator land key
  KILLED     operator kill; terminal

The monitor never issues commands itself: it returns the VelocityCmd the drone
should receive plus the state, and the app maps LANDING/KILLED to drone.land()/kill().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from followme.drone.base import VelocityCmd


class State(Enum):
    ENROLLING = "enrolling"
    FOLLOW = "follow"
    HOLD = "hold"
    LANDING = "landing"
    KILLED = "killed"


@dataclass(frozen=True)
class SafetyLimits:
    max_speed_mps: float = 1.5
    max_vz_mps: float = 0.8
    max_yaw_rate_dps: float = 45.0
    hold_after_s: float = 1.0      # subject lost this long → HOLD
    land_after_s: float = 8.0      # subject lost this long → LANDING
    min_standoff_m: float = 2.0    # closer than this → never command forward
    geofence_radius_m: float = 30.0


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.state = State.ENROLLING
        self._last_seen_t: float | None = None
        self._home_xy: tuple[float, float] | None = None

    # ---- operator inputs -------------------------------------------------
    def confirm_enrollment(self, t: float) -> None:
        if self.state is State.ENROLLING:
            self.state = State.FOLLOW
            self._last_seen_t = t

    def request_land(self) -> None:
        if self.state not in (State.KILLED,):
            self.state = State.LANDING

    def kill(self) -> None:
        self.state = State.KILLED

    def set_home(self, x: float, y: float) -> None:
        self._home_xy = (x, y)

    # ---- per-frame update ---------------------------------------------------
    def step(
        self,
        t: float,
        subject_visible: bool,
        proposed: VelocityCmd,
        standoff_m: float | None = None,
        position_xy: tuple[float, float] | None = None,
    ) -> VelocityCmd:
        """Return the command the drone should actually receive for this frame."""
        if self.state in (State.KILLED, State.LANDING, State.ENROLLING):
            return VelocityCmd.zero()

        if position_xy is not None and self._home_xy is not None:
            if math.dist(position_xy, self._home_xy) > self.limits.geofence_radius_m:
                self.state = State.LANDING
                return VelocityCmd.zero()

        if subject_visible:
            self._last_seen_t = t
            self.state = State.FOLLOW
        else:
            lost_for = t - (self._last_seen_t if self._last_seen_t is not None else t)
            if lost_for >= self.limits.land_after_s:
                self.state = State.LANDING
                return VelocityCmd.zero()
            if lost_for >= self.limits.hold_after_s:
                self.state = State.HOLD
                return VelocityCmd.zero()
            # brief dropout: keep last-frame behaviour but do not advance
            return self._clamp(VelocityCmd(0.0, 0.0, 0.0, proposed.yaw_rate), standoff_m)

        return self._clamp(proposed, standoff_m)

    def _clamp(self, cmd: VelocityCmd, standoff_m: float | None) -> VelocityCmd:
        L = self.limits
        vx = _clip(cmd.vx, L.max_speed_mps)
        vy = _clip(cmd.vy, L.max_speed_mps)
        vz = _clip(cmd.vz, L.max_vz_mps)
        yaw = _clip(cmd.yaw_rate, L.max_yaw_rate_dps)
        if standoff_m is not None and standoff_m < L.min_standoff_m:
            vx = min(vx, 0.0)
        return VelocityCmd(vx, vy, vz, yaw)


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))
