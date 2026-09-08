"""Minimal PID with deadband, anti-windup and output clamp. Pure, deterministic."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    deadband: float = 0.0        # |error| below this → output 0, integrator untouched
    i_max: float | None = None   # clamp on the integral term (anti-windup)
    _i: float = field(default=0.0, init=False, repr=False)
    _prev_err: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if abs(error) < self.deadband:
            self._prev_err = error
            return 0.0
        self._i += error * dt
        if self.i_max is not None:
            self._i = max(-self.i_max, min(self.i_max, self._i))
        d = 0.0 if self._prev_err is None else (error - self._prev_err) / dt
        self._prev_err = error
        out = self.kp * error + self.ki * self._i + self.kd * d
        return max(self.out_min, min(self.out_max, out))
