import pytest
from followme.control.safety import SafetyLimits, SafetyMonitor, State
from followme.drone.base import VelocityCmd

FWD = VelocityCmd(vx=1.0, yaw_rate=10.0)


def armed(t=0.0):
    m = SafetyMonitor(SafetyLimits(hold_after_s=1.0, land_after_s=8.0))
    m.confirm_enrollment(t)
    return m


def test_starts_enrolling_and_outputs_zero():
    m = SafetyMonitor()
    assert m.state is State.ENROLLING
    assert m.step(0.0, True, FWD) == VelocityCmd.zero()


def test_confirm_enrollment_enters_follow_and_passes_commands():
    m = armed()
    assert m.state is State.FOLLOW
    assert m.step(0.1, True, FWD) == FWD


def test_brief_dropout_keeps_yaw_but_stops_translation():
    m = armed()
    out = m.step(0.5, False, FWD)
    assert m.state is State.FOLLOW
    assert out == VelocityCmd(0.0, 0.0, 0.0, 10.0)


def test_lost_past_hold_threshold_holds():
    m = armed()
    m.step(1.0, False, FWD)
    assert m.state is State.HOLD
    assert m.step(2.0, False, FWD) == VelocityCmd.zero()


def test_reacquire_returns_to_follow():
    m = armed()
    m.step(2.0, False, FWD)
    assert m.state is State.HOLD
    assert m.step(2.5, True, FWD) == FWD
    assert m.state is State.FOLLOW


def test_lost_past_land_threshold_lands_and_is_sticky():
    m = armed()
    m.step(8.0, False, FWD)
    assert m.state is State.LANDING
    assert m.step(9.0, True, FWD) == VelocityCmd.zero()
    assert m.state is State.LANDING


def test_kill_is_terminal():
    m = armed()
    m.kill()
    assert m.step(0.1, True, FWD) == VelocityCmd.zero()
    m.confirm_enrollment(0.2)
    assert m.state is State.KILLED


def test_speed_clamps():
    m = SafetyMonitor(SafetyLimits(max_speed_mps=1.0, max_vz_mps=0.5, max_yaw_rate_dps=30))
    m.confirm_enrollment(0)
    out = m.step(0.1, True, VelocityCmd(5, -5, 5, -100))
    assert out == VelocityCmd(1.0, -1.0, 0.5, -30.0)


def test_min_standoff_blocks_forward_but_not_backward():
    m = armed()
    assert m.step(0.1, True, FWD, standoff_m=1.0).vx == 0.0
    assert m.step(0.2, True, VelocityCmd(vx=-0.5), standoff_m=1.0).vx == -0.5
    assert m.step(0.3, True, FWD, standoff_m=3.0).vx == 1.0


def test_geofence_breach_lands():
    m = SafetyMonitor(SafetyLimits(geofence_radius_m=10))
    m.confirm_enrollment(0)
    m.set_home(0, 0)
    assert m.step(0.1, True, FWD, position_xy=(5, 5)) == FWD
    assert m.step(0.2, True, FWD, position_xy=(20, 0)) == VelocityCmd.zero()
    assert m.state is State.LANDING


def test_request_land():
    m = armed()
    m.request_land()
    assert m.state is State.LANDING
