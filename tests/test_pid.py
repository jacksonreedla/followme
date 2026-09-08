import pytest
from followme.control.pid import PID


def test_proportional_only():
    pid = PID(kp=2.0, out_min=-10, out_max=10)
    assert pid.update(1.5, 0.1) == pytest.approx(3.0)


def test_output_clamped():
    pid = PID(kp=100.0, out_min=-1, out_max=1)
    assert pid.update(5.0, 0.1) == 1.0
    assert pid.update(-5.0, 0.1) == -1.0


def test_deadband_zeroes_small_errors():
    pid = PID(kp=1.0, deadband=0.2)
    assert pid.update(0.1, 0.1) == 0.0
    assert pid.update(0.3, 0.1) == pytest.approx(0.3)


def test_integral_accumulates_and_is_clamped():
    pid = PID(kp=0.0, ki=1.0, i_max=0.5, out_min=-10, out_max=10)
    for _ in range(10):
        out = pid.update(1.0, 0.1)
    assert out == pytest.approx(0.5)  # 10 * 0.1 = 1.0 but clamped to 0.5


def test_derivative_uses_error_delta():
    pid = PID(kp=0.0, kd=1.0, out_min=-10, out_max=10)
    assert pid.update(1.0, 0.1) == 0.0          # no previous error → no D kick
    assert pid.update(2.0, 0.1) == pytest.approx(10.0)


def test_reset_clears_state():
    pid = PID(kp=0.0, ki=1.0, kd=1.0, out_min=-10, out_max=10)
    pid.update(1.0, 0.1)
    pid.reset()
    assert pid.update(1.0, 0.1) == pytest.approx(0.1)  # only fresh integral


def test_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        PID(kp=1.0).update(1.0, 0.0)
