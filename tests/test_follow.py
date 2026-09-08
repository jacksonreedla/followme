import pytest
from followme.control.follow import BBox, FollowConfig, FollowController

CFG = FollowConfig(frame_w=640, frame_h=480, focal_px=554.0, person_height_m=1.7, standoff_m=4.0)


def centred_box(h=235.45):  # 554*1.7/4.0 = 235.45 px → exactly at standoff
    return BBox(320 - 40, 240 - h / 2, 320 + 40, 240 + h / 2)


def test_centred_at_standoff_commands_nothing():
    fc = FollowController(CFG)
    cmd, d = fc.update(centred_box(), 0.1)
    assert d == pytest.approx(4.0, abs=0.01)
    assert cmd.vx == 0 and cmd.vz == 0 and cmd.yaw_rate == 0


def test_subject_right_of_centre_yaws_right():
    fc = FollowController(CFG)
    b = centred_box()
    cmd, _ = fc.update(BBox(b.x1 + 160, b.y1, b.x2 + 160, b.y2), 0.1)
    assert cmd.yaw_rate > 0
    assert cmd.yaw_rate <= CFG.max_yaw_rate_dps


def test_subject_left_of_centre_yaws_left():
    fc = FollowController(CFG)
    b = centred_box()
    cmd, _ = fc.update(BBox(b.x1 - 160, b.y1, b.x2 - 160, b.y2), 0.1)
    assert cmd.yaw_rate < 0


def test_subject_low_in_frame_descends():
    fc = FollowController(CFG)
    b = centred_box()
    cmd, _ = fc.update(BBox(b.x1, b.y1 + 100, b.x2, b.y2 + 100), 0.1)
    assert cmd.vz > 0  # NED: positive vz is down


def test_far_subject_moves_forward_and_near_moves_back():
    fc = FollowController(CFG)
    far, _ = fc.update(centred_box(h=100), 0.1)    # ~9.4 m away
    assert far.vx > 0
    fc.reset()
    near, _ = fc.update(centred_box(h=500), 0.1)   # ~1.9 m away
    assert near.vx < 0


def test_standoff_estimate_is_pinhole():
    fc = FollowController(CFG)
    assert fc.estimate_standoff(BBox(0, 0, 10, 554 * 1.7 / 2)) == pytest.approx(2.0)
    assert fc.estimate_standoff(BBox(0, 0, 10, 0)) == float("inf")


def test_outputs_are_clamped():
    fc = FollowController(CFG)
    cmd, _ = fc.update(BBox(600, 0, 640, 5), 0.1)  # tiny box at the far corner
    assert abs(cmd.yaw_rate) <= CFG.max_yaw_rate_dps
    assert abs(cmd.vz) <= CFG.max_vz_mps
    assert abs(cmd.vx) <= CFG.max_speed_mps
