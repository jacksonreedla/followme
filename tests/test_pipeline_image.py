"""End-to-end perception on a real photo (ultralytics' bundled bus.jpg: four people, one bus).
No drone, no sim: proves detector → body/face embedding → matcher → controller → safety wire up."""
from pathlib import Path

import cv2
import pytest
import ultralytics

from followme.app import Pipeline
from followme.control.safety import State

BUS = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(face_every=1)


@pytest.fixture(scope="module")
def img():
    if not BUS.exists():
        pytest.skip("ultralytics sample image missing")
    return cv2.imread(str(BUS))


def test_detects_people_and_stays_enrolling_without_enrollment(pipe, img):
    cmd = pipe.step(img, 0.0)
    assert len(pipe.last_tracks) >= 3
    assert pipe.subject_id is None
    assert pipe.safety.state is State.ENROLLING
    assert cmd.vx == cmd.vz == cmd.yaw_rate == 0.0


def test_enroll_then_follow_locks_same_person(pipe, img):
    assert pipe.enroll(img)
    pipe.safety.confirm_enrollment(0.1)
    cmd = pipe.step(img, 0.1)
    assert pipe.subject_id is not None
    assert pipe.safety.state is State.FOLLOW
    assert pipe.last_standoff is not None and 0 < pipe.last_standoff < 50
    # subject is off-centre in bus.jpg, so at least one axis must be commanding
    assert any(abs(v) > 0 for v in (cmd.vx, cmd.vz, cmd.yaw_rate))


def test_lock_survives_horizontal_flip_of_scene(pipe, img):
    """Same people, mirrored: tracker ids change, body/face re-ID must re-lock on someone."""
    flipped = cv2.flip(img, 1)
    pipe.step(flipped, 0.2)
    pipe.step(flipped, 0.3)
    assert pipe.subject_id is not None
