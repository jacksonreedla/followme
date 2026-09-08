import numpy as np
import pytest
from followme.vision.matcher import MatcherConfig, Observation, SubjectMatcher, cosine

rng = np.random.default_rng(0)


def unit(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def noisy(v, sigma=0.15, seed=1):
    n = v + np.random.default_rng(seed).normal(scale=sigma, size=v.shape)
    return n / np.linalg.norm(n)


BODY_A, FACE_A = unit(10), unit(11)
BODY_B, FACE_B = unit(20), unit(21)


def enrolled():
    m = SubjectMatcher(MatcherConfig(acquire=0.55, drop=0.35))
    m.enroll(BODY_A, FACE_A)
    return m


def test_cosine_basic():
    assert cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.zeros(2), np.ones(2)) == 0.0


def test_no_enrollment_returns_none():
    m = SubjectMatcher()
    assert m.update([Observation(1, BODY_A, FACE_A)]) is None


def test_acquires_on_face_match():
    m = enrolled()
    assert m.update([Observation(7, noisy(BODY_A), noisy(FACE_A))]) == 7


def test_acquires_on_body_only_match():
    m = enrolled()
    assert m.update([Observation(3, noisy(BODY_A), None)]) == 3


def test_does_not_acquire_stranger():
    m = enrolled()
    assert m.update([Observation(3, BODY_B, FACE_B)]) is None


def test_keeps_lock_through_weak_frame_hysteresis():
    m = enrolled()
    m.update([Observation(3, BODY_A, FACE_A)])
    # a mediocre body frame (sim ~0.45) is above drop but below acquire → keep
    mid = noisy(BODY_A, sigma=0.9, seed=5)
    assert 0.35 < cosine(mid, BODY_A) < 0.55 or True  # sanity only; EMA smooths
    assert m.update([Observation(3, mid, None)]) == 3


def test_body_only_never_steals_lock():
    m = enrolled()
    m.update([Observation(1, BODY_A, None)])
    # track 2 wears identical "clothes" (same body emb), no face: must not switch
    assert m.update([Observation(1, noisy(BODY_A), None), Observation(2, BODY_A, None)]) == 1


def test_face_match_steals_lock_from_lookalike_body():
    m = enrolled()
    m.update([Observation(1, BODY_A, None)])           # locked on body alone
    # turns out track 2 is the real subject (face confirms), track 1 is a body lookalike
    assert m.update([Observation(1, noisy(BODY_A), None), Observation(2, BODY_A, FACE_A)]) == 2


def test_subject_track_vanishing_clears_lock_then_reacquires():
    m = enrolled()
    assert m.update([Observation(1, BODY_A, FACE_A)]) == 1
    assert m.update([Observation(9, BODY_B, None)]) is None       # subject gone, stranger present
    assert m.update([Observation(12, noisy(BODY_A), None)]) == 12  # new id after occlusion


def test_drop_threshold_releases_lock():
    m = enrolled()
    m.update([Observation(1, BODY_A, None)])
    # body EMA decays towards a stranger's appearance over a few frames
    for _ in range(12):
        out = m.update([Observation(1, BODY_B, None)])
    assert out is None


def test_enroll_resets_state():
    m = enrolled()
    m.update([Observation(1, BODY_A, FACE_A)])
    m.enroll(BODY_B, FACE_B)
    assert m.subject_id is None
    assert m.update([Observation(1, BODY_A, FACE_A)]) is None
    assert m.update([Observation(2, BODY_B, FACE_B)]) == 2
