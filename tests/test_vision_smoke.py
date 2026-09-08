"""Model-loading smoke tests: every model loads on CPU and behaves sanely on synthetic input.
Real-person accuracy tests live in test_vision_fixtures.py and skip without fixtures."""
import numpy as np
import pytest

from followme.control.follow import BBox
from followme.vision.body_id import BodyEmbedder
from followme.vision.detector import PersonTracker
from followme.vision.face_id import Face, FaceEmbedder

BLANK = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def body():
    return BodyEmbedder()


def test_person_tracker_loads_and_ignores_blank_frame():
    assert PersonTracker().update(BLANK) == []


def test_face_embedder_loads_and_ignores_blank_frame():
    assert FaceEmbedder().detect(BLANK) == []


def test_body_embedding_is_unit_norm_512d(body):
    crop = np.random.default_rng(0).integers(0, 255, (200, 90, 3), dtype=np.uint8)
    f = body.embed_crops([crop, crop])
    assert f.shape == (2, 512)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-4)
    assert np.allclose(f[0], f[1])


def test_body_embedding_distinguishes_different_textures(body):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (200, 90, 3), dtype=np.uint8)
    b = np.full((200, 90, 3), 30, dtype=np.uint8)
    fa, fb = body.embed_crops([a, b])
    assert float(fa @ fb) < 0.95


def test_crop_rejects_tiny_boxes(body):
    assert body.crop(BLANK, BBox(0, 0, 4, 4)) is None
    assert body.crop(BLANK, BBox(600, 400, 700, 600)).shape == (90, 45, 3)  # clipped to frame, 5% pad


def test_assign_faces_to_person_boxes():
    f_in = Face(BBox(110, 60, 130, 80), np.ones(4), 0.9)
    f_big = Face(BBox(100, 50, 140, 95), np.ones(4), 0.9)
    f_out = Face(BBox(500, 60, 520, 80), np.ones(4), 0.9)
    boxes = {1: BBox(80, 40, 160, 300), 2: BBox(300, 40, 380, 300)}
    got = FaceEmbedder.assign_to_boxes([f_in, f_big, f_out], boxes)
    assert set(got) == {1}
    assert got[1] is f_big
