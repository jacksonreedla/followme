"""Decide which tracker id is the enrolled subject, frame by frame.

Inputs per frame: a list of Observation(track_id, body_emb, face_emb|None).
State: the enrolled face + body embeddings, a per-track EMA of body similarity,
and the current subject track id.

Rules (all unit-tested with synthetic embeddings):
  * score(track) = face cosine similarity when a face was seen in the box this
    frame, else the track's EMA body similarity. Face evidence is stronger, so a
    fresh face match also resets that track's body EMA upward.
  * ACQUIRE: with no current subject, the best-scoring track above `acquire`
    becomes the subject.
  * KEEP: the current subject stays while its score is above `drop`
    (hysteresis so a single bad frame does not lose the lock).
  * SWITCH GUARD: while a subject is held, another track can steal the lock only
    with a face match above `acquire` AND a score higher than the subject's.
    Body appearance alone never switches a held lock (two people in similar
    clothes is the classic ID-switch failure).
  * When the subject's track id disappears from the tracker, the lock clears and
    we fall back to ACQUIRE on the next frames (this is "re-acquisition").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


@dataclass(frozen=True)
class Observation:
    track_id: int
    body_emb: np.ndarray
    face_emb: np.ndarray | None = None


@dataclass
class MatcherConfig:
    acquire: float = 0.55
    drop: float = 0.35
    body_ema: float = 0.3          # weight of the newest body similarity
    face_weight: float = 1.0       # face similarity is used raw; body via EMA


@dataclass
class Enrollment:
    face_emb: np.ndarray | None
    body_emb: np.ndarray


@dataclass
class SubjectMatcher:
    cfg: MatcherConfig = field(default_factory=MatcherConfig)
    enrollment: Enrollment | None = None
    subject_id: int | None = None
    _body_ema: dict[int, float] = field(default_factory=dict)
    last_scores: dict[int, float] = field(default_factory=dict)

    def enroll(self, body_emb: np.ndarray, face_emb: np.ndarray | None) -> None:
        self.enrollment = Enrollment(face_emb=face_emb, body_emb=body_emb)
        self.subject_id = None
        self._body_ema.clear()

    def update(self, obs: list[Observation]) -> int | None:
        if self.enrollment is None:
            return None
        e = self.enrollment
        seen = {o.track_id for o in obs}
        # forget tracks that vanished
        for tid in list(self._body_ema):
            if tid not in seen:
                del self._body_ema[tid]
        if self.subject_id is not None and self.subject_id not in seen:
            self.subject_id = None

        scores: dict[int, float] = {}
        face_hit: dict[int, float] = {}
        for o in obs:
            b = cosine(o.body_emb, e.body_emb)
            prev = self._body_ema.get(o.track_id)
            ema = b if prev is None else (1 - self.cfg.body_ema) * prev + self.cfg.body_ema * b
            if o.face_emb is not None and e.face_emb is not None:
                f = cosine(o.face_emb, e.face_emb) * self.cfg.face_weight
                face_hit[o.track_id] = f
                scores[o.track_id] = f
                if f >= self.cfg.acquire:
                    ema = max(ema, f)   # strong face match lifts body confidence
            else:
                scores[o.track_id] = ema
            self._body_ema[o.track_id] = ema
        self.last_scores = scores

        if self.subject_id is not None:
            s_cur = scores[self.subject_id]
            if s_cur < self.cfg.drop:
                self.subject_id = None
            else:
                for tid, f in face_hit.items():
                    if tid != self.subject_id and f >= self.cfg.acquire and f > s_cur:
                        self.subject_id = tid
                        break
                return self.subject_id

        best = max(scores.items(), key=lambda kv: kv[1], default=None)
        if best is not None and best[1] >= self.cfg.acquire:
            self.subject_id = best[0]
        return self.subject_id
