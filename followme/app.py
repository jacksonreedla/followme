"""Main loop: frames → tracks → re-ID → follow controller → safety → drone.

    python -m followme.app --drone replay --source 0          # webcam, perception only
    python -m followme.app --drone sitl                       # PX4 SITL (Phase 3)

Keys (OpenCV window): e = enroll the person nearest frame centre, l = land, q/space = kill.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from followme.control.follow import BBox, FollowConfig, FollowController
from followme.control.safety import SafetyLimits, SafetyMonitor, State
from followme.drone.base import Drone, VelocityCmd
from followme.overlay import draw
from followme.vision.body_id import BodyEmbedder
from followme.vision.detector import PersonTracker, Track
from followme.vision.face_id import FaceEmbedder
from followme.vision.matcher import MatcherConfig, Observation, SubjectMatcher


class Pipeline:
    """Everything per-frame except the drone I/O, so it can run on recorded video in tests."""

    def __init__(self, follow_cfg: FollowConfig | None = None, limits: SafetyLimits | None = None,
                 face_every: int = 3):
        self.tracker = PersonTracker()
        self.faces = FaceEmbedder()
        self.bodies = BodyEmbedder()
        self.matcher = SubjectMatcher(MatcherConfig())
        self.follow = FollowController(follow_cfg)
        self.safety = SafetyMonitor(limits)
        self.face_every = face_every  # face model is the slow one; run it every N frames
        self._n = 0
        self._last_t: float | None = None
        self.last_tracks: list[Track] = []
        self.last_faces: dict[int, BBox] = {}
        self.last_standoff: float | None = None
        self.last_cmd = VelocityCmd.zero()
        self.subject_id: int | None = None

    def enroll(self, frame) -> bool:
        """Enroll the person whose box centre is nearest the frame centre. Needs a face."""
        tracks = self.last_tracks or self.tracker.update(frame)
        if not tracks:
            return False
        h, w = frame.shape[:2]
        t = min(tracks, key=lambda t: (t.box.cx - w / 2) ** 2 + (t.box.cy - h / 2) ** 2)
        faces = self.faces.assign_to_boxes(self.faces.detect(frame), {t.track_id: t.box})
        body = self.bodies.embed(frame, {t.track_id: t.box}).get(t.track_id)
        if body is None:
            return False
        self.matcher.enroll(body, faces[t.track_id].emb if t.track_id in faces else None)
        return True

    def step(self, frame, t: float) -> VelocityCmd:
        dt = 1 / 30 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        self._n += 1
        tracks = self.tracker.update(frame)
        boxes = {tr.track_id: tr.box for tr in tracks}
        body = self.bodies.embed(frame, boxes)
        faces = {}
        if self._n % self.face_every == 0 and boxes:
            faces = self.faces.assign_to_boxes(self.faces.detect(frame), boxes)
        obs = [Observation(tid, body[tid], faces[tid].emb if tid in faces else None)
               for tid in boxes if tid in body]
        self.subject_id = self.matcher.update(obs)
        self.last_tracks, self.last_faces = tracks, {k: v.box for k, v in faces.items()}

        visible = self.subject_id is not None and self.subject_id in boxes
        if visible:
            proposed, standoff = self.follow.update(boxes[self.subject_id], dt)
        else:
            proposed, standoff = VelocityCmd.zero(), None
            self.follow.reset()
        self.last_standoff = standoff
        self.last_cmd = self.safety.step(t, visible, proposed, standoff)
        return self.last_cmd


def make_drone(kind: str, source: str) -> Drone:
    if kind == "replay":
        from followme.drone.replay import ReplayDrone
        return ReplayDrone(source)
    if kind == "sitl":
        from followme.drone.sitl import SitlDrone
        return SitlDrone()
    raise SystemExit(f"unknown drone {kind}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", default="replay", choices=["replay", "sitl"])
    ap.add_argument("--source", default="0", help="webcam index or video path for --drone replay")
    ap.add_argument("--altitude", type=float, default=2.0)
    ap.add_argument("--standoff", type=float, default=4.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log", type=Path, help="write per-frame JSONL here (metrics)")
    ap.add_argument("--record", type=Path, help="write HUD video here")
    args = ap.parse_args(argv)

    drone = make_drone(args.drone, args.source)
    pipe = Pipeline(FollowConfig(standoff_m=args.standoff))
    drone.connect()
    log = args.log.open("w") if args.log else None
    writer = None
    fps, t_prev = 0.0, time.monotonic()
    try:
        for fr in drone.frames():
            cmd = pipe.step(fr.image, fr.t)
            st = pipe.safety.state
            if st is State.FOLLOW or st is State.HOLD:
                drone.send_velocity(cmd)
            elif st is State.LANDING:
                drone.land()
            elif st is State.KILLED:
                drone.kill()
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev); t_prev = now
            if log:
                log.write(json.dumps({"t": fr.t, "seq": fr.seq, "state": st.value, "subject": pipe.subject_id,
                                      "cmd": cmd.__dict__, "standoff": pipe.last_standoff,
                                      "tracks": {tr.track_id: tr.box.__dict__ for tr in pipe.last_tracks},
                                      "latency_s": now - fr.t}) + "\n")
            if not args.headless or args.record:
                hud = draw(fr.image, pipe.last_tracks, pipe.matcher.last_scores, pipe.subject_id, st, cmd,
                           pipe.last_standoff, fps, pipe.last_faces)
                if args.record:
                    if writer is None:
                        h, w = hud.shape[:2]
                        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(hud)
                if not args.headless:
                    cv2.imshow("followme", hud)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("e"):
                        if pipe.enroll(fr.image):
                            if st is State.ENROLLING:
                                drone.takeoff(args.altitude)
                            pipe.safety.confirm_enrollment(fr.t)
                    elif k == ord("l"):
                        pipe.safety.request_land()
                    elif k in (ord("q"), 32):
                        pipe.safety.kill(); drone.kill(); break
            if st in (State.LANDING, State.KILLED) and args.headless:
                break
    finally:
        drone.close()
        if log: log.close()
        if writer: writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
