# Follow-me drone with visual re-identification — implementation plan

## Context

Portfolio project for defense / aerospace hiring (Anduril, SpaceX class). A drone
enrolls a consenting subject from its own camera at takeoff, keeps them framed at a
fixed standoff distance, and re-acquires them after occlusion or crossing people
using face + body-appearance embeddings. This is the ActiveTrack / Skydio product
shape, scoped as a consenting-subject follower: the subject must be present in front
of the camera to enroll, and there is no "find this person from a photo" mode.

Decisions already made with the operator (2026-09-07):

- Compute runs on the **laptop as a ground station** (video in, velocity commands out).
  Onboard companion computer only if research money appears.
- Re-ID = **face (ArcFace) + body appearance (OSNet)**, fused.
- **PX4 SITL + Gazebo Harmonic from the start**; real drone second.
- Deliverables: **public GitHub repo with README + measured metrics, demo video,
  one-page write-up**.
- Physical drone model unknown until the operator visits LSUA (tomorrow). The drone
  is an adapter behind one interface; nothing else depends on it.

Dev box: this laptop (`apollo-IdeaPad-Slim-3-15ABR8`, Linux Mint 22.3 = Ubuntu 24.04
base, Python 3.12, Ryzen APU, **no NVIDIA GPU**, 16 threads, 13 GB RAM, 242 GB free,
webcam at `/dev/video0`). All inference is CPU ONNX / torch-CPU with small models.
This is NOT an OmniChrome project: new repo at `~/followme`, its own git, nothing
touches `~/omniclipper-server` or the HP.

Honest limitation to design around: Gazebo's animated human actors are low-poly with
no usable face. **Sim validates detection, tracking, body re-ID, and the control
loop against ground truth. The face path is validated on the webcam and recorded
video, and the full face+body pipeline on the real drone.** The README states this.

## Architecture

```
 camera frames ──► Detector (YOLOv8n person) ──► Tracker (ByteTrack ids)
                                                        │
                    Enrollment (at takeoff: face emb + body emb of the ONE
                    person centered in frame; operator confirms with a key)
                                                        │
                    SubjectMatcher: per track, cosine sim of face emb (when a
                    face is detected in the box) and body emb vs enrolled;
                    fused score → subject track id or LOST
                                                        │
                    FollowController: bbox → (vx, vy, vz, yaw_rate) via PIDs,
                    standoff distance from bbox height (pinhole, 1.7 m person)
                                                        │
                    SafetyMonitor: lost-timeout hover→land, speed caps,
                    min standoff, geofence, keyboard kill
                                                        │
                    Drone adapter (SITL | Tello | MAVLink real) ◄── velocity cmds
```

Every stage is a plain Python class with one job and a typed interface so each is
unit-testable without a drone or a sim.

## Repo layout (`~/followme`)

```
followme/
  followme/
    drone/
      base.py          # Drone Protocol: connect, takeoff, land, hold,
                       #   send_velocity(vx,vy,vz,yaw_rate), frames(), kill()
      sitl.py          # PX4 via MAVSDK-Python offboard + Gazebo camera topic
      tello.py         # djitellopy adapter (only if LSUA has Tellos)
      mavlink.py       # same MAVSDK code as sitl.py, real serial/UDP link
      replay.py        # video-file / webcam source that ignores commands
                       #   (perception dev + tests, NOT a flight sim)
    vision/
      detector.py      # YOLOv8n person boxes (ultralytics, CPU)
      tracker.py       # ByteTrack via ultralytics model.track()
      face_id.py       # insightface buffalo_s: SCRFD detect + ArcFace embed
      body_id.py       # OSNet-x0.25 ONNX embed (onnxruntime CPU)
      matcher.py       # enrollment store + fused scoring → subject id / LOST
    control/
      pid.py           # PID with anti-windup, output clamp, deadband
      follow.py        # bbox + frame size → VelocityCmd; standoff estimate
      safety.py        # state machine: FOLLOW / HOLD / LANDING / KILLED
    app.py             # main loop, wires everything, CLI flags
    overlay.py         # boxes, ids, scores, state HUD for demo video
    metrics.py         # per-frame log → JSON; summary stats + plots
  sim/
    worlds/follow.sdf  # x500_mono_cam + walking actor(s) + occluder
    scenarios/*.yaml   # actor paths: straight, behind-wall, crosser, turn-away
    run_scenario.py    # launch PX4 SITL + gz, run app headless, dump metrics
  scripts/
    setup_px4.sh       # clone PX4-Autopilot, ubuntu.sh deps, gz-harmonic, MAVSDK
    export_osnet.py    # torchreid → osnet_x0_25.onnx (run once, commit onnx)
    fetch_models.py    # yolov8n.pt, insightface buffalo_s
  tests/               # pytest, no drone/sim needed
  docs/                # architecture.svg, metrics.md, writeup.md
  README.md
```

## Phases

### Phase 0 — Toolchain (highest risk, do first)

1. `scripts/setup_px4.sh`: clone `PX4-Autopilot` (v1.15 tag), run
   `Tools/setup/ubuntu.sh`, install `gz-harmonic` from packages.osrfoundation.org,
   `pip install mavsdk` into `~/followme/.venv` (Python 3.12).
2. Prove the stock loop: `make px4_sitl gz_x500_mono_cam`, then a 30-line
   MAVSDK script does arm → takeoff → offboard velocity → land.
3. Prove camera-in-Python: read the gz camera topic (`/camera`) with the
   `python3-gz-transport13` bindings into a numpy frame. **Fallback if the bindings
   fight Python 3.12:** ROS 2 Jazzy + `ros_gz_bridge` → `cv_bridge`. Decide at this
   step, not later.
4. Prove ground truth: subscribe to `/world/<w>/dynamic_pose/info` for drone + actor
   poses (needed for metrics).

Exit: a frame from the sim camera displayed in OpenCV while the drone hovers under
Python control. Nothing else starts until this works.

### Phase 1 — Perception (webcam + recorded video, TDD)

- `detector.py` + `tracker.py`: wrap ultralytics `model.track(..., tracker=
  "bytetrack.yaml", classes=[0])`. Tests on fixture images.
- `face_id.py`: insightface `buffalo_s` (ONNX, CPU). Enroll = largest face in the
  center third of frame. Tests: same person two photos → sim > 0.5; different → < 0.3.
- `body_id.py`: OSNet-x0.25 ONNX (export once via `export_osnet.py` in a throwaway
  venv; commit the ~2 MB onnx). Tests mirror face tests on cropped fixtures.
- `matcher.py`: enrollment store; per-track running EMA of body sim; face sim when
  a face is visible in the box; fused score = face if available else body, with
  hysteresis (acquire > 0.55, drop < 0.35) and an ID-switch guard (never switch to
  a track whose score is below the current subject's). Output: `subject_track_id |
  None`. Pure logic → thorough unit tests with synthetic embeddings.
- `overlay.py` HUD.

Exit: `python -m followme.app --drone replay --source /dev/video0` enrolls the
operator, keeps their id through a second person walking through frame and the
operator turning around.

### Phase 2 — Control (pure functions, TDD)

- `pid.py`, `follow.py`: error_x → yaw_rate, error_y → vz, standoff error → vx
  (standoff from bbox height via pinhole with focal length from the camera SDF /
  calibration). Deadbands, rate limits, output clamps (defaults: 1.5 m/s, 45°/s).
- `safety.py` state machine: LOST > 1.0 s → HOLD (zero velocity); LOST > 8 s → LAND;
  standoff < 2 m → vx ≤ 0; geofence radius from takeoff point; `q` / `space` → kill.
  Every transition unit-tested.

### Phase 3 — Sim integration + scenarios

- `sim/worlds/follow.sdf`: x500_mono_cam, one walking `<actor>` on a scripted
  trajectory, a wall occluder, second actor for the crosser case.
- `sim/scenarios/*.yaml` + `run_scenario.py`: straight-line follow, behind-wall
  reacquire, crosser (ID-switch test), turn-away, stop-and-start.
- `sitl.py` adapter: MAVSDK offboard `set_velocity_body`, camera frames from Phase 0.

Exit: all five scenarios run headless and produce metrics JSON.

### Phase 4 — Metrics

`metrics.py` computes from the per-frame log + gz ground truth: standoff error
(mean/RMS), lateral pixel error, reacquisition time and success rate per scenario,
ID switches, end-to-end latency (frame timestamp → command send). Plots go to
`docs/`, table into README.

### Phase 5 — Real drone (after LSUA visit)

Finish exactly one of `tello.py` / `mavlink.py`. Field protocol: enroll on the
ground, takeoff to 2 m, follow at 4 m standoff in an open field, kill key tested
before every flight. Record the demo video from the drone feed with HUD.

### Phase 6 — Deliverables

README (architecture diagram, metrics table, safety envelope section, the sim/face
limitation stated plainly), `docs/writeup.md` (one page), demo video edit.

## Reuse / existing assets

- `mediapipe 0.10.14` is already in `~/venv` but has no face *recognizer*; use
  insightface instead. Nothing from the OmniChrome repos is needed
  (`clipper.py`'s face tracker is a crop-centering tool, not re-ID, and importing
  from that mirror leaks the prod `.env` — see OMNICHROME.md).
- ByteTrack comes bundled with ultralytics; do not vendor a tracker.

## Verification

- `pytest` green at every phase (no hardware needed for Phases 1-2 tests).
- Phase 0 exit criteria are the gate for everything else.
- Phase 1: live webcam run with a second person crossing; subject id must not switch.
- Phase 3/4: five scenarios produce metrics; reacquisition rate ≥ 90 % on
  behind-wall, zero ID switches on crosser, is the target before field flight.
- Phase 5: kill switch verified on the ground before every flight.

## Open questions for the operator

- Drone model at LSUA (decides Phase 5 adapter; asked, answer tomorrow).
- Repo name / GitHub org (default `followme`, personal account).
