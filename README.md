# followme

A consenting-subject follow-me drone: the person in front of the camera enrolls
themselves at takeoff, the drone holds a fixed standoff distance and keeps them
framed, and it re-acquires them after occlusion or crossing people using
face + body-appearance embeddings. The product shape is DJI ActiveTrack / Skydio
subject tracking. There is deliberately no "find this person from a photo" mode:
enrollment only works on a person present in the live feed.

**Status (2026-09-07):** perception, re-ID, controller and safety state machine
built and unit-tested on CPU (~8 fps at 640x480 on a Ryzen laptop). PX4 SITL +
Gazebo integration written, not yet run (toolchain install pending). Real-drone
adapter pending the airframe choice.

## Pipeline

```
camera → YOLOv8n person detector → ByteTrack ids
       → OSNet-x0.25 body embedding (every frame) + ArcFace face embedding (every 3rd)
       → SubjectMatcher: hysteresis lock, body-only can never steal a held lock
       → FollowController: bbox offset → yaw / vz, pinhole standoff → vx (PIDs)
       → SafetyMonitor: HOLD after 1 s lost, LAND after 8 s, speed caps, min standoff, geofence, kill key
       → Drone adapter (PX4 SITL via MAVSDK | Tello | MAVLink real)
```

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                   # no hardware needed
.venv/bin/python -m followme.app --drone replay --source 0   # webcam; e = enroll, l = land, q = kill
```

Model weights: `models/yolov8n.pt` (ultralytics auto-download), insightface
`buffalo_s` (auto-download to `~/.insightface`), `models/osnet_x0_25_msmt17.onnx`
(exported from the torchreid model zoo weights with `scripts/export_osnet.py`;
architecture vendored in `followme/vision/osnet.py`, MIT).

## Simulation (PX4 SITL + Gazebo Harmonic)

```bash
git clone --branch v1.15.4 --recursive https://github.com/PX4/PX4-Autopilot ~/PX4-Autopilot
bash scripts/setup_px4.sh                                    # sudo: Gazebo + PX4 deps
.venv/bin/python sim/build_world.py sim/scenarios/behind_wall.yaml
cd ~/PX4-Autopilot && PX4_GZ_WORLD=follow make px4_sitl gz_x500_mono_cam
.venv/bin/python -m followme.app --drone sitl --log runs/behind_wall.jsonl
```

Scenarios in `sim/scenarios/`: straight, behind_wall, crosser, turn_away, stop_start.

**Known limitation:** Gazebo's animated human actors have no usable face, so the
simulator validates detection, tracking, body re-ID and the control loop against
ground truth. The face path is validated on webcam/recorded video and on the real
drone.

## Safety envelope

Enrollment requires a person in the live frame and an operator keypress. Speed is
capped at 1.5 m/s, climb at 0.8 m/s, yaw at 45°/s. Forward motion is refused
inside 2 m. Subject lost for 1 s → hover; 8 s → land. Geofence radius 30 m from
takeoff. `q` / space lands immediately. All transitions are unit-tested in
`tests/test_safety.py`.
