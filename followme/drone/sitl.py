"""PX4 SITL adapter: MAVSDK offboard body-velocity control + Gazebo camera frames.

Run PX4 first:   cd ~/PX4-Autopilot && PX4_GZ_WORLD=follow make px4_sitl gz_x500_mono_cam
Then:            python -m followme.app --drone sitl

MAVSDK is asyncio; the app loop is synchronous, so MAVSDK lives on its own thread
with a small command mailbox. Offboard mode needs a setpoint stream at >2 Hz or
PX4 fails over, hence the 20 Hz sender that re-sends the last command.

Frames come from gz-transport (`python3-gz-transport13` + `python3-gz-msgs10`,
system site-packages; the venv must be created with --system-site-packages or
PYTHONPATH must include /usr/lib/python3/dist-packages). Ground-truth poses of
every model come from /world/<world>/dynamic_pose/info for metrics.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Iterator

import numpy as np

from followme.drone.base import Drone, Frame, VelocityCmd


class GzCamera:
    """Subscribe to a Gazebo image topic and hand back BGR numpy frames."""

    def __init__(self, topic: str = "/camera", world: str = "follow"):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.pose_v_pb2 import Pose_V
        self.node = Node()
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.poses: dict[str, tuple[float, float, float]] = {}
        self._seq = 0
        if not self.node.subscribe(Image, topic, self._on_image):
            raise RuntimeError(f"gz subscribe failed on {topic}")
        self.node.subscribe(Pose_V, f"/world/{world}/dynamic_pose/info", self._on_poses)

    def _on_image(self, msg) -> None:
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)  # RGB8 from the sensor
        bgr = arr[:, :, ::-1].copy()
        fr = Frame(image=bgr, t=time.monotonic(), seq=self._seq)
        self._seq += 1
        if self.q.full():
            try: self.q.get_nowait()   # drop the stale frame, keep latency low
            except queue.Empty: pass
        self.q.put_nowait(fr)

    def _on_poses(self, msg) -> None:
        for p in msg.pose:
            self.poses[p.name] = (p.position.x, p.position.y, p.position.z)

    def frames(self, timeout: float = 5.0) -> Iterator[Frame]:
        while True:
            yield self.q.get(timeout=timeout)


class SitlDrone(Drone):
    def __init__(self, address: str = "udpin://0.0.0.0:14540", camera_topic: str = "/camera",
                 world: str = "follow", rate_hz: float = 20.0):
        self.address = address
        self.camera_topic = camera_topic
        self.world = world
        self.rate_hz = rate_hz
        self.cam: GzCamera | None = None
        self._cmd = VelocityCmd.zero()
        self._mail: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.position_xy: tuple[float, float] | None = None
        self.error: BaseException | None = None

    # ---- Drone protocol ---------------------------------------------------
    def connect(self) -> None:
        self.cam = GzCamera(self.camera_topic, self.world)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(30):
            raise RuntimeError(f"no PX4 heartbeat on {self.address}")
        if self.error:
            raise self.error

    def takeoff(self, altitude_m: float) -> None: self._mail.put(("takeoff", altitude_m))
    def land(self) -> None: self._mail.put(("land", None))
    def hold(self) -> None: self._cmd = VelocityCmd.zero()
    def kill(self) -> None: self._mail.put(("kill", None))
    def send_velocity(self, cmd: VelocityCmd) -> None: self._cmd = cmd

    def frames(self) -> Iterator[Frame]:
        assert self.cam is not None
        return self.cam.frames()

    def close(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(timeout=5)

    # ---- MAVSDK thread ----------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as e:  # surface into connect()/loop
            self.error = e
            self._ready.set()

    async def _main(self) -> None:
        from mavsdk import System
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        drone = System()
        await drone.connect(system_address=self.address)
        async for st in drone.core.connection_state():
            if st.is_connected: break
        async for h in drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok: break
        asyncio.create_task(self._track_position(drone))
        self._ready.set()
        offboard = False
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            try:
                op, arg = self._mail.get_nowait()
            except queue.Empty:
                op = None
            if op == "takeoff":
                await drone.action.arm()
                await drone.action.set_takeoff_altitude(float(arg))
                await drone.action.takeoff()
                await asyncio.sleep(6)
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
                try:
                    await drone.offboard.start(); offboard = True
                except OffboardError as e:
                    self.error = RuntimeError(f"offboard start failed: {e._result.result}")
            elif op == "land":
                if offboard:
                    try: await drone.offboard.stop()
                    except OffboardError: pass
                    offboard = False
                await drone.action.land()
            elif op == "kill":
                if offboard:
                    try: await drone.offboard.stop()
                    except OffboardError: pass
                    offboard = False
                await drone.action.land()   # SITL: a controlled land is the safe "kill"
                self._stop.set()
            if offboard:
                c = self._cmd
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(c.vx, c.vy, c.vz, c.yaw_rate))
            await asyncio.sleep(period)

    async def _track_position(self, drone) -> None:
        async for p in drone.telemetry.position_velocity_ned():
            self.position_xy = (p.position.north_m, p.position.east_m)
