"""Render a Gazebo world for a scenario and install it where PX4 looks for worlds.

    .venv/bin/python sim/build_world.py sim/scenarios/straight.yaml
    -> ~/PX4-Autopilot/Tools/simulation/gz/worlds/follow.sdf

The world = PX4's default.sdf (physics/plugins/ground/sun) + our actors + occluders.
Actors use the Gazebo Fuel walking mesh; a trajectory is a list of [x, y, t_s]
waypoints (yaw is set to face the next waypoint by Gazebo's auto-orientation).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

PX4 = Path.home() / "PX4-Autopilot"
DEFAULT = PX4 / "Tools/simulation/gz/worlds/default.sdf"
WALK = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"


def actor_sdf(name: str, waypoints: list[list[float]], start_yaw: float = 0.0) -> str:
    wps = []
    for i, (x, y, t) in enumerate(waypoints):
        if i + 1 < len(waypoints):
            nx, ny = waypoints[i + 1][0], waypoints[i + 1][1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = wps[-1][1] if wps else start_yaw
        wps.append((t, yaw, x, y))
    wp_xml = "\n".join(
        f"        <waypoint><time>{t}</time><pose>{x} {y} 1.0 0 0 {yaw:.4f}</pose></waypoint>"
        for t, yaw, x, y in wps)
    return f"""
    <actor name="{name}">
      <skin><filename>{WALK}</filename><scale>1.0</scale></skin>
      <animation name="walking"><filename>{WALK}</filename><interpolate_x>true</interpolate_x></animation>
      <script><loop>false</loop><auto_start>true</auto_start>
        <trajectory id="0" type="walking">
{wp_xml}
        </trajectory>
      </script>
    </actor>"""


def box_sdf(name: str, pose: list[float], size: list[float]) -> str:
    p = " ".join(str(v) for v in pose); s = " ".join(str(v) for v in size)
    return f"""
    <model name="{name}"><static>true</static><pose>{p}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{s}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{s}</size></box></geometry>
          <material><ambient>0.5 0.4 0.3 1</ambient><diffuse>0.6 0.5 0.4 1</diffuse></material></visual>
      </link></model>"""


def build(scenario: dict, world_name: str = "follow") -> str:
    base = DEFAULT.read_text()
    base = base.replace('<world name="default">', f'<world name="{world_name}">')
    extra = "".join(actor_sdf(a["name"], a["waypoints"]) for a in scenario.get("actors", []))
    extra += "".join(box_sdf(b["name"], b["pose"], b["size"]) for b in scenario.get("boxes", []))
    return base.replace("  </world>", extra + "\n  </world>")


def main(path: str, out: Path | None = None) -> Path:
    scenario = yaml.safe_load(Path(path).read_text())
    out = out or DEFAULT.with_name("follow.sdf")
    out.write_text(build(scenario))
    return out


if __name__ == "__main__":
    print(main(sys.argv[1]))
