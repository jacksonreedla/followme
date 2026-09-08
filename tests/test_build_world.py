from pathlib import Path

import pytest
import yaml

from sim import build_world

SCEN = Path(__file__).resolve().parents[1] / "sim" / "scenarios"


@pytest.mark.skipif(not build_world.DEFAULT.exists(), reason="PX4 tree not cloned")
@pytest.mark.parametrize("name", [p.stem for p in SCEN.glob("*.yaml")])
def test_scenario_renders_valid_world(name):
    sdf = build_world.build(yaml.safe_load((SCEN / f"{name}.yaml").read_text()))
    assert '<world name="follow">' in sdf
    assert sdf.count("<actor ") == len(yaml.safe_load((SCEN / f"{name}.yaml").read_text())["actors"])
    assert sdf.rstrip().endswith("</sdf>")


def test_actor_faces_next_waypoint():
    xml = build_world.actor_sdf("a", [[0, 0, 0], [0, 5, 5]])
    assert "1.5708" in xml  # atan2(5, 0)
