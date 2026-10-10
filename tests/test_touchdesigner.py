"""TouchDesigner checks. The pure parts (storyboard, camera maths) always run; the live-network checks need my TD
instance (touchdesigner/base.tox, port 9990) with the project built, and are SKIPPED - not passed - otherwise."""
import json
import os
import sys
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "touchdesigner"))
import camera_controller as cc          # noqa: E402
import export_controller as ec          # noqa: E402
import temporal_renderer as tr          # noqa: E402
import ui_controller as uc              # noqa: E402
import beam_renderer as br              # noqa: E402

URL = "http://127.0.0.1:9990/api/td/server/exec"
BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"


def td(code):
    req = urllib.request.Request(URL, data=json.dumps({"script": code + "\nx = 1\n"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    if d.get("error"):
        raise RuntimeError(d["error"])
    return d["data"]


def live():
    try:
        return td(f"result = op('{BASE}') is not None")["result"] is True
    except Exception:
        return False


needs_td = pytest.mark.skipif(not live(), reason="TouchDesigner instance on port 9990 with the built project not reachable")


# ----------------------------------------------------------------------------- pure parts
def test_storyboard_covers_30_seconds_with_required_titles():
    titles = {ec.storyboard(t, 13800.0)["title"] for t in np.arange(0, 30, 0.5)}
    for want in ("DEEP OCEAN ROBOT VISION 4D", "ACOUSTIC SPATIAL PERCEPTION", "BATHYMETRIC RECONSTRUCTION",
                 "UNDERWATER NAVIGATION AND SLAM", "X / Y / Z / TIME", "PERSISTENT OCEAN MEMORY"):
        assert want in titles
    assert ec.SCENES[0][0] == 0.0 and ec.SCENES[-1][1] == 30.0 and ec.FPS == 30
    assert ec.SCENES[-1][3] == "EXPLORING THE INVISIBLE OCEAN"
    s = ec.storyboard(14.0, 13800.0)
    assert s["show_loops"] == 1 and 0 <= s["slam_mix"] <= 1


def test_storyboard_is_deterministic_and_monotone_in_data_time():
    a = [ec.storyboard(t, 10000.0)["data_time"] for t in np.arange(0, 24, 0.25)]
    assert a == [ec.storyboard(t, 10000.0)["data_time"] for t in np.arange(0, 24, 0.25)]
    assert all(y >= x - 1e-9 for x, y in zip(a, a[1:]))


def test_camera_look_at_and_blend():
    W = cc.look_at([0, 100, 0], [0, 0, 0], up=np.array([0, 0, -1.0]))
    assert np.allclose(-W[:3, 2], [0, -1, 0], atol=1e-9)                   # looks straight down
    A, B = cc.look_at([10, 0, 0], [0, 0, 0]), cc.look_at([0, 0, 10], [0, 0, 0])
    M = cc.blend_pose(A, B, 0.5)
    assert np.allclose(M[:3, 3], [5, 0, 5])
    assert np.allclose(cc.blend_pose(A, B, 0.0), A) and np.allclose(cc.blend_pose(A, B, 1.0), B)


def test_free_camera_moves_with_keys_independent_of_time():
    f = cc.FreeCam()
    f.pos = np.array([0.0, 0.0, 0.0])
    f.step(1.0, fwd=1.0, speed=100.0)
    assert np.allclose(f.pos, [0, 0, -100.0])                              # forward = -z at yaw 0
    f.step(1.0, vert=1.0, speed=50.0)
    assert f.pos[1] == pytest.approx(50.0)


def test_cinematic_path_stays_above_floor_and_is_smooth():
    eyes, tgts = cc.cinematic_path([-1200, 1200, -1700, 1700], -250.0, 2.0, y_top=210.0)
    pts = np.array([cc.catmull_rom(eyes, u) for u in np.linspace(0, 1, 200)])
    assert pts[:, 1].min() > 210.0 * 2.0                                  # never under the highest terrain point
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() < 120.0


def test_temporal_ui_parameters():
    assert tr.nearest_history(60) == 50 and tr.nearest_history(400) == 500 and tr.nearest_history(7) == 5
    u = tr.uniforms(4, 100, 40.0, 3.0)
    assert u["visible"] and u["uExplodeMps"] == 3.0 and u["uHistoryS"] == 200.0
    assert tr.uniforms(2, 100, 40.0, 3.0)["uExplodeMps"] == 0.0              # echo mode never displaces points
    assert not tr.uniforms(1, 100, 40.0, 3.0)["visible"]


def test_scientific_style_has_no_cinematic_post_effects():
    p = uc.post_params(0, 1, 0.9, 1)
    assert p["uSparkle"] == 0.0 and p["uScan"] == 0.0 and p["uChroma"] == 0.0
    c = uc.post_params(1, 1, 0.9, 1)
    assert c["uSparkle"] > 0 and c["uGlow"] > p["uGlow"]


def test_beam_fan_geometry_skips_holes_and_respects_swath():
    th = np.radians(np.linspace(-65, 65, 11))
    ends = np.zeros((11, 4))
    ends[:, 3] = 1.0
    ends[4:7, 3] = 0.0                                                      # 3-beam hole
    pairs, ok = br.fan_triangles(np.zeros(3), ends, th, 130.0)
    assert (3, 7) not in pairs and (2, 3) in pairs and len(pairs) == 10 - 4
    pairs2, ok2 = br.fan_triangles(np.zeros(3), ends, th, 60.0)
    assert ok2.sum() < ok.sum()


def test_glsl_sources_have_no_unbalanced_braces_and_required_entrypoints():
    d = os.path.join(ROOT, "shaders")
    files = [f for f in os.listdir(d) if f.endswith((".vert", ".frag"))]
    assert len(files) >= 13
    for f in files:
        s = open(os.path.join(d, f)).read()
        assert s.count("{") == s.count("}"), f
        assert "void main" in s, f
        if f.endswith(".frag") and f not in ("glow.frag", "underwater_fog.frag"):
            assert "TDOutputSwizzle" in s and "fragColor" in s, f


# ----------------------------------------------------------------------------- live network
@needs_td
def test_operators_exist_per_spec():
    want = ["DATA_INPUT/bathymetry_loader", "DATA_INPUT/sonar_loader", "DATA_INPUT/navigation_loader",
            "DATA_INPUT/calibration_loader", "DATA_INPUT/timeline_controller", "SONAR_RENDERER/sonar_positions",
            "SONAR_RENDERER/sonar_attributes", "SONAR_RENDERER/sonar_geometry", "SONAR_RENDERER/sonar_material",
            "BATHYMETRY_RENDERER/terrain_loader", "BATHYMETRY_RENDERER/terrain_geometry",
            "BATHYMETRY_RENDERER/terrain_material", "BATHYMETRY_RENDERER/depth_coloring", "ACOUSTIC_BEAMS/beam_geometry",
            "ACOUSTIC_BEAMS/beam_animation", "ACOUSTIC_BEAMS/beam_material", "VEHICLE_NAVIGATION/vehicle_geometry",
            "VEHICLE_NAVIGATION/uncertainty_geometry", "UNDERWATER_SLAM/pose_graph", "UNDERWATER_SLAM/loop_closures",
            "UNDERWATER_SLAM/map_versions", "OBJECT_PERCEPTION/object_candidates", "OBJECT_PERCEPTION/object_geometry",
            "OBJECT_PERCEPTION/object_labels", "TEMPORAL_4D/sonar_history", "TEMPORAL_4D/scan_echoes",
            "TEMPORAL_4D/temporal_points", "TEMPORAL_4D/time_explosion", "TEMPORAL_4D/time_scrubber",
            "CAMERA_SYSTEM/vehicle_pov", "CAMERA_SYSTEM/chase_camera", "CAMERA_SYSTEM/orbit_camera",
            "CAMERA_SYSTEM/free_camera", "CAMERA_SYSTEM/overhead_camera", "CAMERA_SYSTEM/cinematic_camera",
            "UI_CONTROLS/sonar_controls", "UI_CONTROLS/terrain_controls", "UI_CONTROLS/temporal_controls",
            "UI_CONTROLS/camera_controls", "UI_CONTROLS/diagnostics", "OUTPUT/render3d", "OUTPUT/underwater_atmosphere",
            "OUTPUT/bloom", "OUTPUT/final_composite", "OUTPUT/OUT", "OUTPUT/movie_output"]
    r = td(f"b = op('{BASE}')\nresult = [p for p in {want!r} if b.op(p) is None]")["result"]
    assert r == []


@needs_td
def test_geometry_textures_loaded_with_expected_sizes():
    r = td(f"b = op('{BASE}')\nresult = {{n: [b.op(n).width, b.op(n).height] for n in "
           "['SONAR_RENDERER/sonar_positions', 'SONAR_RENDERER/sonar_attributes', 'BATHYMETRY_RENDERER/terrain_loader',"
           "'BATHYMETRY_RENDERER/recon_loader', 'VEHICLE_NAVIGATION/traj_slam']}")["result"]
    assert r["SONAR_RENDERER/sonar_positions"] == [512, 512]
    assert r["BATHYMETRY_RENDERER/terrain_loader"] == [512, 512]
    assert r["VEHICLE_NAVIGATION/traj_slam"] == [160, 160]


@needs_td
def test_shaders_compile_without_errors():
    code = (f"b = op('{BASE}')\nres = {{}}\n"
            "def walk(c):\n"
            "    for ch in c.children:\n"
            "        if ch.OPType in ('glslMAT', 'glslTOP'): res[ch.path] = ch.errors()\n"
            "        if ch.isCOMP: walk(ch)\n"
            "walk(b)\nresult = res")
    r = td(code)["result"]
    assert len(r) >= 8, f"expected >= 8 GLSL operators, found {len(r)}"
    assert all(v == "" for v in r.values()), {k: v for k, v in r.items() if v}


@needs_td
def test_camera_modes_switch_and_move_the_camera():
    code = (f"b = op('{BASE}')\ncc = b.op('CAMERA_SYSTEM/camera_controller').module\n"
            "tl = b.op('DATA_INPUT/timeline_controller').module\ntl.pause(); tl.set_time(5000.0)\n"
            "cc.clear_override()\nimport numpy as np\npos = {}\n"
            "for m in (1, 2, 3, 5, 6):\n"
            "    W, fov = cc.target_pose(b, m, 5000.0)\n    pos[m] = [round(float(v), 1) for v in W[:3, 3]]\n"
            "result = pos")
    r = td(code)["result"]
    pts = [tuple(v) for v in r.values()]
    assert len(set(pts)) == len(pts)                                           # five distinct camera positions


@needs_td
def test_timeline_controls_play_reverse_step_freeze():
    code = (f"b = op('{BASE}')\ntl = b.op('DATA_INPUT/timeline_controller').module\n"
            "tl.pause(); tl.set_time(1000.0)\nout = {}\n"
            "tl.step(3); out['step'] = tl.state()['scan']\n"
            "tl.set_time(1000.0); k0 = tl.state()['scan']; tl.step(-2); out['back'] = k0 - tl.state()['scan']\n"
            "tl.reverse(); out['dir'] = tl.state()['direction']; tl.play(); out['dir2'] = tl.state()['direction']\n"
            "tl.freeze(True); out['frozen'] = tl.state()['frozen']; tl.freeze(False); tl.pause(); tl.reset()\n"
            "out['t_reset'] = tl.state()['t']\nresult = out")
    r = td(code)["result"]
    assert r["back"] == 2 and r["dir"] == -1 and r["dir2"] == 1 and r["frozen"] is True and r["t_reset"] == 0.0


@needs_td
def test_video_export_configuration(tmp_path):
    r = td(f"b = op('{BASE}')\nm = b.op('OUTPUT/movie_output')\n"
           "result = {'type': m.OPType, 'frames': b.op('OUTPUT/export_controller').module.FPS * "
           "b.op('OUTPUT/export_controller').module.DURATION}")["result"]
    assert r["type"] == "moviefileoutTOP" and r["frames"] == 900
    out = str(tmp_path).replace("\\", "/")
    td(f"b = op('{BASE}')\nec = b.op('OUTPUT/export_controller').module\nec.begin()\nec.render_frame(60, r'{out}')\nec.end()")
    assert os.path.getsize(os.path.join(str(tmp_path), "frame_0060.png")) > 10000
