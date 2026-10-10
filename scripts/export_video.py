"""Render the 30 s cinematic film: PNG frames from the running TouchDesigner project, encoded with ffmpeg (H.264).

    python scripts/export_video.py --port 9990 --out output/videos/deep_ocean_robot_vision_4d.mp4
Requires: TouchDesigner open with the built project (touchdesigner/build_project.py) and ffmpeg on PATH.
The non-commercial TouchDesigner licence caps TOP resolution at 1280 px, so the default film is 1280x720 @ 30 fps;
a 1920x1080 / 4K render needs a commercial licence (set render3d resolution, then --size).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def td_exec(port, code, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/td/server/exec",
                                 data=json.dumps({"script": code}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    if d.get("error"):
        raise RuntimeError(d["error"])
    return (d.get("data") or {}).get("stdout", "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9990)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "videos", "deep_ocean_robot_vision_4d.mp4"))
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--start", type=int, default=0, help="first frame to (re)render; earlier PNGs are kept")
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    frames_dir = os.path.join(ROOT, "output", "videos", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    mod = "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/OUTPUT/export_controller').module"
    if not a.skip_render:
        print(td_exec(a.port, f"print({mod}.begin())"))
        for s in range(a.start, a.frames, a.chunk):
            e = min(s + a.chunk, a.frames)
            print(td_exec(a.port, f"print({mod}.render_range({s}, {e}, r'{frames_dir}'))"))
            sys.stdout.flush()
        print(td_exec(a.port, f"print({mod}.end())"))
    ff = shutil.which("ffmpeg")
    if not ff:
        sys.exit("ffmpeg not found on PATH; frames are in " + frames_dir)
    cmd = [ff, "-y", "-framerate", str(a.fps), "-i", os.path.join(frames_dir, "frame_%04d.png"), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", a.out]
    subprocess.check_call(cmd)
    print("wrote", a.out)
