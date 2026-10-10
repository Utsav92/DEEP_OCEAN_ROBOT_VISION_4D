"""Render one inspection frame inside TouchDesigner and save it as PNG.

usage:  python touchdesigner/tools/shot.py NAME [KEY=VALUE ...]
  keys : t=<data seconds>  cam=<1..6>  style=<0|1>  color=<1..5>  temporal=<1..4>  mix=<0..1>  exag=<f>
         hist=<scans> history  fov=<deg>  terrain=<1|2>  eye=x,y,z target=x,y,z (scripted camera)  w=<px> h=<px>
Prints MAT/TOP/SOP errors found in the network first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdrun  # noqa: E402

CODE = r'''
import os
base = op('/project1/DEEP_OCEAN_ROBOT_VISION_4D')
S = __SETTINGS__
panel = base.op('UI_CONTROLS/panel')
tl = base.op('DATA_INPUT/timeline_controller').module
cc = base.op('CAMERA_SYSTEM/camera_controller').module
tl.pause()
if 't' in S: tl.set_time(float(S['t']))
def radio(prefix, idx):
    for i in range(1, 9):
        w = panel.op('%s%d' % (prefix, i))
        if w is not None: w.par.value0 = 1 if i == int(idx) else 0
for k_, v_ in S.items():
    if k_.startswith('t_') or k_.startswith('s_'):
        w_ = panel.op(k_)
        if w_ is not None: w_.par.value0 = float(v_)
if 'cam' in S: radio('r_camera', S['cam'])
if 'style' in S: radio('r_style', int(S['style']) + 1)
if 'color' in S: radio('r_color', S['color'])
if 'temporal' in S: radio('r_temporal', S['temporal'])
if 'terrain' in S: radio('r_terrain', S['terrain'])
def slider(k, v):
    w = panel.op('s_' + k)
    if w is not None: w.par.value0 = float(v)
for k in ('mix:slam_mix', 'exag:exaggeration', 'hist:scan_history', 'fov:cam_fov', 'gain:gain', 'pt:point_size'):
    a, b = k.split(':')
    if a in S: slider(b, S[a])
if 'fov' in S:
    mc = base.op('UI_CONTROLS/main_controls')
    mc.seq.const[[c.name for c in mc.chans()].index('cam_fov')].par.value = float(S['fov'])
import numpy as np
if 'eye' in S:
    eye = np.array([float(v) for v in S['eye'].split(',')]); tgt = np.array([float(v) for v in S['target'].split(',')])
    cc.set_override(cc.look_at(eye, tgt), float(S.get('fov', 60)))
else:
    cc.clear_override()
if 'w' in S:
    r = base.op('OUTPUT/render3d'); r.par.resolutionw = int(S['w']); r.par.resolutionh = int(S['h'])
x = 1
'''

CODE2 = r'''
import os
base = op('/project1/DEEP_OCEAN_ROBOT_VISION_4D')
cc = base.op('CAMERA_SYSTEM/camera_controller').module
cc.update()
cc._st['from_W'] = None
cc.update()
for p in ('ACOUSTIC_BEAMS/beam_geometry/sop','UNDERWATER_SLAM/loop_closures/sop','UNDERWATER_SLAM/pose_graph/sop'):
    base.op(p).cook(force=True)
op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/vehicle_renderer').module.apply_pose(base)
out = base.op('OUTPUT/OUT')
base.op('OUTPUT/render3d').cook(force=True)
out.cook(force=True)
errs = []
def walk(c):
    for ch in c.children:
        try:
            e = ch.errors()
        except Exception:
            e = ''
        if e: errs.append((ch.path.replace('/project1/DEEP_OCEAN_ROBOT_VISION_4D/', ''), e[:300]))
        if ch.isCOMP: walk(ch)
walk(base)
path = r'__PATH__'
os.makedirs(os.path.dirname(path), exist_ok=True)
out.save(path)
print('errors:', errs)
print('saved', path, out.width, out.height)
x = 1
'''


def main():
    name = sys.argv[1]
    st = dict(a.split("=", 1) for a in sys.argv[2:])
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output",
                        "screenshots", name + ".png")
    tdrun.main(["-c", CODE.replace("__SETTINGS__", repr(st))])
    tdrun.main(["-c", CODE2.replace("__PATH__", path)])
    print(path)


if __name__ == "__main__":
    main()
