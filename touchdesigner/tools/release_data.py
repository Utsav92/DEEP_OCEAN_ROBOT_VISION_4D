"""Release the memory-mapped exports held by the running TouchDesigner (Windows cannot overwrite mapped files)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdrun

CODE = r'''
import gc
b = op('/project1/DEEP_OCEAN_ROBOT_VISION_4D')
if b is not None:
    m = b.op('DATA_INPUT/data_bridge')
    if m is not None:
        mod = m.module
        mod._stores.clear()
    for k in ('attr_t_sorted', 'recon_tfirst_sorted'):
        try:
            b.unstore(k)
        except Exception:
            pass
    tm = b.op('DATA_INPUT/timeline_controller')
    if tm is not None:
        tm.module._g['scrub'] = None
    vr = op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/vehicle_renderer')
    if vr is not None:
        vr.module._cache.clear()
gc.collect()
x = 1
'''
if __name__ == "__main__":
    tdrun.main(["-c", CODE])
    print("released")
