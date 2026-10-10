"""Mode 6 / streaming: serve frame packets from the exported dataset over localhost TCP (JSON lines), or run a
self-test with a local client that deliberately receives shuffled and dropped packets.

    python scripts/start_bridge.py --config config.yaml                 # serve until a client connects
    python scripts/start_bridge.py --config config.yaml --selftest      # loopback test with scrambled/dropped packets
"""
import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "touchdesigner"))
from preprocessing.config import load_config, abspath   # noqa: E402
import data_bridge as db                                 # noqa: E402


def selftest(store, port, n=120):
    ready, stop = threading.Event(), threading.Event()
    srv = threading.Thread(target=db.serve, daemon=True,
                           kwargs=dict(store=store, port=port, rate_hz=0, start=0, stop=n, scramble=6, drop_every=17,
                                       stop_event=stop, ready=ready))
    srv.start()
    ready.wait(5)
    cl = db.SocketClient(port=port, window=8)
    cl.start()
    cl.join(30)
    got = cl.take()
    seqs = [p.seq for p in got]
    print(f"sent 0..{n - 1} with scramble=6, drop_every=17; delivered {len(got)} packets in order={seqs == sorted(seqs)}")
    print("receiver stats:", cl.rx.stats, "error:", cl.error)
    return seqs == sorted(seqs) and len(got) > 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rate", type=float, default=5.0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    store = db.DatasetStore(os.path.join(abspath(cfg, cfg["paths"]["output"]), "pointclouds"))
    port = int(cfg["bridge"]["port"])
    if a.selftest:
        sys.exit(0 if selftest(store, port) else 1)
    print(f"serving {store.n_pings} packets on {cfg['bridge']['host']}:{port} at {a.rate} Hz (waiting for one client)")
    db.serve(store, cfg["bridge"]["host"], port, rate_hz=a.rate)
