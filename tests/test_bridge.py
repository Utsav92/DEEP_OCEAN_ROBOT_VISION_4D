import json
import os
import sys
import threading

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "touchdesigner"))
import data_bridge as db   # noqa: E402


def _pk(i):
    return db.FramePacket(seq=i, timestamp_s=i * 2.0, sensor_pose=list(np.eye(4).ravel()), vehicle_pose=list(np.eye(4).ravel()),
                          pointcloud_ref={"file": "p.npy", "slots": [0, 1]}, attribute_ref={"file": "a.npy", "slots": [0, 1]},
                          map_version=0, validity={"pose": True})


def test_packet_roundtrip_has_required_fields():
    p = db.FramePacket.from_json(_pk(3).to_json())
    d = json.loads(p.to_json())
    for k in ("protocol", "seq", "timestamp_s", "frame", "sensor_pose", "vehicle_pose", "pointcloud_ref",
              "attribute_ref", "map_version", "validity"):
        assert k in d
    assert d["protocol"] == 1 and p.seq == 3


def test_bad_protocol_and_missing_field_rejected():
    d = json.loads(_pk(0).to_json())
    d["protocol"] = 99
    with pytest.raises(ValueError):
        db.FramePacket.from_json(json.dumps(d))
    d = json.loads(_pk(0).to_json())
    del d["vehicle_pose"]
    with pytest.raises(ValueError):
        db.FramePacket.from_json(json.dumps(d))


def test_receiver_reorders_and_reports_missing_and_duplicates():
    rx = db.PacketReceiver(window=4)
    out = []
    for s in (0, 2, 1, 3, 3, 6, 7, 8, 9, 10):          # 4 and 5 never arrive; 3 is duplicated
        out += [p.seq for p in rx.push(_pk(s))]
    out += [p.seq for p in rx.flush()]
    assert out == sorted(out)
    assert out[:4] == [0, 1, 2, 3]
    assert rx.stats["late_dropped"] == 1 and rx.stats["lost"] == 2   # re-sent seq 3 arrives after delivery
    assert 4 not in out and 5 not in out


def test_receiver_drops_stale_packets():
    rx = db.PacketReceiver(window=4)
    for s in range(5):
        rx.push(_pk(s))
    assert rx.push(_pk(1)) == []
    assert rx.stats["late_dropped"] == 1


def _fake_store(tmp_path, n=30):
    d = tmp_path / "pc"
    d.mkdir()
    np.save(d / "ping_t.npy", np.arange(n) * 2.0)
    np.save(d / "vehicle_T_slam.npy", np.tile(np.eye(4), (n, 1, 1)))
    np.save(d / "vehicle_T_dr.npy", np.tile(np.eye(4), (n, 1, 1)))
    np.savez(d / "versions.npz", t_end=np.array([20.0, 60.0]))
    json.dump({"n_pings": n}, open(d / "td_meta.json", "w"))
    return db.DatasetStore(str(d))


def test_socket_stream_handles_scramble_and_drops(tmp_path):
    store = _fake_store(tmp_path, 40)
    ready, stop = threading.Event(), threading.Event()
    port = 9877
    th = threading.Thread(target=db.serve, daemon=True,
                          kwargs=dict(store=store, port=port, rate_hz=0, stop=40, scramble=5, drop_every=10,
                                      stop_event=stop, ready=ready))
    th.start()
    assert ready.wait(5)
    cl = db.SocketClient(port=port, window=8)
    cl.start()
    cl.join(20)
    got = [p.seq for p in cl.take()]
    assert got == sorted(got) and len(got) >= 30
    assert all((s % 10) != 9 for s in got)               # dropped packets stay dropped (never invented)
    assert cl.rx.stats["delivered"] == len(got)


def test_pose_at_interpolates_between_pings(tmp_path):
    store = _fake_store(tmp_path, 10)
    arr = np.load(os.path.join(store.dir, "vehicle_T_slam.npy"))
    arr[:, 0, 3] = np.arange(10) * 10.0
    np.save(os.path.join(store.dir, "vehicle_T_slam.npy"), arr)
    store2 = db.DatasetStore(store.dir)
    T = db.pose_at(store2, 5.0, mix=1.0)
    assert T[0, 3] == pytest.approx(25.0)
