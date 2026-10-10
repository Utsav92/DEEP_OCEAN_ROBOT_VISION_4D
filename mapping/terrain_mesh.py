"""Terrain triangulation and mesh export (OBJ + NPZ). Triangles are emitted only where all vertices carry data, so
cells without observations stay holes instead of being bridged with invented surface."""
import json
import os

import numpy as np


def grid_triangles(valid):
    """Two triangles per grid cell whose four corners are valid. valid: (ny, nx) bool.
    Returns (M,3) int32 indices into the flattened (ny*nx) vertex array (row-major)."""
    ny, nx = valid.shape
    idx = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a, b, c, d = idx[:-1, :-1][ok], idx[:-1, 1:][ok], idx[1:, :-1][ok], idx[1:, 1:][ok]
    # counter-clockwise seen from above (+z): a(x,y) -> b(x+1,y) -> d(x+1,y+1) ; a -> d -> c
    tris = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)], 0)
    return tris.astype(np.int32)


def terrain_mesh(grid, step=1, z_override=None):
    """Mesh from a MetricGrid (ENU: x east, y north, z up). Returns dict(vertices, triangles, normals, valid_mask)."""
    z = grid.z if z_override is None else z_override
    z = z[::step, ::step]
    xs, ys = grid.x[::step], grid.y[::step]
    X, Y = np.meshgrid(xs, ys)
    valid = np.isfinite(z)
    V = np.stack([X, Y, np.where(valid, z, 0.0)], -1).reshape(-1, 3)
    T = grid_triangles(valid)
    N = vertex_normals(V, T)
    return {"vertices": V.astype(np.float32), "triangles": T, "normals": N.astype(np.float32),
            "valid": valid, "x": xs, "y": ys, "step": step}


def vertex_normals(V, T):
    N = np.zeros_like(V, dtype=np.float64)
    e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    fn = np.cross(e1, e2)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.where(n > 1e-12, N / np.maximum(n, 1e-12), np.array([0.0, 0.0, 1.0]))
    return N


def export_obj(mesh, path, comment="terrain mesh, local ENU metres (x east, y north, z up)"):
    V, T, N = mesh["vertices"], mesh["triangles"], mesh["normals"]
    used = np.zeros(len(V), bool)
    used[T.ravel()] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(used.sum())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {comment}\n")
        for v in V[used]:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for n in N[used]:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for t in remap[T] + 1:
            f.write(f"f {t[0]}//{t[0]} {t[1]}//{t[1]} {t[2]}//{t[2]}\n")
    return int(used.sum()), int(len(T))


def read_obj(path):
    V, T = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(s) for s in line.split()[1:4]])
            elif line.startswith("f "):
                T.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(V), np.array(T)


def export_mesh_bundle(mesh, out_dir, name, meta):
    os.makedirs(out_dir, exist_ok=True)
    nv, nt = export_obj(mesh, os.path.join(out_dir, name + ".obj"))
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), vertices=mesh["vertices"], triangles=mesh["triangles"],
                        normals=mesh["normals"], valid=mesh["valid"], x=mesh["x"], y=mesh["y"])
    meta = dict(meta, vertices=nv, triangles=nt, obj=name + ".obj")
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
