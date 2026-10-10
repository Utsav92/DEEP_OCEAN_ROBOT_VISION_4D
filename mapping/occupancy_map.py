"""3D occupancy (log-odds) from sonar ray geometry. Water between the sonar and a valid return is observed FREE; the
return voxel is observed OCCUPIED; everything not crossed by a ray remains UNKNOWN (never assumed free)."""
import numpy as np

L_FREE, L_OCC, L_CLAMP = -0.4, 0.85, 4.0


class OccupancyMap:
    def __init__(self, extent_xy, z_range, voxel=10.0):
        self.v = float(voxel)
        self.x0, self.y0, self.z0 = extent_xy[0], extent_xy[2], z_range[0]
        self.nx = int(np.ceil((extent_xy[1] - extent_xy[0]) / voxel)) + 1
        self.ny = int(np.ceil((extent_xy[3] - extent_xy[2]) / voxel)) + 1
        self.nz = int(np.ceil((z_range[1] - z_range[0]) / voxel)) + 1
        self.L = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.hits = np.zeros(self.L.shape, dtype=np.uint16)

    def _idx(self, p):
        ix = np.floor((p[..., 0] - self.x0) / self.v).astype(np.int64)
        iy = np.floor((p[..., 1] - self.y0) / self.v).astype(np.int64)
        iz = np.floor((p[..., 2] - self.z0) / self.v).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (iz >= 0) & (iz < self.nz)
        return iz, iy, ix, ok

    def integrate(self, origins, endpoints, chunk=40000):
        """origins (n,3), endpoints (n,3): rays from the sonar head to a valid return (world frame)."""
        n_samp = 12
        s = (np.arange(n_samp) + 0.5) / n_samp * 0.97             # stop short of the return voxel
        for a in range(0, len(origins), chunk):
            o, e = origins[a:a + chunk], endpoints[a:a + chunk]
            P = o[:, None, :] + s[None, :, None] * (e - o)[:, None, :]
            iz, iy, ix, ok = self._idx(P)
            lin = np.where(ok, np.ravel_multi_index((np.clip(iz, 0, self.nz - 1), np.clip(iy, 0, self.ny - 1),
                                                    np.clip(ix, 0, self.nx - 1)), self.L.shape), -1)
            lin = np.sort(lin, axis=1)                          # each voxel counts ONCE per ray, once per distinct ray
            first = np.ones_like(lin, dtype=bool)
            first[:, 1:] = lin[:, 1:] != lin[:, :-1]
            fl = lin[first & (lin >= 0)]
            np.add.at(self.L.reshape(-1), fl, L_FREE)
            iz, iy, ix, ok = self._idx(e)
            oc = np.ravel_multi_index((iz[ok], iy[ok], ix[ok]), self.L.shape)
            np.add.at(self.L.reshape(-1), oc, L_OCC)
            np.add.at(self.hits.reshape(-1), oc, 1)
        np.clip(self.L, -L_CLAMP, L_CLAMP, out=self.L)

    def classify(self, free_thr=-0.5, occ_thr=0.5):
        """0 unknown, 1 free, 2 occupied."""
        c = np.zeros(self.L.shape, dtype=np.uint8)
        c[self.L <= free_thr] = 1
        c[self.L >= occ_thr] = 2
        return c

    def stats(self):
        c = self.classify()
        n = c.size
        return {"voxel_m": self.v, "shape": list(self.L.shape), "unknown_fraction": float((c == 0).sum() / n),
                "free_fraction": float((c == 1).sum() / n), "occupied_fraction": float((c == 2).sum() / n),
                "note": "unobserved water volume is UNKNOWN, never free"}
