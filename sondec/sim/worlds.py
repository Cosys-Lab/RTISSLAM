"""Procedural 3D worlds, point-sampled for echo synthesis.

A world is a set of reflector points (the ground truth, EVALUATION ONLY) plus occluders:

    pos     [N, 3]  m
    normal  [N, 3]  outward surface normal; zero for isotropic scatterers (edges, leaves)
    area    [N]     m^2 per surface point (m per edge point, "equivalent area" per leaf)
    mat     [N, 3]  backscatter (d, s, p):  sigma = area * (d cos^2 + s cos^(2p))  (cos = incidence)
                    isotropic points: sigma = area * d
    obj     [N]     object index; per object a size a (m, Rayleigh roll-off (ka)^2 / (1 + (ka)^2) in
                    amplitude; inf = large) and a spectral tilt (dB across 30-90 kHz)
    two     [N]     two-sided surface (thin boards): |cos| instead of max(cos, 0)

Occluders (shadowing): oriented boxes (centre, half extents, yaw) and capsules (p0, p1, radius).
The room shell (floor, walls, ceiling) does not occlude (it bounds the space).

Floor footprint (for route planning): occupancy of everything below 0.65 m.

Simulator version 2 (28 Sep 2026, user request): sparser echo trains and more point-like
reflectors at the sensor's height.
- Surfaces are sampled at 60 instead of 220 points/m^2 (the same power per m^2, fewer and
  stronger points), and the diffuse term of the room shell (walls, floor, ceiling) and the forest
  floor is about 4x lower. Specular returns and objects are unchanged.
- Wall fixtures (`_wall_fixtures`), concentrated at 0.1-1.2 m height (the sensor is at ~0.4 m):
  skirting boards, door frames with ridges and handles, sockets and switches, pipes with brackets,
  valves, fire extinguishers, electrical boxes, vertical panel seams. In the forest: knots and
  branch stubs low on the trunks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIM_VERSION = 2
DENSITY = 60.0           # surface points per m^2 (spacing ~13 cm; version 1: 220)
EDGE_DENSITY = 25.0      # edge points per m

MAT = {  # d (diffuse), s (specular), p (lobe exponent)
    "wall": (0.005, 3.0, 30), "floor": (0.008, 1.0, 20), "ceiling": (0.008, 2.0, 30),   # v1: d 0.02/0.03/0.03
    "wood": (0.03, 3.0, 40), "metal": (0.02, 5.0, 60), "cardboard": (0.05, 1.0, 15),
    "fabric": (0.04, 0.3, 8), "glass": (0.005, 6.0, 80), "bark": (0.15, 0.3, 5),
    "litter": (0.02, 0.1, 3), "plastic": (0.03, 3.0, 40), "fixture": (0.12, 5.0, 40), "ridge": (0.12, 0.0, 0), "edge": (0.015, 0.0, 0), "corner": (0.04, 0.0, 0), "leaf": (0.02, 0.0, 0),   # litter v1: d 0.08
}


@dataclass
class World:
    name: str
    kind: str
    bounds: tuple                          # xmin, xmax, ymin, ymax of the drivable region
    pos: list = field(default_factory=list)
    normal: list = field(default_factory=list)
    area: list = field(default_factory=list)
    mat: list = field(default_factory=list)
    obj: list = field(default_factory=list)
    two: list = field(default_factory=list)
    obj_size: list = field(default_factory=list)
    obj_tilt: list = field(default_factory=list)
    obj_name: list = field(default_factory=list)
    boxes: list = field(default_factory=list)       # (cx, cy, cz, hx, hy, hz, yaw)
    capsules: list = field(default_factory=list)    # (x0, y0, z0, x1, y1, z1, r)
    foot_boxes: list = field(default_factory=list)  # (cx, cy, hx, hy, yaw)
    foot_circles: list = field(default_factory=list)  # (cx, cy, r)
    rng: np.random.Generator = None

    # ------------------------------------------------------------------ objects
    def new_obj(self, name: str, size: float = math.inf, tilt: float | None = None) -> int:
        self.obj_size.append(size)
        self.obj_tilt.append(self.rng.normal(0, 2.0) if tilt is None else tilt)
        self.obj_name.append(name)
        return len(self.obj_size) - 1

    def _add(self, p, n, a, mat, obj, two=False):
        k = len(p)
        self.pos.append(p); self.normal.append(n); self.area.append(np.full(k, a) if np.isscalar(a) else a)
        self.mat.append(np.tile(np.asarray(MAT[mat], float), (k, 1)) if isinstance(mat, str) else mat)
        self.obj.append(np.full(k, obj)); self.two.append(np.full(k, two))

    def rect(self, c, u, v, mat, obj, two=False, density=DENSITY, edges=False):
        """Rectangle centre c, half-edge vectors u, v; normal = u x v (normalised)."""
        c, u, v = (np.asarray(x, float) for x in (c, u, v))
        area = 4 * np.linalg.norm(u) * np.linalg.norm(v)
        n_pts = self.rng.poisson(density * area)
        if n_pts > 0:
            ab = self.rng.uniform(-1, 1, (n_pts, 2))
            p = c + ab[:, :1] * u + ab[:, 1:] * v
            nrm = np.cross(u, v); nrm /= np.linalg.norm(nrm)
            self._add(p, np.tile(nrm, (n_pts, 1)), area / n_pts, mat, obj, two)
        if edges:
            corners = [c + u + v, c + u - v, c - u - v, c - u + v]
            for i in range(4):
                self.line(corners[i], corners[(i + 1) % 4], "edge", obj)

    def line(self, a, b, mat, obj, density=EDGE_DENSITY):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        n_pts = max(1, self.rng.poisson(density * L))
        t = self.rng.uniform(0, 1, (n_pts, 1))
        self._add(a + t * (b - a), np.zeros((n_pts, 3)), L / n_pts, mat, obj)

    def box(self, c, h, yaw, mat, name, size=math.inf, bottom=False, occlude=True, edges=True, edge_mat="edge"):
        """Solid box: centre c, half extents h, yaw about z. Outward-facing faces + edges."""
        obj = self.new_obj(name, size)
        c, h = np.asarray(c, float), np.asarray(h, float)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        ex, ey, ez = np.array([cy_, sy_, 0]), np.array([-sy_, cy_, 0]), np.array([0, 0, 1.0])
        X, Y, Z = ex * h[0], ey * h[1], ez * h[2]
        faces = [(c + X, Y, Z), (c - X, Z, Y), (c + Y, Z, X), (c - Y, X, Z), (c + Z, X, Y)]
        if bottom:
            faces.append((c - Z, Y, X))
        for fc, u, v in faces:
            self.rect(fc, u, v, mat, obj)
        if edges:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    self.line(c + sx * X + sy * Y - Z, c + sx * X + sy * Y + Z, edge_mat, obj)
                    self.line(c + sx * X - Y + sy * Z, c + sx * X + Y + sy * Z, edge_mat, obj)
                    self.line(c - X + sx * Y + sy * Z, c + X + sx * Y + sy * Z, edge_mat, obj)
        if occlude:
            self.boxes.append((*c, *h, yaw))
        if c[2] - h[2] < 0.65:
            self.foot_boxes.append((c[0], c[1], h[0], h[1], yaw))
        return obj

    def cylinder(self, p0, p1, r, mat, name, size=None, occlude=True, caps=False):
        """Cylinder surface between p0 and p1 (radius r), outward normals."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        obj = self.new_obj(name, size if size is not None else (r if r < 0.03 else math.inf))
        ax = p1 - p0; L = np.linalg.norm(ax); ax /= L
        tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(ax, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        area = 2 * math.pi * r * L
        n_pts = max(3, self.rng.poisson(DENSITY * area)) if r >= 0.03 else max(2, self.rng.poisson(EDGE_DENSITY * L))
        t = self.rng.uniform(0, 1, n_pts); ph = self.rng.uniform(0, 2 * math.pi, n_pts)
        nrm = np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2
        self._add(p0 + t[:, None] * (p1 - p0) + r * nrm, nrm, area / n_pts, mat, obj)
        if caps:
            for pc, s in ((p1, 1), (p0, -1)):
                k = max(1, self.rng.poisson(DENSITY * math.pi * r * r))
                rr = r * np.sqrt(self.rng.uniform(0, 1, k)); pp = self.rng.uniform(0, 2 * math.pi, k)
                pts = pc + rr[:, None] * (np.cos(pp)[:, None] * e1 + np.sin(pp)[:, None] * e2)
                self._add(pts, np.tile(s * ax, (k, 1)), math.pi * r * r / k, mat, obj)
        if occlude:
            self.capsules.append((*p0, *p1, r))
        if min(p0[2], p1[2]) < 0.65 and abs(ax[2]) > 0.7:
            self.foot_circles.append((p0[0], p0[1], r))
        return obj

    def leaves(self, c, radius, n, name, size=0.004, strength=None, zmin=None):
        """A cluster of small isotropic scatterers (foliage), Gaussian around c."""
        obj = self.new_obj(name, size, tilt=self.rng.normal(2.0, 2.0))
        p = np.asarray(c, float) + self.rng.normal(0, radius / 2, (n, 3))
        if zmin is not None:
            p[:, 2] = np.maximum(p[:, 2], zmin)
        m = np.tile(np.asarray(MAT["leaf"], float) * (1 if strength is None else strength / MAT["leaf"][0]), (n, 1))
        self._add(p, np.zeros((n, 3)), 1.0, m, obj)
        return obj

    # ---------------------------------------------------------------- finalise
    def arrays(self) -> dict:
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0,))  # noqa: E731
        return {
            "pos": cat(self.pos).astype(np.float32), "normal": cat(self.normal).astype(np.float32),
            "area": cat(self.area).astype(np.float32), "mat": cat(self.mat).astype(np.float32),
            "obj": cat(self.obj).astype(np.int32), "two": cat(self.two).astype(bool),
            "obj_size": np.asarray(self.obj_size, np.float32), "obj_tilt": np.asarray(self.obj_tilt, np.float32),
            "obj_name": np.asarray(self.obj_name), "boxes": np.asarray(self.boxes, np.float32).reshape(-1, 7),
            "capsules": np.asarray(self.capsules, np.float32).reshape(-1, 7),
            "foot_boxes": np.asarray(self.foot_boxes, np.float32).reshape(-1, 5),
            "foot_circles": np.asarray(self.foot_circles, np.float32).reshape(-1, 3),
            "bounds": np.asarray(self.bounds, np.float32), "name": self.name, "kind": self.kind,
        }

    def occupancy(self, res: float = 0.05):
        """Floor occupancy grid (True = blocked) over the bounds; returns grid, origin."""
        x0, x1, y0, y1 = self.bounds
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res, y0 + (np.arange(ny) + 0.5) * res, indexing="ij")
        occ = np.zeros((nx, ny), bool)
        for cx, cy, hx, hy, yaw in self.foot_boxes:
            c, s = math.cos(yaw), math.sin(yaw)
            lx = (X - cx) * c + (Y - cy) * s; ly = -(X - cx) * s + (Y - cy) * c
            occ |= (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
        for cx, cy, r in self.foot_circles:
            occ |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        return occ, (x0, y0), res


# ====================================================================== builders
def _room_shell(w: World, W: float, D: float, H: float, wall_mat="wall"):
    """Floor [0,W]x[0,D], ceiling at H, four inward walls, concave corner lines."""
    o = w.new_obj("floor"); w.rect((W / 2, D / 2, 0), (W / 2, 0, 0), (0, D / 2, 0), "floor", o)
    o = w.new_obj("ceiling"); w.rect((W / 2, D / 2, H), (0, D / 2, 0), (W / 2, 0, 0), "ceiling", o)
    o = w.new_obj("walls", tilt=w.rng.normal(0, 1.5))
    w.rect((0, D / 2, H / 2), (0, D / 2, 0), (0, 0, H / 2), wall_mat, o)          # x = 0, normal +x
    w.rect((W, D / 2, H / 2), (0, 0, H / 2), (0, D / 2, 0), wall_mat, o)          # x = W, normal -x
    w.rect((W / 2, 0, H / 2), (0, 0, H / 2), (W / 2, 0, 0), wall_mat, o)          # y = 0, normal +y
    w.rect((W / 2, D, H / 2), (W / 2, 0, 0), (0, 0, H / 2), wall_mat, o)          # y = D, normal -y
    oc = w.new_obj("corners")
    for x in (0, W):
        for y in (0, D):
            w.line((x, y, 0), (x, y, H), "corner", oc)
    for z in (0, H):
        w.line((0, 0, z), (W, 0, z), "corner", oc); w.line((0, D, z), (W, D, z), "corner", oc)
        w.line((0, 0, z), (0, D, z), "corner", oc); w.line((W, 0, z), (W, D, z), "corner", oc)


class _Placer:
    """Rejection placement of footprints (rotated rectangles) with a clearance margin."""

    def __init__(self, W, D, rng):
        self.W, self.D, self.rng, self.items = W, D, rng, []

    def free(self, cx, cy, hx, hy, margin):
        if cx - hx < 0.05 or cy - hy < 0.05 or cx + hx > self.W - 0.05 or cy + hy > self.D - 0.05:
            return False
        for (ax, ay, bx, by) in self.items:
            if abs(cx - ax) < hx + bx + margin and abs(cy - ay) < hy + by + margin:
                return False
        return True

    def place(self, hx, hy, margin=0.9, along_wall=False, tries=200):
        for _ in range(tries):
            if along_wall:
                side = self.rng.integers(4)
                if side < 2:                       # against x = 0 / x = W, long side along y
                    hx, hy = min(hx, hy), max(hx, hy)
                    cx = hx + 0.02 if side == 0 else self.W - hx - 0.02
                    cy = self.rng.uniform(hy + 0.1, self.D - hy - 0.1)
                else:
                    hx, hy = max(hx, hy), min(hx, hy)
                    cy = hy + 0.02 if side == 2 else self.D - hy - 0.02
                    cx = self.rng.uniform(hx + 0.1, self.W - hx - 0.1)
            else:
                cx = self.rng.uniform(hx + 0.5, self.W - hx - 0.5)
                cy = self.rng.uniform(hy + 0.5, self.D - hy - 0.5)
            if self.free(cx, cy, hx, hy, margin):
                self.items.append((cx, cy, hx, hy))
                return cx, cy, hx, hy
        return None


def _table(w, cx, cy, hx, hy, rng, h=0.75):
    th = 0.015
    w.box((cx, cy, h - th), (hx, hy, th), 0.0, "wood", "table top", occlude=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            w.cylinder((cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), 0), (cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), h - 2 * th),
                       0.02, "metal", "table leg")


def _chair(w, cx, cy, yaw, rng):
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in ((0.2, 0.2), (0.2, -0.2), (-0.2, 0.2), (-0.2, -0.2)):
        x, y = cx + c * lx - s * ly, cy + s * lx + c * ly
        w.cylinder((x, y, 0), (x, y, 0.45), 0.012, "metal", "chair leg")
    w.box((cx, cy, 0.46), (0.22, 0.22, 0.02), yaw, "fabric", "chair seat")
    bx, by = cx - c * 0.21, cy - s * 0.21
    w.box((bx, by, 0.72), (0.02, 0.21, 0.22), yaw, "fabric", "chair back")


def _shelf(w, cx, cy, hx, hy, H, n_boards, rng, mat="wood", fill=0.5):
    """Open shelf: two side panels, back, boards, random boxes/books on the boards."""
    long_x = hx >= hy
    t = 0.01
    if long_x:
        for sx in (-1, 1):
            w.box((cx + sx * (hx - t), cy, H / 2), (t, hy, H / 2), 0, mat, "shelf side")
    else:
        for sy in (-1, 1):
            w.box((cx, cy + sy * (hy - t), H / 2), (hx, t, H / 2), 0, mat, "shelf side")
    zs = np.linspace(0.05, H - 0.02, n_boards)
    for z in zs:
        w.box((cx, cy, z), (hx, hy, t), 0, mat, "shelf board", edges=True)
    for z0, z1 in zip(zs[:-1], zs[1:]):
        L = 2 * (hx if long_x else hy)
        x = -L / 2 + 0.05
        while x < L / 2 - 0.1:
            bw = rng.uniform(0.05, 0.4)
            if rng.random() < fill and x + bw < L / 2 - 0.02:
                bh = rng.uniform(0.3, 0.9) * (z1 - z0 - 0.03)
                bd = rng.uniform(0.5, 0.9) * (hy if long_x else hx)
                off = (x + bw / 2)
                bxc = (cx + off, cy, z0 + t + bh / 2) if long_x else (cx, cy + off, z0 + t + bh / 2)
                hh = (bw / 2, bd, bh / 2) if long_x else (bd, bw / 2, bh / 2)
                w.box(bxc, hh, 0, "cardboard", "box", edges=False)
            x += bw + rng.uniform(0.0, 0.1)


def _lamp(w, x, y, H, z, rng):
    w.cylinder((x, y, z + 0.1), (x, y, H), 0.004, "metal", "lamp cable", size=0.004)
    w.cylinder((x, y, z - 0.08), (x, y, z + 0.1), rng.uniform(0.12, 0.25), "metal", "lamp shade", caps=True)


def _wall_fixtures(w: World, W: float, D: float, H: float, rng, door=None, scale: float = 1.0):
    """Point-like reflectors on the four walls of a W x D room, mostly at 0.1-1.2 m height (the
    sensor is at ~0.4 m): what makes real walls lumpy at a robot's height. `door` = (x, half
    width) on the y = 0 wall. scale > 1: bigger fixtures, sparser (warehouse)."""
    # each wall: origin, direction along the wall, inward normal, length
    walls = [((0, 0), (1, 0), (0, 1), W), ((W, D), (-1, 0), (0, -1), W),
             ((0, D), (0, -1), (1, 0), D), ((W, 0), (0, 1), (-1, 0), D)]
    o_sk, o_seam = w.new_obj("skirting"), w.new_obj("panel seams")
    up = np.array([0, 0, 1.0])
    for (ox, oy), (ax, ay), (nx, ny), L in walls:
        P = lambda s, dn, z: (ox + ax * s + nx * dn, oy + ay * s + ny * dn, z)  # noqa: E731
        along, nrm = np.array([ax, ay, 0.0]), np.array([nx, ny, 0.0])
        # skirting board: an 8 cm strip standing 1.5 cm proud (normal = inward), with its top ridge
        u = along * L / 2
        if np.dot(np.cross(u, up), nrm) < 0:
            u = -u
        w.rect(P(L / 2, 0.015, 0.04), u, up * 0.04, "wood", o_sk)
        w.line(P(0, 0.015, 0.08), P(L, 0.015, 0.08), "corner", o_sk)
        # vertical panel seams / wall columns
        if rng.random() < 0.6:
            s = rng.uniform(0.3, 1.5)
            while s < L - 0.3:
                w.line(P(s, 0.0, 0.0), P(s, 0.0, H), "corner", o_seam)
                s += rng.uniform(1.2, 2.5)
        # discrete fixtures along the wall
        s = rng.uniform(0.2, 1.0)
        while s < L - 0.2:
            if door is not None and oy == 0 and ay == 0 and abs(ox + ax * s - door[0]) < door[1] + 0.15:
                s += 0.3
                continue
            kind = rng.choice(["socket", "switch", "valve", "box", "extinguisher", "pipe"],
                              p=[0.35, 0.15, 0.15, 0.12, 0.08, 0.15])
            if kind in ("socket", "switch", "box"):
                z = (rng.uniform(0.15, 0.45) if kind == "socket" else
                     rng.uniform(0.9, 1.2) if kind == "switch" else rng.uniform(0.2, 1.1))
                hw, hh, hd = (0.04, 0.04, 0.012) if kind != "box" else (0.08 * scale, 0.1 * scale, 0.04 * scale)
                h = (hd, hw, hh) if nx != 0 else (hw, hd, hh)
                w.box(P(s, hd, z), h, 0.0, "fixture", "electrical box" if kind == "box" else kind, occlude=False,
                      edge_mat="ridge")
            elif kind == "valve":                                   # radiator valve / tap: stub + knob
                z = rng.uniform(0.1, 0.6)
                w.cylinder(P(s, 0.0, z), P(s, 0.08, z), rng.uniform(0.01, 0.02), "fixture", "valve", occlude=False)
                w.cylinder(P(s, 0.08, z - 0.03), P(s, 0.08, z + 0.03), 0.025, "fixture", "valve knob",
                           occlude=False, caps=True)
            elif kind == "extinguisher":
                z0 = rng.uniform(0.05, 0.2)
                w.cylinder(P(s, 0.1, z0), P(s, 0.1, z0 + 0.5), 0.075, "metal", "extinguisher", caps=True)
            else:                                                   # pipe run along the wall, with brackets
                z = rng.uniform(0.12, 0.8)
                run = min(L - s - 0.1, rng.uniform(1.0, 4.0))
                if run > 0.5:
                    r = rng.uniform(0.01, 0.03) * scale
                    w.cylinder(P(s, 0.04 + r, z), P(s + run, 0.04 + r, z), r, "metal", "pipe", occlude=False)
                    b = 0.1
                    while b < run:
                        w.box(P(s + b, 0.025, z), (0.015, 0.02, 0.03) if nx != 0 else (0.02, 0.015, 0.03), 0.0,
                              "fixture", "pipe bracket", occlude=False, edge_mat="ridge")
                        b += rng.uniform(0.5, 1.0)
                    s += run
            s += rng.uniform(0.4, 1.8) * scale ** 0.5


def _door(w: World, dx: float, rng) -> float:
    """Door on the y = 0 wall: recessed leaf, a proud frame (vertical ridges), handle, kick plate."""
    o = w.new_obj("door")
    w.rect((dx, 0.03, 1.02), (0, 0, 1.02), (0.45, 0, 0), "wood", o, edges=True)
    of = w.new_obj("door frame")
    for sx in (-1, 1):                                             # frame posts: two ridges each
        for dxx in (0.47, 0.53):
            w.line((dx + sx * dxx, 0.02, 0.0), (dx + sx * dxx, 0.02, 2.08), "ridge", of)
    w.line((dx - 0.53, 0.02, 2.08), (dx + 0.53, 0.02, 2.08), "corner", of)
    hs = rng.choice([-1, 1])
    w.cylinder((dx + hs * 0.35, 0.03, 1.0), (dx + hs * 0.35, 0.10, 1.0), 0.01, "fixture", "door handle", occlude=False)
    w.cylinder((dx + hs * 0.35, 0.10, 1.0), (dx + hs * 0.2, 0.10, 1.0), 0.01, "fixture", "door handle", occlude=False)
    if rng.random() < 0.5:                                         # metal kick plate: its top edge at 0.25 m
        w.line((dx - 0.43, 0.035, 0.25), (dx + 0.43, 0.035, 0.25), "edge", o)
    return dx


def build_office(name: str, seed: int) -> World:
    rng = np.random.default_rng(seed)
    W, D, H = rng.uniform(5.5, 9.0), rng.uniform(4.5, 8.0), rng.uniform(2.5, 3.0)
    w = World(name, "office", (0, W, 0, D), rng=rng)
    _room_shell(w, W, D, H)
    pl = _Placer(W, D, rng)
    # door (a recessed leaf with a frame) and windows (glass) on the walls
    dx = _door(w, rng.uniform(1.0, W - 1.0), rng)
    for _ in range(rng.integers(1, 3)):
        wy = rng.uniform(1.0, D - 1.0)
        o = w.new_obj("window")
        w.rect((W - 0.02, wy, 1.6), (0, 0, 0.6), (0, 0.7, 0), "glass", o, edges=True)
        if rng.random() < 0.7:  # radiator below
            w.box((W - 0.08, wy, 0.45), (0.04, 0.5, 0.25), 0, "metal", "radiator")
            pl.items.append((W - 0.08, wy, 0.04, 0.5))
    for _ in range(rng.integers(1, 4)):                                      # desks with chairs
        r = pl.place(rng.uniform(0.6, 0.9), rng.uniform(0.35, 0.45), margin=0.9)
        if r:
            cx, cy, hx, hy = r
            _table(w, cx, cy, hx, hy, rng)
            if rng.random() < 0.8:
                _chair(w, cx, cy - hy - 0.3 if rng.random() < 0.5 else cy + hy + 0.3, rng.uniform(0, 2 * math.pi), rng)
    for _ in range(rng.integers(1, 4)):                                      # shelves along the walls
        r = pl.place(rng.uniform(0.4, 1.0), 0.18, margin=0.7, along_wall=True)
        if r:
            cx, cy, hx, hy = r
            _shelf(w, cx, cy, hx, hy, rng.uniform(1.2, 2.2), rng.integers(3, 6), rng)
    for _ in range(rng.integers(0, 3)):                                      # cabinets
        r = pl.place(rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.3), margin=0.7, along_wall=rng.random() < 0.7)
        if r:
            cx, cy, hx, hy = r
            w.box((cx, cy, 0.5), (hx, hy, 0.5), 0, "wood", "cabinet")
    for _ in range(rng.integers(0, 3)):                                      # pillars
        r = pl.place(0.15, 0.15, margin=1.0)
        if r:
            cx, cy, _, _ = r
            if rng.random() < 0.5:
                w.box((cx, cy, H / 2), (0.15, 0.15, H / 2), 0, "wall", "pillar")
            else:
                w.cylinder((cx, cy, 0), (cx, cy, H), 0.15, "wall", "pillar")
    for _ in range(rng.integers(1, 4)):                                      # hanging lamps
        _lamp(w, rng.uniform(1, W - 1), rng.uniform(1, D - 1), H, rng.uniform(1.9, 2.3), rng)
    if rng.random() < 0.5:                                                   # a waste bin / plant
        r = pl.place(0.15, 0.15, margin=0.6)
        if r:
            w.cylinder((r[0], r[1], 0), (r[0], r[1], 0.4), 0.15, "cardboard", "bin", caps=True)
    _wall_fixtures(w, W, D, H, rng, door=(dx, 0.55))
    return w


def build_warehouse(name: str, seed: int) -> World:
    rng = np.random.default_rng(seed)
    W, D, H = rng.uniform(10, 15), rng.uniform(8, 12), rng.uniform(4.5, 6.0)
    w = World(name, "warehouse", (0, W, 0, D), rng=rng)
    _room_shell(w, W, D, H)
    pl = _Placer(W, D, rng)
    horizontal = rng.random() < 0.5
    n_rows = rng.integers(2, 4)
    for i in range(n_rows):                                                  # rack rows
        L = rng.uniform(3, 6); depth = rng.uniform(0.4, 0.6); Hr = rng.uniform(2.2, 3.4)
        for _ in range(50):
            if horizontal:
                cx, cy = rng.uniform(L / 2 + 1.2, W - L / 2 - 1.2), rng.uniform(1.8, D - 1.8)
                hx, hy = L / 2, depth
            else:
                cx, cy = rng.uniform(1.8, W - 1.8), rng.uniform(L / 2 + 1.2, D - L / 2 - 1.2)
                hx, hy = depth, L / 2
            if pl.free(cx, cy, hx, hy, 1.6):
                pl.items.append((cx, cy, hx, hy)); break
        else:
            continue
        # posts every ~1.8 m, beams at the shelf levels, boxes on the levels
        n_bay = max(1, int(round(L / 1.8)))
        along = np.linspace(-L / 2, L / 2, n_bay + 1)
        levels = np.linspace(0.1, Hr - 0.2, rng.integers(3, 5))
        for a in along:
            for s in (-1, 1):
                px, py = (cx + a, cy + s * (depth - 0.04)) if horizontal else (cx + s * (depth - 0.04), cy + a)
                w.box((px, py, Hr / 2), (0.04, 0.04, Hr / 2), 0, "metal", "rack post", edges=False)
        for z in levels:
            w.box((cx, cy, z), (hx, hy, 0.02), 0, "metal", "rack level", edges=True)
        for z0, z1 in zip(levels[:-1], list(levels[1:])):
            for b in range(n_bay):
                if rng.random() < 0.75:
                    a0 = 0.5 * (along[b] + along[b + 1])
                    bw, bd, bh = rng.uniform(0.3, 0.8), rng.uniform(0.6, 0.95) * depth, rng.uniform(0.3, 0.9) * (z1 - z0 - 0.05)
                    bx, by = (cx + a0, cy) if horizontal else (cx, cy + a0)
                    w.box((bx, by, z0 + 0.02 + bh), (bw if horizontal else bd, bd if horizontal else bw, bh), 0,
                          "cardboard", "rack box", edges=False)
    for _ in range(rng.integers(2, 5)):                                      # pillars
        r = pl.place(0.2, 0.2, margin=1.2)
        if r:
            w.box((r[0], r[1], H / 2), (0.2, 0.2, H / 2), 0, "wall", "pillar")
    for _ in range(rng.integers(2, 7)):                                      # boxes / pallets on the floor
        r = pl.place(rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), margin=1.0)
        if r:
            cx, cy, hx, hy = r
            w.box((cx, cy, 0.07), (hx, hy, 0.07), 0, "wood", "pallet")
            h = rng.uniform(0.2, 0.7)
            w.box((cx, cy, 0.14 + h), (hx * 0.95, hy * 0.95, h), rng.normal(0, 0.1), "cardboard", "floor box")
    for _ in range(rng.integers(2, 6)):
        _lamp(w, rng.uniform(1, W - 1), rng.uniform(1, D - 1), H, rng.uniform(3.0, 4.0), rng)
    dx = _door(w, rng.uniform(1.5, W - 1.5), rng)
    _wall_fixtures(w, W, D, H, rng, door=(dx, 0.55), scale=1.5)
    return w


def build_forest(name: str, seed: int) -> World:
    rng = np.random.default_rng(seed)
    S = rng.uniform(12, 16)
    M = 3.0                                    # trees beyond the drivable region
    w = World(name, "forest", (0, S, 0, S), rng=rng)
    o = w.new_obj("ground", tilt=rng.normal(0, 1.5))
    w.rect((S / 2, S / 2, 0), ((S + 2 * M) / 2, 0, 0), (0, (S + 2 * M) / 2, 0), "litter", o, density=DENSITY * 0.8)
    n_trees = int(rng.uniform(0.10, 0.2) * (S + 2 * M) ** 2)
    trees = []
    for _ in range(n_trees * 5):
        if len(trees) >= n_trees:
            break
        x, y = rng.uniform(-M, S + M, 2)
        r = rng.uniform(0.07, 0.3)
        if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 1.3) ** 2 for a, b, c in trees):
            trees.append((x, y, r))
    for x, y, r in trees:
        Ht = rng.uniform(6, 10)
        w.cylinder((x, y, 0), (x, y, Ht), r, "bark", "trunk")
        for _ in range(rng.integers(0, 4)):                                 # knots / branch stubs, low
            z = rng.uniform(0.1, 1.3); az = rng.uniform(0, 2 * math.pi); Ls = rng.uniform(0.04, 0.2)
            d = np.array([math.cos(az), math.sin(az), rng.uniform(-0.2, 0.5)])
            p0 = np.array([x, y, z]) + r * np.array([math.cos(az), math.sin(az), 0])
            w.cylinder(p0, p0 + Ls * d, rng.uniform(0.015, 0.04), "bark", "stub", occlude=False, caps=True)
        for _ in range(rng.integers(2, 7)):                                 # branches
            z = rng.uniform(1.2, 5.0); az = rng.uniform(0, 2 * math.pi); up = rng.uniform(0.1, 0.8)
            L = rng.uniform(0.8, 2.5)
            d = np.array([math.cos(az) * math.cos(up), math.sin(az) * math.cos(up), math.sin(up)])
            p0 = np.array([x, y, z]) + r * np.array([math.cos(az), math.sin(az), 0])
            p1 = p0 + L * d
            rb = rng.uniform(0.015, 0.06)
            w.cylinder(p0, p1, rb, "bark", "branch", size=rb)
            w.leaves(p1, rng.uniform(0.4, 1.0), rng.integers(40, 160), "foliage")
        w.leaves((x, y, rng.uniform(3.5, 6.0)), rng.uniform(1.0, 2.0), rng.integers(100, 300), "canopy")
    for _ in range(rng.integers(3, 9)):                                      # bushes (obstacles)
        x, y = rng.uniform(0.5, S - 0.5, 2)
        if all((x - a) ** 2 + (y - b) ** 2 > (c + 1.2) ** 2 for a, b, c in trees):
            rb = rng.uniform(0.25, 0.5)
            w.leaves((x, y, rb * 0.8), rb, rng.integers(80, 200), "bush", size=0.006, zmin=0.02)
            w.foot_circles.append((x, y, rb))
    if rng.random() < 0.6:                                                   # a fallen log
        x, y = rng.uniform(1, S - 1, 2); az = rng.uniform(0, math.pi); L = rng.uniform(1.5, 3)
        d = np.array([math.cos(az), math.sin(az), 0]) * L / 2
        r = rng.uniform(0.1, 0.2)
        w.cylinder(np.array([x, y, r]) - d, np.array([x, y, r]) + d, r, "bark", "log")
        for t in np.linspace(-1, 1, 7):
            w.foot_circles.append((x + t * d[0], y + t * d[1], r + 0.05))
    return w


def build_points(name: str, seed: int, density: float = 0.3, height: float = 3.0) -> World:
    """Random isotropic point reflectors only (28 Sep 2026, user request): no floor, ceiling, walls or
    specular surfaces, so every echo comes from a fixed world point (the model's assumption).
    Density 0.3 per m^3 (user's choice after comparing 1, 0.3 and 0.1).
    density per m^3, uniform in [-M, S+M]^2 x [0, height]; strength log-normal (sigma 5 dB in power),
    size 1-10 cm (Rayleigh roll-off at low frequencies) and a random spectral tilt per point."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(12, 16)
    M = 3.0
    w = World(name, "points", (0, S, 0, S), rng=rng)
    L = S + 2 * M
    n = rng.poisson(density * L * L * height)
    p = np.stack([rng.uniform(-M, S + M, n), rng.uniform(-M, S + M, n), rng.uniform(0, height, n)], 1)
    d = 0.05 * 10 ** (rng.normal(0, 5, n) / 10)                          # backscatter strength (power), m^2
    for i in range(n):
        o = w.new_obj("point", size=float(rng.uniform(0.01, 0.1)))
        w._add(p[i:i + 1], np.zeros((1, 3)), 1.0, np.array([[d[i], 0.0, 0.0]]), o)
    return w


BUILDERS = {"office": build_office, "warehouse": build_warehouse, "forest": build_forest, "points": build_points}


def build(kind: str, name: str, seed: int) -> World:
    return BUILDERS[kind](name, seed)


# 20 training + 4 unseen validation worlds
TRAIN_WORLDS = ([("office", f"t_office_{i}", 100 + i) for i in range(8)]
                + [("warehouse", f"t_warehouse_{i}", 200 + i) for i in range(6)]
                + [("forest", f"t_forest_{i}", 300 + i) for i in range(6)])
# 20 more training worlds (28 Sep 2026): TRAIN_WORLDS_40 = the 40-world training set
TRAIN_WORLDS_EXTRA = ([("office", f"t_office_{i}", 100 + i) for i in range(8, 16)]
                      + [("warehouse", f"t_warehouse_{i}", 200 + i) for i in range(6, 12)]
                      + [("forest", f"t_forest_{i}", 300 + i) for i in range(6, 12)])
TRAIN_WORLDS_40 = TRAIN_WORLDS + TRAIN_WORLDS_EXTRA
# random point-reflector worlds (28 Sep 2026): 40 training + 4 validation
POINT_TRAIN = [("points", f"tp_{i}", 500 + i) for i in range(40)]
POINT_VAL = [("points", f"vp_{i}", 950 + i) for i in range(4)]
WORLD_SETS = {"rooms40": (None, None), "points": (POINT_TRAIN, POINT_VAL),
              "points8": (POINT_TRAIN[:8], POINT_VAL[:2])}                   # a small set for quick tests   # rooms40: TRAIN_WORLDS_40 / VAL_WORLDS
VAL_WORLDS = [("office", "v_office_a", 900), ("office", "v_office_b", 901),
              ("warehouse", "v_warehouse", 902), ("forest", "v_forest", 903)]
