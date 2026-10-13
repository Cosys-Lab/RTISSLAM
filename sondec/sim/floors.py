"""Multi-room office floors for place recognition (2 Oct 2026).

A floor is a W x D rectangle tiled by axis-aligned rectangles, each a room or a corridor. Walls stand
wherever a room borders anything else, as solid boxes (12 cm, floor to ceiling) that occlude, with a
doorway gap (1.0 m) from every room into a corridor. The outer envelope (floor, ceiling, outer walls)
is the usual room shell. Rooms are furnished like `build_office`; corridors get wall fixtures,
extinguishers, benches, bins and lamps.

Layouts (the structure of the floor):
    corridor  a straight corridor along x, rooms on both sides
    ring      a ring corridor around a core of rooms, rooms along the outer walls
    tee       a main corridor along x and a side corridor up from it, rooms in the quadrants

Repetition on purpose: room widths come from a small set per floor and the furnishing from a few
"designs" (each a fixed random seed), so several rooms of a floor are identical, as in real office
buildings. That makes appearance alone ambiguous, which is what a place recognizer must cope with.
"""
from __future__ import annotations

import math

import numpy as np

from .worlds import BUILDERS, World, _chair, _lamp, _room_shell, _shelf, _table

WALL_T = 0.12          # m, wall thickness
DOOR_W = 1.0           # m, doorway width


# ------------------------------------------------------------------ layouts
def _split(L, widths, rng):
    """Cut a length L into pieces drawn from `widths` (the last piece takes the rest)."""
    cuts, x = [0.0], 0.0
    while L - x > max(widths) + min(widths):
        x += float(rng.choice(widths)); cuts.append(x)
    if L - cuts[-1] < 0.75 * min(widths) and len(cuts) > 1:              # no sliver rooms: merge the rest
        cuts.pop()
    cuts.append(L)
    return cuts


def _layout(kind, rng):
    """-> W, D, rectangles [(x0, y0, x1, y1, 'room'|'corridor')]."""
    widths = sorted(rng.choice([3.0, 3.6, 4.2, 4.8, 6.0], size=2, replace=False).tolist())
    dr = float(rng.uniform(4.0, 5.5))                    # room depth
    cw = float(rng.uniform(1.6, 2.4))                    # corridor width
    R = []
    if kind == "corridor":
        W = float(rng.uniform(26, 38)); D = 2 * dr + cw
        R.append((0, dr, W, dr + cw, "corridor"))
        for y0, y1 in ((0, dr), (dr + cw, D)):
            c = _split(W, widths, rng)
            R += [(a, y0, b, y1, "room") for a, b in zip(c[:-1], c[1:])]
    elif kind == "ring":
        W = float(rng.uniform(26, 34)); D = float(rng.uniform(18, 24))
        core = (dr + cw, dr + cw, W - dr - cw, D - dr - cw)
        R += [(dr, dr, W - dr, dr + cw, "corridor"), (dr, D - dr - cw, W - dr, D - dr, "corridor"),
              (dr, dr + cw, dr + cw, D - dr - cw, "corridor"), (W - dr - cw, dr + cw, W - dr, D - dr - cw, "corridor")]
        for y0, y1 in ((0, dr), (D - dr, D)):                                   # outer rooms, long sides
            c = _split(W, widths, rng)
            R += [(a, y0, b, y1, "room") for a, b in zip(c[:-1], c[1:])]
        for x0, x1 in ((0, dr), (W - dr, W)):                                   # outer rooms, short sides
            c = _split(D - 2 * dr, widths, rng)
            R += [(x0, dr + a, x1, dr + b, "room") for a, b in zip(c[:-1], c[1:])]
        cy = (core[1] + core[3]) / 2                                            # core: two rows of rooms
        for y0, y1 in ((core[1], cy), (cy, core[3])):
            c = _split(core[2] - core[0], widths, rng)
            R += [(core[0] + a, y0, core[0] + b, y1, "room") for a, b in zip(c[:-1], c[1:])]
    elif kind == "tee":
        W = float(rng.uniform(28, 36)); D = dr + cw + float(rng.uniform(5.0, 7.0))
        xc = float(rng.uniform(0.35, 0.65) * W)                                  # side corridor position
        R.append((0, dr, W, dr + cw, "corridor"))
        R.append((xc - cw / 2, dr + cw, xc + cw / 2, D, "corridor"))
        c = _split(W, widths, rng)
        R += [(a, 0, b, dr, "room") for a, b in zip(c[:-1], c[1:])]
        for x0, x1 in ((0, xc - cw / 2), (xc + cw / 2, W)):                     # one row of deep rooms per quadrant
            c = _split(x1 - x0, widths, rng)
            R += [(x0 + a, dr + cw, x0 + b, D, "room") for a, b in zip(c[:-1], c[1:])]
    else:
        raise ValueError(kind)
    return W, D, [tuple(float(v) if i < 4 else v for i, v in enumerate(r)) for r in R]


def _shared(a, b):
    """Shared boundary of two rectangles: ('x', x, y0, y1) or ('y', y, x0, x1), or None."""
    for (p, q) in ((a, b), (b, a)):
        if abs(p[2] - q[0]) < 1e-6:                                             # p left of q
            y0, y1 = max(p[1], q[1]), min(p[3], q[3])
            if y1 - y0 > 1e-6:
                return ("x", p[2], y0, y1)
        if abs(p[3] - q[1]) < 1e-6:                                             # p below q
            x0, x1 = max(p[0], q[0]), min(p[2], q[2])
            if x1 - x0 > 1e-6:
                return ("y", p[3], x0, x1)
    return None


# ------------------------------------------------------------------ geometry
def _wall(w, axis, c, a0, a1, H):
    """Solid wall box along a boundary: axis 'x' = the wall lies on x = c, from y = a0 to a1."""
    if a1 - a0 < 0.05:
        return
    m = (a0 + a1) / 2; h = (a1 - a0) / 2
    if axis == "x":
        w.box((c, m, H / 2), (WALL_T / 2, h, H / 2), 0.0, "wall", "wall", edges=True, edge_mat="corner")
    else:
        w.box((m, c, H / 2), (h, WALL_T / 2, H / 2), 0.0, "wall", "wall", edges=True, edge_mat="corner")


def _P(axis, c, s, dn, z):
    """Point at along-wall coordinate s, distance dn off the wall plane (signed), height z."""
    return (c + dn, s, z) if axis == "x" else (s, c + dn, z)


def _doorway(w, axis, c, s, into, rng, closed=False):
    """Door at along-wall position s (centre). `into` = +1/-1: the side of the room (for the leaf).
    Open: a frame (vertical ridges at both jambs, a lintel) and often an open leaf swung into the room.
    Closed: the leaf in the wall plane on both faces, with the frame and a handle."""
    of = w.new_obj("door frame")
    t = WALL_T / 2 + 0.01
    for side in (-1, 1):
        for sgn in (-1, 1):
            w.line(_P(axis, c, s + sgn * DOOR_W / 2, side * t, 0.0), _P(axis, c, s + sgn * DOOR_W / 2, side * t, 2.08),
                   "ridge", of)
        w.line(_P(axis, c, s - DOOR_W / 2, side * t, 2.08), _P(axis, c, s + DOOR_W / 2, side * t, 2.08), "corner", of)
    o = w.new_obj("door")
    if closed:
        for side in (-1, 1):
            n = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 1.0, 0])
            ctr = np.array(_P(axis, c, s, side * (WALL_T / 2 + 0.005), 1.02))
            u = np.array([0, 0, 1.02]); v = np.array([0, DOOR_W / 2, 0]) if axis == "x" else np.array([DOOR_W / 2, 0, 0])
            if np.dot(np.cross(u, v), side * n) < 0:
                u, v = v, u
            w.rect(ctr, u, v, "wood", o, edges=True)
            hx = _P(axis, c, s + 0.35, side * (WALL_T / 2 + 0.07), 1.0)
            w.cylinder(_P(axis, c, s + 0.35, side * (WALL_T / 2 + 0.005), 1.0), hx, 0.01, "fixture", "door handle",
                       occlude=False)
    elif rng.random() < 0.75:                                                  # open leaf, swung into the room
        hinge = s + rng.choice([-1, 1]) * (DOOR_W / 2 - 0.02)
        near, far = WALL_T / 2 * into, (WALL_T / 2 + DOOR_W - 0.05) * into
        ctr = _P(axis, c, hinge, (near + far) / 2, 1.02)
        hl = abs(far - near) / 2
        w.box(ctr, (hl, 0.02, 1.02) if axis == "x" else (0.02, hl, 1.02), 0.0, "wood", "door leaf")


def _fixtures(w, axis, c, s0, s1, normal_sign, H, rng, skip=(), scale=1.0, corridor=False):
    """Skirting, panel seams and point-like fixtures along one face of a wall (cf. worlds._wall_fixtures).
    The face lies at c + normal_sign * WALL_T/2 and faces normal_sign; `skip` = [(s_lo, s_hi)] doorways."""
    f = c + normal_sign * WALL_T / 2
    nrm = np.array([normal_sign, 0, 0.0]) if axis == "x" else np.array([0, normal_sign, 0.0])
    along = np.array([0, 1.0, 0]) if axis == "x" else np.array([1.0, 0, 0])
    up = np.array([0, 0, 1.0])
    P = lambda s, dn, z: _P(axis, f, s, normal_sign * dn, z)  # noqa: E731
    pieces, a = [], s0
    for lo, hi in sorted(skip):
        if lo > a:
            pieces.append((a, lo))
        a = max(a, hi)
    if s1 > a:
        pieces.append((a, s1))
    o_sk, o_seam = w.new_obj("skirting"), w.new_obj("panel seams")
    for a, b in pieces:
        L = b - a
        if L < 0.3:
            continue
        u = along * L / 2
        if np.dot(np.cross(u, up), nrm) < 0:
            u = -u
        w.rect(P((a + b) / 2, 0.015, 0.04), u, up * 0.04, "wood", o_sk)
        w.line(P(a, 0.015, 0.08), P(b, 0.015, 0.08), "corner", o_sk)
        if rng.random() < 0.5:
            s = a + rng.uniform(0.3, 1.5)
            while s < b - 0.3:
                w.line(P(s, 0.0, 0.0), P(s, 0.0, H), "corner", o_seam)
                s += rng.uniform(1.2, 2.5)
        s = a + rng.uniform(0.2, 1.0)
        while s < b - 0.2:
            p = [0.30, 0.12, 0.13, 0.10, 0.20 if corridor else 0.07, 0.15]
            kind = rng.choice(["socket", "switch", "valve", "box", "extinguisher", "pipe"], p=np.array(p) / sum(p))
            hd = 0.012
            if kind in ("socket", "switch", "box"):
                z = (rng.uniform(0.15, 0.45) if kind == "socket" else rng.uniform(0.9, 1.2) if kind == "switch"
                     else rng.uniform(0.2, 1.1))
                hw, hh, hd = (0.04, 0.04, 0.012) if kind != "box" else (0.08 * scale, 0.1 * scale, 0.04 * scale)
                h = (hd, hw, hh) if axis == "x" else (hw, hd, hh)
                w.box(P(s, hd, z), h, 0.0, "fixture", "electrical box" if kind == "box" else kind, occlude=False,
                      edge_mat="ridge")
            elif kind == "valve":
                z = rng.uniform(0.1, 0.6)
                w.cylinder(P(s, 0.0, z), P(s, 0.08, z), rng.uniform(0.01, 0.02), "fixture", "valve", occlude=False)
                w.cylinder(P(s, 0.08, z - 0.03), P(s, 0.08, z + 0.03), 0.025, "fixture", "valve knob", occlude=False,
                           caps=True)
            elif kind == "extinguisher":
                z0 = rng.uniform(0.05, 0.2)
                w.cylinder(P(s, 0.1, z0), P(s, 0.1, z0 + 0.5), 0.075, "metal", "extinguisher", caps=True)
            else:
                z = rng.uniform(0.12, 0.8)
                run = min(b - s - 0.1, rng.uniform(1.0, 4.0))
                if run > 0.5:
                    r = rng.uniform(0.01, 0.03) * scale
                    w.cylinder(P(s, 0.04 + r, z), P(s + run, 0.04 + r, z), r, "metal", "pipe", occlude=False)
                    bb = 0.1
                    while bb < run:
                        w.box(P(s + bb, 0.025, z), (0.015, 0.02, 0.03) if axis == "x" else (0.02, 0.015, 0.03), 0.0,
                              "fixture", "pipe bracket", occlude=False, edge_mat="ridge")
                        bb += rng.uniform(0.5, 1.0)
                    s += run
            s += rng.uniform(0.4, 1.8) * scale ** 0.5


class _LocalPlacer:
    """Rejection placement inside one room [x0, x1] x [y0, y1], keeping clear of the doorway."""

    def __init__(self, x0, y0, x1, y1, rng, keep_clear=()):
        self.x0, self.y0, self.x1, self.y1, self.rng = x0, y0, x1, y1, rng
        self.items = list(keep_clear)

    def free(self, cx, cy, hx, hy, margin):
        m = WALL_T / 2 + 0.05
        if cx - hx < self.x0 + m or cy - hy < self.y0 + m or cx + hx > self.x1 - m or cy + hy > self.y1 - m:
            return False
        return all(abs(cx - ax) >= hx + bx + margin or abs(cy - ay) >= hy + by + margin for ax, ay, bx, by in self.items)

    def place(self, hx, hy, margin=0.9, along_wall=False, tries=200):
        m = WALL_T / 2 + 0.03
        for _ in range(tries):
            u = lambda lo, hi: self.rng.uniform(lo, hi) if hi > lo else None  # noqa: E731  (no room: skip)
            if along_wall:
                side = self.rng.integers(4)
                if side < 2:
                    hx_, hy_ = min(hx, hy), max(hx, hy)
                    cx = self.x0 + m + hx_ if side == 0 else self.x1 - m - hx_
                    cy = u(self.y0 + hy_ + 0.2, self.y1 - hy_ - 0.2)
                else:
                    hx_, hy_ = max(hx, hy), min(hx, hy)
                    cy = self.y0 + m + hy_ if side == 2 else self.y1 - m - hy_
                    cx = u(self.x0 + hx_ + 0.2, self.x1 - hx_ - 0.2)
            else:
                hx_, hy_ = hx, hy
                cx = u(self.x0 + hx_ + 0.6, self.x1 - hx_ - 0.6)
                cy = u(self.y0 + hy_ + 0.6, self.y1 - hy_ - 0.6)
            if cx is None or cy is None:
                continue
            if self.free(cx, cy, hx_, hy_, margin):
                self.items.append((cx, cy, hx_, hy_))
                return cx, cy, hx_, hy_
        return None


def _furnish(w, room, H, design, door):
    """Furniture of one room from a fixed design seed: rooms of the same size and design are identical.
    `door` = (x, y) of the doorway centre, kept clear."""
    x0, y0, x1, y1, _ = room
    rng = np.random.default_rng(design)                                        # the design, not the floor
    pl = _LocalPlacer(x0, y0, x1, y1, rng, keep_clear=[(door[0], door[1], 0.9, 0.9)])
    for _ in range(rng.integers(1, 4)):                                         # desks with chairs
        r = pl.place(rng.uniform(0.6, 0.9), rng.uniform(0.35, 0.45), margin=0.9)
        if r:
            cx, cy, hx, hy = r
            _table(w, cx, cy, hx, hy, rng)
            if rng.random() < 0.8:
                _chair(w, cx, cy - hy - 0.3 if rng.random() < 0.5 else cy + hy + 0.3, rng.uniform(0, 2 * math.pi), rng)
    for _ in range(rng.integers(1, 4)):                                         # shelves along the walls
        r = pl.place(rng.uniform(0.4, 1.0), 0.18, margin=0.7, along_wall=True)
        if r:
            cx, cy, hx, hy = r
            _shelf(w, cx, cy, hx, hy, rng.uniform(1.2, 2.2), rng.integers(3, 6), rng)
    for _ in range(rng.integers(0, 3)):                                         # cabinets
        r = pl.place(rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.3), margin=0.7, along_wall=rng.random() < 0.7)
        if r:
            cx, cy, hx, hy = r
            w.box((cx, cy, 0.5), (hx, hy, 0.5), 0, "wood", "cabinet")
    if rng.random() < 0.4:                                                     # a pillar
        r = pl.place(0.15, 0.15, margin=1.0)
        if r:
            if rng.random() < 0.5:
                w.box((r[0], r[1], H / 2), (0.15, 0.15, H / 2), 0, "wall", "pillar")
            else:
                w.cylinder((r[0], r[1], 0), (r[0], r[1], H), 0.15, "wall", "pillar")
    for _ in range(rng.integers(1, 3)):                                         # hanging lamps
        lx = rng.uniform(x0 + 0.8, x1 - 0.8) if x1 - x0 > 1.6 else (x0 + x1) / 2
        ly = rng.uniform(y0 + 0.8, y1 - 0.8) if y1 - y0 > 1.6 else (y0 + y1) / 2
        _lamp(w, lx, ly, H, rng.uniform(1.9, 2.3), rng)
    if rng.random() < 0.5:                                                     # a bin
        r = pl.place(0.15, 0.15, margin=0.6)
        if r:
            w.cylinder((r[0], r[1], 0), (r[0], r[1], 0.4), 0.15, "cardboard", "bin", caps=True)


def _corridor_items(w, cor, H, rng):
    x0, y0, x1, y1, _ = cor
    along_x = (x1 - x0) >= (y1 - y0)
    L = (x1 - x0) if along_x else (y1 - y0)
    s = rng.uniform(1.0, 4.0)
    while s < L - 1.0:                                                          # lamps along the centre line
        x, y = ((x0 + s, (y0 + y1) / 2) if along_x else ((x0 + x1) / 2, y0 + s))
        _lamp(w, x, y, H, rng.uniform(2.1, 2.4), rng)
        s += rng.uniform(3.0, 6.0)
    for _ in range(rng.integers(0, 3)):                                         # benches / bins against a wall
        s = rng.uniform(1.0, L - 1.0); side = rng.choice([-1, 1])
        if along_x:
            yc = (y0 + WALL_T / 2 + 0.25) if side < 0 else (y1 - WALL_T / 2 - 0.25)
            cx, cy = x0 + s, yc
        else:
            xc = (x0 + WALL_T / 2 + 0.25) if side < 0 else (x1 - WALL_T / 2 - 0.25)
            cx, cy = xc, y0 + s
        if rng.random() < 0.5:
            hx, hy = (0.6, 0.2) if along_x else (0.2, 0.6)
            w.box((cx, cy, 0.22), (hx, hy, 0.22), 0, "wood", "bench")
        else:
            w.cylinder((cx, cy, 0), (cx, cy, 0.45), 0.17, "plastic", "bin", caps=True)


def build_floor(name: str, seed: int, layout: str = "corridor") -> World:
    rng = np.random.default_rng(seed)
    W, D, rects = _layout(layout, rng)
    H = float(rng.uniform(2.6, 3.0))
    w = World(name, f"floor_{layout}", (0, W, 0, D), rng=rng)
    _room_shell(w, W, D, H)
    designs = rng.integers(0, 2 ** 31, size=int(rng.integers(3, 5)))          # a few room designs per floor
    walls = {}                                                                  # (axis, c, a0, a1) -> doorways
    # interior boundaries: a wall wherever a room meets anything; corridor-corridor stays open
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            sh = _shared(rects[i], rects[j])
            if sh and "room" in (rects[i][4], rects[j][4]):
                walls.setdefault(sh, [])
    # one doorway per room, on its longest boundary with a corridor
    for i, r in enumerate(rects):
        if r[4] != "room":
            continue
        best = None
        for j, q in enumerate(rects):
            if q[4] == "corridor":
                sh = _shared(r, q)
                if sh and sh[3] - sh[2] >= DOOR_W + 0.6 and (best is None or sh[3] - sh[2] > best[3] - best[2]):
                    best = sh
        if best is None:
            continue
        axis, c, a0, a1 = best
        s = float(rng.uniform(a0 + DOOR_W / 2 + 0.3, a1 - DOOR_W / 2 - 0.3))
        mid = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        into = 1 if (mid[0] if axis == "x" else mid[1]) > c else -1
        walls[best].append((s, into, rng.random() < 0.15, i))                  # (centre, side, closed, room)
    # build the walls with their doorway gaps, and the doors
    for (axis, c, a0, a1), doors in walls.items():
        a = a0
        for s, into, closed, _ in sorted(doors):
            if closed:
                continue
            _wall(w, axis, c, a, s - DOOR_W / 2, H); a = s + DOOR_W / 2
        _wall(w, axis, c, a, a1, H)
        for s, into, closed, _ in doors:
            _doorway(w, axis, c, s, into, rng, closed=closed)
    # furniture, wall fixtures on every face that faces a free space
    door_of = {i: (axis, c, s) for (axis, c, a0, a1), ds in walls.items() for s, into, closed, i in ds}
    for i, r in enumerate(rects):
        if r[4] == "room":
            axis, c, s = door_of.get(i, ("x", r[0], (r[1] + r[3]) / 2))
            door_xy = (c, s) if axis == "x" else (s, c)
            w_r, d_r = r[2] - r[0], r[3] - r[1]
            design = int(designs[int(round(10 * w_r) + 7 * round(10 * d_r)) % len(designs)])   # same size -> same design
            _furnish(w, r, H, design, door_xy)
        else:
            _corridor_items(w, r, H, rng)

    def kind_at(x, y):
        for q in rects:
            if q[0] <= x <= q[2] and q[1] <= y <= q[3]:
                return q[4]
        return None

    for (axis, c, a0, a1), doors in walls.items():                            # interior walls, both faces
        skip = [(s - DOOR_W / 2 - 0.15, s + DOOR_W / 2 + 0.15) for s, _, _, _ in doors]
        m = (a0 + a1) / 2
        for sign in (-1, 1):
            k = kind_at(c + sign * 0.2, m) if axis == "x" else kind_at(m, c + sign * 0.2)
            _fixtures(w, axis, c, a0, a1, sign, H, rng, skip=skip, corridor=k == "corridor")
    for r in rects:                                                            # outer walls, from the inside
        sides = [("x", 0.0, r[1], r[3], 1, r[0] == 0), ("x", W, r[1], r[3], -1, abs(r[2] - W) < 1e-6),
                 ("y", 0.0, r[0], r[2], 1, r[1] == 0), ("y", D, r[0], r[2], -1, abs(r[3] - D) < 1e-6)]
        for axis, b, a0, a1, sign, on in sides:
            if not on:
                continue
            c = b - sign * WALL_T / 2                                          # so that the face is the outer wall
            _fixtures(w, axis, c, a0, a1, sign, H, rng, corridor=r[4] == "corridor")
            if r[4] == "room" and a1 - a0 > 2.0 and rng.random() < 0.7:        # window with a radiator below
                s = (a0 + a1) / 2
                o = w.new_obj("window")
                n = np.array([sign, 0, 0.0]) if axis == "x" else np.array([0, sign, 0.0])
                u = np.array([0, 0.6, 0]) if axis == "x" else np.array([0.6, 0, 0])
                v = np.array([0, 0, 0.6])
                if np.dot(np.cross(u, v), n) < 0:
                    u, v = v, u
                w.rect(_P(axis, b, s, sign * 0.02, 1.6), u, v, "glass", o, edges=True)
                if rng.random() < 0.7:
                    w.box(_P(axis, b, s, sign * 0.08, 0.45), (0.04, 0.5, 0.25) if axis == "x" else (0.5, 0.04, 0.25),
                          0, "metal", "radiator")
    w.layout = rects
    return w


for _k in ("corridor", "ring", "tee"):
    BUILDERS[f"floor_{_k}"] = (lambda k: (lambda name, seed: build_floor(name, seed, k)))(_k)
