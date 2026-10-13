"""2D floor plans for RTISSLAM.

Jan, 2026-10-12: "smooth navigation is key, then some loops, floor plans; not too many reflectors, specular walls";
later the same day: "make more complex environments, these are way too simple".

A floor plan is a set of walls (thin occluding boxes with specular faces, 1.2 m high; the sensor plane is at 0.4 m),
with door gaps whose wall ends carry a door frame (two strong vertical ridges) or a plain diffracting edge,
furniture (cabinets, desks, counters, racks as occluding boxes; tables and chairs as legs plus a top/seat; bins, pots
and pillars as cylinders), a few point-like objects (poles, fixtures), and the concave corners of every wall junction
(L- and T-junctions, any angle: strong vertical lines, the single-bounce stand-in for the dihedral
retro-reflection of a corner). Everything is built with the sondec World class, so ArraySynth renders it and
World.occupancy gives the ground-truth grid (walls, furniture footprints).

Driving: a plan has a TOUR (waypoints visited in order, closed); the driving line is the A* path through the tour
on the ground-truth grid, elastically smoothed (rtissim.planner). The three simple test plans (loop, figure8, hall)
keep their hand-made centre lines (rtissim.robot.smooth_closed_path).

Plans (metres):
    office     36 x 22: lobby with pillars and an angled reception counter, two corridors joined by the lobby and an
               east hall, 13 furnished rooms (south row with an inner connecting door, a middle block with doors to
               both corridors and a glass meeting room, north row); many loops at different scales
    warehouse  32 x 26, L-shaped (loading-dock notch), a walled office in a corner, 8 double-sided rack rows with
               uprights in 4 aisles and a cross aisle, pillars, pallets
    atrium     an octagonal hall (radius 12) around a round glass core, angled planters and screens, two side rooms
               off angled walls
    loop, figure8, hall   the simple test plans of session 1
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sondec.sim.worlds import MAT, World

MAT.setdefault("corner2d", (0.3, 0.0, 0.0))            # concave corner: strong isotropic line (retro-reflector)
MAT.setdefault("frame", (0.25, 0.0, 0.0))              # door frame ridge
H_WALL = 1.2                                           # m
T_WALL = 0.10                                          # m
DOOR_W = 1.4                                           # m


@dataclass
class FloorPlan:
    name: str
    walls: list                                        # [(x0, y0, x1, y1[, mat])]
    tour: list = field(default_factory=list)           # waypoints of the mapping drive (closed)
    loops: list = field(default_factory=list)          # simple plans: hand-made centre lines
    pillars: list = field(default_factory=list)        # [(x, y, r)] metal cylinders, floor to 2 m
    objects: list = field(default_factory=list)        # [(x, y, strength)] point-like poles at 0.25-0.6 m
    furniture: list = field(default_factory=list)      # tuples, see _furnish
    frames: list = field(default_factory=list)         # [(x, y)] door-frame ridges
    edges: list = field(default_factory=list)          # [(x, y)] free wall ends (diffracting vertical edges)
    bounds: tuple = (0.0, 10.0, 0.0, 10.0)


# ====================================================================== helpers
def _rect(x0, y0, x1, y1, mat="wall"):
    return [(x0, y0, x1, y0, mat), (x1, y0, x1, y1, mat), (x1, y1, x0, y1, mat), (x0, y1, x0, y0, mat)]


def _poly(P, mat="wall", closed=True):
    P = list(P) + ([P[0]] if closed else [])
    return [(a[0], a[1], b[0], b[1], mat) for a, b in zip(P[:-1], P[1:])]


class _Walls:
    """Collects walls; doors cut gaps and record their frames (or bare wall ends)."""

    def __init__(self):
        self.walls, self.frames, self.edges = [], [], []

    def add(self, x0, y0, x1, y1, doors=(), mat="wall", width=DOOR_W, frame=True):
        """Wall from (x0, y0) to (x1, y1) with door gaps centred at the given distances along it."""
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        s = 0.0
        for d in sorted(doors):
            e0, e1 = d - width / 2, d + width / 2
            if e0 > s + 1e-6:
                self.walls.append((*(a + s * u), *(a + e0 * u), mat))
            for e in (e0, e1):
                p = a + e * u
                if frame:
                    self.frames += [tuple(p + 0.06 * n), tuple(p - 0.06 * n)]
                else:
                    self.edges += [tuple(p + T_WALL / 2 * n), tuple(p - T_WALL / 2 * n)]
            s = e1
        if L > s + 1e-6:
            self.walls.append((*(a + s * u), *b, mat))

    def rect(self, x0, y0, x1, y1, mat="wall"):
        for w in _rect(x0, y0, x1, y1, mat):
            self.walls.append(w)


def _room_items(x0, y0, x1, y1, door_side: str, style: str, rng):
    """Furniture for a rectangular room, keeping a free driving loop in the middle and the door zone clear.
    door_side: the wall with the (main) door: 'S', 'N', 'W', 'E'. Items go against the other walls."""
    F = []
    W, D = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    far = {"S": "N", "N": "S", "W": "E", "E": "W"}[door_side]

    def against(side, along, hx_along, depth, h, mat):
        """Box against the wall `side` centred `along` (0..1 along it), half length hx_along, depth (full)."""
        g = 0.05 + T_WALL / 2
        if side in "SN":
            x = x0 + along * W
            y = (y0 + g + depth / 2) if side == "S" else (y1 - g - depth / 2)
            return ("box", x, y, hx_along, depth / 2, 0.0, h, mat)
        y = y0 + along * D
        x = (x0 + g + depth / 2) if side == "W" else (x1 - g - depth / 2)
        return ("box", x, y, depth / 2, hx_along, 0.0, h, mat)

    sides = [s for s in "SNWE" if s != door_side]
    if style == "desks":
        F.append(against(far, 0.3, 0.7, 0.7, 0.75, "wood"))                      # two desks on the far wall
        F.append(against(far, 0.72, 0.7, 0.7, 0.75, "wood"))
        side = sides[1] if sides[0] == far else sides[0]
        F.append(against(side, 0.5, 0.45, 0.45, 1.1, "metal"))                   # a cabinet on a side wall
        dy = {"N": -1.0, "S": 1.0, "E": 0.0, "W": 0.0}[far]
        dx = {"E": -1.0, "W": 1.0, "N": 0.0, "S": 0.0}[far]
        for al in (0.3, 0.72):
            if far in "NS":
                F.append(("chair", x0 + al * W, (y1 - 1.25 if far == "N" else y0 + 1.25), rng.uniform(-0.4, 0.4) + math.atan2(dy, dx)))
            else:
                F.append(("chair", (x1 - 1.25 if far == "E" else x0 + 1.25), y0 + al * D, rng.uniform(-0.4, 0.4) + math.atan2(dy, dx)))
        F.append(("cyl", *_corner_pt(x0, y0, x1, y1, far, sides, 0.35), 0.15, 0.45, "plastic"))
    elif style == "meeting":
        hx, hy = (0.9, 0.5) if W >= D else (0.5, 0.9)
        F.append(("table", cx, cy, hx, hy, 0.0))
        for s in (-1, 1):
            if W >= D:
                F.append(("chair", cx + s * 0.45, cy + hy + 0.35, -math.pi / 2)); F.append(("chair", cx + s * 0.45, cy - hy - 0.35, math.pi / 2))
            else:
                F.append(("chair", cx + hx + 0.35, cy + s * 0.45, math.pi)); F.append(("chair", cx - hx - 0.35, cy + s * 0.45, 0.0))
        F.append(against(far, 0.5, 0.8, 0.45, 0.9, "wood"))                      # sideboard
    elif style == "storage":
        for s in sides:
            if s != far:
                F.append(against(s, 0.5, min(W, D) * 0.3, 0.45, 1.8, "wood"))   # shelving on both side walls
        F.append(("cyl", *_corner_pt(x0, y0, x1, y1, far, sides, 0.4), 0.2, 0.9, "cardboard"))
        F.append(("box", cx + rng.uniform(-0.3, 0.3), cy + rng.uniform(-0.3, 0.3), 0.3, 0.3, rng.uniform(0, 1), 0.6, "cardboard"))
    elif style == "lab":
        F.append(against(far, 0.5, 0.42 * (W if far in "NS" else D), 0.8, 0.9, "metal"))   # a long bench
        for s in sides:
            if s != far:
                F.append(against(s, 0.35, 0.3, 0.5, 1.4, "metal"))
        F.append(("cyl", cx, cy, 0.04, 0.9, "metal"))                              # a free-standing pole (stool)
    return F


def _corner_pt(x0, y0, x1, y1, far, sides, off):
    side = [s for s in sides if s != far][0]
    x = {"W": x0 + off, "E": x1 - off}.get(side, None)
    y = {"S": y0 + off, "N": y1 - off}.get(far, None)
    if x is None:
        x = {"W": x0 + off, "E": x1 - off}.get(far, x0 + off)
        y = {"S": y0 + off, "N": y1 - off}[side]
    return x, y


# ====================================================================== plans
def plan_office() -> FloorPlan:
    rng = np.random.default_rng(7)
    W, D = 36.0, 22.0
    ys0, ys1 = 5.5, 7.9           # south corridor
    yn0, yn1 = 14.1, 16.5         # north corridor
    xl = 7.0                      # lobby east edge
    xe = 31.0                     # east hall west edge
    w = _Walls()
    w.rect(0, 0, W, D)
    # south rooms [7, 13, 19, 25, 31] + SE room [31, 36]
    w.add(xl, ys0, xe, ys0, doors=[16 - xl, 22 - xl, 28 - xl])
    w.add(xe, ys0, W, ys0, doors=[2.5])
    w.add(xl, 0, xl, ys0, doors=[3.0])                          # room S1 <-> lobby
    w.add(13, 0, 13, ys0, doors=[2.5])                          # S1 <-> S2 (inner door: a small loop)
    w.add(19, 0, 19, ys0)
    w.add(25, 0, 25, ys0, doors=[2.0])                          # S3 <-> S4
    w.add(xe, 0, xe, ys0)
    # middle block [7, 15.5, 22.5, 31]
    w.add(xl, ys1, xe, ys1, doors=[11 - xl, 19 - xl, 27 - xl])
    w.add(xl, yn0, xe, yn0, doors=[11 - xl, 27 - xl])
    w.add(xl, ys1, xl, yn0, doors=[3.1])                        # M1 <-> lobby
    w.add(15.5, ys1, 15.5, yn0, mat="glass")
    w.add(22.5, ys1, 22.5, yn0, mat="glass")
    w.add(xe, ys1, xe, yn0)
    # north rooms [7, 15, 23, 31, 36]
    w.add(xl, yn1, W, yn1, doors=[11 - xl, 19 - xl, 27 - xl, 33.5 - xl])
    w.add(xl, yn1, xl, D)
    for x in (15, 23, 31):
        w.add(x, yn1, x, D)
    F = []
    rooms = [((7, 0, 13, ys0), "N", "desks"), ((13, 0, 19, ys0), "N", "meeting"), ((19, 0, 25, ys0), "N", "lab"),
             ((25, 0, 31, ys0), "N", "storage"), ((31, 0, 36, ys0), "N", "desks"),
             ((7, ys1, 15.5, yn0), "W", "meeting"), ((15.5, ys1, 22.5, yn0), "S", "meeting"), ((22.5, ys1, 31, yn0), "S", "storage"),
             ((7, yn1, 15, D), "S", "storage"), ((15, yn1, 23, D), "S", "desks"), ((23, yn1, 31, D), "S", "meeting"),
             ((31, yn1, 36, D), "S", "lab")]
    for (a, b, c, d), side, style in rooms:
        F += _room_items(a, b, c, d, side, style, rng)
    # lobby: angled reception counter, sofa block, plant pots; east hall: benches, a vending machine
    F += [("box", 3.2, 9.0, 1.3, 0.35, math.radians(30), 1.05, "wood"), ("box", 1.0, 19.5, 0.45, 1.2, 0.0, 0.8, "fabric"),
          ("cyl", 6.3, 5.0, 0.28, 0.9, "plastic"), ("cyl", 6.3, 17.0, 0.28, 0.9, "plastic"), ("cyl", 0.6, 0.6, 0.3, 0.9, "plastic"),
          ("box", 35.4, 11.0, 0.35, 0.5, 0.0, 1.8, "metal"), ("box", 35.5, 14.2, 0.22, 0.9, 0.0, 0.45, "wood"),
          ("cyl", 31.6, 6.2, 0.18, 0.5, "plastic")]
    pillars = [(3.5, 4.5, 0.25), (3.5, 13.5, 0.25), (3.5, 17.5, 0.25)]
    objects = [(9.0, 6.0, 0.8), (21.0, 7.8 - 0.06, 0.6), (26.0, 14.25, 0.8), (17.0, 16.4, 0.6), (33.0, 9.0, 0.7),
               (0.2, 11.0, 0.8)]
    tour = [(4.8, 11.0), (10, 6.7), (16, 4.6), (18.0, 2.8), (14.2, 2.8), (11.5, 3.2), (9.0, 3.5), (5.0, 2.5),
            (5.0, 6.7), (14, 6.7), (20, 6.7), (22, 3.6), (23.8, 2.3), (27.0, 3.8), (28, 6.7), (33.5, 3.0),
            (33.5, 9.0), (33.2, 13.5), (32.4, 19.3), (33.5, 15.3), (27, 15.3), (27, 7.0), (19, 6.8), (11, 7.0),
            (11, 15.2), (19, 15.3), (20.5, 18.6), (17.5, 18.6), (19, 15.4), (11, 15.4), (5.5, 15.5), (2.0, 15.5)]
    return FloorPlan("office", w.walls, tour=tour, pillars=pillars, objects=objects, furniture=F, frames=w.frames,
                     edges=w.edges, bounds=(0, W, 0, D))


def plan_warehouse() -> FloorPlan:
    rng = np.random.default_rng(11)
    W, D = 32.0, 26.0
    w = _Walls()
    outer = [(0, 0), (W, 0), (W, 20), (26, 20), (26, D), (0, D)]
    for x0, y0, x1, y1, m in _poly(outer):
        w.add(x0, y0, x1, y1)
    w.add(0, 20, 8, 20, doors=[6.0])                             # corner office (NW)
    w.add(8, 20, 8, D, doors=[2.0], frame=False)
    F = []
    rows = [4.0, 8.0, 12.0, 16.0]
    for y in rows:
        for xa, xb in ((5.0, 14.0), (18.0, 27.0)):
            F.append(("rack", (xa + xb) / 2, y, (xb - xa) / 2, 0.5, 2.6))
    F += _room_items(0, 20, 8, D, "S", "desks", rng)
    F += [("box", 30.5, 2.0, 0.6, 0.4, 0.0, 1.0, "wood"), ("box", 30.5, 3.4, 0.6, 0.4, 0.0, 0.7, "cardboard"),
          ("box", 2.0, 17.8, 0.4, 0.6, 0.2, 1.2, "cardboard"), ("box", 28.4, 18.5, 0.6, 0.4, 0.0, 1.0, "wood"),
          ("box", 24.5, 24.6, 0.5, 0.5, 0.3, 0.9, "cardboard"), ("cyl", 16.0, 23.0, 0.3, 1.0, "metal")]
    pillars = [(x, y, 0.2) for x in (2.6, 16.0, 29.4) for y in (10.0,)] + [(16.0, 19.5, 0.2)]
    objects = [(0.2, 6.0, 1.0), (31.8, 14.0, 1.0), (12.0, 25.8, 0.8), (20.0, 0.2, 0.8)]
    tour = [(3.0, 2.0), (29.5, 2.0), (29.5, 6.0), (16.0, 6.0), (3.0, 6.0), (3.0, 14.0), (16.0, 14.0),
            (17.2, 10.0), (29.5, 10.0), (29.5, 18.0), (17.2, 18.0), (17.2, 22.5), (9.5, 18.6), (6.0, 21.5),
            (3.5, 22.8), (6.0, 18.6), (1.5, 13.5)]
    return FloorPlan("warehouse", w.walls, tour=tour, pillars=pillars, objects=objects, furniture=F, frames=w.frames,
                     edges=w.edges, bounds=(0, W, 0, D))


def plan_atrium() -> FloorPlan:
    rng = np.random.default_rng(13)
    cx, cy, R = 16.0, 14.0, 12.0
    w = _Walls()
    oct_ = [(cx + R * math.cos(math.radians(22.5 + 45 * k)), cy + R * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    seg = list(zip(oct_, oct_[1:] + oct_[:1]))
    side_rooms = (7, 3)                                         # doors in edges 7 (east) and 3 (west)
    for k, (a, b) in enumerate(seg):
        L = math.dist(a, b)
        w.add(*a, *b, doors=[L / 2] if k in side_rooms else [])
    # side rooms behind the east and west walls (rectangles attached to the octagon edges)
    F = []
    for k, sgn in ((7, 1), (3, -1)):
        a, b = seg[k]
        x_wall = a[0]
        x_far = x_wall + sgn * 6.0
        y0, y1 = min(a[1], b[1]), max(a[1], b[1])
        y0, y1 = y0 - 1.5, y1 + 1.5
        w.add(x_wall, y0, x_far, y0); w.add(x_far, y0, x_far, y1); w.add(x_far, y1, x_wall, y1)
        w.add(x_wall, y0, x_wall, min(a[1], b[1])); w.add(x_wall, max(a[1], b[1]), x_wall, y1)
        xa, xb = sorted((x_wall, x_far))
        F += _room_items(xa, y0, xb, y1, "W" if sgn > 0 else "E", "meeting" if sgn > 0 else "storage", rng)
    # round glass core, angled planters and screens, benches
    F.append(("cyl", cx, cy, 2.6, 1.2, "glass"))
    for k in range(4):
        ang = math.radians(45 + 90 * k)
        px, py = cx + 7.3 * math.cos(ang), cy + 7.3 * math.sin(ang)
        F.append(("box", px, py, 1.1, 0.45, ang + math.pi / 2, 0.7, "wood"))
    for k in (0, 2):
        ang = math.radians(90 * k)
        x0, y0 = cx + 5.0 * math.cos(ang - 0.25), cy + 5.0 * math.sin(ang - 0.25)
        w.add(x0, y0, x0 + 1.8 * math.cos(ang + 1.2), y0 + 1.8 * math.sin(ang + 1.2), mat="glass", frame=False)
    F += [("box", cx, cy + 9.6, 1.2, 0.3, 0.0, 0.45, "wood"), ("box", cx, cy - 9.6, 1.2, 0.3, 0.0, 0.45, "wood"),
          ("cyl", cx + 9.0, cy + 5.5, 0.25, 0.9, "plastic"), ("cyl", cx - 9.0, cy - 5.5, 0.25, 0.9, "plastic")]
    pillars = [(cx + 4.2 * math.cos(math.radians(a)), cy + 4.2 * math.sin(math.radians(a)), 0.18) for a in (0, 120, 240)]
    objects = [(cx + 11.0, cy + 3.0, 0.8), (cx - 3.0, cy + 11.0, 0.8), (cx - 10.5, cy - 4.0, 0.6)]
    tour = [(cx + 5.5, cy + 2.5), (cx, cy + 5.2), (cx - 5.5, cy + 2.5), (cx - 5.5, cy - 2.5), (cx, cy - 5.2),
            (cx + 5.0, cy - 2.0), (cx + 9.8, cy), (cx + 13.5, cy + 3.3), (cx + 15.6, cy + 2.5), (cx + 15.6, cy - 2.5), (cx + 13.5, cy - 3.3),
            (cx + 9.5, cy - 0.5), (cx + 7.5, cy - 6.5), (cx, cy - 9.0), (cx - 7.5, cy - 6.5), (cx - 9.8, cy),
            (cx - 14.0, cy - 1.6), (cx - 15.5, cy + 1.0), (cx - 13.0, cy + 2.3), (cx - 9.5, cy + 0.5),
            (cx - 7.5, cy + 6.5), (cx, cy + 9.0), (cx + 7.5, cy + 6.5)]
    return FloorPlan("atrium", w.walls, tour=tour, pillars=pillars, objects=objects, furniture=F, frames=w.frames,
                     edges=w.edges, bounds=(cx - R - 7, cx + R + 7, cy - R, cy + R))


# ---------------------------------------------------------------------- simple test plans (session 1)
def plan_loop() -> FloorPlan:
    W, D, c = 16.0, 10.0, 2.6
    walls = _rect(0, 0, W, D) + _rect(c, c, W - c, D - c)
    h = c / 2
    loop = [(h, h), (W - h, h), (W - h, D - h), (h, D - h)]
    objects = [(5.0, 0.25, 1.0), (11.0, D - 0.25, 1.0), (W - 0.25, 5.0, 0.6), (0.25, 3.0, 0.6), (8.0, c - 0.25, 0.8)]
    return FloorPlan("loop", walls, loops=[loop], objects=objects, bounds=(0, W, 0, D))


def plan_figure8() -> FloorPlan:
    W, D, c = 24.0, 10.0, 2.6
    xm = W / 2
    walls = _rect(0, 0, W, D) + _rect(c, c, xm - c / 2, D - c) + _rect(xm + c / 2, c, W - c, D - c)
    h = c / 2
    eight = [(xm, h), (W - h, h), (W - h, D - h), (xm, D - h), (xm, h), (h, h), (h, D - h), (xm, D - h)]
    objects = [(4.0, 0.25, 1.0), (19.0, 0.25, 0.8), (20.0, D - 0.25, 1.0), (6.0, D - 0.25, 0.7), (xm - c / 2 + 0.25, D / 2, 0.5),
               (0.25, 6.0, 0.6), (W - 0.25, 3.5, 0.6)]
    return FloorPlan("figure8", walls, loops=[eight], objects=objects, bounds=(0, W, 0, D))


def plan_hall() -> FloorPlan:
    W, D = 20.0, 12.0
    walls = _rect(0, 0, W, D)
    pillars = [(x, y, 0.15) for x in (6.0, 10.0, 14.0) for y in (4.5, 7.5)]
    t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    oval = [(10.0 + 7.0 * math.cos(a), 6.0 + 3.8 * math.sin(a)) for a in t]
    objects = [(2.0, 1.0, 0.8), (18.5, 10.5, 0.8), (3.0, 11.0, 0.6), (17.0, 1.5, 0.6)]
    return FloorPlan("hall", walls, loops=[oval], pillars=pillars, objects=objects, bounds=(0, W, 0, D))


PLANS = {"office": plan_office, "warehouse": plan_warehouse, "atrium": plan_atrium,
         "loop": plan_loop, "figure8": plan_figure8, "hall": plan_hall}


# ====================================================================== world
def build_world(plan: FloorPlan, seed: int = 0) -> World:
    """FloorPlan -> sondec World."""
    rng = np.random.default_rng(seed)
    w = World(plan.name, "rtissim2d", plan.bounds, rng=rng)
    for wall in plan.walls:
        x0, y0, x1, y1 = wall[:4]
        mat = wall[4] if len(wall) > 4 else "wall"
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-3:
            continue
        w.box(((x0 + x1) / 2, (y0 + y1) / 2, H_WALL / 2), (L / 2 + T_WALL / 2, T_WALL / 2, H_WALL / 2),
              math.atan2(y1 - y0, x1 - x0), mat, "wall", edges=False)
    for x, y, r in plan.pillars:
        w.cylinder((x, y, 0.0), (x, y, 2.0), r, "metal", "pillar")
    _furnish(w, plan.furniture)
    if plan.frames:
        o = w.new_obj("door frame")
        for x, y in plan.frames:
            w.line((x, y, 0.0), (x, y, H_WALL), "frame", o)
    if plan.edges:
        o = w.new_obj("wall end")
        for x, y in plan.edges:
            w.line((x, y, 0.0), (x, y, H_WALL), "edge", o)
    for x, y, s in plan.objects:
        o = w.new_obj("object", size=0.05)
        MAT.setdefault(f"object{s:.2f}", (0.2 * s, 0.0, 0.0))
        w.line((x, y, 0.25), (x, y, 0.6), f"object{s:.2f}", o)
    oc = w.new_obj("corners")
    for x, y in concave_corners(plan, w):
        w.line((x, y, 0.0), (x, y, H_WALL), "corner2d", oc)
    return w


def _furnish(w: World, items):
    for it in items:
        kind = it[0]
        if kind == "box":                                   # cabinet / desk / counter: a solid block
            _, x, y, hx, hy, yaw, h, mat = it
            w.box((x, y, h / 2), (hx, hy, h / 2), yaw, mat, "furniture", edges=True, edge_mat="edge")
        elif kind == "cyl":
            _, x, y, r, h, mat = it
            w.cylinder((x, y, 0.0), (x, y, h), r, mat, "furniture", caps=True)
        elif kind == "table":                               # four legs, top at 0.75 m; footprint blocks the robot
            _, x, y, hx, hy, yaw = it
            c, s = math.cos(yaw), math.sin(yaw)
            w.box((x, y, 0.735), (hx, hy, 0.015), yaw, "wood", "table top", edges=True, edge_mat="edge")
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx, ly = sx * (hx - 0.06), sy * (hy - 0.06)
                    px, py = x + c * lx - s * ly, y + s * lx + c * ly
                    w.cylinder((px, py, 0), (px, py, 0.72), 0.025, "metal", "table leg")
            w.foot_boxes.append((x, y, hx, hy, yaw))
        elif kind == "chair":                               # four thin legs + seat at 0.46 m + back
            _, x, y, yaw = it
            c, s = math.cos(yaw), math.sin(yaw)
            for lx, ly in ((0.2, 0.2), (0.2, -0.2), (-0.2, 0.2), (-0.2, -0.2)):
                px, py = x + c * lx - s * ly, y + s * lx + c * ly
                w.cylinder((px, py, 0), (px, py, 0.45), 0.012, "metal", "chair leg")
            w.box((x, y, 0.46), (0.22, 0.22, 0.02), yaw, "fabric", "chair seat", edges=False)
            w.box((x - c * 0.21, y - s * 0.21, 0.72), (0.02, 0.21, 0.24), yaw, "fabric", "chair back", edges=False)
            w.foot_boxes.append((x, y, 0.24, 0.24, yaw))
        elif kind == "rack":                                # double-sided pallet rack: boxed goods + uprights
            _, x, y, hx, hy, h = it
            w.box((x, y, h / 2), (hx, hy - 0.05, h / 2), 0.0, "cardboard", "rack goods", edges=False)
            n_bay = max(1, int(round(2 * hx / 2.7)))
            for a in np.linspace(-hx, hx, n_bay + 1):
                for sy in (-1, 1):
                    w.cylinder((x + a, y + sy * hy, 0), (x + a, y + sy * hy, h), 0.04, "metal", "rack upright")
            for z in (0.12, 1.3):                           # beam fronts (horizontal ridges)
                o = w.new_obj("rack beam")
                for sy in (-1, 1):
                    w.line((x - hx, y + sy * hy, z), (x + hx, y + sy * hy, z), "ridge", o, density=6.0)
        else:
            raise ValueError(kind)


# ====================================================================== corners and free space
def concave_corners(plan: FloorPlan, world: World | None = None):
    """Concave corners of all wall junctions (L and T, any angle), on the drivable side: for every pair of walls that
    meet, every angle < 180 deg between their arms, a point on the bisector 1.5 cm off both faces; kept if it lies in
    the free space connected to the start of the drive."""
    free = drivable(plan, world)
    W = [np.asarray(w[:4], float) for w in plan.walls]
    out = []
    for i in range(len(W)):
        a0, a1 = W[i][:2], W[i][2:]
        ua = a1 - a0; La = np.linalg.norm(ua); ua /= La
        for j in range(i + 1, len(W)):
            b0, b1 = W[j][:2], W[j][2:]
            ub = b1 - b0; Lb = np.linalg.norm(ub); ub /= Lb
            den = ua[0] * ub[1] - ua[1] * ub[0]
            if abs(den) < 1e-3:
                continue
            d = b0 - a0
            ta = (d[0] * ub[1] - d[1] * ub[0]) / den
            tb = (d[0] * ua[1] - d[1] * ua[0]) / den
            tol = T_WALL / 2 + 1e-3
            if not (-tol <= ta <= La + tol and -tol <= tb <= Lb + tol):
                continue
            p = a0 + ta * ua
            arms_a = [s * ua for s, ok in ((1, La - ta > T_WALL), (-1, ta > T_WALL)) if ok]
            arms_b = [s * ub for s, ok in ((1, Lb - tb > T_WALL), (-1, tb > T_WALL)) if ok]
            for da in arms_a:
                for db in arms_b:
                    th = math.acos(float(np.clip(da @ db, -1, 1)))
                    if th > math.radians(170):
                        continue
                    bis = (da + db) / np.linalg.norm(da + db)
                    c = p + (T_WALL / 2 + 0.015) / math.sin(th / 2) * bis
                    if free(c):
                        out.append((float(c[0]), float(c[1])))
    return out


def drivable(plan: FloorPlan, world: World | None = None, res: float = 0.02):
    """-> predicate p -> bool: p lies in the free space connected to the start of the drive (walls rasterised with
    their thickness, plus the world's footprints if given; flood fill from the first tour / loop waypoint)."""
    from scipy import ndimage
    xmin, xmax, ymin, ymax = plan.bounds
    nx, ny = int(math.ceil((xmax - xmin) / res)), int(math.ceil((ymax - ymin) / res))
    X, Y = np.meshgrid(xmin + (np.arange(nx) + 0.5) * res, ymin + (np.arange(ny) + 0.5) * res, indexing="ij")
    occ = np.zeros((nx, ny), bool)
    for wall in plan.walls:
        x0, y0, x1, y1 = wall[:4]
        a, b = np.array([x0, y0]), np.array([x1, y1])
        ab = b - a
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        occ |= np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]) <= T_WALL / 2
    for x, y, r in plan.pillars:
        occ |= np.hypot(X - x, Y - y) <= r
    lab, _ = ndimage.label(~occ)
    sx, sy = (plan.tour or plan.loops[0])[0]
    k = lab[min(nx - 1, int((sx - xmin) / res)), min(ny - 1, int((sy - ymin) / res))]

    def free(p):
        i, j = int((p[0] - xmin) / res), int((p[1] - ymin) / res)
        return 0 <= i < nx and 0 <= j < ny and lab[i, j] == k and k > 0
    return free


# ====================================================================== driving line
def driving_line(plan: FloorPlan, world: World, ds: float = 0.05) -> np.ndarray:
    """Closed smooth driving line [K, 2]: the A* tour on the ground-truth grid, or the simple plans' centre line."""
    if not plan.tour:
        from .robot import smooth_closed_path
        return smooth_closed_path(plan.loops[0], ds=ds)
    from .planner import GridPlanner
    occ, org, res = world.occupancy(0.05)
    return GridPlanner(occ, org, res).tour(plan.tour, ds=ds, closed=True)
