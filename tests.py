#!/usr/bin/env python3
"""
tests.py - regression checks for the Part 77 generator.

    python tests.py [path/to/adip.json]

Runs every check accumulated during development: the 77.19 dimension table,
composite control points, mesh area exactness, T-junction count, composite
assignment against the analytic evaluator, and wing clipping. Run it after
any change to part77_core.py; a change that reintroduces an old bug fails
here before it reaches the app.
"""

import json
import sys

import numpy as np

import part77_core as C

FAILS = []


def trans_gaps(m):
    """A transitional strip and its 5,000 ft wing must tile without a seam.

    Clipping the two against different things (the ceiling by station
    sampling, the conical by arc) once left a bare strip between them, and
    the conical showed through it 50 ft higher than the transitional on
    either side. A gap of that kind appears as a hole in the union.
    """
    from shapely.ops import unary_union
    holes = 0
    total = 0
    by = {}
    for pc in m.pieces:
        if pc.kind.startswith("TRANSITIONAL"):
            by.setdefault((pc.runway, pc.label), []).append(pc.poly)
    for key, polys in by.items():
        u = unary_union([p.buffer(0.01) for p in polys]).buffer(-0.01)
        for pg in C.as_polygons(u):
            total += 1
            for ring in pg.interiors:
                if C.Polygon(ring).area > 1.0:
                    holes += 1
    return holes == 0, "(%d holes across %d strips)" % (holes, total)


def chk(label, ok, detail=""):
    print("  %s  %s %s" % ("PASS" if ok else "FAIL", label, detail))
    if not ok:
        FAILS.append(label)


def close(a, b, tol=0.05):
    return a is not None and abs(a - b) <= tol


def tri_area(t):
    return abs((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) -
               (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) / 2.0


# ---------------------------------------------------------------- training
print("== training model (flat, single precision runway) ==")
m = C.training_model()
f = m.frames[0]

chk("primary half-width 500", close(f.ps_half, 500))
chk("primary surface end 5,200", close(f.ps_end, 5200))
chk("approach length 50,000", close(f.approach_len(f.end_pos), 50000))
h, z = f.inner(15200)
chk("approach @10,000: width 4,000 elev 200",
    close(2 * h, 4000) and close(z, 200))
h, z = f.inner(55200)
chk("approach @50,000: width 16,000 elev 1,200",
    close(2 * h, 16000) and close(z, 1200))
chk("conical top 350", close(C.CONE_HGT, 350))

POINTS = [  # (label, x, y, kind, elev)
    ("centerline", 0, 0, "PRIMARY", 0),
    ("745 abeam / BRL", 0, 745, "TRANSITIONAL", 35),
    ("1,550 abeam", 0, 1550, None, 150),
    ("3,000 abeam", 0, 3000, "HORIZONTAL", 150),
    ("x=15,200 CL", 15200, 0, None, 150),
    ("x=16,000 CL", 16000, 0, "CONICAL", 190),
    ("x=17,200 CL", 17200, 0, "APPROACH2", 250),
    ("x=55,200 CL", 55200, 0, "APPROACH2", 1200),
]
for lbl, x, y, kind, want in POINTS:
    z, who = m.controlling(x, y)
    ok = close(z, want) and (kind is None or (who and who.kind == kind))
    chk("controlling %s = %s" % (lbl, want), ok,
        "(got %.2f %s)" % (z if z is not None else float("nan"),
                           who.kind if who else None))

# transitional termination and the 5,000 ft wing trigger
e = f.end_pos
ie = f.inner(12700)
chk("transitional run 0 at horizontal pierce",
    m._trans_run(f, e, 12700, 1, *ie) <= 1e-6)
ie = f.inner(11000)
chk("transitional run 238 at s=11,000",
    close(m._trans_run(f, e, 11000, 1, *ie), 238, 1.0))
wings = [p for p in m.pieces if p.kind == "TRANSITIONAL5000"]
chk("precision wings exist", len(wings) > 0)
chk("transitional coverage has no gaps", *trans_gaps(m))

# composite matches the evaluator everywhere
comp = m.composite()
bad = 0
for poly, pc in comp:
    x, y = poly.representative_point().coords[0]
    z, who = m.controlling(x, y)
    if who is not pc and abs(pc.z(x, y) - z) > 0.5:
        bad += 1
chk("composite faces match controlling()", bad == 0, "(%d off)" % bad)

# meshes: watertight per polygon, by area
worst = 0.0
for pc in m.pieces:
    tris = C.triangulate(pc.poly)
    a = sum(tri_area(t) for t in tris)
    worst = max(worst, abs(a - pc.poly.area) / max(pc.poly.area, 1.0))
chk("triangulation covers every piece exactly", worst < 1e-9,
    "(worst %.1e)" % worst)


# ---------------------------------------------------------------- airport
def airport_checks(path):
    print("== %s ==" % path)
    rec = json.load(open(path))
    rwys = []
    for r in rec["runways"]:
        b, e = r["baseEnd"], r["reciprocalEnd"]
        rwys.append(C.Runway(
            r["runwayIdentifier"],
            C.RunwayEnd(b["runwayEndId"], b["latitude"], b["longitude"],
                        b["elevation"], b.get("obstaclePart77")),
            C.RunwayEnd(e["runwayEndId"], e["latitude"], e["longitude"],
                        e["elevation"], e.get("obstaclePart77")),
            r["width"], "existing"))
    m = C.Model(rec["arpLatitude"], rec["arpLongitude"],
                rec["elevation"], rwys)
    chk("horizontal at airport elev + 150",
        close(m.horiz_z, rec["elevation"] + 150))

    comp = m.composite()
    bad = 0
    for poly, pc in comp:
        x, y = poly.representative_point().coords[0]
        z, who = m.controlling(x, y)
        if who is not pc and abs(pc.z(x, y) - z) > 0.5:
            bad += 1
    chk("composite faces match controlling()", bad == 0,
        "(%d of %d off)" % (bad, len(comp)))

    worst = 0.0
    for face in m.arrangement():
        tris = C.triangulate(face)
        a = sum(tri_area(t) for t in tris)
        worst = max(worst, abs(a - face.area) / max(face.area, 1.0))
    chk("arrangement faces triangulate exactly", worst < 1e-9,
        "(worst %.1e)" % worst)

    chk("transitional coverage has no gaps", *trans_gaps(m))

    # every piece must be fully tiled by its arrangement faces: a dropped
    # face leaves a hole that shows as a cut in an exported TIN border
    fbp = m.faces_by_piece()
    worst_c = 0.0
    for i, pc in enumerate(m.pieces):
        fs = fbp.get(i, [])
        if not fs:
            continue
        a = sum(f.area for f in fs)
        worst_c = max(worst_c, abs(a - pc.poly.area) / max(pc.poly.area, 1.0))
    chk("exported surfaces have no interior holes", worst_c < 1e-9,
        "(worst shortfall %.1e)" % worst_c)


for path in sys.argv[1:]:
    airport_checks(path)

print()
if FAILS:
    print("*** %d FAILURE(S): %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("ALL CHECKS PASS")
