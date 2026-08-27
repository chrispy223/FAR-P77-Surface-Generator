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


def group_checks(m, comp):
    """The four export groups must partition the composite exactly.

    A group that loses area has a hole; a group that gains it double-covers,
    which a TIN cannot hold. Both are invisible in 3D and obvious as cuts or
    folds once the surface is built in Civil 3D.
    """
    from shapely.ops import unary_union
    g = C.composite_groups(m, comp)
    total = sum(p.area for p, _ in comp)
    ga = sum(p.area for items in g.values() for p, _ in items)
    chk("groups conserve composite area",
        abs(ga - total) / max(total, 1.0) < 1e-9,
        "(%.4e vs %.4e)" % (ga, total))

    nface = sum(len(v) for v in g.values())
    chk("every composite face lands in exactly one group",
        nface == len(comp), "(%d of %d)" % (nface, len(comp)))

    # Each group must be single valued: its faces may not overlap in plan.
    # Tested pairwise, not by unary_union area, which loses thousands of
    # square feet to rounding when it dissolves a few hundred faces at state
    # plane magnitudes and reads as an overlap that is not there.
    from shapely.strtree import STRtree
    worst = 0.0
    for name, items in g.items():
        polys = [p for p, _ in items]
        tree = STRtree(polys)
        for i, p in enumerate(polys):
            for j in tree.query(p):
                if j > i:
                    worst = max(worst, p.intersection(polys[j]).area)
    chk("no group folds over itself in plan", worst < 1.0,
        "(worst pair overlap %.3f sf)" % worst)

    # No outer-approach piece may sit in the inner group: the inner group is
    # the below-horizontal set by definition.
    stray = [pc.kind for p, pc in g.get(C.GROUP_INNER, [])
             if pc.kind in ("APPROACH2", "TRANSITIONAL5000")
             and min(pc.z(x, y) for x, y in p.exterior.coords) > m.horiz_z + 0.5]
    chk("no outer-approach face lands in the inner group", not stray,
        "(%d stray)" % len(stray))

    # The inner group is the below-horizontal set by definition. Before v20
    # the transitional terminated against the conical where the approach edge
    # lay outside the horizontal perimeter, which pushed a narrow strip of it
    # up to +350 and read as a wall in 3D. It now stops on the horizontal
    # elevation and that band is carried by the 5,000 ft wing instead.
    hi = 0.0
    for p, pc in g.get(C.GROUP_INNER, []):
        hi = max(hi, max(pc.z(x, y) for x, y in p.exterior.coords))
    chk("inner group stays at or below the horizontal", hi <= m.horiz_z + 0.5,
        "(max %.2f vs horizontal %.2f)" % (hi, m.horiz_z))

    # An outer group holds only outer-approach geometry. It is NOT required
    # to sit above the horizontal: where a runway end is below the airport
    # elevation the outer segment starts under it and climbs through.
    for name in sorted(g):
        if name.endswith("OUTER APR"):
            kinds = set(pc.kind for _, pc in g[name])
            chk("%s holds only outer-approach geometry" % name,
                kinds <= set(C.APPROACH_FAMILY), "(%s)" % ", ".join(sorted(kinds)))


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
group_checks(m, comp)

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
    group_checks(m, comp)

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
