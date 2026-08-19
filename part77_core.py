#!/usr/bin/env python3
"""
part77_core.py - 14 CFR Part 77.19 surfaces from an editable runway table.

Design notes
------------
Every Part 77 surface except the conical is PLANAR over its own footprint:

  primary        z = runway centerline profile, linear between end elevations
  approach       z = end elevation + d/slope, linear in d
  transitional   z = z_inner(station) + offset/7, linear in both

so each piece carries polygon + plane coefficients (z = a + b*x + c*y). That
makes the composite exact: boundaries between two planes are straight lines,
not sampled cells, and a viewer can evaluate elevation at the cursor in
closed form.

The conical is radial off the horizontal surface boundary and is the one
surface that needs a curve where it crosses something else.
"""

import io
import json
import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union, split as shp_split

US_FT = 1200.0 / 3937.0
EARTH_R_FT = 20925721.8  # mean earth radius, US survey feet

# --- 77.19 criteria, keyed by ADIP obstaclePart77 code ----------------------
PART77 = {
    "A(V)":  dict(label="Utility, visual", short="Utility visual",
                  primary=250,  outer=1250,  slopes=[(5000, 20)],
                  horiz_rad=5000,  precision=False),
    "A(NP)": dict(label="Utility, nonprecision", short="Utility NPI",
                  primary=500,  outer=2000,  slopes=[(5000, 20)],
                  horiz_rad=5000,  precision=False),
    "B(V)":  dict(label="Other than utility, visual", short="Visual",
                  primary=500,  outer=1500,  slopes=[(5000, 20)],
                  horiz_rad=5000,  precision=False),
    "B(NP)": dict(label="Other than utility, nonprecision, vis > 3/4 mi",
                  short="NPI vis > 3/4 mi",
                  primary=500,  outer=3500,  slopes=[(10000, 34)],
                  horiz_rad=10000, precision=False),
    "C":     dict(label="Other than utility, nonprecision, vis as low as 3/4 mi",
                  short="NPI vis <= 3/4 mi",
                  primary=1000, outer=4000,  slopes=[(10000, 34)],
                  horiz_rad=10000, precision=False),
    "PIR":   dict(label="Precision instrument", short="Precision",
                  primary=1000, outer=16000, slopes=[(10000, 50), (40000, 40)],
                  horiz_rad=10000, precision=True),
}
PART77["MIR"] = dict(PART77["PIR"], label="Military (treated as precision)",
                     short="Military")
CODES = ["A(V)", "A(NP)", "B(V)", "B(NP)", "C", "PIR", "MIR"]

PRIMARY_EXT = 200.0
TRANS_SLOPE = 7.0
TRANS_RUN_PRECISION = 5000.0
HORIZ_HGT = 150.0
CONE_SLOPE = 20.0
CONE_HORIZ = 4000.0
CONE_HGT = HORIZ_HGT + CONE_HORIZ / CONE_SLOPE

COLORS = {
    "PRIMARY": "#3ddc84", "APPROACH": "#ff5a5a", "APPROACH2": "#ffab3d",
    "TRANSITIONAL": "#5b97ff", "TRANSITIONAL5000": "#a8c8ff",
    "HORIZONTAL": "#22ded0", "CONICAL": "#ec6bec", "RUNWAY": "#f5c542",
}


# ===========================================================================
# Local projection: equirectangular about a reference point, feet.
# Good to a few feet over the ~20 mile extent of a Part 77 model and keeps
# the app free of a projection dependency. DXF export can restate in state
# plane if a CRS is given.
# ===========================================================================
class LocalFrame:
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.kx = EARTH_R_FT * math.cos(math.radians(lat0)) * math.pi / 180.0
        self.ky = EARTH_R_FT * math.pi / 180.0

    def fwd(self, lat, lon):
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)

    def inv(self, x, y):
        return (self.lat0 + y / self.ky, self.lon0 + x / self.kx)

    def inv_ring(self, coords):
        return [[self.inv(x, y)[0], self.inv(x, y)[1]] for x, y in coords]


# ===========================================================================
# Runway
# ===========================================================================
class RunwayEnd:
    def __init__(self, ident, lat, lon, elev, code):
        self.id = ident
        self.lat, self.lon = float(lat), float(lon)
        self.elev = float(elev)
        self.code = code if code in PART77 else "B(V)"
        self.crit = PART77[self.code]


class Runway:
    def __init__(self, ident, base, recip, width=150.0, status="existing"):
        self.id = ident
        self.base, self.recip = base, recip
        self.width = float(width)
        self.status = status

    @property
    def ends(self):
        return [self.base, self.recip]


class Frame:
    """Runway-local coordinates: s along centerline (0 at midpoint, positive
    toward the reciprocal end), t offset positive to the left."""

    def __init__(self, rwy, lf, apt_elev):
        a = np.array(lf.fwd(rwy.base.lat, rwy.base.lon))
        b = np.array(lf.fwd(rwy.recip.lat, rwy.recip.lon))
        d = b - a
        self.len = float(np.hypot(*d))
        if self.len < 1.0:
            raise ValueError("Runway %s: the two ends are at the same place."
                             % rwy.id)
        self.u = d / self.len
        self.v = np.array([-self.u[1], self.u[0]])
        self.mid = (a + b) / 2.0
        self.rwy = rwy
        self.half = self.len / 2.0
        self.ps_end = self.half + PRIMARY_EXT
        self.ps_half = max(e.crit["primary"] for e in rwy.ends) / 2.0
        self.end_neg, self.end_pos = rwy.base, rwy.recip
        self.z_neg, self.z_pos = rwy.base.elev, rwy.recip.elev
        self.horiz_z = apt_elev + HORIZ_HGT

    def world(self, s, t):
        p = self.mid + self.u * s + self.v * t
        return (float(p[0]), float(p[1]))

    def local(self, x, y):
        d = np.array([x, y]) - self.mid
        return float(d @ self.u), float(d @ self.v)

    def end_for(self, s):
        return self.end_pos if s >= 0 else self.end_neg

    def centerline_z(self, s):
        if s <= -self.half:
            return self.z_neg
        if s >= self.half:
            return self.z_pos
        return self.z_neg + (s + self.half) / self.len * (self.z_pos - self.z_neg)

    def approach_len(self, e):
        return sum(L for L, _ in e.crit["slopes"])

    def inner(self, s):
        """(half width, elevation) of the primary or approach edge."""
        a = abs(s)
        if a <= self.ps_end:
            return self.ps_half, self.centerline_z(s)
        e = self.end_for(s)
        d = a - self.ps_end
        total = self.approach_len(e)
        if d > total + 1e-6:
            return None
        half = self.ps_half + (e.crit["outer"] / 2.0 - self.ps_half) * (d / total)
        z, rem = e.elev, d
        for L, slope in e.crit["slopes"]:
            seg = min(rem, L)
            z += seg / slope
            rem -= seg
            if rem <= 1e-9:
                break
        return half, z

    # -- plane coefficients in world xy for a linear model in (s, t) --------
    def plane_from_st(self, z0, dz_ds, dz_dt):
        """z = z0 + dz_ds*s + dz_dt*t, expressed as z = a + b*x + c*y."""
        b = dz_ds * self.u[0] + dz_dt * self.v[0]
        c = dz_ds * self.u[1] + dz_dt * self.v[1]
        a = z0 - (b * self.mid[0] + c * self.mid[1])
        return (float(a), float(b), float(c))


# ===========================================================================
# Surface pieces
# ===========================================================================
class Piece:
    __slots__ = ("kind", "poly", "plane", "label", "runway")

    def __init__(self, kind, poly, plane, label, runway=None):
        self.kind = kind
        self.poly = poly
        self.plane = plane          # (a, b, c) or None for the conical
        self.label = label
        self.runway = runway

    def z(self, x, y):
        a, b, c = self.plane
        return a + b * x + c * y

    def z_grid(self, X, Y):
        a, b, c = self.plane
        return a + b * X + c * Y


class ConicalPiece(Piece):
    def __init__(self, poly, boundary, base_z, label):
        super().__init__("CONICAL", poly, None, label)
        self.boundary = boundary
        self.base_z = base_z
        v = np.asarray(boundary.coords, float)
        self.p0 = v[:-1]
        self.d = v[1:] - v[:-1]
        self.dd = np.maximum((self.d ** 2).sum(1), 1e-12)

    def _dist(self, X, Y):
        """Distance from each point to the horizontal surface boundary,
        over all segments at once."""
        px = X.ravel()[:, None] - self.p0[None, :, 0]
        py = Y.ravel()[:, None] - self.p0[None, :, 1]
        t = np.clip((px * self.d[None, :, 0] + py * self.d[None, :, 1])
                    / self.dd[None, :], 0.0, 1.0)
        dx = px - t * self.d[None, :, 0]
        dy = py - t * self.d[None, :, 1]
        return np.sqrt((dx * dx + dy * dy).min(axis=1)).reshape(X.shape)

    def z(self, x, y):
        return self.base_z + Point(x, y).distance(self.boundary) / CONE_SLOPE

    def z_grid(self, X, Y):
        return self.base_z + self._dist(X, Y) / CONE_SLOPE


# ===========================================================================
# Model
# ===========================================================================
class Model:
    def __init__(self, apt_lat, apt_lon, apt_elev, runways):
        self.lf = LocalFrame(apt_lat, apt_lon)
        self.elev = float(apt_elev)
        self.horiz_z = self.elev + HORIZ_HGT
        self.cone_z = self.elev + CONE_HGT
        self.runways = runways
        self.frames = [Frame(r, self.lf, self.elev) for r in runways]
        self.notes = []
        self._build_horizontal()
        self.pieces = []
        for f in self.frames:
            self._runway_pieces(f)
        self._hc_pieces()

    # -- horizontal / conical ------------------------------------------
    def _build_horizontal(self):
        circles = []
        for f in self.frames:
            for s, e in ((-f.ps_end, f.end_neg), (f.ps_end, f.end_pos)):
                circles.append(Point(f.world(s, 0.0))
                               .buffer(e.crit["horiz_rad"], quad_segs=256))
        # 77.19(c) tangent-line construction. Arcs are drawn fine and then
        # thinned to a quarter-foot, which keeps the boundary honest without
        # dragging thousands of vertices through every later operation.
        self.hpoly = unary_union(circles).convex_hull.simplify(0.25)
        self.cpoly = self.hpoly.buffer(CONE_HORIZ, quad_segs=256)
        self.hbound = self.hpoly.exterior

    def hc_ceiling(self, x, y):
        p = Point(x, y)
        if self.hpoly.covers(p):
            return self.horiz_z
        if self.cpoly.covers(p):
            return self.horiz_z + p.distance(self.hbound) / CONE_SLOPE
        return None

    def _hc_pieces(self):
        self.pieces.append(Piece("HORIZONTAL", self.hpoly,
                                 (self.horiz_z, 0.0, 0.0), "Horizontal"))
        ring = Polygon(self.cpoly.exterior).difference(self.hpoly)
        self.pieces.append(ConicalPiece(ring, self.hbound, self.horiz_z,
                                        "Conical"))

    # -- per runway ----------------------------------------------------
    def _runway_pieces(self, f):
        rid = f.rwy.id
        # primary, split at the runway ends so each part is planar
        segs = [(-f.ps_end, -f.half), (-f.half, f.half), (f.half, f.ps_end)]
        for s0, s1 in segs:
            z0, z1 = f.centerline_z(s0), f.centerline_z(s1)
            dz = (z1 - z0) / (s1 - s0)
            plane = f.plane_from_st(f.centerline_z(0.0) if s0 < 0 < s1
                                    else z0 - dz * s0, dz, 0.0)
            # rebuild exactly: z(s) = z0 + dz*(s - s0)
            plane = f.plane_from_st(z0 - dz * s0, dz, 0.0)
            poly = Polygon([f.world(s0, -f.ps_half), f.world(s1, -f.ps_half),
                            f.world(s1, f.ps_half), f.world(s0, f.ps_half)])
            self.pieces.append(Piece("PRIMARY", poly, plane,
                                     "%s primary" % rid, rid))

        # approach, one piece per slope segment
        for sign, e in ((-1, f.end_neg), (1, f.end_pos)):
            d0 = 0.0
            for k, (L, slope) in enumerate(e.crit["slopes"]):
                d1 = d0 + L
                s0, s1 = sign * (f.ps_end + d0), sign * (f.ps_end + d1)
                h0, z0 = f.inner(s0)
                h1, z1 = f.inner(s1)
                dz_ds = sign * (1.0 / slope)
                plane = f.plane_from_st(z0 - dz_ds * s0, dz_ds, 0.0)
                poly = Polygon([f.world(s0, -h0), f.world(s1, -h1),
                                f.world(s1, h1), f.world(s0, h0)])
                kind = "APPROACH" if k == 0 else "APPROACH2"
                self.pieces.append(Piece(
                    kind, poly, plane,
                    "%s approach %s%s" % (rid, e.id,
                                          "" if len(e.crit["slopes"]) == 1
                                          else " %d:1" % slope), rid))
                d0 = d1

        # transitional: one piece per (end, side, underlying segment)
        for sign, e in ((-1, f.end_neg), (1, f.end_pos)):
            bounds = [(0.0, f.half), (f.half, f.ps_end)]
            d0 = 0.0
            for L, _ in e.crit["slopes"]:
                bounds.append((f.ps_end + d0, f.ps_end + d0 + L))
                d0 += L
            for side in (-1, 1):
                for a0, a1 in bounds:
                    self._trans_piece(f, e, sign, side, a0, a1)

    def _trans_piece(self, f, e, sign, side, a0, a1):
        """Transitional strip over |s| in [a0, a1]. Linear in (s, t), so the
        plane is exact; only the outer footprint edge needs stepping, since
        it follows wherever the surface meets the horizontal or conical."""
        ie0, ie1 = f.inner(sign * a0), f.inner(sign * a1)
        if ie0 is None or ie1 is None:
            return
        # z_in(s) = zi0 + dzi*(s - s0); half(s) = h0 + dh*(s - s0)
        s0, s1 = sign * a0, sign * a1
        ds = s1 - s0
        if abs(ds) < 1e-6:
            return
        dzi = (ie1[1] - ie0[1]) / ds
        dh = (ie1[0] - ie0[0]) / ds
        # z = z_in + (side*t - half)/7   for the strip on this side
        dz_ds = dzi - dh / TRANS_SLOPE
        dz_dt = side / TRANS_SLOPE
        z_at_s0 = ie0[1] - side * (side * ie0[0]) / TRANS_SLOPE
        plane = f.plane_from_st(z_at_s0 - dz_ds * s0, dz_ds, dz_dt)

        n = max(4, int(abs(ds) / 200.0))
        stations = [s0 + ds * i / n for i in range(n + 1)]
        cross = self._conical_crossing_s(f, e, 1 if s0 >= 0 else -1, side,
                                         a0, a1)
        if cross is not None:
            sgn = 1 if s0 >= 0 else -1
            sc = sgn * cross
            stations += [sc - sgn * 0.5, sc + sgn * 0.5]
            stations.sort(key=lambda v: (v - s0) / ds)
        inner_pts, outer_pts = [], []
        for s in stations:
            ie = f.inner(s)
            if ie is None:
                continue
            half, z_in = ie
            run = self._trans_run(f, e, s, side, half, z_in)
            inner_pts.append(f.world(s, side * half))
            outer_pts.append(f.world(s, side * (half + max(run, 0.0))))
        if len(inner_pts) < 2:
            return
        ring = inner_pts + outer_pts[::-1]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1.0:
            return
        kind = "TRANSITIONAL"
        if e.crit["precision"]:
            mid = len(inner_pts) // 2
            if abs(np.hypot(*(np.array(outer_pts[mid]) - np.array(inner_pts[mid])))
                   - TRANS_RUN_PRECISION) < 1.0:
                kind = "TRANSITIONAL5000"
        self.pieces.append(Piece(kind, poly, plane,
                                 "%s transitional %s" % (f.rwy.id, e.id),
                                 f.rwy.id))

    def beyond_conical(self, f, s, side, half):
        """True where the approach edge at this station lies outside the
        conical surface limit. 77.19(d) attaches the 5,000 ft transitional
        only to portions of a precision approach that project through and
        beyond the conical."""
        x, y = f.world(s, side * half)
        return not self.cpoly.covers(Point(x, y))

    def _trans_run(self, f, e, s, side, half, z_in):
        """Outward run of the transitional at this station.

        Inside the conical limit the transitional rises 7:1 and terminates
        where it meets the horizontal or conical surface. Once the approach
        has climbed above both there is nothing left to terminate against and
        the transitional simply ends. Only past the conical limit does the
        flat 5,000 ft extension of 77.19(d) apply, and only for a precision
        approach.
        """
        if self.beyond_conical(f, s, side, half):
            return TRANS_RUN_PRECISION if e.crit["precision"] else 0.0

        def above(run):
            x, y = f.world(s, side * (half + run))
            ceil = self.hc_ceiling(x, y)
            if ceil is None:
                return True
            return z_in + run / TRANS_SLOPE >= ceil

        if above(0.0):
            return 0.0
        hi = (self.cone_z - z_in) * TRANS_SLOPE + CONE_HORIZ
        if not above(hi):
            return hi
        lo = 0.0
        for _ in range(40):
            m = (lo + hi) / 2.0
            if above(m):
                hi = m
            else:
                lo = m
        return lo

    def _conical_crossing_s(self, f, e, sign, side, a_lo, a_hi):
        """Station where the approach edge crosses the conical limit, so the
        footprint edge lands flush on it instead of on a sample station."""
        def out(a):
            ie = f.inner(sign * a)
            return ie is not None and self.beyond_conical(f, sign * a, side, ie[0])
        if out(a_lo) == out(a_hi):
            return None
        lo, hi = a_lo, a_hi
        for _ in range(40):
            m = (lo + hi) / 2.0
            if out(m) == out(a_lo):
                lo = m
            else:
                hi = m
        return (lo + hi) / 2.0

    # -- point query ---------------------------------------------------
    def controlling(self, x, y):
        best, who = None, None
        p = Point(x, y)
        for pc in self.pieces:
            if not pc.poly.covers(p) and pc.poly.distance(p) > 1e-6:
                continue
            z = pc.z(x, y)
            if best is None or z < best - 1e-9:
                best, who = z, pc
        return best, who

    # ===================================================================
    # Composite: exact vector partition
    # ===================================================================
    def composite(self):
        lines = [pc.poly.exterior for pc in self.pieces]
        for pc in self.pieces:
            lines.extend(pc.poly.interiors)
        merged = unary_union(lines)
        faces = [f for f in polygonize(merged) if f.area > 25.0]

        out = []
        for face in faces:
            out.extend(self._assign(face, 0))
        return out

    def _candidates(self, face):
        p = face.representative_point()
        return [pc for pc in self.pieces if pc.poly.covers(p)]

    def _assign(self, face, depth):
        cand = self._candidates(face)
        if not cand:
            return []
        if len(cand) == 1:
            return [(face, cand[0])]

        # sample the face: centroid plus boundary vertices
        pts = [face.representative_point().coords[0]]
        pts += list(face.exterior.coords)[:-1]
        winners = set()
        for x, y in pts:
            best, who = None, None
            for pc in cand:
                z = pc.z(x, y)
                if best is None or z < best - 1e-9:
                    best, who = z, pc
            winners.add(id(who))

        if len(winners) == 1 or depth >= 3:
            x, y = face.representative_point().coords[0]
            who = min(cand, key=lambda pc: pc.z(x, y))
            return [(face, who)]

        # two or more surfaces cross inside this face: cut on the crossing
        top = [pc for pc in cand if id(pc) in winners][:2]
        cut = self._crossing(face, top[0], top[1])
        if cut is None:
            x, y = face.representative_point().coords[0]
            return [(face, min(cand, key=lambda pc: pc.z(x, y)))]
        parts = []
        try:
            for piece in shp_split(face, cut).geoms:
                if piece.area > 25.0:
                    parts.extend(self._assign(piece, depth + 1))
        except Exception:
            x, y = face.representative_point().coords[0]
            return [(face, min(cand, key=lambda pc: pc.z(x, y)))]
        return parts or [(face, top[0])]

    def _crossing(self, face, A, B):
        """Line or curve where two surfaces are at equal elevation."""
        minx, miny, maxx, maxy = face.bounds
        if A.plane and B.plane:
            da = A.plane[0] - B.plane[0]
            db = A.plane[1] - B.plane[1]
            dc = A.plane[2] - B.plane[2]
            if abs(db) < 1e-12 and abs(dc) < 1e-12:
                return None
            # exact straight line da + db*x + dc*y = 0, extended past the face
            L = 2.0 * math.hypot(maxx - minx, maxy - miny) + 1000.0
            if abs(dc) >= abs(db):
                x0, x1 = minx - L, maxx + L
                return LineString([(x0, -(da + db * x0) / dc),
                                   (x1, -(da + db * x1) / dc)])
            y0, y1 = miny - L, maxy + L
            return LineString([(-(da + dc * y0) / db, y0),
                               (-(da + dc * y1) / db, y1)])
        # conical against a plane: trace the equal-elevation contour
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 48
        xs = np.linspace(minx, maxx, n)
        ys = np.linspace(miny, maxy, n)
        X, Y = np.meshgrid(xs, ys)
        Z = A.z_grid(X, Y) - B.z_grid(X, Y)
        fig = plt.figure()
        try:
            cs = plt.contour(X, Y, Z, levels=[0.0])
            segs = []
            for path in cs.get_paths():
                v = path.vertices
                if len(v) > 1:
                    segs.append(LineString(v))
        finally:
            plt.close(fig)
        if not segs:
            return None
        return MultiLineString(segs) if len(segs) > 1 else segs[0]

    # ===================================================================
    def obstacle_report(self, obstacles):
        """obstacles: list of dicts with description, lat, lon, elev."""
        rows = []
        for o in obstacles:
            x, y = self.lf.fwd(float(o["lat"]), float(o["lon"]))
            z, who = self.controlling(x, y)
            if z is None:
                rows.append(dict(o, surface="(outside Part 77)",
                                 surface_elev=None, penetration=None,
                                 penetrates=False))
                continue
            pen = float(o["elev"]) - z
            rows.append(dict(o, surface=who.label, surface_kind=who.kind,
                             surface_elev=round(z, 1),
                             penetration=round(pen, 1), penetrates=pen > 0))
        return rows


# ===========================================================================
# GeoJSON for the map, with plane coefficients so the viewer can read out
# elevation at the cursor without another round trip
# ===========================================================================
def to_geojson(model, composite=None):
    lf = model.lf

    def feat(poly, props):
        if poly.is_empty:
            return None
        rings = [lf.inv_ring(poly.exterior.coords)]
        for r in poly.interiors:
            rings.append(lf.inv_ring(r.coords))
        return {"type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": [[[c[1], c[0]] for c in r]
                                             for r in rings]},
                "properties": props}

    def props(pc):
        p = {"kind": pc.kind, "label": pc.label,
             "color": COLORS.get(pc.kind, "#888888")}
        if pc.plane:
            p["plane"] = list(pc.plane)
        else:
            p["conical"] = {"base_z": pc.base_z, "slope": CONE_SLOPE}
        return p

    layers = {}
    for pc in model.pieces:
        layers.setdefault(pc.kind, []).append(feat(pc.poly, props(pc)))
    comp, outlines = [], []
    if composite:
        by_kind = {}
        for poly, pc in composite:
            comp.append(feat(poly, props(pc)))
            by_kind.setdefault(pc.kind, []).append(poly)
        # Every atomic face has its own edges, and drawing all of them buries
        # the map in interior lines. Dissolve the faces of each surface so only
        # the outline of the region it governs is stroked.
        for kind, polys in by_kind.items():
            merged = unary_union([p.buffer(0.5) for p in polys]).buffer(-0.5)
            parts = [merged] if merged.geom_type == "Polygon" else \
                list(getattr(merged, "geoms", []))
            for pg in parts:
                if pg.is_empty or pg.area < 100.0:
                    continue
                f = feat(pg, {"kind": kind, "label": kind,
                              "color": COLORS.get(kind, "#888888")})
                if f:
                    outlines.append(f)

    rwy = []
    for f in model.frames:
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        rwy.append(feat(poly, {"kind": "RUNWAY", "label": f.rwy.id,
                               "color": COLORS["RUNWAY"],
                               "status": f.rwy.status}))
    return {
        "origin": [lf.lat0, lf.lon0], "kx": lf.kx, "ky": lf.ky,
        "horiz_z": model.horiz_z, "cone_z": model.cone_z,
        "hbound": lf.inv_ring(model.hpoly.exterior.coords),
        "layers": {k: [f for f in v if f] for k, v in layers.items()},
        "composite": [f for f in comp if f],
        "composite_outlines": outlines,
        "runways": [f for f in rwy if f],
    }


def triangulate(poly):
    """Triangles covering a polygon, holes included.

    Fanning from the centroid only works for convex rings; the conical is an
    annulus and composite faces can be concave, so both need a real
    triangulation. Delaunay over the boundary vertices, keeping only the
    triangles whose centroid is actually inside the polygon, handles every
    case here without another dependency.
    """
    from matplotlib.tri import Triangulation
    rings = [np.asarray(poly.exterior.coords)[:-1]]
    for r in poly.interiors:
        rings.append(np.asarray(r.coords)[:-1])
    pts = np.vstack(rings)
    if len(pts) < 3:
        return []
    try:
        tri = Triangulation(pts[:, 0], pts[:, 1])
    except Exception:
        return []
    out = []
    for a, b, c in tri.triangles:
        t = pts[[a, b, c]]
        if poly.covers(Point(t[:, 0].mean(), t[:, 1].mean())):
            out.append(t)
    return out


def to_mesh3d(model, composite=None, use_composite=False):
    """Triangles in local feet, grouped by surface, for the 3D view."""
    groups = {}

    def add(kind, poly, z_at):
        if poly.is_empty:
            return
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        buf = groups.setdefault(kind, [])
        for pg in polys:
            for t in triangulate(pg):
                for x, y in t:
                    buf.extend([round(float(x), 1), round(float(y), 1),
                                round(float(z_at(x, y)), 2)])

    if use_composite and composite:
        for poly, pc in composite:
            add("COMPOSITE:" + pc.kind, poly, pc.z)
    else:
        for pc in model.pieces:
            add(pc.kind, pc.poly, pc.z)
    for f in model.frames:
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        add("RUNWAY", poly, lambda x, y, f=f: f.centerline_z(f.local(x, y)[0]))

    colors = {}
    for k in groups:
        colors[k] = COLORS.get(k.split(":")[-1], "#888888")
    return {"groups": groups, "colors": colors,
            "elev": model.elev, "horiz_z": model.horiz_z,
            "cone_z": model.cone_z}


# ===========================================================================
def to_dxf(model, composite=None, epsg=None):
    import ezdxf
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 2
    msp = doc.modelspace()

    xf = None
    if epsg:
        from pyproj import CRS, Transformer
        crs = CRS.from_epsg(int(epsg))
        unit = crs.axis_info[0].unit_name.lower()
        k = 1.0 if ("foot" in unit or "feet" in unit) else 1.0 / US_FT
        tr = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

        def xf(x, y):
            lat, lon = model.lf.inv(x, y)
            px, py = tr.transform(lon, lat)
            return px * k, py * k

    def emit(poly, z_at, layer):
        if poly.is_empty:
            return 0
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        n = 0
        for pg in polys:
            for tri in triangulate(pg):
                out = []
                for x, y in tri:
                    zz = z_at(x, y)
                    X, Y = xf(x, y) if xf else (x, y)
                    out.append((X, Y, zz))
                out.append(out[2])
                msp.add_3dface(out, dxfattribs={"layer": layer})
                n += 1
        return n

    stats = {}
    for pc in model.pieces:
        lay = "P77-%s" % pc.kind
        stats[lay] = stats.get(lay, 0) + emit(pc.poly, pc.z, lay)
    for f in model.frames:
        lay = "P77-RUNWAY-%s" % f.rwy.id.replace("/", "-")
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        stats[lay] = stats.get(lay, 0) + emit(
            poly, lambda x, y, f=f: f.centerline_z(f.local(x, y)[0]), lay)
    if composite:
        for poly, pc in composite:
            stats["P77-COMPOSITE"] = stats.get("P77-COMPOSITE", 0) + \
                emit(poly, pc.z, "P77-COMPOSITE")

    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8"), stats


# ===========================================================================
# Presets
# ===========================================================================
def training_model():
    """The flat single-runway precision case the visualizer was built on:
    10,000 x 150 ft, precision both ends, airport elevation 0."""
    lat0, lon0 = 40.0, -100.0
    lf = LocalFrame(lat0, lon0)
    half = 5000.0
    b = lf.inv(-half, 0.0)
    r = lf.inv(half, 0.0)
    rwy = Runway("09/27",
                 RunwayEnd("09", b[0], b[1], 0.0, "PIR"),
                 RunwayEnd("27", r[0], r[1], 0.0, "PIR"),
                 width=150.0, status="training")
    return Model(lat0, lon0, 0.0, [rwy])
