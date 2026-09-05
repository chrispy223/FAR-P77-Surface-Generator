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

__version__ = "v20"

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

# A precision approach is one surface under 77.19(b) that changes slope at
# 10,000 ft, and the transitional plus its 77.19(d) extension is one surface
# too. They are modelled as separate planar pieces because that is what the
# maths needs, but drawing them in different colours puts a seam across a
# continuous surface. Merge them for display.
DISPLAY_MERGE = {"APPROACH2": "APPROACH", "TRANSITIONAL5000": "TRANSITIONAL"}


def display_kind(kind, merge=True):
    if not merge:
        return kind
    pre, _, k = kind.rpartition(":")
    k = DISPLAY_MERGE.get(k, k)
    return (pre + ":" + k) if pre else k


# Where two surfaces meet, they are equal along the shared edge, and which
# one "wins" there is decided by noise. The transitional terminates exactly
# on the horizontal, and its outer edge is a polyline through stations 200 ft
# apart, so between stations the chord bulges a hair past the true curve and
# the transitional reads fractionally high. Lowest-wins then hands the
# horizontal a sliver a few hundred square feet in size, sitting inside the
# inner surface as a notch. Ties within TIE_TOL go to the surface with the
# lower priority number instead, which keeps the inner surfaces whole and
# matches how these are drawn.
TIE_TOL = 0.10

PRIORITY = {"PRIMARY": 0, "APPROACH": 1, "APPROACH2": 1,
            "TRANSITIONAL": 2, "TRANSITIONAL5000": 2,
            "CONICAL": 3, "HORIZONTAL": 4}


def lowest(cand, x, y):
    """The controlling surface at a point, ties broken by priority."""
    zs = [(pc.z(x, y), pc) for pc in cand]
    lo = min(z for z, _ in zs)
    tied = [(PRIORITY.get(pc.kind, 9), i, z, pc)
            for i, (z, pc) in enumerate(zs) if z <= lo + TIE_TOL]
    tied.sort()
    return tied[0][3]


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
def as_polygons(geom):
    """Flatten to simple polygons. Cleaning a self-touching strip can split it
    into several parts, and a piece is assumed to be one polygon everywhere
    downstream."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(as_polygons(g))
        return out
    return []


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
        from shapely.geometry import box as _box
        minx, miny, maxx, maxy = ring.bounds
        cx, cy = self.hpoly.centroid.x, self.hpoly.centroid.y
        d = 2.0 * max(maxx - minx, maxy - miny)
        # An annulus has a hole, and a hole cannot be ear clipped directly.
        # Quartering it about the centre yields four simple polygons.
        for q in (_box(cx - d, cy - d, cx, cy), _box(cx, cy - d, cx + d, cy),
                  _box(cx - d, cy, cx, cy + d), _box(cx, cy, cx + d, cy + d)):
            for pg in as_polygons(ring.intersection(q)):
                if pg.area > 1.0:
                    self.pieces.append(ConicalPiece(
                        pg, self.hbound, self.horiz_z, "Conical"))

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
        """Two parts per strip. Inside the conical limit the transitional
        rises 7:1 until it terminates on the horizontal or conical surface.
        Outside it, a precision approach carries the 5,000 ft extension of
        77.19(d); that wing is generated at full width and then clipped to
        the conical limit, so its front edge follows the arc itself rather
        than a straight station cut."""
        ie0, ie1 = f.inner(sign * a0), f.inner(sign * a1)
        if ie0 is None or ie1 is None:
            return
        s0, s1 = sign * a0, sign * a1
        ds = s1 - s0
        if abs(ds) < 1e-6:
            return
        dzi = (ie1[1] - ie0[1]) / ds
        dh = (ie1[0] - ie0[0]) / ds
        dz_ds = dzi - dh / TRANS_SLOPE
        dz_dt = side / TRANS_SLOPE
        z_at_s0 = ie0[1] - side * (side * ie0[0]) / TRANS_SLOPE
        plane = f.plane_from_st(z_at_s0 - dz_ds * s0, dz_ds, dz_dt)

        n = max(4, int(abs(ds) / 50.0))
        stations = [s0 + ds * i / n for i in range(n + 1)]

        def build(rings):
            inner_pts, outer_pts = rings
            if len(inner_pts) < 2:
                return None
            poly = Polygon(inner_pts + outer_pts[::-1])
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly

        def emit(poly, kind):
            if poly is None:
                return
            for pg in as_polygons(poly):
                if pg.area >= 1.0:
                    self.pieces.append(Piece(
                        kind, pg, plane,
                        "%s transitional %s" % (f.rwy.id, e.id), f.rwy.id))

        # terminated transitional, inside the conical limit
        inner_pts, outer_pts = [], []
        for st in stations:
            ie = f.inner(st)
            if ie is None:
                continue
            half, z_in = ie
            if self.beyond_conical(f, st, side, half):
                continue
            run = self._trans_run(f, e, st, side, half, z_in)
            if run <= 1e-6:
                continue
            inner_pts.append(f.world(st, side * half))
            outer_pts.append(f.world(st, side * (half + run)))
        term = build((inner_pts, outer_pts))
        emit(term, "TRANSITIONAL")

        # Precision wing. Building it at full width over every station and
        # subtracting the terminated strip leaves hairline slivers: the strip
        # is emitted as several pieces on different station bounds, and
        # differencing a multi-part polygon out of a single one does not
        # cancel exactly along the seams. Those slivers survive the composite
        # in the thin band where the wing crosses the horizontal and export
        # as detached threads. Build the wing off the strip's own outer edge
        # on the same stations instead, so the two are adjacent by
        # construction and nothing can fall between or overlap.
        if e.crit["precision"]:
            inner_pts, outer_pts = [], []
            for st in stations:
                ie = f.inner(st)
                if ie is None:
                    continue
                half, z_in = ie
                run = 0.0
                if not self.beyond_conical(f, st, side, half):
                    run = self._trans_run(f, e, st, side, half, z_in)
                if run >= TRANS_RUN_PRECISION:
                    continue
                inner_pts.append(f.world(st, side * (half + run)))
                outer_pts.append(
                    f.world(st, side * (half + TRANS_RUN_PRECISION)))
            emit(build((inner_pts, outer_pts)), "TRANSITIONAL5000")

    def beyond_conical(self, f, s, side, half):
        """True where the approach edge at this station lies outside the
        conical surface limit. 77.19(d) attaches the 5,000 ft transitional
        only to portions of a precision approach that project through and
        beyond the conical."""
        x, y = f.world(s, side * half)
        return not self.cpoly.covers(Point(x, y))

    def trans_ceiling(self, x, y):
        """Elevation the transitional terminates against.

        77.19(e) runs the transitional up to the horizontal surface, which
        is a plane 150 ft above the established airport elevation. It does
        not chase the conical: where the approach edge has already climbed
        past that plane the transitional has nothing left to reach and does
        not exist. Terminating against the conical instead fills the band
        just inside the conical limit with a narrow strip rising to +350,
        which reads as a wall in 3D and does not appear on the reference
        drawings.
        """
        p = Point(x, y)
        if self.hpoly.covers(p) or self.cpoly.covers(p):
            return self.horiz_z
        return None

    def _trans_run(self, f, e, s, side, half, z_in):
        """Outward run of the transitional at this station.

        Inside the conical limit the transitional rises 7:1 and terminates
        on the horizontal surface elevation. Once the approach has climbed
        above it there is nothing to terminate against and the transitional
        simply ends. Only past the conical limit does the flat 5,000 ft
        extension of 77.19(d) apply, and only for a precision approach.
        """
        if self.beyond_conical(f, s, side, half):
            return 0.0

        def above(run):
            x, y = f.world(s, side * (half + run))
            ceil = self.trans_ceiling(x, y)
            if ceil is None:
                return True
            return z_in + run / TRANS_SLOPE >= ceil

        if above(0.0):
            return 0.0
        hi = max(0.0, self.horiz_z - z_in) * TRANS_SLOPE
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
        if getattr(self, "_pb", None) is None or len(self._pb) != len(self.pieces):
            self._pb = np.array([pc.poly.bounds for pc in self.pieces])
        b = self._pb
        hit = np.nonzero((x >= b[:, 0] - 1.0) & (x <= b[:, 2] + 1.0) &
                         (y >= b[:, 1] - 1.0) & (y <= b[:, 3] + 1.0))[0]
        for i in hit:
            pc = self.pieces[i]
            if not pc.poly.covers(p) and pc.poly.distance(p) > 1e-3:
                continue
            z = pc.z(x, y)
            if best is None or z < best - 1e-9:
                best, who = z, pc
        return best, who

    # ===================================================================
    # Composite: exact vector partition
    # ===================================================================
    def arrangement(self):
        """Every piece boundary noded together into one planar subdivision.

        Faces produced this way share their edges vertex for vertex, so
        meshing the faces rather than the pieces removes the T-junctions you
        get when two surfaces abut but were sampled at different spacings.
        Cached: both the composite and the meshes are built from it.
        """
        if getattr(self, "_faces", None) is None:
            lines = [pc.poly.exterior for pc in self.pieces]
            for pc in self.pieces:
                lines.extend(pc.poly.interiors)
            merged = unary_union(lines)
            # Every face is kept. Dropping thin ones was left over from
            # chasing 3D fins, whose real cause turned out to be conical
            # chord error; all it does now is punch holes in the exported
            # surfaces, which a TIN border makes obvious.
            self._faces = [f for f in polygonize(merged) if f.area > 1e-6]
        return self._faces

    def faces_by_piece(self):
        """Which arrangement faces each piece covers."""
        if getattr(self, "_fbp", None) is None:
            out = {}
            for face in self.arrangement():
                p = face.representative_point()
                for i, pc in enumerate(self.pieces):
                    if pc.poly.covers(p):
                        out.setdefault(i, []).append(face)
            self._fbp = out
        return self._fbp

    def composite(self):
        faces = self.arrangement()
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

        # sample the face: representative point, boundary vertices, and edge
        # midpoints. A surface can dip below the assigned one strictly inside
        # an edge, and vertex-only sampling never sees it.
        pts = [face.representative_point().coords[0]]
        ext = list(face.exterior.coords)
        pts += ext[:-1]
        for i in range(len(ext) - 1):
            pts.append(((ext[i][0] + ext[i + 1][0]) / 2.0,
                        (ext[i][1] + ext[i + 1][1]) / 2.0))
        # The conical is a convex function, so it can dip below a plane in
        # the middle of a face while sitting above it along the whole
        # boundary. Boundary sampling alone never sees that dip; a coarse
        # interior grid does, and the contour split then finds the exact
        # crossing curve.
        minx, miny, maxx, maxy = face.bounds
        for gx in np.linspace(minx, maxx, 7)[1:-1]:
            for gy in np.linspace(miny, maxy, 7)[1:-1]:
                if face.contains(Point(gx, gy)):
                    pts.append((float(gx), float(gy)))
        winners = set()
        for x, y in pts:
            winners.add(id(lowest(cand, x, y)))

        if len(winners) == 1 or depth >= 8:
            x, y = face.representative_point().coords[0]
            return [(face, lowest(cand, x, y))]

        # Two or more surfaces are lowest somewhere in this face. Try every
        # winning pair until a cut actually divides it: with several
        # overlapping corridors, the first pair's crossing can lie entirely
        # outside the face, and shp_split hands the face back whole.
        tops = [pc for pc in cand if id(pc) in winners]
        for ai in range(len(tops)):
            for bi in range(ai + 1, len(tops)):
                cut = self._crossing(face, tops[ai], tops[bi])
                if cut is None:
                    continue
                try:
                    pieces = list(shp_split(face, cut).geoms)
                except Exception:
                    continue
                if len(pieces) < 2:
                    continue
                parts = []
                for piece in pieces:
                    if piece.area > 4.0:
                        parts.extend(self._assign(piece, depth + 1))
                if parts:
                    return parts
        x, y = face.representative_point().coords[0]
        return [(face, lowest(cand, x, y))]

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
        dk = display_kind(pc.kind)
        p = {"kind": dk, "label": pc.label,
             "color": COLORS.get(dk, "#888888")}
        if pc.plane:
            p["plane"] = list(pc.plane)
        else:
            p["conical"] = {"base_z": pc.base_z, "slope": CONE_SLOPE}
        return p

    layers = {}
    for pc in model.pieces:
        layers.setdefault(display_kind(pc.kind), []).append(
            feat(pc.poly, props(pc)))
    comp, outlines = [], []
    if composite:
        by_kind = {}
        for poly, pc in composite:
            comp.append(feat(poly, props(pc)))
            by_kind.setdefault(display_kind(pc.kind), []).append(poly)
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


def _earcut(ring, coll=1e-4):
    """Ear clipping on a simple CCW ring, with the containment test done in
    numpy across all remaining vertices at once. The scalar version spent
    tens of seconds on 26 million individual point-in-triangle checks for a
    single airport; this is the same algorithm at array speed.

    coll is a perpendicular distance in feet: on long boundary edges a raw
    cross-product epsilon misreads collinear vertices as convex, clips them,
    and leaves T-junction cracks along the boundary.
    """
    P = np.asarray(ring, float)
    n = len(P)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []
    while len(idx) > 3:
        m = len(idx)
        R = P[idx]
        cut = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            x = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if x <= 0:
                continue
            base = math.hypot(c[0] - a[0], c[1] - a[1])
            if base > 0 and x / base <= coll:
                continue
            d1 = (b[0]-a[0])*(R[:,1]-a[1]) - (b[1]-a[1])*(R[:,0]-a[0])
            d2 = (c[0]-b[0])*(R[:,1]-b[1]) - (c[1]-b[1])*(R[:,0]-b[0])
            d3 = (a[0]-c[0])*(R[:,1]-c[1]) - (a[1]-c[1])*(R[:,0]-c[0])
            ins = (d1 >= 0) & (d2 >= 0) & (d3 >= 0)
            ins[k - 1] = ins[k] = ins[(k + 1) % m] = False
            if ins.any():
                continue
            tris.append((tuple(a), tuple(b), tuple(c)))
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tris.append(tuple(tuple(P[i]) for i in idx))
        return tris
    if coll > 0.0:
        return _earcut(ring, coll / 100.0)
    return tris


def _tri_area(t):
    return abs((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) -
               (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) / 2.0


def triangulate(poly):
    """Watertight triangles covering a polygon.

    Split faces out of the composite can carry near-degenerate boundaries the
    ear clip stalls on, and a stall is a hole in the mesh. Every result is
    therefore checked by area, and shortfalls retried on cleaned geometry.
    """
    from shapely.geometry.polygon import orient

    def rings(pg):
        if pg.interiors:
            return [q for q in as_polygons(_split_quadrants(pg))]
        return [pg]

    def attempt(pg):
        out = []
        for q in rings(pg):
            out.extend(_earcut(list(orient(q, 1.0).exterior.coords)[:-1]))
        return out

    result = []
    for pg in as_polygons(poly):
        tris = attempt(pg)
        got = sum(_tri_area(t) for t in tris)
        if abs(got - pg.area) > max(pg.area, 1.0) * 1e-6:
            for fix in (pg.buffer(0), pg.simplify(0.02).buffer(0)):
                tris2 = []
                ok = True
                tot = 0.0
                for pg2 in as_polygons(fix):
                    tt = attempt(pg2)
                    tris2.extend(tt)
                    tot += sum(_tri_area(t) for t in tt)
                if abs(tot - pg.area) <= max(pg.area, 1.0) * 1e-4:
                    tris = tris2
                    break
        result.extend(tris)
    return [np.asarray(t, float) for t in result]


def _split_quadrants(poly):
    """Cut a ring into four simple pieces about its centroid, so no piece has
    a hole."""
    from shapely.geometry import box
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    d = max(maxx - minx, maxy - miny)
    quads = [box(cx - d, cy - d, cx, cy), box(cx, cy - d, cx + d, cy),
             box(cx - d, cy, cx, cy + d), box(cx, cy, cx + d, cy + d)]
    parts = []
    for q in quads:
        parts.extend(as_polygons(poly.intersection(q)))
    return parts


def mesh_polys(model, pc, poly, step=200.0):
    """Polygons to triangulate for one piece of one surface.

    Planar surfaces triangulate as they are: any chord across a plane still
    lies in the plane. The conical is curved, and a triangle whose vertices
    all sit near the rim spans the bowl at rim height — up to 175 ft above
    the true surface on a face the size of the conical ring. Cutting the
    polygon into concentric bands off the horizontal boundary puts vertices
    on offset rings every `step` ft of radius, and the residual chord error
    drops to inches.
    """
    if pc.plane is not None:
        return [poly]
    if getattr(model, "_bands", None) is None or model._band_step != step:
        bands, prev, k = [], model.hpoly, 0
        while k * step < CONE_HORIZ - 1e-6:
            outer = model.hpoly.buffer(min((k + 1) * step, CONE_HORIZ),
                                       quad_segs=64)
            bands.append(outer.difference(prev))
            prev = outer
            k += 1
        model._bands, model._band_step = bands, step
    out = []
    for band in model._bands:
        cut = poly.intersection(band)
        out.extend(pg for pg in as_polygons(cut) if pg.area > 1.0)
    return out or [poly]


def triangulate_conical(model, poly):
    """Delaunay over the polygon boundary plus interior points seeded on the
    conical offset rings. Ear clipping is exact for planes but free to draw
    a 3,000 ft diagonal along the rim of a curved surface, and that chord
    sags almost 100 ft in plan — a 5 ft elevation lie. Delaunay keeps edges
    short and ring-aligned, so chords sag inches. Coverage is verified by
    area; on any shortfall the banded ear clip takes over."""
    from matplotlib.tri import Triangulation
    pts = [np.asarray(poly.exterior.coords)[:-1]]
    for r in poly.interiors:
        pts.append(np.asarray(r.coords)[:-1])
    for band in model._bands or []:
        ring = band.exterior
        arc = np.asarray(ring.coords)
        seg = np.hypot(*np.diff(arc, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        want = np.arange(0.0, cum[-1], 250.0)
        xs = np.interp(want, cum, arc[:, 0])
        ys = np.interp(want, cum, arc[:, 1])
        keep = [(x, y) for x, y in zip(xs, ys)
                if poly.contains(Point(x, y))]
        if keep:
            pts.append(np.asarray(keep))
    P = np.vstack(pts)
    if len(P) < 3:
        return None
    try:
        tri = Triangulation(P[:, 0], P[:, 1])
    except Exception:
        return None
    out, cover = [], 0.0
    for a, b, c in tri.triangles:
        t = P[[a, b, c]]
        if poly.covers(Point(t[:, 0].mean(), t[:, 1].mean())):
            out.append(t)
            cover += abs((t[1][0]-t[0][0])*(t[2][1]-t[0][1]) -
                         (t[1][1]-t[0][1])*(t[2][0]-t[0][0])) / 2.0
    if abs(cover - poly.area) > max(poly.area, 1.0) * 1e-6:
        return None
    return out


def to_mesh3d(model, composite=None, use_composite=False, merge=True):
    """Triangles in local feet, grouped by surface, for the 3D view."""
    groups = {}

    def add(kind, poly, z_at, pc=None):
        kind = display_kind(kind, merge)
        if poly.is_empty:
            return
        buf = groups.setdefault(kind, [])
        polys = []
        for pg0 in as_polygons(poly):
            if pc is not None and pc.plane is None:
                polys.extend(mesh_polys(model, pc, pg0))
            elif pc is not None:
                polys.extend(mesh_polys(model, pc, pg0))
            else:
                polys.append(pg0)
        for pg in polys:
            for t in triangulate(pg):
                for x, y in t:
                    buf.extend([round(float(x), 1), round(float(y), 1),
                                round(float(z_at(x, y)), 2)])

    if use_composite and composite:
        for poly, pc in composite:
            add("COMPOSITE:" + pc.kind, poly, pc.z, pc)
    else:
        fbp = model.faces_by_piece()
        for i, pc in enumerate(model.pieces):
            for face in fbp.get(i, []):
                add(pc.kind, face, pc.z, pc)
    for f in model.frames:
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        add("RUNWAY", poly, lambda x, y, f=f: f.centerline_z(f.local(x, y)[0]))

    # Outlines drawn from the real piece boundaries. Deriving them from the
    # triangles instead pulls in every interior diagonal, and a fan across a
    # 37,000 ft approach corridor then stripes the whole surface.
    edges = {}

    def add_edge(kind, poly, z_at):
        buf = edges.setdefault(display_kind(kind, merge), [])
        for pg in as_polygons(poly):
            for ring in [pg.exterior] + list(pg.interiors):
                c = list(ring.coords)
                for i in range(len(c) - 1):
                    (x0, y0), (x1, y1) = c[i], c[i + 1]
                    z0, z1 = z_at(x0, y0), z_at(x1, y1)
                    # Where a region boundary runs along a cliff in the
                    # envelope, consecutive vertices share a plan position but
                    # differ by hundreds of feet, and the stroke draws a
                    # vertical line up the face of the drop. The drop is real;
                    # drawing a line on it is not, since nothing is there.
                    run = math.hypot(x1 - x0, y1 - y0)
                    if abs(z1 - z0) > max(2.0, run):
                        continue
                    buf.extend([round(float(x0), 1), round(float(y0), 1),
                                round(float(z0), 2),
                                round(float(x1), 1), round(float(y1), 1),
                                round(float(z1), 2)])

    def env_z(x, y):
        z, _ = model.controlling(x, y)
        return z if z is not None else model.horiz_z

    if use_composite and composite:
        by = {}
        for poly, pc in composite:
            by.setdefault(display_kind(pc.kind, merge), []).append((poly, pc))
        for kind, items in by.items():
            merged = unary_union([p.buffer(0.5) for p, _ in items]).buffer(-0.5)
            for pg in as_polygons(merged):
                add_edge("COMPOSITE:" + kind, pg, env_z)
    else:
        by = {}
        for pc in model.pieces:
            by.setdefault(display_kind(pc.kind, merge), []).append(pc)
        for kind, pcs in by.items():
            merged = unary_union([pc.poly.buffer(0.5) for pc in pcs]).buffer(-0.5)

            def kind_z(x, y, pcs=pcs):
                best = None
                pt = Point(x, y)
                for pc in pcs:
                    if pc.poly.covers(pt) or pc.poly.distance(pt) < 1.0:
                        z = pc.z(x, y)
                        if best is None or z < best:
                            best = z
                return best if best is not None else model.horiz_z

            for pg in as_polygons(merged):
                add_edge(kind, pg, kind_z)
    for f in model.frames:
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        add_edge("RUNWAY", poly,
                 lambda x, y, f=f: f.centerline_z(f.local(x, y)[0]))

    colors = {}
    for k in groups:
        colors[k] = COLORS.get(k.split(":")[-1], "#888888")
    return {"groups": groups, "edges": edges, "colors": colors,
            "elev": model.elev, "horiz_z": model.horiz_z,
            "cone_z": model.cone_z}


# Names in the EPSG database do not say "State Plane" — Louisiana South is
# published as "NAD83 / Louisiana South (ftUS)" — so zones are identified by
# excluding the projections that are clearly not a state plane zone.
_CRS_REJECT = ("utm", "mercator", "albers", "ease-grid", "equidistant",
               "pdc ", "world", "pseudo", "polar", "lcc", "laea", "gnomonic",
               "oblique", "conus", "north pole", "south pole", "island grid",
               # BLM zones are UTM under another name, and these statewide
               # or national systems are not state plane zones either
               "blm", "canada", "atlas", "gic ", "teale", "statewide")


def suggest_crs(lat, lon, limit=8):
    """Projected CRSs whose published area of use contains this airport.

    Ranked so the plain NAD83 state plane zone in US survey feet — what a
    Civil 3D drawing normally sits in — comes first, with the realization
    variants after it. Returns [] rather than raising if pyproj or its
    database is unavailable.
    """
    try:
        from pyproj.database import query_crs_info
        from pyproj.aoi import AreaOfInterest
        from pyproj import CRS
    except Exception:
        return []
    try:
        infos = query_crs_info(
            auth_name="EPSG", pj_types=("PROJECTED_CRS",),
            area_of_interest=AreaOfInterest(lon - 0.02, lat - 0.02,
                                            lon + 0.02, lat + 0.02),
            contains=True)
    except Exception:
        return []

    out = []
    for i in infos:
        name = i.name
        low = name.lower()
        if any(r in low for r in _CRS_REJECT):
            continue
        if "nad83" not in low and "nad27" not in low:
            continue
        try:
            unit = CRS.from_epsg(int(i.code)).axis_info[0].unit_name.lower()
        except Exception:
            continue
        feet = "foot" in unit or "feet" in unit
        datum = name.split("/")[0].strip()
        rank = (0 if datum == "NAD83" else
                1 if datum.startswith("NAD83(2011)") else
                2 if datum.startswith("NAD83") else 3,
                0 if feet else 1,
                1 if "offshore" in low else 0,
                name)
        out.append((rank, {"code": int(i.code), "name": name,
                           "units": "US survey feet" if feet else "meters"}))
    out.sort(key=lambda r: r[0])
    return [d for _, d in out[:limit]]


def _projector(model, epsg):
    """World transform for export, or None to stay in local feet."""
    if not epsg:
        return None
    from pyproj import CRS, Transformer
    crs = CRS.from_epsg(int(epsg))
    unit = crs.axis_info[0].unit_name.lower()
    k = 1.0 if ("foot" in unit or "feet" in unit) else 1.0 / US_FT
    tr = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    def xf(x, y):
        lat, lon = model.lf.inv(x, y)
        px, py = tr.transform(lon, lat)
        return px * k, py * k

    return xf


def piece_end(pc):
    """Runway end a piece belongs to, read back off its label."""
    parts = (pc.label or "").split()
    if len(parts) >= 3 and parts[1] in ("approach", "transitional"):
        return parts[2]
    return None


def tin_name(pc, merge=True):
    """Surface name carrying the runway and end it applies to."""
    kind = display_kind(pc.kind, merge)
    if not pc.runway:
        return "P77-%s" % kind
    rwy = pc.runway.replace("/", "-")
    end = piece_end(pc)
    return "P77-%s-%s-%s" % (rwy, end, kind) if end else \
           "P77-%s-%s" % (rwy, kind)


# ===========================================================================
# Grouped composite export (v20)
#
# The composite as one TIN ramps across every vertical step, because a TIN
# holds one elevation per plan position and cannot carry a cliff. Splitting
# it into four groups turns each cliff into a gap *between* surfaces instead
# of a ramp *within* one. Every group is cut from composite faces, which are
# already a mutually exclusive lower envelope, so each group is single
# valued and internally continuous with no new geometry.
#
#   INNER RWY SURF   primary + transitional + approach below the horizontal
#   <end> OUTER APR   approach + its 5,000 ft wings above the horizontal
#   HORIZONTAL        as-is
#   CONICAL           as-is
#
# The inner/outer split is by elevation against the horizontal, not by the
# approach's 10,000 ft slope break. An approach leaves the inner group where
# it pierces the horizontal, the horizontal and then the conical own the
# band above, and the approach resurfaces where it drops back below the
# conical. Those crossings are already composite face boundaries, so testing
# a face against the horizontal elevation never splits anything.
# ===========================================================================
GROUP_INNER = "INNER RWY SURF"
GROUP_HORIZONTAL = "HORIZONTAL"
GROUP_CONICAL = "CONICAL"
APPROACH_FAMILY = ("APPROACH", "APPROACH2", "TRANSITIONAL5000")

_SIDE_ORDER = {"L": 0, "C": 1, "R": 2, "": 3}


def _end_parts(end_id):
    """Split '04R' into its designator and side: ('4', 'R')."""
    e = (end_id or "").strip().upper()
    i = len(e)
    while i > 0 and e[i - 1] in "LRC":
        i -= 1
    return (e[:i].lstrip("0") or e[:i] or "?"), e[i:]


def _end_display(end_id):
    n, s = _end_parts(end_id)
    return n + s


def approach_group_name(end_ids):
    """'28L/28R OUTER APR' for a parallel pair, '4R OUTER APR' for one end."""
    ends = sorted(set(end_ids),
                  key=lambda e: (_SIDE_ORDER.get(_end_parts(e)[1], 9), e))
    return "%s OUTER APR" % "/".join(_end_display(e) for e in ends)


def _face_min_z(pc, poly):
    return min(pc.z(x, y) for x, y in poly.exterior.coords)


def composite_groups(model, composite=None, tol=0.5):
    """Composite faces bucketed into the four export surfaces.

    Parallel ends share a bucket because they share a designator: 28L and
    28R both reduce to '28'. Where two approach surfaces overlap in plan the
    composite has already resolved them to the lower one, so the higher
    surface carries a bite out of its side rather than folding over itself.

    The outer group is keyed on kind, not elevation. A runway end can sit
    below the established airport elevation — BNA 20L is 540 against an
    airport elevation of 599 — and then the outer approach segment begins
    below the horizontal and climbs through it. Elevation alone would file
    those faces as inner. The elevation test survives only as an escape for
    an inner-family face that lies wholly above the horizontal.
    """
    if composite is None:
        composite = model.composite()
    inner, horizontal, conical = [], [], []
    outer, outer_ends = {}, {}
    for poly, pc in composite:
        if pc.kind == "HORIZONTAL":
            horizontal.append((poly, pc))
        elif pc.kind == "CONICAL":
            conical.append((poly, pc))
        elif (pc.kind in ("APPROACH2", "TRANSITIONAL5000")
                or (pc.kind in ("APPROACH", "TRANSITIONAL")
                    and _face_min_z(pc, poly) > model.horiz_z + tol)):
            end = piece_end(pc) or "?"
            num = _end_parts(end)[0]
            outer.setdefault(num, []).append((poly, pc))
            outer_ends.setdefault(num, set()).add(end)
        else:
            inner.append((poly, pc))

    out = {}
    if inner:
        out[GROUP_INNER] = inner
    if horizontal:
        out[GROUP_HORIZONTAL] = horizontal
    if conical:
        out[GROUP_CONICAL] = conical
    for num, items in outer.items():
        out[approach_group_name(outer_ends[num])] = items
    return _node_groups(_absorb_slivers(out))


def _split_at_cliffs(groups, tol=0.5):
    """Break a group wherever it steps vertically, so each surface is continuous.

    Merging parallel ends puts a cliff inside a group. At PDX 10L's 5,000 ft
    wing ends where 10R's approach is still 422 ft higher, and lowest-wins
    correctly takes the step. A TIN holds one elevation per plan position
    and cannot carry that, so the two sides meet as unmatched edges and the
    border draws a slit down the middle of the approach.

    This is the reason the composite was split into groups in the first
    place: a cliff has to fall *between* surfaces, not inside one. Faces are
    joined only where they share an edge and agree on elevation across it,
    and each connected run is exported on its own. A group that never steps
    is untouched and keeps its name.
    """
    from shapely.strtree import STRtree

    out = {}
    for name, items in groups.items():
        if len(items) < 2:
            out[name] = items
            continue
        polys = [p for p, _ in items]
        tree = STRtree(polys)
        parent = list(range(len(items)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, (pa, ca) in enumerate(items):
            for j in tree.query(pa.buffer(0.5)):
                if j <= i:
                    continue
                pb, cb = items[j]
                try:
                    shared = pa.buffer(0.01).intersection(pb)
                except Exception:
                    continue
                if shared.is_empty or shared.length <= 1.0:
                    continue
                pt = shared.representative_point()
                if abs(ca.z(pt.x, pt.y) - cb.z(pt.x, pt.y)) > tol:
                    continue                      # a cliff: leave them apart
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

        runs = {}
        for i in range(len(items)):
            runs.setdefault(find(i), []).append(items[i])
        if len(runs) < 2:
            out[name] = items
            continue
        ordered = sorted(runs.values(), key=lambda v: -sum(p.area for p, _ in v))
        for k, part in enumerate(ordered, 1):
            out["%s %d" % (name, k)] = part
    return out


def _node_groups(groups):
    """Node every group's faces against each other, as arrangement() does.

    Composite faces abut without sharing vertices. _assign splits a face on
    a crossing line, which plants a vertex on that face's boundary but not
    on the neighbour across it, and the two then share a segment
    geometrically while disagreeing about its vertices. Their triangles fail
    to meet, and the surface border traces up one side of the seam and back
    down the other - a slit that draws as a line through the middle of a
    sound surface. At PDX one ran 6,120 ft down the 28 approach, between
    28L's approach at 817 vertices and 28R's wing at 170.

    Inserting the missing vertices one edge at a time was tried first and
    only moved the problem around: the totals across the five surfaces were
    identical before and after. This does what Model.arrangement() already
    does for pieces - unions all the boundaries so every crossing becomes a
    node, then rebuilds the faces from that - which is what removed
    T-junctions in the first place. Faces come back split more finely; the
    covered area is unchanged, and each new face inherits the piece whose
    face contains it.
    """
    from shapely.strtree import STRtree

    out = {}
    for name, items in groups.items():
        if len(items) < 2:
            out[name] = items
            continue
        lines = []
        for poly, _ in items:
            lines.append(LineString(poly.exterior.coords))
            for r in poly.interiors:
                lines.append(LineString(r.coords))
        try:
            noded = list(polygonize(unary_union(lines)))
        except Exception:
            out[name] = items
            continue
        if not noded:
            out[name] = items
            continue

        polys = [p for p, _ in items]
        tree = STRtree(polys)
        rebuilt, claimed = [], 0.0
        for face in noded:
            pt = face.representative_point()
            hit = None
            for j in tree.query(pt):
                if polys[j].covers(pt):
                    hit = j
                    break
            if hit is None:
                continue
            rebuilt.append((face, items[hit][1]))
            claimed += face.area

        before = sum(p.area for p in polys)
        if not rebuilt or abs(claimed - before) > max(1.0, before * 1e-9):
            out[name] = items          # noding lost area; keep the original
            continue
        out[name] = rebuilt
    return out


def _weld_tjunctions(groups, tol=0.05):
    """Insert a neighbour's vertex into any edge that runs through it.

    Splitting a face on a crossing line puts a new vertex on that face's
    boundary. The face on the other side of the boundary keeps its original
    edge, so the two no longer share vertices along a segment they share
    geometrically. Their triangles then fail to meet and the surface border
    traces up one side of the seam and back down the other: a hole of zero
    area that draws as a line across the middle of an otherwise sound
    surface. This is the same T-junction failure that cut the exported TINs
    in v19, one level up - between composite faces rather than inside a
    triangulation.

    Only collinear vertices are added, so every polygon keeps its exact
    shape and area.
    """
    from shapely.strtree import STRtree

    out = {}
    for name, items in groups.items():
        pts = set()
        for poly, _ in items:
            for ring in [poly.exterior] + list(poly.interiors):
                for x, y in ring.coords[:-1]:
                    pts.add((x, y))
        if not pts:
            out[name] = items
            continue
        plist = sorted(pts)
        tree = STRtree([Point(p) for p in plist])

        def weld(ring):
            src = list(ring.coords)
            new = []
            for i in range(len(src) - 1):
                ax, ay = src[i]
                bx, by = src[i + 1]
                new.append((ax, ay))
                dx, dy = bx - ax, by - ay
                dd = dx * dx + dy * dy
                if dd <= 0:
                    continue
                seg = LineString([(ax, ay), (bx, by)])
                hits = []
                for j in tree.query(seg.buffer(tol)):
                    px, py = plist[j]
                    if (abs(px - ax) < tol and abs(py - ay) < tol) or \
                       (abs(px - bx) < tol and abs(py - by) < tol):
                        continue
                    t = ((px - ax) * dx + (py - ay) * dy) / dd
                    if t <= 0.0 or t >= 1.0:
                        continue
                    if abs((px - ax) * dy - (py - ay) * dx) / (dd ** 0.5) > tol:
                        continue
                    hits.append((t, px, py))
                hits.sort()
                for _, px, py in hits:
                    new.append((px, py))
            new.append(src[-1])
            return new

        fixed = []
        for poly, pc in items:
            ext = weld(poly.exterior)
            ints = [weld(r) for r in poly.interiors]
            try:
                q = Polygon(ext, ints)
                if not q.is_valid:
                    q = q.buffer(0)
                if q.is_empty or abs(q.area - poly.area) > 1.0:
                    q = poly
            except Exception:
                q = poly
            fixed.append((q, pc))
        out[name] = fixed
    return out


def _absorb_slivers(groups, max_width=8.0):
    """Move hairline faces into the group that surrounds them.

    Where two surfaces cross at a shallow angle the arrangement throws a
    face a couple of feet wide and hundreds of feet long. Assigned strictly
    by elevation it can land in a different group from everything around it,
    and then it draws as a second boundary line running alongside the first.
    Dropping it is not an option — a dropped face is a hole in whatever
    surface covered it, which is how v19 cut notches into exported TINs.
    Reassigning it preserves the partition exactly: the same faces come out,
    only labelled by the neighbour they are embedded in.
    """
    from shapely.strtree import STRtree

    flat = [(name, poly, pc)
            for name, items in groups.items() for poly, pc in items]
    polys = [p for _, p, _ in flat]
    if not polys:
        return groups
    tree = STRtree(polys)

    moved = 0
    for i, (name, poly, pc) in enumerate(flat):
        per = poly.length
        if per <= 0 or 4.0 * poly.area / per > max_width:
            continue
        share = {}
        for j in tree.query(poly.buffer(0.5)):
            if j == i or flat[j][0] == name:
                continue
            try:
                ln = poly.buffer(0.01).intersection(polys[j]).length
            except Exception:
                continue
            if ln > 0:
                share[flat[j][0]] = share.get(flat[j][0], 0.0) + ln
        if not share:
            continue
        best = max(share, key=share.get)
        if share[best] > per * 0.25:
            flat[i] = (best, poly, pc)
            moved += 1

    if not moved:
        return groups
    out = {}
    for name, poly, pc in flat:
        out.setdefault(name, []).append((poly, pc))
    return out


def dxf_layer(name):
    """DXF forbids / \\ : ; * ? \" < > | in a layer name."""
    for ch in "/\\:;*?\"<>|":
        name = name.replace(ch, "-")
    return "P77-" + name.replace(" ", "-")


def tin_groups(model, composite=None, individual=True, use_composite=False,
               use_groups=False):
    """Triangles grouped into sets that are each valid as a TIN.

    A TIN holds one elevation per plan position, so a group must not fold
    over itself. Surfaces are split per runway end for that reason: two
    runways' approach surfaces routinely overlap in plan at different
    elevations, and merging them would be wrong wherever they cross. The
    composite is a lower envelope, so it is single valued by construction.
    """
    out = {}

    def add(name, poly, pc):
        tris = out.setdefault(name, [])
        for pg0 in as_polygons(poly):
            for pg in mesh_polys(model, pc, pg0):
                for t in triangulate(pg):
                    tris.append([(float(x), float(y), float(pc.z(x, y)))
                                 for x, y in t])

    if individual:
        fbp = model.faces_by_piece()
        for i, pc in enumerate(model.pieces):
            for face in fbp.get(i, []):
                add(tin_name(pc), face, pc)
    if use_groups:
        for name, items in composite_groups(model, composite).items():
            for poly, pc in items:
                add(name, poly, pc)
    if use_composite and composite:
        for poly, pc in composite:
            add("P77-COMPOSITE", poly, pc)
    return {k: v for k, v in out.items() if v}


def to_landxml(model, composite=None, epsg=None, individual=True,
               use_composite=True, use_groups=False):
    """LandXML 1.2 TIN surfaces.

    Civil 3D imports these as surfaces directly. A DXF carries 3DFACE
    entities instead, which have to be rebuilt into a surface by hand.
    """
    import datetime

    xf = _projector(model, epsg)
    now = datetime.datetime.now()
    buf = ["<?xml version='1.0' encoding='UTF-8'?>",
           "<LandXML xmlns='http://www.landxml.org/schema/LandXML-1.2' "
           "version='1.2' date='%s' time='%s'>"
           % (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
           "<Units><Imperial areaUnit='squareFoot' "
           "linearUnit='USSurveyFoot' volumeUnit='cubicFeet' "
           "temperatureUnit='fahrenheit' pressureUnit='inHG'/></Units>",
           "<Application name='FAR Part 77 Surface Generator' "
           "version='%s'/>" % __version__,
           "<Surfaces>"]
    groups = tin_groups(model, composite, individual, use_composite,
                        use_groups)

    def weld_faces(tris, tol=0.02):
        """Split a triangle edge that runs through a neighbour's vertex.

        Noding fixes the faces, but triangles are built afterwards and the
        conical is cut into ring bands at that point, so its mesh carries
        vertices its neighbours never see. Where one triangle's edge passes
        through another's vertex the two do not share that edge, and the
        border traces up one side of the seam and back down the other - the
        slit that draws as a line across a sound surface.

        Insertions are resolved per shared edge, not per triangle. Deciding
        independently lets the two triangles either side of an edge disagree
        near the tolerance and breaks a join that was previously sound; done
        this way both get the same points in the same order.

        A vertex is only inserted where its elevation matches what the edge
        interpolates there, which keeps the weld off a genuine step: at a
        cliff two points share a plan position at different heights and have
        to stay apart.
        """
        from shapely.strtree import STRtree

        # the original vertex is kept, not the rounded key - inserting the
        # rounded coordinate puts the point up to tol off the edge and
        # leaves a hairline sliver along every weld
        keys = {}
        for t in tris:
            for x, y, z in t:
                keys.setdefault((round(x, 3), round(y, 3)), []).append((x, y, z))
        plist = sorted(keys)
        if not plist:
            return tris
        tree = STRtree([Point(p) for p in plist])

        def ekey(a, b):
            ka = (round(a[0], 3), round(a[1], 3), round(a[2], 3))
            kb = (round(b[0], 3), round(b[1], 3), round(b[2], 3))
            return (ka, kb) if ka <= kb else (kb, ka)

        cuts = {}
        for t in tris:
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                k = ekey(a, b)
                if k in cuts:
                    continue
                ax, ay, az = a
                bx, by, bz = b
                dx, dy = bx - ax, by - ay
                dd = dx * dx + dy * dy
                if dd <= 0:
                    cuts[k] = []
                    continue
                L = dd ** 0.5
                seg = LineString([(ax, ay), (bx, by)])
                hits = []
                for j in tree.query(seg.buffer(tol)):
                    px, py = plist[j]
                    if (abs(px - ax) < tol and abs(py - ay) < tol) or \
                       (abs(px - bx) < tol and abs(py - by) < tol):
                        continue
                    if abs((px - ax) * dy - (py - ay) * dx) / L > tol:
                        continue
                    u = ((px - ax) * dx + (py - ay) * dy) / dd
                    if u <= 1e-9 or u >= 1.0 - 1e-9:
                        continue
                    zi = az + (bz - az) * u
                    zs = [v for v in keys[(px, py)] if abs(v[2] - zi) <= 0.01]
                    if not zs:
                        continue          # a step: leave this edge alone
                    hits.append((u, zs[0]))
                hits.sort()
                # store against the canonical direction of the edge key
                fwd = ((round(ax, 3), round(ay, 3), round(az, 3)) == k[0])
                cuts[k] = [p for _, p in hits] if fwd else \
                          [p for _, p in reversed(hits)]

        out = []
        for t in tris:
            ring, added = [], False
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                ring.append(a)
                k = ekey(a, b)
                ins = cuts.get(k) or []
                if not ins:
                    continue
                fwd = ((round(a[0], 3), round(a[1], 3), round(a[2], 3)) == k[0])
                ring.extend(ins if fwd else list(reversed(ins)))
                added = True
            if not added:
                out.append(t)
                continue
            for m in range(1, len(ring) - 1):
                out.append([ring[0], ring[m], ring[m + 1]])
        return out

    def separate(tris):
        """Split vertices that join two otherwise separate pieces of surface.

        A group can touch itself at a single vertex. Two cases arise. One is
        a corner meeting across a vertical step, where a runway's wing domain
        ends while the parallel runway's approach is still hundreds of feet
        higher. The other is a pinch at equal elevation, where the horizontal
        wraps around the inner surfaces and its hole boundary closes to a
        point; PDX has nineteen of those. Either way the areas do not
        overlap, so the group stays single valued, but a TIN cannot carry a
        vertex belonging to two disconnected fans: Civil 3D welds it, and the
        border runs out to the point and back as a thin spike.

        Triangles around each vertex are grouped into fans by shared edges.
        A vertex with one fan is ordinary and left alone. Where there are
        more, every fan after the first has its copy moved a hundredth of a
        foot, which separates the pieces without moving anything a survey
        could measure.
        """
        at = {}
        for ti, t in enumerate(tris):
            for ci, (x, y, z) in enumerate(t):
                at.setdefault((round(x, 3), round(y, 3), round(z, 3)),
                              []).append((ti, ci))

        for key, uses in at.items():
            if len(uses) < 2:
                continue
            # two triangles belong to the same fan if they share an edge
            # through this vertex, i.e. another vertex in common
            others = []
            for ti, ci in uses:
                t = tris[ti]
                others.append(set((round(p[0], 3), round(p[1], 3),
                                   round(p[2], 3))
                                  for j, p in enumerate(t) if j != ci))
            n = len(uses)
            comp = list(range(n))

            def find(a):
                while comp[a] != a:
                    comp[a] = comp[comp[a]]
                    a = comp[a]
                return a

            for i in range(n):
                for j in range(i + 1, n):
                    if others[i] & others[j]:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            comp[ri] = rj
            roots = {}
            for i in range(n):
                roots.setdefault(find(i), []).append(i)
            if len(roots) < 2:
                continue
            for k, (_, members) in enumerate(sorted(roots.items())):
                if k == 0:
                    continue
                for i in members:
                    ti, ci = uses[i]
                    x, y, z = tris[ti][ci]
                    tris[ti][ci] = (x + 0.01 * k, y + 0.01 * k, z)
        return tris

    for name, tris in sorted(groups.items()):
        tris = separate(weld_faces([list(t) for t in tris]))
        # Snap to a thousandth of a foot before indexing. Shapely's boolean
        # operations leave vertices apart in the fourth decimal - a
        # thousandth of an inch - which are the same point in every sense
        # that matters. Emitted as distinct points they overrun Civil 3D's
        # own merge tolerance: it welds them, the face list stops describing
        # a valid triangulation, and it rebuilds the surface by Delaunay.
        # That rebuild is what cut 148 million square feet out of the 28
        # approach at PDX, and it is why imported triangle counts came back
        # at a third of what was written.
        ids, pts, faces = {}, [], []
        for t in tris:
            f = []
            for x, y, z in t:
                X, Y = xf(x, y) if xf else (x, y)
                key = (round(X, 3), round(Y, 3), round(z, 3))
                n = ids.get(key)
                if n is None:
                    n = len(pts) + 1
                    ids[key] = n
                    pts.append(key)
                f.append(n)
            if len(set(f)) != 3:
                continue
            # Civil 3D flags zero-area faces when building the surface, so
            # drop slivers that collapse once coordinates are rounded.
            (ax, ay, _), (bx, by, _), (cx, cy, _) = (pts[f[0] - 1],
                                                     pts[f[1] - 1],
                                                     pts[f[2] - 1])
            if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) < 1e-6:
                continue
            faces.append(f)
        if not faces:
            continue
        buf.append("<Surface name='%s' desc='14 CFR Part 77.19'>"
                   "<Definition surfType='TIN'>" % name)
        buf.append("<Pnts>")
        for i, (X, Y, z) in enumerate(pts, 1):
            # LandXML orders a point northing, easting, elevation
            buf.append("<P id='%d'>%.4f %.4f %.4f</P>" % (i, Y, X, z))
        buf.append("</Pnts><Faces>")
        for f in faces:
            buf.append("<F>%d %d %d</F>" % tuple(f))
        buf.append("</Faces></Definition></Surface>")
    buf.append("</Surfaces></LandXML>")
    return "\n".join(buf).encode("utf-8")


# ===========================================================================
def to_dxf(model, composite=None, epsg=None, individual=True,
           use_composite=True, use_groups=False):
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
    def emit_piece(pc, poly, lay):
        n = 0
        for pg0 in as_polygons(poly):
            for pg in mesh_polys(model, pc, pg0):
                n += emit(pg, pc.z, lay)
        return n

    fbp = model.faces_by_piece()
    if individual:
        for i, pc in enumerate(model.pieces):
            lay = tin_name(pc, merge=False)
            for face in fbp.get(i, []):
                stats[lay] = stats.get(lay, 0) + emit_piece(pc, face, lay)
    for f in model.frames:
        lay = "P77-RUNWAY-%s" % f.rwy.id.replace("/", "-")
        poly = Polygon([f.world(-f.half, -f.rwy.width / 2),
                        f.world(f.half, -f.rwy.width / 2),
                        f.world(f.half, f.rwy.width / 2),
                        f.world(-f.half, f.rwy.width / 2)])
        stats[lay] = stats.get(lay, 0) + emit(
            poly, lambda x, y, f=f: f.centerline_z(f.local(x, y)[0]), lay)
    if use_groups:
        for name, items in composite_groups(model, composite).items():
            lay = dxf_layer(name)
            for poly, pc in items:
                stats[lay] = stats.get(lay, 0) + emit_piece(pc, poly, lay)
    if use_composite and composite:
        for poly, pc in composite:
            stats["P77-COMPOSITE"] = stats.get("P77-COMPOSITE", 0) + \
                emit_piece(pc, poly, "P77-COMPOSITE")

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
