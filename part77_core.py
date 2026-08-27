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

__version__ = "2026-08-21.14"

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

        n = max(4, int(abs(ds) / 200.0))
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

        # Precision wing. Cutting this against the conical arc while the
        # terminated strip is cut against the ceiling by station sampling
        # left a gap between them: two clips of the same edge that do not
        # agree. Subtract the terminated strip itself instead, so the two
        # share an exact boundary and nothing can fall between.
        if e.crit["precision"]:
            inner_pts, outer_pts = [], []
            for st in stations:
                ie = f.inner(st)
                if ie is None:
                    continue
                half, z_in = ie
                inner_pts.append(f.world(st, side * half))
                outer_pts.append(
                    f.world(st, side * (half + TRANS_RUN_PRECISION)))
            full = build((inner_pts, outer_pts))
            if full is not None:
                wing = full.difference(term) if term is not None else full
                emit(wing, "TRANSITIONAL5000")

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
            return 0.0

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
            faces = []
            for f in polygonize(merged):
                if f.area <= 25.0:
                    continue
                # Nearly coincident boundaries (one runway's wing edge along
                # another's approach edge) leave faces thousands of feet long
                # and under a foot wide. They carry no area worth drawing, but
                # edge-on under vertical exaggeration they render as fins.
                if 4.0 * f.area / max(f.length, 1.0) < 1.0:
                    continue
                faces.append(f)
            self._faces = faces
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
            for poly, pc in self._assign(face, 0):
                # splitting on crossing lines can create its own slivers
                if 4.0 * poly.area / max(poly.length, 1.0) < 1.0:
                    continue
                out.append((poly, pc))
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
            best, who = None, None
            for pc in cand:
                z = pc.z(x, y)
                if best is None or z < best - 1e-9:
                    best, who = z, pc
            winners.add(id(who))

        if len(winners) == 1 or depth >= 8:
            x, y = face.representative_point().coords[0]
            who = min(cand, key=lambda pc: pc.z(x, y))
            return [(face, who)]

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
        return [(face, min(cand, key=lambda pc: pc.z(x, y)))]

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


def tin_groups(model, composite=None, individual=True, use_composite=False):
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
    if use_composite and composite:
        for poly, pc in composite:
            add("P77-COMPOSITE", poly, pc)
    return {k: v for k, v in out.items() if v}


def to_landxml(model, composite=None, epsg=None, individual=True,
               use_composite=True):
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
    groups = tin_groups(model, composite, individual, use_composite)
    for name, tris in sorted(groups.items()):
        ids, pts, faces = {}, [], []
        for t in tris:
            f = []
            for x, y, z in t:
                X, Y = xf(x, y) if xf else (x, y)
                key = (round(X, 4), round(Y, 4), round(z, 4))
                n = ids.get(key)
                if n is None:
                    n = len(pts) + 1
                    ids[key] = n
                    pts.append(key)
                f.append(n)
            if len(set(f)) == 3:
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
           use_composite=True):
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
