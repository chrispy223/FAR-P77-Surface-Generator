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
        parts = [pg for pg in as_polygons(poly) if pg.area >= 1.0]
        if not parts:
            return
        kind = "TRANSITIONAL"
        if e.crit["precision"]:
            mid = len(inner_pts) // 2
            if abs(np.hypot(*(np.array(outer_pts[mid]) - np.array(inner_pts[mid])))
                   - TRANS_RUN_PRECISION) < 1.0:
                kind = "TRANSITIONAL5000"
        for pg in parts:
            self.pieces.append(Piece(kind, pg, plane,
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
            self._faces = [f for f in polygonize(merged) if f.area > 25.0]
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


def _earcut(ring, coll=1e-4):
    """Ear clipping on a simple CCW ring.

    `coll` is a perpendicular distance in feet, not a cross product. Testing
    the raw cross product against a fixed epsilon fails on long edges: three
    collinear points 200 ft apart on a 40,000 ft approach boundary still give
    a cross product far above any absolute epsilon, so the vertex reads as
    convex, gets clipped, and the replacement diagonal runs along the boundary
    and strands every vertex between its ends. Those stranded vertices are
    T-junctions, and in CAD they are cracks.
    """
    n = len(ring)
    if n < 3:
        return []
    idx = list(range(n))

    def cross(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0]))

    def inside(p, a, b, c):
        return (cross(a, b, p) >= 0 and cross(b, c, p) >= 0
                and cross(c, a, p) >= 0)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        cut = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = ring[i0], ring[i1], ring[i2]
            x = cross(a, b, c)
            if x <= 0:
                continue                      # reflex
            base = math.hypot(c[0] - a[0], c[1] - a[1])
            if base > 0 and x / base <= coll:
                continue                      # collinear within tolerance
            bad = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if inside(ring[j], a, b, c):
                    bad = True
                    break
            if bad:
                continue
            tris.append((a, b, c))
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tris.append(tuple(ring[i] for i in idx))
        return tris
    if coll > 0.0:
        # refusing every collinear ear can stall; loosen and try again rather
        # than return a partial cover
        return _earcut(ring, coll / 100.0)
    return tris


def triangulate(poly):
    """Watertight triangles covering a polygon."""
    from shapely.geometry.polygon import orient
    out = []
    for pg in as_polygons(poly):
        if pg.interiors:
            # nothing here should carry a hole once the conical is split into
            # quadrants, but fall back rather than silently drop the piece
            for q in as_polygons(_split_quadrants(pg)):
                out.extend(_earcut(list(orient(q, 1.0).exterior.coords)[:-1]))
            continue
        out.extend(_earcut(list(orient(pg, 1.0).exterior.coords)[:-1]))
    return [np.asarray(t, float) for t in out]


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
        fbp = model.faces_by_piece()
        for i, pc in enumerate(model.pieces):
            for face in fbp.get(i, []):
                add(pc.kind, face, pc.z)
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
        buf = edges.setdefault(kind, [])
        for pg in as_polygons(poly):
            for ring in [pg.exterior] + list(pg.interiors):
                c = list(ring.coords)
                for i in range(len(c) - 1):
                    for x, y in (c[i], c[i + 1]):
                        buf.extend([round(float(x), 1), round(float(y), 1),
                                    round(float(z_at(x, y)), 2)])

    if use_composite and composite:
        by = {}
        for poly, pc in composite:
            by.setdefault(pc.kind, []).append((poly, pc))
        for kind, items in by.items():
            merged = unary_union([p.buffer(0.5) for p, _ in items]).buffer(-0.5)
            _, pc0 = items[0]
            for pg in as_polygons(merged):
                add_edge("COMPOSITE:" + kind, pg, pc0.z)
    else:
        for pc in model.pieces:
            add_edge(pc.kind, pc.poly, pc.z)
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
    fbp = model.faces_by_piece()
    for i, pc in enumerate(model.pieces):
        lay = "P77-%s" % pc.kind
        for face in fbp.get(i, []):
            stats[lay] = stats.get(lay, 0) + emit(face, pc.z, lay)
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
