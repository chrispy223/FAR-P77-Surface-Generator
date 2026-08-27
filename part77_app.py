#!/usr/bin/env python3
"""
FAR Part 77 Airspace Surface Generator

    pip install streamlit shapely numpy matplotlib ezdxf pyproj openpyxl pandas
    streamlit run part77_app.py
"""

import json
import math
import re
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

import part77_core as C
import part77_map as M
import part77_view3d as V3

st.set_page_config(page_title="Part 77 Surface Generator",
                   page_icon="✈", layout="wide")

# ADIP's public portal POSTs the identifier as a JSON body and sends a fixed
# client key on every API call. The key is baked into the public Angular app,
# not tied to a user login. If ADIP rotates it, Search will start returning
# 400 or 401 — pull the current one out of devtools (any request to
# /agisServices/, Headers tab, Authorization) and paste it below.
ADIP_URL = "https://adip.faa.gov/agisServices/public-api/getAirportDetails"
ADIP_KEY = "Basic 3f647d1c-a3e7-415e-96e1-6e8415e6f209-ADIP"
UA = {"Accept": "application/json, text/plain, */*",
      "Accept-Language": "en-US,en;q=0.9",
      "Authorization": ADIP_KEY,
      "Cache-Control": "no-cache",
      "Content-Type": "application/json;charset=UTF-8",
      "Origin": "https://adip.faa.gov",
      "Pragma": "no-cache",
      "Referer": "https://adip.faa.gov/agis/public/",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"}

CAT_LABELS = {c: "[%d] %s" % (i + 1, C.PART77[c]["short"])
              for i, c in enumerate(C.CODES)}
LABEL_CAT = {v: k for k, v in CAT_LABELS.items()}


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


MISSING_CODES = []


def _code(v, end=None):
    raw = (v or "").strip().upper()
    if raw in C.PART77:
        return raw
    if end:
        MISSING_CODES.append((end, raw or "(blank)"))
    return "B(V)"


def fetch_adip(loc, url=ADIP_URL):
    loc = loc.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", loc):
        raise ValueError("Identifier should be 3 or 4 letters or digits.")
    body = json.dumps({"locId": loc}).encode()
    req = urllib.request.Request(url, data=body, headers=UA, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "replace")
    rec = json.loads(raw)
    if not rec or not rec.get("runways"):
        raise ValueError("ADIP returned no runway data for %s." % loc)
    return rec


def adip_to_rows(rec):
    MISSING_CODES.clear()
    rows = []
    for rw in rec.get("runways", []):
        b, r = rw.get("baseEnd", {}), rw.get("reciprocalEnd", {})
        rows.append({
            "Include": True,
            "Runway": rw.get("runwayIdentifier"),
            "Status": "existing",
            "Width (ft)": _f(rw.get("width"), 150.0),
            "End 1": b.get("runwayEndId"),
            "Lat 1": _f(b.get("latitude")), "Lon 1": _f(b.get("longitude")),
            "Elev 1": _f(b.get("elevation")),
            "Cat 1": CAT_LABELS[_code(b.get("obstaclePart77"),
                                      b.get("runwayEndId"))],
            "End 2": r.get("runwayEndId"),
            "Lat 2": _f(r.get("latitude")), "Lon 2": _f(r.get("longitude")),
            "Elev 2": _f(r.get("elevation")),
            "Cat 2": CAT_LABELS[_code(r.get("obstaclePart77"),
                                      r.get("runwayEndId"))],
        })
    return rows


def apt_from_rec(rec):
    return {"locId": rec.get("locId"), "name": rec.get("name"),
            "elev": _f(rec.get("elevation"), 0.0),
            "lat": _f(rec.get("arpLatitude")),
            "lon": _f(rec.get("arpLongitude"))}


def load_dtpp(upload):
    """Approach categories derived from the FAA approach charts.

    Part 77 category is not published directly on a chart; it follows from
    the procedure type and the published visibility minimums, which live
    inside the chart PDF. Rather than guess at that pipeline, this reads a
    two-column table you supply (end, category). Disagreements with ADIP are
    reported, never applied silently.
    """
    if upload is None:
        return {}
    name = upload.name.lower()
    df = pd.read_excel(upload) if name.endswith((".xlsx", ".xls")) \
        else pd.read_csv(upload)
    cols = {str(c).lower().strip(): c for c in df.columns}
    ec = cols.get("end") or cols.get("runway end") or df.columns[0]
    cc = cols.get("category") or cols.get("cat") or df.columns[1]
    out = {}
    for _, r in df.iterrows():
        k, v = str(r[ec]).strip().upper(), str(r[cc]).strip().upper()
        if k and v in C.PART77:
            out[k] = v
    return out


def rows_to_runways(df):
    """Build runways from the table, rejecting anything that would reach the
    geometry with a blank or nonsense coordinate. A NaN survives float() and
    only fails much later inside shapely, as an error nobody can act on."""
    rwys, problems = [], []

    def num(row, key, lo, hi, what):
        v = row.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("%s is blank" % what)
        if not math.isfinite(v):
            raise ValueError("%s is blank" % what)
        if not (lo <= v <= hi):
            raise ValueError("%s of %g is out of range" % (what, v))
        return v

    for _, r in df.iterrows():
        if not bool(r.get("Include", True)):
            continue
        name = str(r.get("Runway") or "").strip()
        if not name or name.lower() == "nan":
            continue                      # empty row from the editor
        try:
            ends = []
            for n, side in (("1", "end 1"), ("2", "end 2")):
                lat = num(r, "Lat " + n, -90.0, 90.0, "%s latitude" % side)
                lon = num(r, "Lon " + n, -180.0, 180.0, "%s longitude" % side)
                elev = num(r, "Elev " + n, -1500.0, 30000.0,
                           "%s elevation" % side)
                if abs(lat) < 1e-9 and abs(lon) < 1e-9:
                    raise ValueError("%s still sits at 0, 0 — enter its "
                                     "coordinates" % side)
                ends.append(C.RunwayEnd(
                    str(r.get("End " + n) or n), lat, lon, elev,
                    LABEL_CAT.get(r.get("Cat " + n), "B(V)")))
            width = 150.0
            try:
                w = float(r.get("Width (ft)"))
                if math.isfinite(w) and w > 0:
                    width = w
            except (TypeError, ValueError):
                pass
            rwys.append(C.Runway(name, ends[0], ends[1], width,
                                 str(r.get("Status", "existing"))))
        except Exception as ex:
            problems.append("%s: %s" % (name, ex))
    return rwys, problems


def blank_row():
    return {"Include": True, "Runway": "NEW 00/18", "Status": "proposed",
            "Width (ft)": 150.0,
            "End 1": "00", "Lat 1": 0.0, "Lon 1": 0.0, "Elev 1": 0.0,
            "Cat 1": CAT_LABELS["B(V)"],
            "End 2": "18", "Lat 2": 0.0, "Lon 2": 0.0, "Elev 2": 0.0,
            "Cat 2": CAT_LABELS["B(V)"]}


def training_rows():
    r = C.training_model().runways[0]
    return [{"Include": True, "Runway": r.id, "Status": "training",
             "Width (ft)": r.width,
             "End 1": r.base.id, "Lat 1": r.base.lat, "Lon 1": r.base.lon,
             "Elev 1": 0.0, "Cat 1": CAT_LABELS["PIR"],
             "End 2": r.recip.id, "Lat 2": r.recip.lat, "Lon 2": r.recip.lon,
             "Elev 2": 0.0, "Cat 2": CAT_LABELS["PIR"]}]


# ---------------------------------------------------------------------------
S = st.session_state
for k, v in (("rows", None), ("apt", None), ("model", None),
             ("composite", None), ("obstacles", []), ("dtpp", {})):
    S.setdefault(k, v)

st.title("✈ FAR Part 77 Airspace Surface Generator")
st.caption("14 CFR Part 77.19 · multi-runway · existing and proposed · "
           "surfaces reduced to the lowest controlling elevation · "
           "engine v%s" % C.__version__)

# A form so Enter in the identifier field submits, rather than only the
# button working.
with st.form("search", clear_on_submit=False):
    c1, c2, c3 = st.columns([2, 1, 1])
    loc = c1.text_input("Airport ICAO or FAA identifier", value="LFT",
                        help="Press Enter or click Search")
    c2.write("")
    go = c2.form_submit_button("Search ADIP", type="primary",
                               use_container_width=True)
    c3.write("")
    train = c3.form_submit_button("Load training example",
                                  use_container_width=True)

up = st.file_uploader("…or load a saved ADIP getAirportDetails JSON "
                      "(use this if ADIP will not answer directly)",
                      type=["json", "txt"])

if train:
    S.rows = training_rows()
    S.apt = {"locId": "TRAINING", "lat": 40.0, "lon": -100.0, "elev": 0.0,
             "name": "Training example — flat, single runway, "
                     "precision instrument both ends"}
    S.model = S.composite = None

if go:
    try:
        rec = fetch_adip(loc)
        S.rows, S.apt = adip_to_rows(rec), apt_from_rec(rec)
        S.model = S.composite = None
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        st.error("Could not reach ADIP (%s). Upload a saved "
                 "getAirportDetails file instead." % e)
    except Exception as e:
        st.error(str(e))

if up is not None and st.button("Load uploaded file"):
    rec = json.load(up)
    S.rows, S.apt = adip_to_rows(rec), apt_from_rec(rec)
    S.model = S.composite = None

if not S.rows:
    st.info("Search an identifier, upload a saved ADIP record, or load the "
            "training example to begin.")
    st.stop()

apt = S.apt
st.subheader("%s — %s" % (apt["locId"] or "", apt["name"] or ""))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Airport elevation", "%.1f ft" % apt["elev"])
m2.metric("Horizontal surface", "%.1f ft MSL" % (apt["elev"] + C.HORIZ_HGT))
m3.metric("Conical outer edge", "%.1f ft MSL" % (apt["elev"] + C.CONE_HGT))
m4.metric("Runways loaded", str(len(S.rows)))

with st.expander("Approach category cross-check (d-TPP vs ADIP)"):
    st.caption("ADIP's obstaclePart77 field is the authoritative record and "
               "is what prefills the table. Upload a two-column table of end "
               "and category derived from the FAA approach charts to compare. "
               "Disagreements are flagged here — nothing is overwritten.")
    dt = st.file_uploader("d-TPP category table (CSV or Excel: end, category)",
                          type=["csv", "xlsx", "xls"], key="dtpp_up")
    if dt is not None:
        try:
            S.dtpp = load_dtpp(dt)
            st.success("Loaded %d runway end categories." % len(S.dtpp))
        except Exception as e:
            st.error("Could not read that file: %s" % e)

conflicts = []
if S.dtpp:
    for r in S.rows:
        for n in ("1", "2"):
            end = str(r["End " + n]).upper()
            adip = LABEL_CAT.get(r["Cat " + n], "B(V)")
            other = S.dtpp.get(end)
            if other and other != adip:
                conflicts.append({
                    "Runway end": end,
                    "ADIP": C.PART77[adip]["short"],
                    "d-TPP": C.PART77[other]["short"],
                    "Same primary width":
                        C.PART77[adip]["primary"] == C.PART77[other]["primary"],
                })
if conflicts:
    st.warning("%d runway end(s) disagree between ADIP and d-TPP. Where the "
               "primary width matches, the surfaces are unaffected."
               % len(conflicts))
    st.dataframe(pd.DataFrame(conflicts), use_container_width=True,
                 hide_index=True)

if MISSING_CODES:
    st.warning("ADIP has no Part 77 category for: " +
               ", ".join("%s (%s)" % mc for mc in MISSING_CODES) +
               ". Defaulted to visual — verify these before generating, the "
               "way you would against the approach charts.")

st.subheader("Runway configuration")
st.caption("Everything below is editable. Add a row for a proposed runway, or "
           "move an end's coordinates to model an extension — proposed "
           "geometry is treated identically to existing in both the "
           "individual surfaces and the composite.")

cat_opts = list(CAT_LABELS.values())
edited = st.data_editor(
    pd.DataFrame(S.rows), use_container_width=True, num_rows="dynamic",
    key="rwy_editor",
    column_config={
        "Include": st.column_config.CheckboxColumn(width="small"),
        "Status": st.column_config.SelectboxColumn(
            options=["existing", "proposed", "training"], width="small"),
        "Cat 1": st.column_config.SelectboxColumn("Category — End 1",
                                                  options=cat_opts),
        "Cat 2": st.column_config.SelectboxColumn("Category — End 2",
                                                  options=cat_opts),
        "Lat 1": st.column_config.NumberColumn(format="%.6f"),
        "Lon 1": st.column_config.NumberColumn(format="%.6f"),
        "Lat 2": st.column_config.NumberColumn(format="%.6f"),
        "Lon 2": st.column_config.NumberColumn(format="%.6f"),
        "Elev 1": st.column_config.NumberColumn(format="%.1f"),
        "Elev 2": st.column_config.NumberColumn(format="%.1f"),
    })
S.rows = edited.to_dict("records")

e1, e2, e3 = st.columns([1, 1, 2])
if e1.button("Add proposed runway"):
    S.rows = S.rows + [blank_row()]
    st.rerun()
apt["elev"] = e3.number_input(
    "Established airport elevation (ft MSL)", value=float(apt["elev"]),
    step=0.1, help="77.19(c) puts the horizontal surface 150 ft above this, "
                   "regardless of the runway profile.")
if e2.button("Generate Part 77 surfaces", type="primary"):
    rwys, problems = rows_to_runways(edited)
    for p in problems:
        st.error(p)
    if rwys:
        lat = apt["lat"] or rwys[0].base.lat
        lon = apt["lon"] or rwys[0].base.lon
        try:
            with st.spinner("Building surfaces…"):
                S.model = C.Model(lat, lon, apt["elev"], rwys)
                S.composite = S.model.composite()
                S.pop("mesh_comp", None)
                S.pop("mesh_ind", None)
        except Exception as ex:
            S.model = S.composite = None
            st.error("Could not build the surfaces: %s\n\nCheck the runway "
                     "table for a row with missing or placeholder "
                     "coordinates." % ex)
    else:
        st.error("No runways selected.")

if S.model is None:
    st.stop()
model, comp = S.model, S.composite

st.subheader("Part 77 surfaces")
tab_map, tab_3d = st.tabs(["Map", "3D view"])

with tab_map:
    mode = st.radio("View", ["Composite (controlling surface)",
                             "Individual surfaces"],
                    horizontal=True, label_visibility="collapsed")
    is_comp = mode.startswith("Composite")
    st.caption("Move the cursor over the map for the exact Part 77 elevation "
               "at that point. Overlapping areas show only the lowest surface.")
    st.components.v1.html(
        M.render(C.to_geojson(model, comp if is_comp else None),
                 height=580, composite=is_comp), height=580)

with tab_3d:
    use_comp = st.toggle("Show composite instead of individual surfaces",
                         value=False, key="3d_comp")
    st.caption("Drag to orbit, shift-drag or right-drag to pan, wheel to "
               "zoom. One finger orbits and two fingers pan and pinch on a "
               "touch screen. At true scale these surfaces are nearly "
               "invisible against a plan extent this wide, so start with the "
               "exaggeration slider up.")
    key = "mesh_comp" if use_comp else "mesh_ind"
    if key not in S:
        with st.spinner("Building mesh (once per generation)…"):
            S[key] = V3.render(
                C.to_mesh3d(model, comp, use_composite=use_comp), height=620)
    st.components.v1.html(S[key], height=620)

with st.expander("Surface parameters", expanded=True):
    rows = []
    for f in model.frames:
        a, b = f.rwy.base, f.rwy.recip
        rows.append({
            "Runway": f.rwy.id, "Status": f.rwy.status,
            "Length (ft)": round(f.len),
            "End 1 type": a.crit["short"], "End 2 type": b.crit["short"],
            "Primary width (ft)": f.ps_half * 2,
            "App 1 length (ft)": f.approach_len(a),
            "App 2 length (ft)": f.approach_len(b),
            "App 1 slope": " → ".join("%d:1" % s for _, s in a.crit["slopes"]),
            "App 2 slope": " → ".join("%d:1" % s for _, s in b.crit["slopes"]),
            "Horiz radius (ft)": max(a.crit["horiz_rad"], b.crit["horiz_rad"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Primary surface width follows the more demanding approach at "
               "either end per 77.19(a). Precision approaches run 10,000 ft "
               "at 50:1 then 40,000 ft at 40:1 per 77.19(b)(2)(ii).")

st.subheader("Obstacle analysis")
st.caption("Only the lowest, most critical surface penetrated is reported "
           "per obstacle.")
t1, t2 = st.tabs(["Add obstacle manually", "Import Excel / CSV"])
with t1:
    o1, o2, o3, o4, o5 = st.columns([2, 1.4, 1.4, 1.2, 1])
    desc = o1.text_input("Description", value="Tower")
    olat = o2.number_input("Latitude (°)", value=float(model.lf.lat0),
                           format="%.6f")
    olon = o3.number_input("Longitude (°)", value=float(model.lf.lon0),
                           format="%.6f")
    oelev = o4.number_input("Elevation (ft MSL)", value=float(model.horiz_z))
    o5.write("")
    if o5.button("Add", use_container_width=True):
        S.obstacles.append({"description": desc, "lat": olat, "lon": olon,
                            "elev": oelev})
with t2:
    ob = st.file_uploader("Columns: description, lat, lon, elev",
                          type=["csv", "xlsx", "xls"], key="obs_up")
    if ob is not None and st.button("Import obstacles"):
        odf = pd.read_excel(ob) if ob.name.lower().endswith(("xlsx", "xls")) \
            else pd.read_csv(ob)
        cols = {str(c).lower().strip(): c for c in odf.columns}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        cd = pick("description", "desc", "name", "obstacle")
        cla, clo = pick("lat", "latitude"), pick("lon", "long", "longitude")
        ce = pick("elev", "elevation", "amsl", "ft msl")
        if not (cla and clo and ce):
            st.error("Need latitude, longitude, and elevation columns.")
        else:
            for _, r in odf.iterrows():
                S.obstacles.append({
                    "description": str(r[cd]) if cd else "obstacle",
                    "lat": float(r[cla]), "lon": float(r[clo]),
                    "elev": float(r[ce])})
            st.success("Imported %d obstacles." % len(odf))

if S.obstacles:
    b1, b2, b3 = st.columns([1, 1, 3])
    if b1.button("Evaluate", type="primary"):
        S["obs_report"] = model.obstacle_report(S.obstacles)
    if b2.button("Clear all"):
        S.obstacles = []
        S.pop("obs_report", None)
        st.rerun()
    b3.caption("%d obstacle(s) staged." % len(S.obstacles))

if S.get("obs_report"):
    rep = pd.DataFrame(S["obs_report"])
    show = rep[["description", "lat", "lon", "elev", "surface",
                "surface_elev", "penetration", "penetrates"]]
    show.columns = ["Description", "Lat", "Lon", "Obstacle elev",
                    "Controlling surface", "Surface elev",
                    "Penetration (ft)", "Penetrates"]
    st.dataframe(show, use_container_width=True, hide_index=True)
    n = int(rep["penetrates"].fillna(False).sum())
    (st.error if n else st.success)(
        "%d of %d obstacles penetrate a Part 77 surface." % (n, len(rep)))

st.subheader("Export")
st.caption("DXF carries 3DFACE entities on layers. LandXML carries TIN "
           "surfaces, which Civil 3D imports as surfaces directly rather "
           "than something you rebuild by hand. Each surface is named for "
           "the runway end and surface it belongs to.")
x1, x2, x3, x4 = st.columns([1.1, 1.1, 1, 1])
fmt = x1.radio("Format", ["LandXML (TIN surfaces)", "DXF (3DFACE)"],
               label_visibility="collapsed")
epsg = x2.text_input("EPSG (blank = local feet about the ARP)", value="")
inc_ind = x3.checkbox("Individual surfaces", value=True)
inc_comp = x4.checkbox("Composite", value=True)

if st.button("Build export", type="primary"):
    if not (inc_ind or inc_comp):
        st.error("Select individual surfaces, the composite, or both.")
    else:
        tag = apt["locId"] or "airport"
        try:
            with st.spinner("Writing export…"):
                if fmt.startswith("LandXML"):
                    data = C.to_landxml(model, comp, epsg.strip() or None,
                                        individual=inc_ind,
                                        use_composite=inc_comp)
                    fname, mime = "%s_Part77.xml" % tag, "application/xml"
                    names = sorted(C.tin_groups(model, comp, inc_ind,
                                                inc_comp).keys())
                    stats = None
                else:
                    data, stats = C.to_dxf(model, comp, epsg.strip() or None,
                                           individual=inc_ind,
                                           use_composite=inc_comp)
                    fname, mime = "%s_Part77.dxf" % tag, "application/dxf"
                    names = None
            st.download_button("Download " + fname, data, file_name=fname,
                               mime=mime)
            if names:
                st.dataframe(pd.DataFrame({"TIN surface": names}),
                             use_container_width=True, hide_index=True)
            if stats:
                st.dataframe(pd.DataFrame(sorted(stats.items()),
                                          columns=["Layer", "3DFACE count"]),
                             use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error("Export failed: %s" % ex)

st.caption("Surfaces are 14 CFR Part 77.19 imaginary surfaces. The horizontal "
           "perimeter uses the 77.19(c) tangent-line construction. "
           "Transitional surfaces terminate on the horizontal or conical "
           "surface, with the 77.19(d) 5,000 ft extension where a precision "
           "approach has climbed past both. Not a substitute for a surveyed "
           "obstruction analysis.")
