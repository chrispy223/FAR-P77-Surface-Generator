# FAR Part 77 Airspace Surface Generator

14 CFR Part 77.19 imaginary surfaces for any number of runways, existing or
proposed, with an exact vector composite, obstacle evaluation, and DXF export.

## Run it locally

    pip install -r requirements.txt
    streamlit run part77_app.py

A browser tab opens. Type an identifier, or press **Load training example**.

## Put it on the web

Push these four files plus `requirements.txt` to a GitHub repo, then point
share.streamlit.io at it. `part77_app.py` is the entry point. Nothing else
needs configuring.

## Files

| File | Purpose |
|---|---|
| `part77_app.py` | The application |
| `part77_core.py` | Geometry, composite, obstacle evaluation, DXF |
| `part77_map.py` | Leaflet map with the cursor elevation readout |
| `requirements.txt` | Dependencies |

## How the composite stays clean

Every Part 77 surface except the conical is planar over its own footprint, so
each piece carries plane coefficients rather than a sampled mesh. The composite
is built by overlaying all footprints into atomic faces, then assigning each
face its lowest surface. Where two planes cross inside a face, the face is cut
on the exact crossing line; where the conical crosses a plane, the equal
elevation contour is traced. No gridding, so no stair-stepping, and the map can
report elevation at the cursor in closed form.

## Sources and what is checked

`obstaclePart77` in the ADIP record sets the approach category per runway end
and drives every dimension. A two-column table of end and category derived from
the FAA approach charts can be uploaded to cross-check; disagreements are
reported, never applied.

## Known limits

- The rendered/exported conical shell is a triangulated approximation of a
  curved surface, faithful to within 10 ft (locally, near hull corners) and
  far less elsewhere. All planar surfaces are exact. Obstacle evaluation
  never uses the mesh: it evaluates the surface equations directly and is
  exact everywhere, conical included.

- The horizontal perimeter uses the convex hull of the arcs, which equals the
  77.19(c) tangent-line construction for a convex runway arrangement. An
  L-shaped airfield with an outlying runway could differ.
- Runway profile is linear between end elevations.
- Local projection is equirectangular about the ARP, good to a few feet over
  the extent of the model. DXF export can restate in a state plane CRS.
- Not a substitute for a surveyed obstruction analysis.
