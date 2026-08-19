MAP_HTML = r"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;padding:0;background:transparent;}
  #map{width:100%;height:__H__px;border-radius:6px;}
  .readout{
    position:absolute;right:12px;bottom:24px;z-index:900;
    background:rgba(12,20,30,.9);color:#e8eef5;padding:8px 11px;
    border:1px solid #2b3b4d;border-radius:5px;
    font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;pointer-events:none;
    min-width:190px;
  }
  .readout b{color:#ffd479;font-weight:600;}
  .readout .s{color:#8fa6bd;}
  .leaflet-container{background:#0b1723;}
</style>
<div style="position:relative">
  <div id="map"></div>
  <div class="readout" id="ro"><span class="s">move cursor over the map</span></div>
</div>
<script>
const D = __DATA__;
const SHOW_COMPOSITE = __COMPOSITE__;

const map = L.map('map', {preferCanvas:true});
L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
  {maxZoom:19, attribution:'&copy; OpenStreetMap, &copy; CARTO'}).addTo(map);

function style(f){
  return {color:f.properties.color, weight:1, opacity:.9,
          fillColor:f.properties.color, fillOpacity:SHOW_COMPOSITE?0.55:0.22};
}
/* composite faces are drawn without strokes; the dissolved outline of each
   surface is drawn over the top instead, so interior face edges disappear */
function fillOnly(f){
  return {stroke:false, fillColor:f.properties.color, fillOpacity:0.55};
}
function lineOnly(f){
  return {color:f.properties.color, weight:1.4, opacity:.95, fill:false};
}
const groups = {};
const order = ["CONICAL","HORIZONTAL","TRANSITIONAL5000","TRANSITIONAL",
               "APPROACH2","APPROACH","PRIMARY"];

function addLayer(name, feats, show){
  if(!feats || !feats.length) return;
  const g = L.geoJSON({type:"FeatureCollection",features:feats},
                      {style:style, interactive:false});
  groups[name] = g;
  if(show) g.addTo(map);
}

if(SHOW_COMPOSITE){
  const g = L.geoJSON({type:"FeatureCollection",features:D.composite},
                      {style:fillOnly, interactive:false}).addTo(map);
  groups["Composite (controlling surface)"] = g;
  if(D.composite_outlines && D.composite_outlines.length){
    const o = L.geoJSON({type:"FeatureCollection",
                         features:D.composite_outlines},
                        {style:lineOnly, interactive:false}).addTo(map);
    groups["Surface boundaries"] = o;
  }
} else {
  order.forEach(k => addLayer(k, D.layers[k], true));
}
addLayer("Runways", D.runways, true);

const ctl = {};
Object.keys(groups).forEach(k => ctl[k]=groups[k]);
L.control.layers(null, ctl, {collapsed:false, position:'topright'}).addTo(map);

/* fit to the conical extent */
let all = [];
(D.composite.length?D.composite:[].concat(...Object.values(D.layers)))
  .forEach(f => f.geometry.coordinates[0].forEach(c => all.push([c[1],c[0]])));
if(all.length) map.fitBounds(L.latLngBounds(all).pad(0.05));
else map.setView(D.origin, 12);

/* ---- cursor elevation -------------------------------------------------
   Every surface but the conical is planar, so elevation is exact from the
   plane coefficients. The conical is radial off the horizontal boundary. */
const KX = D.kx, KY = D.ky, LAT0 = D.origin[0], LON0 = D.origin[1];
function toLocal(lat,lng){ return [(lng-LON0)*KX, (lat-LAT0)*KY]; }

const HB = D.hbound.map(p => toLocal(p[0],p[1]));
function distToBoundary(x,y){
  let best = Infinity;
  for(let i=0;i<HB.length-1;i++){
    const ax=HB[i][0], ay=HB[i][1], bx=HB[i+1][0], by=HB[i+1][1];
    const dx=bx-ax, dy=by-ay, dd=dx*dx+dy*dy;
    let t = dd>0 ? ((x-ax)*dx+(y-ay)*dy)/dd : 0;
    t = t<0?0:(t>1?1:t);
    const ex=x-(ax+t*dx), ey=y-(ay+t*dy);
    const d=ex*ex+ey*ey;
    if(d<best) best=d;
  }
  return Math.sqrt(best);
}
function inRing(ring, lng, lat){
  let ins=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++){
    const xi=ring[i][0], yi=ring[i][1], xj=ring[j][0], yj=ring[j][1];
    if(((yi>lat)!=(yj>lat)) && (lng < (xj-xi)*(lat-yi)/(yj-yi)+xi)) ins=!ins;
  }
  return ins;
}
function inFeature(f, lng, lat){
  const c=f.geometry.coordinates;
  if(!inRing(c[0],lng,lat)) return false;
  for(let i=1;i<c.length;i++) if(inRing(c[i],lng,lat)) return false;
  return true;
}
function elevAt(f, lat, lng){
  const p = f.properties;
  const xy = toLocal(lat,lng);
  if(p.plane) return p.plane[0] + p.plane[1]*xy[0] + p.plane[2]*xy[1];
  return p.conical.base_z + distToBoundary(xy[0],xy[1])/p.conical.slope;
}

const POOL = D.composite.length ? D.composite
           : order.reduce((a,k)=>a.concat(D.layers[k]||[]),[]);
const ro = document.getElementById('ro');
map.on('mousemove', e => {
  const lat=e.latlng.lat, lng=e.latlng.lng;
  let bz=null, bf=null;
  for(const f of POOL){
    if(!inFeature(f,lng,lat)) continue;
    const z = elevAt(f,lat,lng);
    if(bz===null || z<bz){ bz=z; bf=f; }
  }
  if(bz===null){
    ro.innerHTML = '<span class="s">outside Part 77 surfaces</span><br>'+
      '<span class="s">'+lat.toFixed(5)+', '+lng.toFixed(5)+'</span>';
    return;
  }
  ro.innerHTML =
    '<b>'+bz.toLocaleString(undefined,{maximumFractionDigits:1})+' ft MSL</b><br>'+
    bf.properties.label+'<br>'+
    '<span class="s">'+lat.toFixed(5)+', '+lng.toFixed(5)+'</span>';
});
map.on('mouseout', ()=>{ ro.innerHTML='<span class="s">move cursor over the map</span>'; });
</script>
"""


def render(payload, height=560, composite=False):
    import json
    return (MAP_HTML
            .replace("__DATA__", json.dumps(payload))
            .replace("__COMPOSITE__", "true" if composite else "false")
            .replace("__H__", str(height - 20)))
