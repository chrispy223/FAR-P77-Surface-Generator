VIEW_HTML = r"""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  html,body{margin:0;padding:0;background:transparent;overscroll-behavior:none;}
  #wrap{position:relative;width:100%;height:__H__px;border-radius:6px;
        overflow:hidden;background:#0b1723;}
  canvas{display:block;touch-action:none;}
  .panel{position:absolute;left:10px;top:10px;z-index:5;
    background:rgba(11,23,35,.88);border:1px solid #2b3b4d;border-radius:6px;
    padding:9px 11px;color:#dbe6f0;
    font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;max-height:__PH__px;
    overflow-y:auto;touch-action:pan-y;min-width:190px;}
  .panel h4{margin:0 0 6px;font-size:10px;letter-spacing:.12em;
    text-transform:uppercase;color:#8fa6bd;font-weight:600;}
  .row{display:flex;align-items:center;gap:7px;padding:1px 0;cursor:pointer;}
  .sw{width:11px;height:11px;border-radius:2px;flex:none;}
  .views{position:absolute;right:10px;top:10px;z-index:5;display:flex;
    flex-direction:column;gap:5px;}
  .views button{background:rgba(11,23,35,.88);color:#dbe6f0;
    border:1px solid #2b3b4d;border-radius:5px;padding:5px 11px;cursor:pointer;
    font:11px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.08em;}
  .views button:hover{border-color:#5b97ff;}
  .ve{position:absolute;left:10px;bottom:10px;right:10px;z-index:5;
    background:rgba(11,23,35,.88);border:1px solid #2b3b4d;border-radius:6px;
    padding:7px 11px;color:#dbe6f0;display:flex;align-items:center;gap:11px;
    font:11px/1 ui-monospace,Menlo,Consolas,monospace;}
  .ve input{flex:1;touch-action:pan-x;}
</style>
<div id="wrap">
  <div class="panel" id="legend"><h4>Surfaces</h4></div>
  <div class="views">
    <button data-v="iso">ISO</button><button data-v="plan">PLAN</button>
    <button data-v="prof">PROFILE</button><button data-v="end">END</button>
  </div>
  <div class="ve">
    <span>VERTICAL EXAGGERATION</span>
    <input type="range" id="ve" min="1" max="30" step="1" value="8">
    <span id="veL">8&times;</span>
  </div>
</div>
<script>
const D = __DATA__;
const wrap = document.getElementById('wrap');
let W = wrap.clientWidth, H = wrap.clientHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1723);
const camera = new THREE.PerspectiveCamera(45, W/H, 10, 4000000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W,H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
wrap.appendChild(renderer.domElement);

/* stage carries the vertical exaggeration so the grid stays true */
const stage = new THREE.Group();
scene.add(stage);

let cx=0, cy=0, n=0, maxR=1;
const groups = {};
Object.keys(D.groups).forEach(k => {
  const a = D.groups[k];
  if(!a.length) return;
  const pos = new Float32Array(a.length);
  for(let i=0;i<a.length;i+=3){
    pos[i]=a[i]; pos[i+1]=a[i+2]; pos[i+2]=-a[i+1];   /* y up */
    cx+=a[i]; cy+=-a[i+1]; n++;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.computeVertexNormals();
  const isComp = k.indexOf('COMPOSITE:')===0 || k==='RUNWAY';
  const m = new THREE.MeshBasicMaterial({
    color:new THREE.Color(D.colors[k]),
    transparent:!isComp, opacity:isComp?1.0:0.42,
    side:THREE.DoubleSide, depthWrite:isComp});
  const mesh = new THREE.Mesh(g,m);
  const grp = new THREE.Group(); grp.add(mesh);
  /* outline comes from the surface boundary, not from triangle edges */
  const ea = D.edges && D.edges[k];
  if(ea && ea.length){
    const ep = new Float32Array(ea.length);
    for(let i=0;i<ea.length;i+=3){ ep[i]=ea[i]; ep[i+1]=ea[i+2]; ep[i+2]=-ea[i+1]; }
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.BufferAttribute(ep,3));
    grp.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({
      color:new THREE.Color(D.colors[k]), transparent:true, opacity:0.85})));
  }
  stage.add(grp); groups[k]=grp;
});
cx/=Math.max(n,1); cy/=Math.max(n,1);
Object.values(D.groups).forEach(a=>{
  for(let i=0;i<a.length;i+=3){
    const r=Math.hypot(a[i]-cx, -a[i+1]-cy); if(r>maxR) maxR=r;
  }
});
stage.position.set(-cx,0,-cy);

const grid = new THREE.GridHelper(maxR*2.4, 40, 0x22344a, 0x18293b);
grid.position.y = D.elev;
scene.add(grid);

/* legend */
const leg = document.getElementById('legend');
Object.keys(groups).sort().forEach(k=>{
  const d=document.createElement('div'); d.className='row';
  d.innerHTML='<span class="sw" style="background:'+D.colors[k]+'"></span>'+
              '<span>'+k.replace('COMPOSITE:','')+'</span>';
  d.onclick=()=>{ groups[k].visible=!groups[k].visible;
                  d.style.opacity=groups[k].visible?1:.35; };
  leg.appendChild(d);
});

/* orbit, written by hand: r128 has no OrbitControls */
let az=0.7, el=0.5, dist=maxR*2.6;
const target = new THREE.Vector3(0, D.elev, 0);
function apply(){
  camera.position.set(
    target.x + dist*Math.cos(el)*Math.sin(az),
    target.y + dist*Math.sin(el),
    target.z + dist*Math.cos(el)*Math.cos(az));
  camera.lookAt(target);
  renderer.render(scene,camera);
}
function setVE(v){ stage.scale.y=v; document.getElementById('veL').innerHTML=v+'&times;'; apply(); }
document.getElementById('ve').addEventListener('input',e=>setVE(+e.target.value));

const views={iso:[0.7,0.5],plan:[0,1.5533],prof:[0,0.02],end:[1.5708,0.02]};
document.querySelectorAll('.views button').forEach(b=>{
  b.onclick=()=>{ const v=views[b.dataset.v]; az=v[0]; el=v[1];
                  target.set(0,D.elev,0); dist=maxR*2.6; apply(); };
});

/* pointer + touch. Touch must be native with passive:false and every event
   prevented, or the browser reads a drag as a scroll and the view is
   unusable on a phone. */
let drag=null;
const cv=renderer.domElement;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};});
window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan) pan(dx,dy); else { az-=dx*0.006; el=clamp(el+dy*0.006); }
  apply();
});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{e.preventDefault();
  dist*=Math.pow(1.0015,e.deltaY); apply();},{passive:false});

function clamp(v){ return Math.max(-1.5533,Math.min(1.5533,v)); }
function pan(dx,dy){
  const s=dist*0.0016;
  const rt=new THREE.Vector3(Math.cos(az),0,-Math.sin(az));
  const up=new THREE.Vector3(-Math.sin(el)*Math.sin(az),Math.cos(el),
                             -Math.sin(el)*Math.cos(az));
  target.addScaledVector(rt,-dx*s); target.addScaledVector(up,dy*s);
}

let touch=null;
function td(e){ const a=e.touches[0],b=e.touches[1];
  return Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY); }
cv.addEventListener('touchstart',e=>{
  e.preventDefault(); e.stopPropagation();
  touch = e.touches.length===1
    ? {n:1,x:e.touches[0].clientX,y:e.touches[0].clientY}
    : {n:2,d:td(e),x:(e.touches[0].clientX+e.touches[1].clientX)/2,
       y:(e.touches[0].clientY+e.touches[1].clientY)/2};
},{passive:false});
cv.addEventListener('touchmove',e=>{
  e.preventDefault(); e.stopPropagation();
  if(!touch) return;
  if(e.touches.length===1 && touch.n===1){
    const dx=e.touches[0].clientX-touch.x, dy=e.touches[0].clientY-touch.y;
    touch.x=e.touches[0].clientX; touch.y=e.touches[0].clientY;
    az-=dx*0.006; el=clamp(el+dy*0.006);
  } else if(e.touches.length===2){
    const d=td(e), mx=(e.touches[0].clientX+e.touches[1].clientX)/2,
          my=(e.touches[0].clientY+e.touches[1].clientY)/2;
    if(touch.n===2){ dist*=touch.d/d; pan(mx-touch.x,my-touch.y); }
    touch={n:2,d:d,x:mx,y:my};
  }
  apply();
},{passive:false});
cv.addEventListener('touchend',e=>{e.preventDefault(); touch=null;},{passive:false});
document.addEventListener('gesturestart',e=>e.preventDefault());

/* follow the container: Streamlit hands the component the full column
   width, which changes when the browser window does */
function resize(){
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if(!w || !h || (w===W && h===H)) return;
  W = w; H = h;
  camera.aspect = W/H;
  camera.updateProjectionMatrix();
  renderer.setSize(W,H);
  apply();
}
if(window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
window.addEventListener('resize', resize);

setVE(8);
</script>
"""


def render(mesh, height=620):
    import json
    return (VIEW_HTML
            .replace("__DATA__", json.dumps(mesh))
            .replace("__PH__", str(height - 140))
            .replace("__H__", str(height - 20)))
