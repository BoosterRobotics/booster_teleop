const {t, known} = window.dashboardI18n || {t: k => k, known: x => x};
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HumanView } from './app.js';

const request = new AbortController();
let renderer, controls, scene, human, clip, raf = null, disposed = false;
let playing = true, elapsed = 0, last = 0, frameIndex = 0;
const root = document.getElementById('view');
const play = document.getElementById('play');
window.disposePreview = () => {
  if (disposed) return;
  disposed = true;
  request.abort();
  cancelAnimationFrame(raf); raf = null;
  controls?.dispose();
  scene?.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); });
  renderer?.dispose(); renderer?.forceContextLoss(); renderer?.domElement.remove();
  clip = null;
};
window.addEventListener('pagehide', window.disposePreview, { once: true });
let previewError=null;
window.addEventListener('dashboard-language', ()=>{
  play.textContent=t(playing?'ui109':'ui110');
  if(previewError){document.getElementById('filename').textContent=t('ui111');document.getElementById('error').textContent=t('ui112',{p0:known(previewError)});}
});
play.onclick = () => {
  playing = !playing;
  play.textContent = playing ? t("ui109") : t("ui110");
};
try {
  const response = await fetch('/api/trajectory-preview', { signal: request.signal });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  if (disposed) throw new DOMException('closed', 'AbortError');
  clip = data;
  document.getElementById('filename').removeAttribute('data-i18n');
  document.getElementById('filename').textContent = `${clip.filename} · ${clip.format}`;
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x080b10);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  camera.up.set(0, 0, 1);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  root.appendChild(renderer.domElement);
  human = new HumanView(scene);
  human.update({ positions: clip.positions[0], names: clip.names, edges: clip.edges });
  const bounds = new THREE.Box3().setFromObject(human.points);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
  camera.position.copy(center).add(new THREE.Vector3(Math.cos(human.yaw)*radius*1.6, Math.sin(human.yaw)*radius*1.6, radius*.35));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x263041, 2));
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center); controls.update();
  play.disabled = false;
  // Debuggable runtime state for local acceptance, no network playback loop.
  window.preview = { scene, human, renderer, controls, clip, get frameIndex() { return frameIndex; }, get raf() { return raf; }, get disposed() { return disposed; } };
  function draw(now) {
    if (disposed) return;
    if (last && playing) elapsed += (now - last) / 1000;
    last = now;
    const duration = clip.time_s.at(-1);
    const t = duration > 0 ? elapsed % duration : 0;
    // Binary search the actual recorded timestamps, not nominal FPS.
    let lo = 0, hi = clip.time_s.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (clip.time_s[mid] <= t) lo = mid + 1; else hi = mid; }
    frameIndex = Math.max(0, lo - 1);
    human.update({ positions: clip.positions[frameIndex], names: clip.names, edges: clip.edges });
    document.getElementById('time').textContent = `${t.toFixed(1)} / ${duration.toFixed(1)} s`;
    const w = root.clientWidth, h = Math.max(1, root.clientHeight);
    if (renderer.domElement.width !== Math.floor(w * renderer.getPixelRatio()) || renderer.domElement.height !== Math.floor(h * renderer.getPixelRatio())) renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    controls.update(); renderer.render(scene, camera);
    raf = requestAnimationFrame(draw);
  }
  raf = requestAnimationFrame(draw);
} catch (error) {
  if (!disposed) {
    previewError=error.message;
    window.disposePreview();
    document.getElementById('filename').textContent = t("ui111");
    const box = document.getElementById('error'); box.hidden = false;
    box.textContent = t("ui112", {p0: (known(error.message))});
  }
}
