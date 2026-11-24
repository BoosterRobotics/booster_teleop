import * as THREE from 'three';
import { CameraRig, RobotView, HumanView, parseGeometryBundle, addBaseScene } from '/webviz/app.js';

const TEXT = {
  zh: { title: '遥操日志回放', play: '▶ 播放', pause: '⏸ 暂停', locked: '视角锁定', unlocked: '自由视角',
        reset: '重置视角', loading: '加载中…', failed: '加载失败', noModel: '缺少模型，无法渲染机器人窗',
        states: { walking: '行走', tracking: '遥操中', tracking_paused: '暂停', arming_tracking: '准备进入',
                  exiting_to_walk: '退出中', simulation: '仿真', fault: '故障' },
        streams: { human: '原始人体', retarget: '重定向', sim: '仿真执行', real: '真机反馈' },
        missing: { inactive_walking: '不在此区间（已隐藏）', no_sample: '无样本', recording_gap: '记录缺口' },
        gap: '数据缺口', fixed: '根平移未知', keys: '按键' },
  en: { title: 'Telemetry replay', play: '▶ Play', pause: '⏸ Pause', locked: 'Camera locked',
        unlocked: 'Free camera', reset: 'Reset camera', loading: 'Loading…', failed: 'Load failed',
        noModel: 'Model unavailable: robot panes cannot render',
        states: { walking: 'Walking', tracking: 'Tracking', tracking_paused: 'Paused',
                  arming_tracking: 'Arming', exiting_to_walk: 'Exiting', simulation: 'Simulation', fault: 'Fault' },
        streams: { human: 'Raw human', retarget: 'Retarget', sim: 'Simulation', real: 'Real feedback' },
        missing: { inactive_walking: 'outside tracking (hidden)', no_sample: 'no sample', recording_gap: 'recording gap' },
        gap: 'data gap', fixed: 'root translation unknown', keys: 'Keys' },
};
const COLORS = { human: 0x38bdf8, retarget: 0xa78bfa, sim: 0x6ee7b7, real: 0xfbbf24 };
let lang = 'zh';
const t = (key) => TEXT[lang][key] ?? key;

const root = document.getElementById('replay-root');
const status = document.getElementById('status');
const keysBox = document.getElementById('keys');
const timeline = document.getElementById('timeline');
const timeLabel = document.getElementById('time-label');
const stateLabel = document.getElementById('state-label');
const playButton = document.getElementById('play-toggle');
const speedSelect = document.getElementById('speed');
const lockButton = document.getElementById('camera-lock');
const resetButton = document.getElementById('camera-reset');
const langButton = document.getElementById('language');

const view = {
  manifest: null, model: null, geometry: null, entries: [], duration: 0,
  time: 0, playing: false, speed: 1, locked: true, generation: 0, snapshot: null, error: null,
};

function fmt(seconds) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60;
  const s = Math.floor(ms / 1000) % 60, r = ms % 1000;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(r).padStart(3, '0')}`;
}

function fail(message) {
  view.error = String(message);
  status.hidden = false;
  status.className = 'error';
  status.textContent = `${t('failed')}: ${message}`;
}

function updateChrome() {
  document.getElementById('title').textContent = t('title');
  playButton.textContent = view.playing ? t('pause') : t('play');
  lockButton.textContent = view.locked ? t('locked') : t('unlocked');
  lockButton.className = view.locked ? 'on' : '';
  resetButton.textContent = t('reset');
  langButton.textContent = lang === 'zh' ? 'EN' : '中文';
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  timeLabel.textContent = `${fmt(view.time)} / ${fmt(view.duration)}`;
  const state = view.snapshot?.state;
  stateLabel.textContent = state ? (t('states')[state] || state) : '—';
  renderLabels();
  renderKeys();
}

function renderKeys() {
  const recent = (view.snapshot?.keys || []).slice(-4);
  keysBox.textContent = recent.length
    ? `${t('keys')}: ` + recent.map(k => `${k.key || k.label || '?'} @ ${fmt(k.seconds)}`).join('  ')
    : '';
}

function renderLabels() {
  const snapshot = view.snapshot;
  view.entries.forEach((entry, index) => {
    const rect = entry.rect;
    if (!rect) return;
    const label = entry.label;
    label.style.left = `${rect.left + 6}px`;
    label.style.top = `${rect.top + 6}px`;
    label.style.maxWidth = `${Math.max(48, rect.w - 12)}px`;
    const reason = snapshot?.missing?.[entry.name];
    const inactive = Boolean(reason);
    entry.label.firstChild.textContent = t('streams')[entry.name] || entry.name;
    entry.label.querySelector('small').textContent = inactive
      ? (t('missing')[reason] || reason)
      : (entry.name === 'real' ? t('fixed') : '');
    label.classList.toggle('hidden-stream', inactive);
    if (snapshot) setStreamVisible(entry, !inactive);
  });
}

function setStreamVisible(entry, visible) {
  if (entry.visible === visible) return;
  entry.visible = visible;
  if (typeof entry.view.setVisible === 'function') entry.view.setVisible(visible);
  const bodies = entry.view.bodies;
  if (Array.isArray(bodies)) bodies.forEach(body => { body.visible = visible; });
}

function layout() {
  const width = Math.max(1, root.clientWidth), height = Math.max(1, root.clientHeight);
  const count = Math.max(1, view.entries.length);
  const columns = width < 720 && count > 2 ? 2 : count;
  const rows = Math.ceil(count / columns);
  const paneWidth = width / columns;
  const paneHeight = height / rows;
  view.entries.forEach((entry, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const left = column * paneWidth;
    const top = row * paneHeight;
    entry.rect = {
      x: Math.round(left),
      y: Math.round(height - top - paneHeight),
      w: Math.round(paneWidth),
      h: Math.round(paneHeight),
      left,
      top,
    };
  });
  if (view.camera) {
    view.camera.aspect = width / height;
    view.camera.updateProjectionMatrix();
    view.renderer.setSize(width, height, false);
  }
  renderLabels();
}

function applyFrame(snapshot) {
  view.snapshot = snapshot;
  view.time = snapshot.seconds;
  for (const entry of view.entries) {
    const frame = snapshot.frames?.[entry.name];
    if (!frame) continue;
    entry.view.update(frame);
    let anchor = frame.root_pos;
    if (!anchor && frame.positions && frame.positions.length >= 3) {
      // HumanView draws the skeleton shifted so the lowest joint sits on the floor,
      // so the camera must apply the same offset or it frames the raw body.
      const names = frame.names || [];
      let index = names.indexOf('Pelvis');
      if (index < 0) index = 0;
      const zOffset = Number.isFinite(entry.view.zOffset) ? entry.view.zOffset : 0;
      anchor = [frame.positions[index * 3], frame.positions[index * 3 + 1],
                frame.positions[index * 3 + 2] + zOffset];
    }
    entry.rootQuat = frame.root_quat || null;
    if (anchor && entry.rig) entry.rig.follow(anchor, entry.view.yaw ?? null, entry.rootQuat);
  }
  const slider = view.duration > 0 ? Math.round((view.time / view.duration) * 1000) : 0;
  if (document.activeElement !== timeline) timeline.value = String(slider);
  updateChrome();
}

async function requestFrame(seconds) {
  const generation = ++view.generation;
  try {
    const response = await fetch(`/api/replay/frame?time_s=${encodeURIComponent(seconds.toFixed(3))}`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`HTTP ${response.status} ${detail.slice(0, 200)}`);
    }
    const payload = await response.json();
    if (generation !== view.generation) return;
    status.hidden = true;
    applyFrame(payload);
  } catch (error) {
    if (generation === view.generation) fail(error.message || error);
  }
}

function setTime(seconds, { fetch: doFetch = true } = {}) {
  view.time = Math.max(0, Math.min(seconds, view.duration));
  if (doFetch) requestFrame(view.time);
  else updateChrome();
}

function setPlaying(playing) {
  view.playing = Boolean(playing) && view.duration > 0;
  updateChrome();
}

function applyLock(locked) {
  view.locked = Boolean(locked);
  view.entries.forEach((entry, index) => {
    entry.rig.setLocked(view.locked);
    if (index > 0) entry.rig.controls.enabled = false;
  });
  updateChrome();
}

function resetCamera() {
  applyLock(true);
  view.entries.forEach(entry => entry.rig.reset());
}

function linkCameras() {
  if (view.locked || view.entries.length < 2) return;
  const master = view.entries[0];
  const offset = master.camera.position.clone().sub(master.rig.controls.target);
  for (let index = 1; index < view.entries.length; index++) {
    const entry = view.entries[index];
    entry.rig.target.copy(entry.rig.goalTarget);
    entry.rig.controls.target.copy(entry.rig.goalTarget);
    entry.camera.up.copy(master.camera.up);
    entry.camera.position.copy(entry.rig.goalTarget).add(offset);
    entry.camera.lookAt(entry.rig.goalTarget);
  }
}

function renderLoop(now) {
  const dt = view.lastNow ? Math.min(0.1, (now - view.lastNow) / 1000) : 0;
  view.lastNow = now;
  if (view.playing) {
    const next = view.time + dt * view.speed;
    if (next >= view.duration) { setTime(view.duration); setPlaying(false); }
    else setTime(next);
  }
  const width = Math.max(1, root.clientWidth), height = Math.max(1, root.clientHeight);
  if (view.renderer.domElement.width !== Math.floor(width * view.renderer.getPixelRatio())
      || view.renderer.domElement.height !== Math.floor(height * view.renderer.getPixelRatio())) layout();
  view.renderer.setScissorTest(true);
  if (view.locked) view.entries.forEach(entry => entry.rig.update(dt));
  else { view.entries[0].rig.update(dt); linkCameras(); }
  for (const entry of view.entries) {
    const rect = entry.rect;
    if (!rect) continue;
    entry.camera.aspect = rect.w / Math.max(1, rect.h);
    entry.camera.updateProjectionMatrix();
    view.renderer.setViewport(rect.x, rect.y, rect.w, rect.h);
    view.renderer.setScissor(rect.x, rect.y, rect.w, rect.h);
    view.renderer.render(entry.scene, entry.camera);
  }
  requestAnimationFrame(renderLoop);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} → HTTP ${response.status}`);
  return response.json();
}

async function boot() {
  status.textContent = t('loading');
  const manifest = await getJSON('/api/replay/manifest');
  view.manifest = manifest;
  view.duration = Number(manifest.duration_s) || 0;

  const canvas = document.createElement('canvas');
  canvas.id = 'replay-canvas';
  root.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setScissorTest(true);
  view.renderer = renderer;
  view.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);

  if (manifest.model_available) {
    view.model = await getJSON('/api/visualization/model');
    const blob = await fetch('/api/visualization/geometry');
    if (!blob.ok) throw new Error(`geometry → HTTP ${blob.status}`);
    view.geometry = parseGeometryBundle(await blob.arrayBuffer());
  }

  for (const name of manifest.streams) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);
    if (name !== 'human' && !view.model) throw new Error(t('noModel'));
    if (name === 'human') {
      addBaseScene(scene);
    } else {
      addBaseScene(scene);
    }
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    let entity;
    if (name === 'human') entity = new HumanView(scene);
    else {
      entity = new RobotView(scene, view.model, COLORS[name] ?? 0xffffff);
      if (view.geometry) entity.attach(view.geometry);
    }
    const rig = new CameraRig(camera, canvas, { distance: name === 'human' ? 4.6 : 3.0, height: 0.55, look: -0.15 });
    const label = document.createElement('div');
    label.className = 'pane-label';
    label.innerHTML = '<span></span><small></small>';
    root.appendChild(label);
    view.entries.push({ name, scene, camera, view: entity, rig, label, visible: true, rect: null });
  }
  applyLock(true);
  layout();
  window.dispatchEvent(new CustomEvent('replay-ready'));
  await requestFrame(0);
  requestAnimationFrame(renderLoop);
}

playButton.addEventListener('click', () => setPlaying(!view.playing));
speedSelect.addEventListener('change', () => { view.speed = Number(speedSelect.value) || 1; });
timeline.addEventListener('input', () => {
  const ratio = Number(timeline.value) / 1000;
  setTime(ratio * view.duration);
});
lockButton.addEventListener('click', () => applyLock(!view.locked));
resetButton.addEventListener('click', resetCamera);
langButton.addEventListener('click', () => {
  lang = lang === 'zh' ? 'en' : 'zh';
  keysBox.textContent = keysBox.textContent;
  updateChrome();
});
root.addEventListener('pointerdown', () => { if (view.locked) applyLock(false); }, true);
window.addEventListener('resize', layout);
window.addEventListener('keydown', event => {
  if (event.code === 'Space' && event.target === document.body) {
    event.preventDefault();
    setPlaying(!view.playing);
  }
});

window.__replay = {
  get manifest() { return view.manifest; },
  get entries() { return view.entries; },
  get currentTime() { return view.time; },
  get playing() { return view.playing; },
  get speed() { return view.speed; },
  get locked() { return view.locked; },
  get snapshot() { return view.snapshot; },
  get error() { return view.error; },
  get language() { return lang; },
  setTime: seconds => setTime(seconds),
  setPlaying,
  setLock: locked => applyLock(locked),
  resetCamera,
  setLanguage: value => { lang = value === 'en' ? 'en' : 'zh'; updateChrome(); },
  layout,
};

boot().catch(error => fail(error.message || error));
