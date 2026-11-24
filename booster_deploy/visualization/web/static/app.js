const {t, known} = window.dashboardI18n || {t: k => k, known: x => x};
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const STREAMS = ['human', 'retarget', 'sim'];
const COLORS = { human: 0x38bdf8, retarget: 0xa78bfa, sim: 0x6ee7b7 };
const GEOMETRY_ATTEMPTS = 3;

function quatFromWxyz(values, offset) {
  return new THREE.Quaternion(values[offset + 1], values[offset + 2], values[offset + 3], values[offset]);
}

// MuJoCo/T2 faces +X, so the frontal camera sits on the robot's +X axis rotated
// by its yaw.  Every view uses its own robot yaw, which is what makes the three
// viewports show the *same* front view even when the roots are yawed differently.
function yawFromQuatWxyz(quat) {
  if (!quat || quat.length < 4) return 0;
  const [w, x, y, z] = quat;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function addBaseScene(scene) {
  scene.background = new THREE.Color(0x05070a);
  scene.add(new THREE.HemisphereLight(0xdbeafe, 0x263041, 1.6));
  const light = new THREE.DirectionalLight(0xffffff, 1.5);
  light.position.set(4, -3, 6);
  scene.add(light);
  const grid = new THREE.GridHelper(40, 40, 0x46566e, 0x202b3a);
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);
  return grid;
}

class CameraRig {
  constructor(camera, canvas, {
    distance = 2.8, height = 0.55, look = -0.15,
  } = {}) {
    this.camera = camera;
    // OrbitControls snapshots camera.up when it is constructed and never re-reads
    // it, so the Z-up frame must be in place *before* the controls exist.
    camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    // Unlocked means "look around", not "move the robot": panning would be
    // immediately undone by the per-frame root follow.
    this.controls.enablePan = false;
    // Locked is the default: leaving orbit live would let a stray drag fight the
    // per-frame front-view snap and read as "the lock button does nothing".
    this.controls.enabled = false;
    this.distance = distance;
    this.height = height;
    this.look = look;
    this.locked = true;
    this.yaw = 0;
    this.target = new THREE.Vector3(0, 0, 1.0);
    this.goalTarget = this.target.clone();
    camera.fov = 45;
    camera.updateProjectionMatrix();
    this._place();
  }
  _desired(cameraPos) {
    cameraPos.set(
      this.goalTarget.x + Math.cos(this.yaw) * this.distance,
      this.goalTarget.y + Math.sin(this.yaw) * this.distance,
      this.goalTarget.z + this.height,
    );
  }
  _place() {
    this.target.copy(this.goalTarget);
    this._desired(this.camera.position);
    this.controls.target.copy(this.target);
    this.controls.update();
  }
  reset() { this._place(); }
  setLocked(locked) {
    this.locked = Boolean(locked);
    this.controls.enabled = !this.locked;
    if (this.locked) this._place();
  }
  follow(root, yaw) {
    if (!root) return;
    for (let i = 0; i < 3; i++) {
      if (!Number.isFinite(root[i])) return;   // never let NaN reach the camera
    }
    this.goalTarget.set(root[0], root[1], Math.max(0.4, root[2]) + this.look);
    if (typeof yaw === 'number' && Number.isFinite(yaw)) this.yaw = yaw;
    if (this.locked) return;
    // Unlocked: keep the orbit centre on the robot, never fight the operator.
    this.target.copy(this.goalTarget);
    this.controls.target.copy(this.target);
  }
  update(dt) {
    if (this.locked) {
      const k = 1 - Math.exp(-dt / 0.06);
      const want = new THREE.Vector3();
      this._desired(want);
      this.camera.position.lerp(want, k);
      this.target.lerp(this.goalTarget, k);
      this.camera.lookAt(this.target);
      this.controls.target.copy(this.target);
    }
    this.controls.update();
  }
}

class RobotView {
  constructor(scene, manifest, color) {
    this.scene = scene;
    this.manifest = manifest;
    this.color = color;
    this.bodies = manifest.bodies.map(() => new THREE.Group());
    this.bodies.forEach(body => scene.add(body));
    this.root = [0, 0, 0.9];
    this.rootQuat = [1, 0, 0, 0];
    this.yaw = 0;
    this.linePositions = new Float32Array(Math.max(0, manifest.bodies.length - 2) * 6);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3).setUsage(THREE.DynamicDrawUsage));
    this.lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    scene.add(this.lines);
  }
  setVisible(v) {
    for (const body of this.bodies) body.visible = v;
    if (this.lines) this.lines.visible = v;
  }
  /** Attach every visual from the already-decoded geometry map (no per-part fetch). */
  attach(geometries) {
    if (!geometries) return 0;
    // Retries must not stack duplicate meshes on the same body.
    for (const body of this.bodies) {
      for (const child of body.children.slice()) body.remove(child);
    }
    let attached = 0;
    for (const visual of this.manifest.visuals) {
      const geometry = geometries.get(visual.mesh_id);
      if (!geometry) continue;
      const rgba = visual.rgba || [0.75, 0.75, 0.75, 1];
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]).lerp(new THREE.Color(this.color), 0.12),
        roughness: 0.68,
        metalness: 0.04,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.fromArray(visual.pos || [0, 0, 0]);
      mesh.quaternion.copy(quatFromWxyz(visual.quat || [1, 0, 0, 0], 0));
      this.bodies[visual.body].add(mesh);
      attached += 1;
    }
    return attached;
  }
  missingVisuals(geometries) {
    if (!geometries) return this.manifest.visuals.map(v => v.mesh);
    return this.manifest.visuals.filter(v => !geometries.get(v.mesh_id)).map(v => v.mesh);
  }
  update(frame) {
    const pos = frame.body_pos || [], quat = frame.body_quat || [];
    for (let i = 0; i < this.bodies.length; i++) {
      this.bodies[i].position.set(pos[i * 3] || 0, pos[i * 3 + 1] || 0, pos[i * 3 + 2] || 0);
      if (quat.length >= i * 4 + 4) this.bodies[i].quaternion.copy(quatFromWxyz(quat, i * 4));
    }
    this.root = frame.root_pos || this.root;
    this.rootQuat = frame.root_quat || this.rootQuat;
    this.yaw = yawFromQuatWxyz(this.rootQuat);
    let edge = 0;
    for (let i = 1; i < this.manifest.bodies.length; i++) {
      const parent = this.manifest.bodies[i].parent;
      if (parent <= 0) continue;
      const a = this.bodies[parent].position, b = this.bodies[i].position;
      this.linePositions.set([a.x, a.y, a.z, b.x, b.y, b.z], edge * 6);
      edge++;
    }
    this.lines.geometry.setDrawRange(0, edge * 2);
    this.lines.geometry.attributes.position.needsUpdate = true;
  }
}

// Display heading only: L = left - right; F = L cross world up.
function humanHeading(names, positions, previous = 0) {
  const index = new Map(names.map((n, i) => [n.toLowerCase().replace(/[^a-z]/g, ''), i]));
  for (const [left, right] of [['leftshoulder','rightshoulder'], ['leftarm','rightarm'], ['lefthip','righthip'], ['leftupleg','rightupleg']]) {
    const a = index.get(left), b = index.get(right);
    if (a === undefined || b === undefined) continue;
    const x = positions[a*3] - positions[b*3], y = positions[a*3+1] - positions[b*3+1];
    if (!Number.isFinite(x+y) || Math.hypot(x,y) < 1e-6) continue;
    const yaw = Math.atan2(-x,y);
    return previous + Math.atan2(Math.sin(yaw-previous), Math.cos(yaw-previous));
  }
  return previous;
}

const handJoint = name => /hand|wrist|thumb|index|middle|ring|pinky/i.test(name);
const sideColor = name => /^left/i.test(name) ? 0x7D9EB4 : /^right/i.test(name) ? 0xBAA16B : 0xB3B7BC;
function humanBoneRadius(a, b) {
  const ah = handJoint(a), bh = handJoint(b);
  if (!ah && bh) return .012; // wrist (drawers.py precedence)
  if (ah !== bh || (ah && (/hand$/i.test(a) || /hand$/i.test(b) || /inhand/i.test(a+b)))) return .008;
  return ah && bh ? .0025 : .015;
}

class HumanView {
  constructor(scene) {
    this.scene = scene; this.root = [0,0,1]; this.yaw = 0;
    this.zOffset = null; this.points = null; this.lines = null;
    this.sphere = new THREE.SphereGeometry(1, 12, 8);
    this.cylinder = new THREE.CylinderGeometry(1, 1, 1, 10);
    this.material = new THREE.MeshStandardMaterial({roughness:.75, metalness:0});
    this.matrixObject = new THREE.Object3D();
    this.start = new THREE.Vector3(); this.end = new THREE.Vector3();
    this.direction = new THREE.Vector3(); this.up = new THREE.Vector3(0,1,0);
    this.color = new THREE.Color(); this.visible = true;
  }
  attach() { return 0; }
  setVisible(v) {
    this.visible = v;
    if (this.points) this.points.visible = v;
    if (this.lines) this.lines.visible = v;
  }
  update(frame) {
    const raw = frame.positions || [], names = frame.names || [], edges = frame.edges || [];
    if (!raw.length || !raw.every(Number.isFinite)) return;
    if (this.zOffset === null) {
      let lowest = Infinity;
      for (let i=2;i<raw.length;i+=3) lowest = Math.min(lowest,raw[i]);
      this.zOffset = Number.isFinite(lowest) ? -lowest : 0;
    }
    if (!this.positions || this.positions.length !== raw.length) this.positions = new Float32Array(raw.length);
    this.positions.set(raw);
    for (let i=2;i<this.positions.length;i+=3) this.positions[i] += this.zOffset;
    const count = raw.length/3;
    if (!this.points || this.points.count !== count) {
      if (this.points) this.scene.remove(this.points);
      this.points = new THREE.InstancedMesh(this.sphere,this.material,count);
      this.points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.points.frustumCulled = false; this.scene.add(this.points);
    }
    if (!this.lines || this.lines.count !== edges.length) {
      if (this.lines) this.scene.remove(this.lines);
      this.lines = new THREE.InstancedMesh(this.cylinder,this.material,edges.length);
      this.lines.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.lines.frustumCulled = false; this.scene.add(this.lines);
    }
    this.jointRadii = this.jointRadii || [];
    this.linkRadii = this.linkRadii || [];
    const o = this.matrixObject;
    for (let i=0;i<count;i++) {
      const radius = handJoint(names[i] || '') ? .008 : .03;
      this.jointRadii[i] = radius;
      o.position.fromArray(this.positions,i*3); o.quaternion.identity(); o.scale.setScalar(radius); o.updateMatrix();
      this.points.setMatrixAt(i,o.matrix);
      this.points.setColorAt(i,this.color.setHex(sideColor(names[i] || '')));
    }
    edges.forEach(([a,b],i) => {
      this.start.fromArray(this.positions,a*3); this.end.fromArray(this.positions,b*3);
      this.direction.subVectors(this.end,this.start);
      const length = this.direction.length(), radius = humanBoneRadius(names[a]||'',names[b]||'');
      this.linkRadii[i] = radius;
      o.position.copy(this.start).add(this.end).multiplyScalar(.5);
      o.quaternion.setFromUnitVectors(this.up, length > 1e-8 ? this.direction.multiplyScalar(1/length) : this.up);
      o.scale.set(radius,length,radius); o.updateMatrix(); this.lines.setMatrixAt(i,o.matrix);
      this.lines.setColorAt(i,this.color.setHex(handJoint(names[a]||'') && handJoint(names[b]||'') ? sideColor(names[b]||'') : 0x8B929B));
    });
    this.points.instanceMatrix.needsUpdate = true; this.lines.instanceMatrix.needsUpdate = true;
    if(this.points.instanceColor) this.points.instanceColor.needsUpdate = true;
    if(this.lines.instanceColor) this.lines.instanceColor.needsUpdate = true;
    this.points.computeBoundingSphere(); this.points.computeBoundingBox();
    const hips = names.findIndex(n => /^(hips|pelvis)$/i.test(n));
    this.root = Array.from(this.positions.slice(Math.max(0,hips)*3,Math.max(0,hips)*3+3));
    this.yaw = humanHeading(names,raw,this.yaw);
    this.setVisible(this.visible);
  }
}

function parseGeometryBundle(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'BVZ2') throw new Error(`geometry magic ${magic}`);
  const version = view.getUint16(4, true);
  if (version !== 2) throw new Error(`geometry version ${version}`);
  const count = view.getUint16(6, true);
  const entries = [];
  for (let i = 0; i < count; i++) {
    const base = 8 + i * 28;
    entries.push({
      meshId: view.getUint32(base, true),
      vertices: view.getUint32(base + 4, true),
      faces: view.getUint32(base + 8, true),
      flags: view.getUint32(base + 12, true),
      vertexByte: view.getUint32(base + 16, true),
      normalByte: view.getUint32(base + 20, true),
      faceByte: view.getUint32(base + 24, true),
    });
  }
  const geometries = new Map();
  for (const entry of entries) {
    const indexBytes = (entry.flags & 1) ? 2 : 4;
    const hasNormals = Boolean(entry.flags & 2);
    const needed = entry.faceByte + entry.faces * 3 * indexBytes;
    if (needed > buffer.byteLength) {
      throw new Error(`geometry truncated: mesh ${entry.meshId} needs ${needed} bytes`);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(
      new Float32Array(buffer, entry.vertexByte, entry.vertices * 3), 3));
    if (hasNormals) {
      // MuJoCo normals, quantised to int8: no computeVertexNormals over ~1M faces.
      geometry.setAttribute('normal', new THREE.BufferAttribute(
        new Int8Array(buffer, entry.normalByte, entry.vertices * 3), 3, true));
    }
    geometry.setIndex(new THREE.BufferAttribute(
      indexBytes === 2
        ? new Uint16Array(buffer, entry.faceByte, entry.faces * 3)
        : new Uint32Array(buffer, entry.faceByte, entry.faces * 3), 1));
    if (!hasNormals) geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    geometries.set(entry.meshId, geometry);
  }
  return geometries;
}

const LOADING_MIN_MS = 700;
const GHOST_MS = 3000;
function nowMs() {
  return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}
async function holdFor(shownAt) {
  const left = LOADING_MIN_MS - (nowMs() - shownAt);
  if (left > 0) await new Promise(done => setTimeout(done, left));
}

async function readWithProgress(response, onProgress) {
  const total = Number(response.headers.get('content-length') || 0);
  if (!response.body || !response.body.getReader) {
    const buffer = await response.arrayBuffer();
    onProgress(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let done = 0;
  for (;;) {
    const step = await reader.read();
    if (step.done) break;
    chunks.push(step.value);
    done += step.value.byteLength;
    onProgress(done, total);
  }
  const out = new Uint8Array(done);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.byteLength; }
  return out.buffer;
}

class WebVisualizer {
  constructor(root) {
    this.root = root;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'webviz-canvas';
    root.appendChild(this.canvas);
    this.overlay = document.createElement('div');
    this.overlay.className = 'webviz-error';
    this.overlay.hidden = true;
    root.appendChild(this.overlay);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setScissorTest(true);
    this.views = STREAMS.slice();
    this.layout = '3';
    this.locked = true;
    this.entries = {};
    this.lastSeq = {};
    this.geometries = null;
    this.clock = new THREE.Clock();
    this.init();
  }
  async init() {
    window.addEventListener('dashboard-language', ()=>{
      if(this.overlay) this.overlay.textContent=window.dashboardI18n.refreshText(this.overlay.textContent);
      if(this.loading) this.loading.textContent=window.dashboardI18n.refreshText(this.loading.textContent);
      this.updateBadges(this.displayStatus||{}, this.displayErrors||{});
    });
    try {
      this.manifest = await this.loadManifest();
    } catch (err) {
      // No manifest means no scenes at all: say so instead of showing a black box.
      this.showError(t("ui101", {p0: (err && err.message ? err.message : err)}));
      return;
    }
    for (const name of STREAMS) {
      const scene = new THREE.Scene();
      const grid = addBaseScene(scene);
      const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 200);
      const rig = name === 'human'
        ? new CameraRig(camera, this.canvas, { distance: 3.0, height: 0.3, look: -0.15 })
        : new CameraRig(camera, this.canvas);
      const entity = name === 'human' ? new HumanView(scene) : new RobotView(scene, this.manifest, COLORS[name]);
      this.entries[name] = { scene, grid, camera, rig, entity };
    }
    if (this.manifest.default_frame) {
      this.applyPayload({streams: {sim:this.manifest.default_frame, retarget:this.manifest.default_frame},
        status:{sim:{display:'default'},retarget:{display:'default'}}});
    }
    this.connectStream();
    this.render();
    window.dispatchEvent(new Event('webviz-ready'));
    await this.loadGeometry();
  }
  async loadManifest() {
    let lastError = null;
    for (let attempt = 1; attempt <= GEOMETRY_ATTEMPTS; attempt++) {
      try {
        const response = await fetch('/api/visualization/model', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
      } catch (err) {
        lastError = err;
        if (attempt < GEOMETRY_ATTEMPTS) await new Promise(done => setTimeout(done, 400 * attempt));
      }
    }
    throw lastError;
  }
  async loadGeometry() {
    // Show the hint before the request so it is visible even on a warm cache, and
    // keep it on screen for a moment: a local 13 MB bundle can arrive in <100 ms and
    // a single-frame flash is indistinguishable from "no hint at all".
    const shownAt = nowMs();
    this.showLoading(0, null);
    let lastError = null;
    for (let attempt = 1; attempt <= GEOMETRY_ATTEMPTS; attempt++) {
      try {
        const url = this.manifest.geometry || '/api/visualization/geometry';
        const response = await fetch(url, { cache: 'default' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await readWithProgress(response, (done, total) => {
          this.showError(null);
          this.showLoading(done, total);
        });
        this.geometries = parseGeometryBundle(buffer);
        for (const name of STREAMS) this.entries[name].entity.attach(this.geometries);
        const missing = this.entries.retarget.entity.missingVisuals(this.geometries);
        if (missing.length) throw new Error(t("ui102", {p0: (missing.length), p1: (missing.slice(0, 4).join(', '))}));
        this.showError(null);
        await holdFor(shownAt);
        this.showLoading(null, null);
        return;
      } catch (err) {
        lastError = err;
        if (attempt < GEOMETRY_ATTEMPTS) await new Promise(done => setTimeout(done, 400 * attempt));
      }
    }
    this.showError(t("ui103", {p0: (lastError && lastError.message ? lastError.message : lastError)}));
  }
  showLoading(done, total) {
    if (!this.loading) {
      this.loading = document.createElement('div');
      this.loading.className = 'webviz-error webviz-loading';
      this.root.appendChild(this.loading);
    }
    if (done === null) { this.loading.hidden = true; return; }
    const mb = (v) => (v / 1048576).toFixed(1);
    this.loading.hidden = false;
    this.loading.textContent = total
      ? t("ui104", {p0: (mb(done)), p1: (mb(total))})
      : t("ui105", {p0: (mb(done))});
  }
  showError(message) {
    this.geometryError = message || null;
    this.overlay.textContent = message || '';
    this.overlay.hidden = !message;
    const badge = document.querySelector('[data-webviz-badge="retarget"]');
    if (badge && message) { badge.textContent = t("ui106"); badge.classList.add('stale'); }
  }
  setViews(views, layout) { this.views = views.length ? views : ['human']; this.layout = layout || '1'; }
  setLocked(locked) { this.locked = Boolean(locked); Object.values(this.entries).forEach(e => e.rig.setLocked(this.locked)); }
  reset() { Object.values(this.entries).forEach(e => e.rig.reset()); }
  connectStream() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${scheme}://${location.host}/ws/visualization`);
    socket.onmessage = event => this.applyPayload(JSON.parse(event.data));
    socket.onerror = () => this.updateBadges({}, { dashboard: 'visualization socket error' });
    socket.onclose = () => setTimeout(() => this.connectStream(), 1000);
    this.socket = socket;
  }
  applyPayload(payload) {
    this.displayStatus = payload.status || {};
    this.lastPayloadAt = nowMs();
    this.lastFrameAt = this.lastFrameAt || {};
    for (const name of STREAMS) {
      const frame = (payload.streams || {})[name];
      const status = this.displayStatus[name] || {};
      const identity = `${status.generation ?? ''}:${status.display === 'preview' ? 'preview' : frame?.seq}`;
      if (!frame || identity === this.lastSeq[name]) { this.syncVisibility(name); continue; }
      this.lastSeq[name] = identity;
      this.lastFrameAt[name] = nowMs();
      const entry = this.entries[name];
      entry.entity.update(frame);
      entry.rig.follow(entry.entity.root, entry.entity.yaw);
      const r = entry.entity.root || [0, 0, 0];
      entry.grid.position.set(Math.round(r[0] / 5) * 5, Math.round(r[1] / 5) * 5, 0);
    }
    for (const name of STREAMS) this.syncVisibility(name);
    this.updateBadges(payload.status || {}, payload.errors || {});
  }
  // A stream with no data (or older than GHOST_MS) is hidden instead of being drawn
  // at its last/identity pose — an idle robot at the origin looks like a real pose.
  syncVisibility(name) {
    const entry = this.entries[name];
    if (!entry || !entry.entity.setVisible) return;
    const at = (this.lastFrameAt || {})[name];
    const stale = at === undefined || (nowMs() - at) > GHOST_MS;
    const state = ((this.displayStatus || {})[name] || {});
    const isDefault = state.display === 'default';
    const retained = ['preview','frozen','stopped'].includes(state.display) && nowMs()-this.lastPayloadAt <= GHOST_MS;
    entry.entity.setVisible(isDefault || retained || !stale);
  }
  updateBadges(status, errors) {
    this.displayErrors=errors;
    for (const name of STREAMS) {
      const badge = document.querySelector(`[data-webviz-badge="${name}"]`);
      if (!badge) continue;
      if (this.geometryError && name !== 'human') {
        badge.textContent = t("ui106");
        badge.classList.add('stale');
        continue;
      }
      const state = status[name];
      if (state && state.display === 'default') {
        badge.textContent = t("ui107"); badge.classList.add('stale'); continue;
      }
      if (state && ['preview','frozen','stopped'].includes(state.display)) {
        badge.textContent = t('display_'+state.display); badge.classList.toggle('stale', state.display !== 'preview'); continue;
      }
      badge.textContent = errors[name] ? t("ui108") : (!state ? t("ui38") : (state.stale ? t('viz_stale',{p0:state.age_s.toFixed(1)}) : `${(state.age_s * 1000).toFixed(0)}ms`));
      badge.classList.toggle('stale', !state || state.stale || Boolean(errors[name]));
    }
  }
  rectangles(width, height) {
    const names = this.views.slice(0, this.layout === '3v' ? 3 : Number(this.layout));
    if (this.layout === '3v') return names.map((name, i) => ({ name, x: 0, y: height - (i + 1) * height / names.length, w: width, h: height / names.length }));
    return names.map((name, i) => ({ name, x: i * width / names.length, y: 0, w: width / names.length, h: height }));
  }
  render() {
    requestAnimationFrame(() => this.render());
    const dt = this.clock.getDelta();
    const width = this.root.clientWidth, height = this.root.clientHeight;
    const dpr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.floor(width * dpr) || this.canvas.height !== Math.floor(height * dpr)) this.renderer.setSize(width, height, false);
    this.renderer.setScissorTest(true);
    for (const rect of this.rectangles(width, height)) {
      const entry = this.entries[rect.name]; if (!entry) continue;
      entry.camera.aspect = rect.w / Math.max(1, rect.h); entry.camera.updateProjectionMatrix();
      entry.rig.update(dt);
      this.renderer.setViewport(rect.x, rect.y, rect.w, rect.h);
      this.renderer.setScissor(rect.x, rect.y, rect.w, rect.h);
      this.renderer.render(entry.scene, entry.camera);
    }
  }
}

export { HumanView };

const root = document.getElementById('webviz-root');
if (root) {
  const visualizer = new WebVisualizer(root);
  window.webviz = visualizer;
}
