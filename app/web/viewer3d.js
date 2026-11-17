/* scan3d 3D viewer: the splat, the textured mesh and the collision mesh of one job in one canvas.
 *
 * An ES module, loaded by jobview.js only when somebody opens a 3D tab. The libraries are local
 * files under vendor/ (see vendor/README.md), found through the import map in index.html.
 * Nothing here looks at the user agent: every setting that the splat renderer would otherwise
 * guess from it is passed explicitly below, so all devices get the same behaviour.
 *
 * Coordinates: the splat's numbers equal the vertex numbers of both GLB files. The GLB files
 * carry "rotation to +Y up" on their root node; the same rotation (from report.json) is put on
 * the splat here, so everything stands upright and lines up.
 *
 * The splat comes in three qualities (jobview.js decides which): a reduced SPZ preview without
 * view-dependent colour, the full SPZ preview with all of it, and splat.ply itself. The last two
 * are loaded "precise" (Spark's extended splats: float positions, near-exact SH); by default
 * Spark packs positions into 16-bit floats and the SH bands into 7, 8 and 6 bits, which is what
 * the reduced preview gets, to save memory. Splat files are decoded while they download.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const MAX_PIXEL_RATIO = 2;          // sharper than this costs a phone far more than it shows
const BACKGROUND = 0x1b1f24;
const WIRE_COLOUR = 0x39e58c;
const SOLID_COLOUR = 0x9fb2c6;
const CAMERA_STOPS = 12;            // "next viewpoint" walks through about this many cameras

export function webgl2Supported() {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    if (!gl) { return false; }
    const lose = gl.getExtension("WEBGL_lose_context");
    if (lose) { lose.loseContext(); }
    return true;
  } catch (e) {
    return false;
  }
}

const LOST = "下载中断了：连接不上电脑上的扫描程序";

// Ask for a file. Rejects with an Error whose message is already fit for the user.
async function fetchOk(url, signal) {
  let response;
  try {
    response = await fetch(url, { signal, credentials: "same-origin" });
  } catch (e) {
    if (signal.aborted) { throw e; }
    throw new Error("连接不上电脑上的扫描程序");
  }
  if (!response.ok) {
    let message = "下载失败（" + response.status + "）";
    try { const data = await response.json(); if (data && data.message) { message = data.message; } } catch (e) { /* not JSON */ }
    throw new Error(message);
  }
  return response;
}

// Download a file completely, reporting progress.
async function fetchBytes(url, signal, onProgress) {
  return readAll(await fetchOk(url, signal), signal, onProgress);
}

async function readAll(response, signal, onProgress) {
  const total = parseInt(response.headers.get("Content-Length") || "0", 10) || 0;
  if (!response.body || !response.body.getReader) {
    const whole = new Uint8Array(await response.arrayBuffer());
    onProgress(whole.length, whole.length);
    return whole;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    let part;
    try {
      part = await reader.read();
    } catch (e) {
      if (signal.aborted) { throw e; }
      throw new Error(LOST);
    }
    if (part.done) { break; }
    chunks.push(part.value);
    loaded += part.value.length;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  return bytes;
}

function disposeMaterial(material) {
  for (const key of Object.keys(material)) {
    const value = material[key];
    if (value && value.isTexture) { value.dispose(); }
  }
  material.dispose();
}

function disposeObject(root) {
  root.traverse((node) => {
    if (node.geometry) { node.geometry.dispose(); }
    if (node.material) {
      (Array.isArray(node.material) ? node.material : [node.material]).forEach(disposeMaterial);
    }
  });
}

export class Viewer3D {
  /* stage: the element the canvas goes into.
   * hooks: onProgress({label, loaded, total}) while loading (total 0: unknown),
   *        onReady(info), onError(message). */
  constructor(stage, hooks) {
    this.stage = stage;
    this.hooks = hooks;
    this.generation = 0;
    this.abort = null;
    this.loaded = { splat: null, mesh: null, collision: null };   // kind -> {url, object}
    this.mode = null;
    this.style = "wire";
    this.rotation = new THREE.Quaternion();
    this.cameras = [];          // training poses in viewer coordinates
    this.cameraIndex = 0;
    this.focus = null;          // the point the training cameras look at
    this.framed = false;
    this.frames = 0;            // rendered frames (diagnostics)
    this.forceFrames = 0;
    this.disposed = false;
    this.state = "idle";

    const renderer = this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    renderer.setClearColor(BACKGROUND);
    this.canvas = renderer.domElement;
    this.canvas.className = "viewer-canvas";
    this.canvas.setAttribute("aria-label", "三维视图");
    stage.insertBefore(this.canvas, stage.firstChild);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    this.scene.add(this.camera);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x4a5563, 1.4));
    const headlight = new THREE.DirectionalLight(0xffffff, 1.6);
    headlight.position.set(0.4, 0.7, 1);
    this.camera.add(headlight);

    // The splat renderer. The last four values are the ones it would otherwise choose by
    // user agent; level-of-detail is off, so they only matter by being the same everywhere.
    this.spark = new SparkRenderer({
      renderer, onDirty: () => { this.dirty = true; },
      enableLod: false, maxPagedSplats: 256 * 65536, lodRaycast: 25000, lodSplatCount: 2500000
    });
    this.scene.add(this.spark);
    this.splatRoot = new THREE.Group();
    this.scene.add(this.splatRoot);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.zoomToCursor = true;
    this.controls.screenSpacePanning = true;

    this.onContextLost = (event) => {
      event.preventDefault();
      this.fail("浏览器收回了显卡资源（内存不足？）。可以改用“轻量”再试。");
    };
    this.canvas.addEventListener("webglcontextlost", this.onContextLost);

    this.resizer = new ResizeObserver(() => this.resize());
    this.resizer.observe(stage);
    this.resize();

    this.dirty = true;
    const tick = () => {
      if (this.disposed) { return; }
      this.raf = requestAnimationFrame(tick);
      const moved = this.controls.update();
      if (moved || this.dirty || this.forceFrames > 0) {
        this.dirty = false;
        if (this.forceFrames > 0) { this.forceFrames--; }
        this.renderer.render(this.scene, this.camera);
        this.frames++;
      }
    };
    this.raf = requestAnimationFrame(tick);
    stage.viewer3d = this;      // for diagnostics and the browser tests
  }

  resize() {
    if (this.disposed) { return; }
    const width = Math.max(1, this.stage.clientWidth), height = Math.max(1, this.stage.clientHeight);
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    this.camera.aspect = aspect;
    // 55 degrees vertically on a wide view; on a tall one keep the horizontal angle instead,
    // so turning a phone upright does not zoom in.
    const wide = 55, tall = 2 * Math.atan(Math.tan(wide * Math.PI / 360) / aspect) * 180 / Math.PI;
    this.camera.fov = aspect >= 1 ? wide : Math.min(85, tall);
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  setState(state) {
    this.state = state;
    this.stage.setAttribute("data-state", state);
  }

  fail(message) {
    if (this.disposed) { return; }
    this.setState("error");
    this.hooks.onError(message);
  }

  /* config: {mode: "splat" | "mesh" | "collision" | "overlay",
   *          urls: {splat, mesh, collision}  (only the ones the mode needs),
   *          splat: {type: "spz" | "ply", precise: bool}  (how to read urls.splat),
   *          rotation: [x, y, z, w] or null, cameras: [{position, forward}],
   *          style: "wire" | "solid" | "both"} */
  async show(config) {
    const generation = ++this.generation;
    if (this.abort) { this.abort.abort(); }
    const abort = this.abort = new AbortController();
    const stale = () => this.disposed || generation !== this.generation;
    this.mode = config.mode;
    this.style = config.style || this.style;
    this.stage.setAttribute("data-mode", config.mode);
    this.setState("loading");

    const need = {
      splat: config.mode === "splat" || config.mode === "overlay",
      mesh: config.mode === "mesh",
      collision: config.mode === "collision" || config.mode === "overlay"
    };
    // Free what this view does not show (a phone cannot hold everything at once).
    for (const kind of Object.keys(this.loaded)) {
      const have = this.loaded[kind];
      if (have && (!need[kind] || have.url !== config.urls[kind])) { this.drop(kind); }
    }
    this.setCameras(config.rotation, config.cameras || []);
    this.dirty = true;

    const labels = { splat: "高斯点云", mesh: "网格模型", collision: "碰撞网格" };
    try {
      for (const kind of ["collision", "mesh", "splat"]) {
        if (!need[kind] || this.loaded[kind]) { continue; }
        const url = config.urls[kind];
        if (!url) { throw new Error("这个任务没有" + labels[kind] + "文件"); }
        this.hooks.onProgress({ label: "正在下载" + labels[kind], loaded: 0, total: 0 });
        const progress = (loaded, total) => {
          if (stale()) { return; }
          if (total > 0 && loaded >= total) { this.hooks.onProgress({ label: "正在处理" + labels[kind] + "…", loaded: 0, total: 0 }); }
          else { this.hooks.onProgress({ label: "正在下载" + labels[kind], loaded, total }); }
        };
        let object;
        if (kind === "splat") {
          object = await this.buildSplat(url, config.splat || {}, abort.signal, progress);
        } else {
          const bytes = await fetchBytes(url, abort.signal, progress);
          if (stale()) { return; }
          this.hooks.onProgress({ label: "正在处理" + labels[kind] + "…", loaded: 0, total: 0 });
          object = await this.buildGlb(bytes, kind);
        }
        if (stale()) { this.disposeLoaded(kind, object); return; }
        this.loaded[kind] = { url, object };
        (kind === "splat" ? this.splatRoot : this.scene).add(object);
        this.dirty = true;
      }
    } catch (error) {
      if (stale() || (error && error.name === "AbortError")) { return; }
      this.fail((error && error.message) || "加载失败");
      return;
    }
    this.applyStyle();
    if (!this.framed) { this.resetView(); this.framed = true; }
    this.forceFrames = 30;     // sorting the splats takes a few frames to settle
    this.setState("ready");
    this.hooks.onReady({ splats: this.splatCount(), shDegree: this.splatShDegree() });
  }

  splatCount() {
    const have = this.loaded.splat;
    try { return have && have.object.splats ? have.object.splats.getNumSplats() : 0; } catch (e) { return 0; }
  }

  splatShDegree() {
    const have = this.loaded.splat;
    try { return have && have.object.splats ? have.object.splats.getNumSh() : 0; } catch (e) { return 0; }
  }

  // Fetch and decode a splat file (SPZ preview or the training PLY). The file is handed to
  // the decoder piece by piece as it arrives, so a 240 MB PLY is never held here as a whole.
  async buildSplat(url, how, signal, onProgress) {
    const response = await fetchOk(url, signal);
    const total = parseInt(response.headers.get("Content-Length") || "0", 10) || 0;
    const options = { fileType: how.type === "ply" ? "ply" : "spz", lod: false };
    if (how.precise) { options.extSplats = true; }
    let failure = null, loaded = 0;
    const streamed = !!(response.body && response.body.getReader && total > 0 && typeof ReadableStream === "function");
    if (streamed) {
      const reader = response.body.getReader();
      options.streamLength = total;
      options.stream = new ReadableStream({
        async pull(controller) {
          try {
            const part = await reader.read();
            if (part.done) { controller.close(); return; }
            loaded += part.value.length;
            onProgress(loaded, total);
            controller.enqueue(part.value);
          } catch (error) {
            // The connection broke, or the user moved on. The decoder must not be left
            // waiting for the rest: end its input, and report the real reason below.
            failure = error || new Error(LOST);
            controller.close();
          }
        },
        cancel(reason) { return reader.cancel(reason).catch(() => {}); }
      });
    } else {
      options.fileBytes = await readAll(response, signal, onProgress);
    }
    const mesh = new SplatMesh(options);
    let broken = null;
    try { await mesh.initialized; } catch (error) { broken = error || new Error("unreadable"); }
    if (!broken && !failure && streamed && loaded < total) { failure = new Error(LOST); }
    if (broken || failure) {
      try { mesh.dispose(); } catch (e) { /* never got that far */ }
      if (signal.aborted) { throw (failure && failure.name === "AbortError") ? failure : new DOMException("aborted", "AbortError"); }
      if (failure) { throw new Error(LOST); }
      throw new Error("无法读取高斯点云文件" + (broken.message ? "（" + String(broken.message).slice(0, 120) + "）" : ""));
    }
    return mesh;
  }

  async buildGlb(bytes, kind) {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const gltf = await new GLTFLoader().parseAsync(buffer, "");
    const root = gltf.scene;
    const meshes = [];
    root.traverse((node) => { if (node.isMesh) { meshes.push(node); } });
    for (const node of meshes) {
      const old = node.material;
      if (kind === "mesh") {
        // The texture is a photo of the real surface with its light baked in: show it as it
        // is, without adding lighting on top.
        const map = old && old.map ? old.map : null;
        node.material = new THREE.MeshBasicMaterial({ map, color: map ? 0xffffff : SOLID_COLOUR, side: THREE.DoubleSide });
        if (old) { old.map = null; disposeMaterial(old); }
      } else {
        if (old) { disposeMaterial(old); }
        node.material = new THREE.MeshLambertMaterial({
          color: SOLID_COLOUR, flatShading: true, side: THREE.DoubleSide,
          polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1
        });
        node.userData.role = "solid";
        const wire = new THREE.Mesh(node.geometry, new THREE.MeshBasicMaterial({ color: WIRE_COLOUR, wireframe: true }));
        wire.position.copy(node.position);
        wire.quaternion.copy(node.quaternion);
        wire.scale.copy(node.scale);
        wire.userData.role = "wire";
        node.parent.add(wire);
      }
    }
    return root;
  }

  disposeLoaded(kind, object) {
    if (object.parent) { object.parent.remove(object); }
    if (kind === "splat") { object.dispose(); } else { disposeObject(object); }
  }

  drop(kind) {
    const have = this.loaded[kind];
    if (!have) { return; }
    this.loaded[kind] = null;
    this.disposeLoaded(kind, have.object);
    this.dirty = true;
  }

  // How the collision mesh is drawn. Alone: ordinary shaded faces and/or lines. Over the
  // splat: the lines are drawn on top of everything (see-through), so the two can be compared.
  applyStyle() {
    const have = this.loaded.collision;
    if (!have) { return; }
    const overlay = this.mode === "overlay";
    have.object.traverse((node) => {
      if (node.userData.role === "solid") {
        node.visible = this.style !== "wire";
      } else if (node.userData.role === "wire") {
        node.visible = this.style !== "solid";
        node.material.depthTest = !overlay;
        node.material.depthWrite = !overlay;
        node.material.transparent = overlay;
        node.material.opacity = overlay ? 0.6 : 1;
        node.renderOrder = overlay ? 10 : 0;
        node.material.needsUpdate = true;
      }
    });
    this.dirty = true;
  }

  setStyle(style) {
    this.style = style;
    this.applyStyle();
  }

  // ---- camera

  setCameras(rotation, cameras) {
    const q = new THREE.Quaternion();
    if (Array.isArray(rotation) && rotation.length === 4 && rotation.every((v) => typeof v === "number" && isFinite(v))) {
      q.fromArray(rotation).normalize();
    }
    this.rotation.copy(q);
    this.splatRoot.quaternion.copy(q);
    if (this.cameras.length || !cameras.length) { return; }
    for (const item of cameras) {
      const position = new THREE.Vector3().fromArray(item.position).applyQuaternion(q);
      const forward = new THREE.Vector3().fromArray(item.forward).applyQuaternion(q).normalize();
      if (isFinite(position.lengthSq()) && forward.lengthSq() > 0.5) { this.cameras.push({ position, forward }); }
    }
    this.focus = this.lookAtPoint();
  }

  // The point closest to all the training cameras' lines of sight (least squares): what the
  // photographer was looking at.
  lookAtPoint() {
    if (this.cameras.length < 2) { return null; }
    const a = [0, 0, 0, 0, 0, 0, 0, 0, 0], b = [0, 0, 0];
    for (const { position, forward } of this.cameras) {
      const d = [forward.x, forward.y, forward.z], p = [position.x, position.y, position.z];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const m = (r === c ? 1 : 0) - d[r] * d[c];
          a[r * 3 + c] += m;
          b[r] += m * p[c];
        }
      }
    }
    const m = new THREE.Matrix3().set(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    if (Math.abs(m.determinant()) < 1e-9) { return null; }
    const point = new THREE.Vector3(b[0], b[1], b[2]).applyMatrix3(m.invert());
    return isFinite(point.lengthSq()) ? point : null;
  }

  contentBox() {
    const box = new THREE.Box3();
    for (const kind of ["collision", "mesh"]) {
      if (this.loaded[kind]) { box.expandByObject(this.loaded[kind].object); }
    }
    if (box.isEmpty() && this.loaded.splat) {
      // No mesh to go by: the middle 90 % of the splat centres (stray far-away ones ignored).
      const xs = [], ys = [], zs = [];
      const count = this.splatCount(), step = Math.max(1, Math.floor(count / 20000));
      const point = new THREE.Vector3();
      this.loaded.splat.object.forEachSplat((index, center) => {
        if (index % step === 0) {
          point.copy(center).applyQuaternion(this.rotation);
          xs.push(point.x); ys.push(point.y); zs.push(point.z);
        }
      });
      if (xs.length) {
        const span = (values) => { values.sort((p, q) => p - q); return [values[Math.floor(values.length * 0.05)], values[Math.floor(values.length * 0.95)]]; };
        const x = span(xs), y = span(ys), z = span(zs);
        box.set(new THREE.Vector3(x[0], y[0], z[0]), new THREE.Vector3(x[1], y[1], z[1]));
      }
    }
    return box;
  }

  setClip(distance) {
    this.camera.near = Math.max(distance * 0.01, 1e-5);
    this.camera.far = distance * 400;
    this.camera.updateProjectionMatrix();
  }

  resetView() {
    if (this.cameras.length) { this.gotoCamera(this.cameraIndex); return; }
    const box = this.contentBox();
    if (box.isEmpty()) { return; }
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-4);
    this.camera.position.copy(center).add(new THREE.Vector3(0.55, 0.42, 0.72).normalize().multiplyScalar(radius * 2.2));
    this.camera.up.set(0, 1, 0);
    this.controls.target.copy(center);
    this.setClip(radius * 2.2);
    this.controls.update();
    this.dirty = true;
  }

  // Stand where one of the photos was taken, looking the way that camera looked.
  gotoCamera(index) {
    const count = this.cameras.length;
    if (!count) { return; }
    this.cameraIndex = ((index % count) + count) % count;
    const { position, forward } = this.cameras[this.cameraIndex];
    let distance = this.focus ? this.focus.clone().sub(position).dot(forward) : 0;
    const typical = this.focus ? this.focus.distanceTo(position) : 1;
    if (!(distance > typical * 0.25)) { distance = Math.max(typical, 1e-4); }
    this.camera.position.copy(position);
    this.camera.up.set(0, 1, 0);
    this.controls.target.copy(position).addScaledVector(forward, distance);
    this.setClip(distance);
    this.controls.update();
    this.dirty = true;
  }

  nextCamera() {
    const count = this.cameras.length;
    if (!count) { return; }
    this.gotoCamera(this.cameraIndex + Math.max(1, Math.round(count / CAMERA_STOPS)));
  }

  // ---- the end

  dispose() {
    if (this.disposed) { return; }
    this.disposed = true;
    this.generation++;
    if (this.abort) { this.abort.abort(); }
    cancelAnimationFrame(this.raf);
    this.resizer.disconnect();
    this.canvas.removeEventListener("webglcontextlost", this.onContextLost);
    for (const kind of Object.keys(this.loaded)) { this.drop(kind); }
    this.controls.dispose();
    try { this.scene.remove(this.spark); this.spark.dispose(); } catch (e) { /* already gone */ }
    this.renderer.dispose();
    this.renderer.forceContextLoss();     // hand the GPU memory back now, not at the next GC
    this.canvas.remove();
    this.stage.removeAttribute("data-state");
    this.stage.removeAttribute("data-mode");
    if (this.stage.viewer3d === this) { this.stage.viewer3d = null; }
  }
}
