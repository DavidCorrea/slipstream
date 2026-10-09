// The 3D stage: renderer, lights, sky, and the isometric camera with its three modes.
//
// The sim works on a flat x/y plane in metres. Here x stays x, the sim's y becomes -z, and up is +y, so a car
// heading of θ is a rotation of θ about the vertical axis and left turns still look like left turns.
import * as THREE from 'three';
import { createPixelRenderer } from './pixel.js';

export const toWorld = (x, y, height = 0) => new THREE.Vector3(x, height, -y);

export function createStage(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  // Full Retina resolution with antialiasing quadruples the pixels for little visible gain on a low-poly scene.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = true;
  // The soft filter samples the shadow map many more times per pixel; the plain one looks the same from this far up.
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#9cc7e8');
  scene.fog = new THREE.Fog('#b9d6ea', 700, 1500);

  const hemisphere = new THREE.HemisphereLight('#dcefff', '#5b7a3a', 1.1);
  scene.add(hemisphere);
  const sun = new THREE.DirectionalLight('#fff3dc', 2.3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -90; sun.shadow.camera.right = 90;
  sun.shadow.camera.top = 90; sun.shadow.camera.bottom = -90;
  sun.shadow.camera.near = 10; sun.shadow.camera.far = 400;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.6;
  scene.add(sun, sun.target);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 2400);
  const view = new CameraRig(camera, sun);

  const resize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    view.resize(window.innerWidth / window.innerHeight);
  };
  window.addEventListener('resize', resize);
  resize();

  // 'classic' draws straight to the screen; 'pixel' goes through the pixel renderer (see pixel.js).
  const pixel = createPixelRenderer(renderer);
  let style = 'classic';
  return {
    renderer, scene, camera, view, sun,
    // The location's light (see scenery.js): sky and ambient colours and the sun. The weather dims it from there.
    setLocation(location) {
      const [skyLight, groundLight, ambient] = location.hemisphere;
      hemisphere.color.set(skyLight);
      hemisphere.groundColor.set(groundLight);
      hemisphere.intensity = ambient;
      sun.color.set(location.sun[0]);
    },
    get style() { return style; },
    setStyle(value) {
      if (!['classic', 'pixel'].includes(value)) throw new Error(`Unknown style ${value}`);
      style = value;
    },
    render(activeCamera) {
      if (style === 'pixel') pixel.render(scene, activeCamera);
      else renderer.render(scene, activeCamera);
    },
    // How many pixels tall the image being drawn is, for anything sized in pixels.
    pixelsTall() {
      return style === 'pixel' ? pixel.height : renderer.getDrawingBufferSize(new THREE.Vector2()).y;
    },
  };
}

// Isometric-style camera that glides toward a target. 'follow' tracks one car, 'tv' picks the closest fight
// on its own, 'overview' frames the whole circuit. The view can be turned in quarter steps and zoomed.
class CameraRig {
  constructor(camera, sun) {
    this.camera = camera;
    this.sun = sun;
    this.mode = 'follow';
    this.azimuth = Math.PI / 4;
    this.targetAzimuth = this.azimuth;
    this.elevation = 0.62;
    this.zoom = 28;            // half the visible height, in metres
    this.targetZoom = 28;
    this.focus = new THREE.Vector3();
    this.goal = new THREE.Vector3();
    this.aspect = 1;
    this.overviewZoom = 300;
  }

  resize(aspect) {
    this.aspect = aspect;
    this.applyProjection();
  }

  applyProjection() {
    const camera = this.camera;
    camera.left = -this.zoom * this.aspect;
    camera.right = this.zoom * this.aspect;
    camera.top = this.zoom;
    camera.bottom = -this.zoom;
    camera.updateProjectionMatrix();
  }

  rotate(quarterTurns) { this.targetAzimuth += quarterTurns * Math.PI / 2; }

  zoomBy(factor) {
    const limits = this.mode === 'overview' ? [60, 900] : [8, 260];
    this.targetZoom = Math.min(limits[1], Math.max(limits[0], this.targetZoom * factor));
  }

  setMode(mode, circuit) {
    this.mode = mode;
    if (mode === 'overview' && circuit) {
      this.goal.copy(circuit.centre);
      this.targetZoom = this.overviewZoom = circuit.radius * 0.75 + 30;
    } else if (this.targetZoom > 260) {
      this.targetZoom = 28;
    }
  }

  // `point` is where the camera should look this frame (ignored in overview).
  update(seconds, point, speed = 0) {
    if (this.mode !== 'overview' && point) this.goal.copy(point);
    const ease = 1 - Math.exp(-seconds * (this.mode === 'overview' ? 2 : 4));
    this.focus.lerp(this.goal, ease);
    this.azimuth += (this.targetAzimuth - this.azimuth) * (1 - Math.exp(-seconds * 6));
    // Pull back a little at speed, so a fast car has more road in front of it.
    const wanted = this.mode === 'overview' ? this.targetZoom : this.targetZoom * (1 + Math.min(speed, 80) / 260);
    this.zoom += (wanted - this.zoom) * (1 - Math.exp(-seconds * 3));
    this.applyProjection();

    const distance = 600;
    const offset = new THREE.Vector3(
      Math.cos(this.azimuth) * Math.cos(this.elevation),
      Math.sin(this.elevation),
      Math.sin(this.azimuth) * Math.cos(this.elevation),
    ).multiplyScalar(distance);
    this.camera.position.copy(this.focus).add(offset);
    this.camera.lookAt(this.focus);

    // The sun's shadow box follows what's on screen, so shadows stay sharp wherever the camera goes.
    const span = Math.min(260, this.zoom * Math.max(this.aspect, 1) * 1.3);
    const shadow = this.sun.shadow.camera;
    shadow.left = shadow.bottom = -span;
    shadow.right = shadow.top = span;
    shadow.updateProjectionMatrix();
    this.sun.target.position.copy(this.focus);
    this.sun.position.copy(this.focus).add(new THREE.Vector3(-120, 220, 80));
  }
}
