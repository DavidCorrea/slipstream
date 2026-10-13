// The 3D stage: renderer, lights, sky, and the isometric camera with its three modes.
//
// The light comes from a sun placed for the race's time of day (midday overhead, late afternoon low and warm),
// a sky dome that fades from a deep zenith to a hazy horizon and glows around the sun, and that same sky baked
// into an environment map, so paint, glass and a wet track have something to reflect.
//
// The sim works on a flat x/y plane in metres. Here x stays x, the sim's y becomes -z, and up is +y, so a car
// heading of θ is a rotation of θ about the vertical axis and left turns still look like left turns.
import * as THREE from 'three';
import { createPost } from './post.js';

export const toWorld = (x, y, height = 0) => new THREE.Vector3(x, height, -y);

// Times of day a race can be run at: the sun's height above the horizon (radians), its colour and strength
// relative to the location's own, and how warm the horizon haze gets.
const TIMES_OF_DAY = [
  { name: 'midday', elevation: 1.05, warmth: 0.0, strength: 1.0 },
  { name: 'afternoon', elevation: 0.62, warmth: 0.25, strength: 0.95 },
  { name: 'late afternoon', elevation: 0.3, warmth: 0.7, strength: 0.8 },
];
const WARM_SUN = new THREE.Color('#ffb070'), WARM_HAZE = new THREE.Color('#f3c08f');
const SKY_RADIUS = 1800;
// The environment map is re-baked only when the sky has changed this much (rain greying it over), as baking costs
// a few milliseconds.
const REBAKE_CHANGE = 0.03;

const skyVertex = `
  varying vec3 direction;
  void main() {
    direction = normalize(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const skyFragment = `
  uniform vec3 zenith;
  uniform vec3 horizon;
  uniform vec3 haze;
  uniform vec3 sunDirection;
  uniform vec3 sunColor;
  uniform float glow;
  varying vec3 direction;
  void main() {
    float up = clamp(direction.y, 0.0, 1.0);
    vec3 colour = mix(horizon, zenith, pow(up, 0.55));
    // A band of haze just above the horizon, and the ground fading into it below.
    colour = mix(colour, haze, exp(-up * 9.0) * 0.6);
    if (direction.y < 0.0) colour = haze;
    float towardsSun = max(dot(direction, sunDirection), 0.0);
    colour += sunColor * (pow(towardsSun, 8.0) * 0.25 + pow(towardsSun, 300.0) * 1.5) * glow;
    gl_FragColor = vec4(colour, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

// The same seed always gives a race the same time of day.
export function timeOfDay(seed) {
  return TIMES_OF_DAY[Math.abs(Math.floor(seed)) % TIMES_OF_DAY.length];
}

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

  const skyUniforms = {
    zenith: { value: new THREE.Color('#5d9bd6') }, horizon: { value: new THREE.Color('#9cc7e8') }, haze: { value: new THREE.Color('#b9d6ea') },
    sunDirection: { value: new THREE.Vector3(0, 1, 0) }, sunColor: { value: new THREE.Color('#fff3dc') }, glow: { value: 1 },
  };
  const skyMaterial = new THREE.ShaderMaterial({
    vertexShader: skyVertex, fragmentShader: skyFragment, uniforms: skyUniforms, side: THREE.BackSide, depthWrite: false, fog: false,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(SKY_RADIUS, 32, 16), skyMaterial);
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  // Kept centred on whichever camera is drawing, so it's always at the horizon.
  sky.onBeforeRender = (_, __, camera) => sky.position.copy(camera.position);
  scene.add(sky);

  // The sky on its own, for baking into the environment map.
  const skyOnly = new THREE.Scene();
  const bakedSky = new THREE.Mesh(new THREE.SphereGeometry(100, 32, 16), skyMaterial);
  skyOnly.add(bakedSky);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let environment = null, bakedGloom = null;
  const bake = (gloom, always = false) => {
    if (!always && bakedGloom !== null && Math.abs(gloom - bakedGloom) < REBAKE_CHANGE) return;
    const previous = environment;
    environment = pmrem.fromScene(skyOnly, 0, 1, 300);
    scene.environment = environment.texture;
    previous?.dispose();
    bakedGloom = gloom;
  };

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
  const sunDirection = new THREE.Vector3(-0.45, 0.8, 0.3).normalize();
  const view = new CameraRig(camera, sun, sunDirection);

  const post = createPost(renderer, scene, camera);
  const resize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    post.resize(window.innerWidth, window.innerHeight);
    view.resize(window.innerWidth / window.innerHeight);
  };
  window.addEventListener('resize', resize);
  resize();

  let daylight = { sun: new THREE.Color('#fff3dc'), strength: 1, haze: new THREE.Color('#b9d6ea'), zenith: new THREE.Color('#5d9bd6'), glow: 1 };
  return {
    renderer, scene, camera, view, sun, sky: skyUniforms, post,
    // The location's light (see scenery.js) at the race's time of day (see timeOfDay): sky and ambient colours and
    // the sun. The weather greys it from there (see setSky).
    setLocation(location, time = TIMES_OF_DAY[0]) {
      const [skyLight, groundLight, ambient] = location.hemisphere;
      hemisphere.color.set(skyLight);
      hemisphere.groundColor.set(groundLight);
      // The environment map now lights the scene from the sky too, so the hemisphere light gives less.
      hemisphere.intensity = ambient * 0.6;
      const night = Boolean(location.night);
      const elevation = night ? 0.9 : time.elevation;
      sunDirection.set(-Math.cos(elevation) * 0.83, Math.sin(elevation), Math.cos(elevation) * 0.55).normalize();
      const sunColor = new THREE.Color(location.sun[0]);
      if (!night) sunColor.lerp(WARM_SUN, time.warmth * 0.6);
      sun.color.copy(sunColor);
      const haze = new THREE.Color(location.fog);
      if (!night) haze.lerp(WARM_HAZE, time.warmth * 0.5);
      // A night sky has no sun glowing in it, and reflects little.
      daylight = { sun: sunColor, strength: night ? 1 : time.strength, haze, glow: night ? 0 : 1,
                   zenith: new THREE.Color(location.sky).multiplyScalar(night ? 0.6 : 0.72) };
      skyUniforms.sunDirection.value.copy(sunDirection);
      skyUniforms.sunColor.value.copy(sunColor);
      scene.environmentIntensity = night ? 0.25 : 0.8;
      this.setSky(new THREE.Color(location.sky), haze, 0);
      bake(0, true);
    },
    // How strong the sun is for this race's time of day, before rain dims it.
    get daylight() { return daylight; },
    // The sky's horizon and haze colours, greyed by rain by `gloom` (0-1), and the environment map baked from it.
    setSky(horizon, haze, gloom) {
      skyUniforms.horizon.value.copy(horizon);
      skyUniforms.haze.value.copy(haze);
      skyUniforms.zenith.value.copy(daylight.zenith).lerp(horizon, gloom);
      skyUniforms.glow.value = daylight.glow * Math.max(0.05, 1 - gloom);
      scene.background.copy(horizon);
      bake(gloom);
    },
    // Through the effects (see post.js), or straight to the screen with them off. `look` is how the effects see
    // this frame.
    render(activeCamera, look) {
      if (post.enabled) post.render(activeCamera, look);
      else renderer.render(scene, activeCamera);
    },
    // How many pixels tall the image being drawn is, for anything sized in pixels.
    pixelsTall() {
      return renderer.getDrawingBufferSize(new THREE.Vector2()).y;
    },
  };
}

// Isometric-style camera that glides toward a target. 'follow' tracks one car, 'tv' picks the closest fight
// on its own, 'overview' frames the whole circuit. The view can be turned in quarter steps and zoomed.
class CameraRig {
  constructor(camera, sun, sunDirection) {
    this.camera = camera;
    this.sun = sun;
    this.sunDirection = sunDirection;
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
    this.sun.position.copy(this.focus).addScaledVector(this.sunDirection, 260);
  }
}
