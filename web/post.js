// Effects drawn over the finished image, the way a camera and a broadcast would show it:
//   ambient occlusion  soft darkening where things meet: a car on the road, a wall on the grass, a grandstand's feet
//                      (from the isometric camera only: close up, at half size, it speckles the cars)
//   glow               bright lights (brake discs, brake and rain lights, headlights, street lamps) bleed a little
//   speed blur         on the chase, onboard and wheel cameras, the edges of the picture streak at speed
//   heat haze          the air shimmers behind the exhaust of a car on the throttle, seen close up
//   depth of field     in replays, a long lens keeps the cars sharp and lets what's near and far go soft
//
// They cost GPU time, so they adapt: the occlusion is drawn at half size, and when frames take too long the
// costliest effects are dropped one at a time (and stay dropped for the session) until the frame rate recovers.
// The pixel style draws without any of them.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const GLOW = { strength: 0.35, radius: 0.3, threshold: 1.6 };
// Occlusion reaches about a car's height into the corners it darkens.
const OCCLUSION = { radius: 1.6, distanceExponent: 1.5, thickness: 2.5, scale: 1.0, samples: 12, distanceFallOff: 1.0 };
const SPEED_BLUR = { samples: 8, reach: 0.06 };     // reach: how far the edges streak at full strength, of the screen
const LENS = { aperture: 0.0016, maxblur: 0.008 };
const HEAT_SOURCES = 6;                              // the nearest exhausts that shimmer
const HEAT = { wobble: 0.08, rise: 1.6 };           // how far the picture shifts, of a source's size; how far it rises
// Dropping effects: frames slower than this on average over a few seconds of racing (the race is drawn at up to 60
// a second, a paused one at 30, which isn't counted). A gap longer than a hitch (a hidden tab) isn't counted either.
const SLOW_FRAME_MS = 1000 / 42;
const HITCH_MS = 250;
const MEASURE_SECONDS = 4;
// Most costly first: the occlusion renders the whole scene a second time, the lens a third time in replays.
const DROP_ORDER = ['occlusion', 'lens', 'glow', 'heat'];

const speedBlurShader = {
  uniforms: { tDiffuse: { value: null }, strength: { value: 0 } },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float strength;
    varying vec2 vUv;
    void main() {
      vec2 fromCentre = vUv - 0.5;
      // Nothing at the middle of the picture, where the car is; more towards the edges.
      float edge = smoothstep(0.12, 0.5, length(fromCentre));
      vec2 step = fromCentre * strength * edge * ${SPEED_BLUR.reach.toFixed(3)} / ${SPEED_BLUR.samples.toFixed(1)};
      vec4 sum = vec4(0.0);
      for (int i = 0; i < ${SPEED_BLUR.samples}; i++) sum += texture2D(tDiffuse, vUv - step * float(i));
      gl_FragColor = sum / ${SPEED_BLUR.samples.toFixed(1)};
    }
  `,
};

const heatShader = {
  uniforms: { tDiffuse: { value: null }, sources: { value: Array.from({ length: HEAT_SOURCES }, () => new THREE.Vector3()) },
              count: { value: 0 }, time: { value: 0 }, aspect: { value: 1 } },
  vertexShader: speedBlurShader.vertexShader,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec3 sources[${HEAT_SOURCES}];
    uniform int count;
    uniform float time;
    uniform float aspect;
    varying vec2 vUv;
    void main() {
      vec2 shift = vec2(0.0);
      for (int i = 0; i < ${HEAT_SOURCES}; i++) {
        if (i >= count) break;
        vec3 source = sources[i];
        // A plume above and behind the exhaust: strongest at the pipe, fading as it rises.
        vec2 away = (vUv - source.xy) * vec2(aspect, 1.0) / source.z;
        away.y = away.y > 0.0 ? away.y / ${HEAT['rise']} : away.y * 2.0;
        float plume = exp(-dot(away, away) * 2.5);
        shift += plume * source.z * vec2(sin(vUv.y * 340.0 / max(source.z * 40.0, 1.0) + time * 23.0),
                                         cos(vUv.x * 300.0 / max(source.z * 40.0, 1.0) + time * 19.0));
      }
      gl_FragColor = texture2D(tDiffuse, vUv + shift * ${HEAT['wobble']});
    }
  `,
};

export function createPost(renderer, scene, camera) {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  // Drawing into a texture loses the canvas's own antialiasing, so the texture is multisampled instead.
  const target = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, target);

  const scenePass = new RenderPass(scene, camera);
  const occlusion = new GTAOPass(scene, camera, size.x / 2, size.y / 2);
  occlusion.updateGtaoMaterial(OCCLUSION);
  occlusion.blendIntensity = 0.85;
  const fullSize = occlusion.setSize.bind(occlusion);
  occlusion.setSize = (width, height) => fullSize(Math.max(1, Math.round(width / 2)), Math.max(1, Math.round(height / 2)));
  const glow = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), GLOW.strength, GLOW.radius, GLOW.threshold);
  const speedBlur = new ShaderPass(speedBlurShader);
  const heat = new ShaderPass(heatShader);
  const lens = new BokehPass(scene, camera, { focus: 20, ...LENS });
  for (const pass of [scenePass, occlusion, heat, glow, speedBlur, lens, new OutputPass()]) composer.addPass(pass);

  const dropped = new Set();
  let enabled = true, measuredMs = 0, measuredFrames = 0, lastFrame = null;


  // Frames that take too long drop the costliest effect still on.
  function measure(now) {
    if (lastFrame !== null && now - lastFrame < HITCH_MS) {
      measuredMs += now - lastFrame;
      measuredFrames++;
    }
    lastFrame = now;
    if (measuredMs < MEASURE_SECONDS * 1000) return;
    const average = measuredMs / measuredFrames;
    measuredMs = measuredFrames = 0;
    if (average <= SLOW_FRAME_MS) return;
    const next = DROP_ORDER.find(effect => !dropped.has(effect));
    if (next) dropped.add(next);
  }

  return {
    get enabled() { return enabled; },
    setEnabled(value) { enabled = value; lastFrame = null; measuredMs = measuredFrames = 0; },
    // Which effects were dropped to keep the frame rate up.
    get dropped() { return [...dropped]; },

    resize(width, height) {
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(width, height);
    },

    // `look`: `speed` (0-1) streaks the edges, `focus` (metres from the camera) puts the lens on, `heat` lists the
    // exhausts that shimmer as [x, y, size] on the screen (0-1, size a share of its height), and `timed` says
    // whether this frame counts towards the frame rate (not while paused).
    render(activeCamera, { speed = 0, focus = null, heat: exhausts = [], timed = true } = {}, now = performance.now()) {
      if (timed) measure(now);
      else lastFrame = null;
      scenePass.camera = lens.camera = activeCamera;
      occlusion.enabled = activeCamera === camera && !dropped.has('occlusion');
      glow.enabled = !dropped.has('glow');
      speedBlur.enabled = speed > 0.02;
      heat.enabled = exhausts.length > 0 && !dropped.has('heat');
      if (heat.enabled) {
        exhausts.slice(0, HEAT_SOURCES).forEach(([x, y, size], index) => heat.uniforms.sources.value[index].set(x, y, size));
        heat.uniforms.count.value = Math.min(exhausts.length, HEAT_SOURCES);
        heat.uniforms.time.value = now / 1000;
        heat.uniforms.aspect.value = renderer.domElement.width / Math.max(renderer.domElement.height, 1);
      }
      speedBlur.uniforms.strength.value = speed;
      lens.enabled = focus !== null && activeCamera.isPerspectiveCamera && !dropped.has('lens');
      if (lens.enabled) lens.uniforms.focus.value = focus;
      composer.render();
    },

  };
}
