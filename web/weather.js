// The weather you can see: rain streaks falling around whatever the camera is looking at, a sky and light that
// turn grey as it rains, and a tarmac that darkens and shines as it gets wet. The sim decides how wet it is.
import * as THREE from 'three';

const DROPS = 2600;
const RAIN_BOX = { width: 150, height: 70 };   // metres around the camera's subject
const FALL_SPEED = 38;                          // metres per second
const STREAK = 1.3;                             // metres long

const vertexShader = `
  attribute vec3 start;
  attribute float tail;
  uniform float time;
  uniform vec3 centre;
  uniform float box;
  uniform float height;
  uniform vec2 wind;
  void main() {
    // Each drop falls from its own start and wraps around a box that follows the camera, so it never runs out.
    vec3 drop = start;
    drop.y = mod(start.y - time * ${FALL_SPEED.toFixed(1)}, height);
    drop.xz = centre.xz + mod(start.xz - centre.xz + box * 0.5, box) - box * 0.5;
    // Wind blows the drops sideways as they fall, so streaks lean with it.
    drop.xz += wind * (height - drop.y) / ${FALL_SPEED.toFixed(1)};
    drop += vec3(wind.x * ${(STREAK / FALL_SPEED).toFixed(3)}, ${STREAK.toFixed(1)}, wind.y * ${(STREAK / FALL_SPEED).toFixed(3)}) * tail;
    gl_Position = projectionMatrix * viewMatrix * vec4(drop, 1.0);
  }
`;
const fragmentShader = `
  uniform float opacity;
  void main() { gl_FragColor = vec4(0.78, 0.84, 0.92, opacity); }
`;

const WET_SKY = new THREE.Color('#7d8794'), WET_FOG = new THREE.Color('#8e979f');

export function createWeather(stage) {
  const starts = new Float32Array(DROPS * 2 * 3), tails = new Float32Array(DROPS * 2);
  for (let drop = 0; drop < DROPS; drop++) {
    const x = Math.random() * RAIN_BOX.width, y = Math.random() * RAIN_BOX.height, z = Math.random() * RAIN_BOX.width;
    starts.set([x, y, z, x, y, z], drop * 6);
    tails.set([0, 1], drop * 2);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(DROPS * 2 * 3), 3));
  geometry.setAttribute('start', new THREE.BufferAttribute(starts, 3));
  geometry.setAttribute('tail', new THREE.BufferAttribute(tails, 1));
  const material = new THREE.ShaderMaterial({
    vertexShader, fragmentShader, transparent: true, depthWrite: false,
    uniforms: { time: { value: 0 }, centre: { value: new THREE.Vector3() }, box: { value: RAIN_BOX.width }, height: { value: RAIN_BOX.height }, opacity: { value: 0 },
                wind: { value: new THREE.Vector2() } },
  });
  const rain = new THREE.LineSegments(geometry, material);
  rain.frustumCulled = false;
  rain.visible = false;
  stage.scene.add(rain);

  let tarmac = null, shownWetness = 0, shownRain = 0;
  // The location's clear-weather sky, fog and sun (see scenery.js), which rain greys over.
  const clearSky = new THREE.Color('#9cc7e8'), clearFog = new THREE.Color('#b9d6ea');
  let sunshine = 2.3, darkness = 0;

  return {
    setLocation(location) {
      clearSky.set(location.sky);
      clearFog.set(location.fog);
      sunshine = location.sun[1];
      // At night rain clouds have no daylight to grey: the sky stays dark.
      darkness = location.night ? 1 : 0;
    },

    // The circuit's tarmac material, which gets darker and glossier as the track gets wet.
    setTarmac(material) {
      tarmac = material;
      tarmac.userData.dryRoughness ??= tarmac.roughness;
    },

    // `wetness` and `rain` from the sim (0-1); `centre` is what the camera is looking at.
    // `wind` is [x, y] in the race's coordinates (m/s).
    update(wetness, rainIntensity, centre, seconds, time, wind = [0, 0]) {
      material.uniforms.wind.value.set(wind[0], -wind[1]);
      // Ease towards the sim's values, so frames at 20 Hz don't make the sky flicker.
      const ease = 1 - Math.exp(-seconds * 2);
      shownWetness += (wetness - shownWetness) * ease;
      shownRain += (rainIntensity - shownRain) * ease;

      rain.visible = shownRain > 0.02;
      material.uniforms.time.value = time;
      material.uniforms.centre.value.set(centre.x, 0, centre.z);
      material.uniforms.opacity.value = 0.15 + shownRain * 0.4;
      geometry.setDrawRange(0, Math.round(DROPS * Math.min(1, shownRain * 1.2)) * 2);

      const gloom = Math.min(1, shownRain * 1.4);
      stage.scene.background.copy(clearSky).lerp(WET_SKY, gloom * (1 - darkness));
      stage.scene.fog.color.copy(clearFog).lerp(WET_FOG, gloom * (1 - darkness));
      stage.sun.intensity = sunshine * (1 - gloom * 0.65);
      if (tarmac) {
        tarmac.roughness = tarmac.userData.dryRoughness - shownWetness * 0.55;
        tarmac.color.setScalar(1 - shownWetness * 0.4);
      }
    },

    get wetness() { return shownWetness; },
  };
}
