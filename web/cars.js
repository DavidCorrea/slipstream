// The cars: a low-poly open-wheeler per driver, posed every frame from the sim, with every part showing the
// car's setup and condition.
//   wheels      spin with speed, fronts steer; the sidewall stripe is the compound's colour, greying as it wears
//   brakes      discs glow orange under hard braking and cool off after
//   body        leans in corners, dips under braking, and sits lower on a full tank
//   rear wing   steeper the more wing it runs, with an extra flap at high settings
//   damage      the front wing droops, then breaks in two: one half goes at 15% damage and the other at 35%;
//               past 50% the rear wing hangs off its pillar, past 65% a front wheel splays out, and the paint
//               darkens throughout. Lost halves of wing are left on the track.
//   exhaust     a flame on full throttle, bigger and hotter in the more powerful engine modes
// The crew can lift the car on jacks, take the rear wing off, open the engine cover and swap the nose.
import * as THREE from 'three';
import { toWorld } from './scene.js';

const WHEEL_RADIUS = 0.36;
const WHEELBASE = 3.0;
const TRACK_WIDTH = 1.7;
export const COMPOUND_COLORS = { soft: '#e63946', medium: '#ffd23f', hard: '#f2f2f2', intermediate: '#3ddc84', wet: '#2f8cff' };
const WORN = new THREE.Color('#6b6b6b');

// One wheel: tyre, rim, and a stripe in the compound's colour. Also used for the loose wheels in a pit stop.
export function createWheel(compound = 'medium', front = true) {
  const wheel = new THREE.Group();
  const width = front ? 0.34 : 0.42;
  const rubber = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, width, 18), new THREE.MeshStandardMaterial({ color: '#141414', roughness: 0.85 }));
  rubber.rotation.x = Math.PI / 2;
  rubber.castShadow = true;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_RADIUS * 0.6, WHEEL_RADIUS * 0.6, width + 0.01, 6), new THREE.MeshStandardMaterial({ color: '#b9bec7', roughness: 0.3, metalness: 0.8 }));
  rim.rotation.x = Math.PI / 2;
  const stripeMaterial = new THREE.MeshStandardMaterial({ color: COMPOUND_COLORS[compound] ?? COMPOUND_COLORS.medium, roughness: 0.5 });
  for (const side of [-1, 1]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(WHEEL_RADIUS * 0.82, 0.035, 6, 24), stripeMaterial);
    ring.position.z = side * (width / 2 + 0.005);
    wheel.add(ring);
  }
  // A notch on the sidewall, so the spin reads at a glance.
  const notch = new THREE.Mesh(new THREE.BoxGeometry(WHEEL_RADIUS * 1.4, 0.07, width + 0.02), new THREE.MeshStandardMaterial({ color: '#2b2b2b' }));
  wheel.add(rubber, rim, notch);
  wheel.userData = { stripeMaterial };
  return wheel;
}

// The livery's second colour: white on dark paint, near-black on light, so the stripes always read.
function accentFor(color) {
  const { l } = new THREE.Color(color).getHSL({});
  return l < 0.5 ? '#f4f4f2' : '#16181d';
}

// The race number as a decal: drawn once into a small texture.
function numberDecal(number, color, accent) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const paint = canvas.getContext('2d');
  paint.fillStyle = color;
  paint.fillRect(0, 0, 64, 64);
  paint.fillStyle = accent;
  paint.font = '900 46px "Titillium Web", sans-serif';
  paint.textAlign = 'center';
  paint.textBaseline = 'middle';
  paint.fillText(String(number), 32, 36);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshStandardMaterial({ map: texture, roughness: 0.4 });
}

export function createCar(color, name, number = null) {
  const car = new THREE.Group();
  car.name = name;
  const baseColor = new THREE.Color(color);
  const accentColor = accentFor(color);
  const paint = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.25 });
  const accent = new THREE.MeshStandardMaterial({ color: accentColor, roughness: 0.35, metalness: 0.2 });
  const carbon = new THREE.MeshStandardMaterial({ color: '#1c1f24', roughness: 0.6, metalness: 0.3 });

  // The body leans and pitches on its own, while the wheels stay planted.
  const body = new THREE.Group();
  body.position.y = 0.3;
  car.add(body);
  const part = (geometry, surface, x, y, z, parent = body) => {
    const mesh = new THREE.Mesh(geometry, surface);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    parent.add(mesh);
    return mesh;
  };
  part(new THREE.BoxGeometry(2.6, 0.38, 0.95), paint, 0.1, 0.12, 0);             // monocoque
  part(new THREE.BoxGeometry(1.4, 0.32, 1.45), paint, -0.35, 0.08, 0);           // sidepods
  part(new THREE.BoxGeometry(0.75, 0.12, 0.62), carbon, 0.45, 0.36, 0);          // cockpit rim
  const helmet = part(new THREE.SphereGeometry(0.2, 12, 10), new THREE.MeshStandardMaterial({ color: accentColor, roughness: 0.3 }), 0.3, 0.5, 0);
  helmet.scale.set(1.1, 1, 1);
  part(new THREE.BoxGeometry(0.22, 0.05, 0.95), carbon, 0.45, 0.62, 0);          // halo bar
  part(new THREE.BoxGeometry(2.4, 0.02, 0.2), accent, 0.2, 0.32, 0);             // livery stripe down the spine
  for (const side of [-1, 1]) {
    part(new THREE.BoxGeometry(1.2, 0.08, 0.02), accent, -0.35, -0.02, side * 0.73); // stripe along each sidepod
    part(new THREE.BoxGeometry(0.06, 0.2, 0.45), carbon, 0.36, 0.1, side * 0.5);     // sidepod inlet
    part(new THREE.BoxGeometry(0.04, 0.3, 0.04), carbon, 0.75, 0.42, side * 0.42);   // mirror stalk
    part(new THREE.BoxGeometry(0.06, 0.08, 0.16), paint, 0.75, 0.58, side * 0.48);   // mirror
  }
  part(new THREE.BoxGeometry(0.3, 0.22, 0.28), carbon, -0.05, 0.62, 0);          // airbox above the driver
  part(new THREE.BoxGeometry(3.9, 0.04, 1.3), carbon, 0, -0.1, 0);               // floor
  for (const fin of [-0.4, 0, 0.4]) part(new THREE.BoxGeometry(0.6, 0.18, 0.03), carbon, -1.75, -0.04, fin);   // diffuser fins
  if (number !== null) {
    const decal = numberDecal(number, color, accentColor);
    const onNose = part(new THREE.PlaneGeometry(0.34, 0.34), decal, 1.6, 0.17, 0);
    onNose.rotation.x = -Math.PI / 2;
    onNose.rotation.z = -Math.PI / 2;
    for (const side of [-1, 1]) {
      const onPod = part(new THREE.PlaneGeometry(0.3, 0.3), decal, -0.7, 0.14, side * 0.731);
      onPod.rotation.y = side > 0 ? 0 : Math.PI;
    }
  }

  // Engine cover, hinged at its front so the crew can open it.
  const cover = new THREE.Group();
  cover.position.set(-0.55, 0.56, 0);
  body.add(cover);
  part(new THREE.BoxGeometry(1.0, 0.42, 0.55), paint, -0.5, -0.21, 0, cover);

  // Nose and front wing: one assembly that droops with damage and comes off for repairs.
  const nose = new THREE.Group();
  nose.position.set(1.9, 0.05, 0);
  body.add(nose);
  part(new THREE.BoxGeometry(1.5, 0.22, 0.42), paint, 0, 0, 0, nose);
  // The front wing is two halves (plane, flap and endplate each), so a hit can break one off and leave the other.
  const wingHalves = [-1, 1].map(side => {
    const half = new THREE.Group();
    half.position.set(0.65, -0.17, side * 0.52);
    nose.add(half);
    part(new THREE.BoxGeometry(0.45, 0.06, 1.02), paint, 0, 0, 0, half);
    part(new THREE.BoxGeometry(0.5, 0.22, 0.05), paint, 0, 0.09, side * 0.5, half);
    return half;
  });
  const frontFlaps = wingHalves.map(half => part(new THREE.BoxGeometry(0.3, 0.05, 0.95), carbon, -0.2, 0.07, 0, half));
  const firstToGo = Math.random() < 0.5 ? 0 : 1;

  // Rear wing: main plane and flap tilt with the wing setting; the whole assembly lifts off in a stop.
  const rearWing = new THREE.Group();
  rearWing.position.set(-1.9, 0.75, 0);
  body.add(rearWing);
  const mainPlane = part(new THREE.BoxGeometry(0.5, 0.07, 1.55), carbon, 0, 0, 0, rearWing);
  const flap = part(new THREE.BoxGeometry(0.32, 0.05, 1.5), paint, -0.12, 0.17, 0, rearWing);
  for (const side of [-1, 1]) part(new THREE.BoxGeometry(0.75, 0.6, 0.06), paint, 0.05, -0.25, side * 0.78, rearWing);
  part(new THREE.BoxGeometry(0.08, 0.5, 0.08), carbon, -1.65, 0.45, 0);          // wing pillar

  const brakeLight = part(new THREE.BoxGeometry(0.06, 0.1, 0.22), new THREE.MeshStandardMaterial({ color: '#400', emissive: '#ff1a1a', emissiveIntensity: 0 }), -2.17, 0.32, 0);
  // Headlights for racing at night, set into the nose.
  const headlightGlow = new THREE.MeshStandardMaterial({ color: '#333', emissive: '#fff4d6', emissiveIntensity: 0 });
  for (const side of [-1, 1]) part(new THREE.BoxGeometry(0.05, 0.07, 0.12), headlightGlow, 2.66, 0.05, side * 0.13);
  const flame = part(new THREE.ConeGeometry(0.12, 0.7, 8), new THREE.MeshBasicMaterial({ color: '#ffb347', transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }), -2.0, 0.28, 0);
  flame.rotation.z = Math.PI / 2;
  flame.visible = false;
  flame.castShadow = false;

  const wheels = [], hubs = [], steering = [], discs = [];
  for (const [x, front] of [[WHEELBASE / 2 + 0.05, true], [-WHEELBASE / 2 + 0.05, false]]) {
    for (const side of [-1, 1]) {
      const hub = new THREE.Group();
      hub.position.set(x, WHEEL_RADIUS, side * TRACK_WIDTH / 2);
      const wheel = createWheel('medium', front);
      hub.add(wheel);
      // The brake disc sits inside the wheel on the hub, so it glows without spinning with the tyre's notch.
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_RADIUS * 0.5, WHEEL_RADIUS * 0.5, 0.05, 14), new THREE.MeshStandardMaterial({ color: '#555', emissive: '#ff5a00', emissiveIntensity: 0, roughness: 0.4, metalness: 0.7 }));
      disc.rotation.x = Math.PI / 2;
      disc.position.z = -side * 0.24;
      hub.add(disc);
      car.add(hub);
      wheels.push(wheel);
      hubs.push(hub);
      discs.push(disc);
      if (front) steering.push(hub);
    }
  }

  // A soft round shadow under the car, so it sits on the ground even far from the sun's shadow box.
  const blob = new THREE.Mesh(new THREE.CircleGeometry(1, 20), new THREE.MeshBasicMaterial({ color: '#000', transparent: true, opacity: 0.22, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2;
  blob.scale.set(2.6, 1.15, 1);
  blob.position.y = 0.15;
  car.add(blob);

  const marker = selectionRing(color);
  car.add(marker);
  // A gold chevron hovering over your racer, so you can find it in the pack.
  const chevron = new THREE.Mesh(new THREE.ConeGeometry(0.7, 1.3, 4), new THREE.MeshStandardMaterial({ color: '#ffd23f', emissive: '#ffb000', emissiveIntensity: 0.6, roughness: 0.4 }));
  chevron.rotation.x = Math.PI;
  chevron.visible = false;
  car.add(chevron);

  car.userData = {
    body, wheels, hubs, steering, discs, brakeLight, headlightGlow, flame, marker, chevron, cover, nose, wingHalves, frontFlaps, rearWing, mainPlane, flap,
    // Set from outside each frame: lights on (night) and a wet track (the rear light flashes in the rain).
    lightsOn: false, raining: false,
    // Parts that come off as damage builds, in the order they go, with the damage that takes each one.
    breakable: [{ part: wingHalves[firstToGo], at: 0.15 }, { part: wingHalves[1 - firstToGo], at: 0.35 }],
    splayedWheel: Math.random() < 0.5 ? 0 : 1,
    justLost: [],
    paint, baseColor, compound: 'medium',
    spin: 0, lean: 0, pitch: 0, lastSpeed: 0, brakeHeat: 0,
    // Set by the pit crew while it works on the car.
    lift: 0, rearWingLift: 0, coverOpen: 0, noseOff: 0,
  };
  return car;
}

function selectionRing(color) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(3.2, 3.6, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.16;
  ring.visible = false;
  return ring;
}

// Poses a car from interpolated sim values for this render frame. `seconds` is the real time since the last
// render frame and `simSeconds` how much race time it covered, so wheels spin at the race's pace.
export function poseCar(car, state, seconds, simSeconds, selected, yours, time) {
  const data = car.userData;
  car.position.copy(toWorld(state.x, state.y));
  car.rotation.y = state.heading;

  data.spin -= state.speed * simSeconds / WHEEL_RADIUS;
  for (const wheel of data.wheels) wheel.rotation.z = data.spin;
  for (const hub of data.steering) hub.rotation.y = state.steer * 0.32;

  // Tyres: compound colour, greying as they wear.
  if (state.compound) data.compound = state.compound;
  const stripe = new THREE.Color(COMPOUND_COLORS[data.compound] ?? COMPOUND_COLORS.medium).lerp(WORN, Math.min(1, state.tyreWear ?? 0) * 0.8);
  for (const wheel of data.wheels) wheel.userData.stripeMaterial.color.copy(stripe);

  // Brake discs glow with the race's own brake temperature: dull red past about 400°C, bright orange near 1000°C.
  const heat = Math.min(1, Math.max(0, ((state.brakeTemp ?? 0.1) - 0.3) / 0.6));
  data.brakeHeat += (heat - data.brakeHeat) * (1 - Math.exp(-seconds * 6));
  for (const disc of data.discs) disc.material.emissiveIntensity = data.brakeHeat * 3;
  // A puncture: the rear tyre sags flat on its rim.
  data.wheels[3].scale.y = state.punctured ? 0.55 : 1;

  // Lean against the turn and pitch with the pedals, both eased so the car settles like it has springs.
  const ease = 1 - Math.exp(-seconds * 8);
  const accelerating = simSeconds > 0 ? (state.speed - data.lastSpeed) / simSeconds : 0;
  data.lastSpeed = state.speed;
  // Turning left rolls the body to the right (positive about the forward axis); braking dips the nose.
  data.lean += (state.steer * Math.min(state.speed, 60) / 60 * 0.06 - data.lean) * ease;
  data.pitch += (THREE.MathUtils.clamp(accelerating / 25, -1, 1) * 0.035 - data.pitch) * ease;
  data.body.rotation.x = data.lean;
  data.body.rotation.z = data.pitch;
  // A full tank sits the car a little lower; the jacks lift it in the pits.
  data.body.position.y = 0.3 - (state.fuel ?? 0) * 0.05 + data.lift;
  for (const hub of data.hubs) hub.position.y = WHEEL_RADIUS + data.lift;

  // Wing: steeper planes for more wing, the extra flap only on higher settings; lifted off by the crew.
  const wing = state.wing ?? 0.5;
  data.mainPlane.rotation.z = -0.05 - wing * 0.3;
  data.flap.rotation.z = -0.2 - wing * 0.55;
  data.flap.visible = wing > 0.25;
  data.rearWing.position.y = 0.75 + data.rearWingLift;
  for (const flap of data.frontFlaps) flap.rotation.z = -wing * 0.4;

  // Damage: the front wing droops, then breaks; the rear wing hangs off, a wheel splays, the paint darkens. The
  // nose comes away while the crew swaps it.
  // Where the damage is shows where the car was hit: the front wing from hits to the nose, a splayed wheel from
  // hits to the side, and the rest (darker, rear wing hanging) from everything.
  const damage = state.damage ?? 0, wingDamage = state.wingDamage ?? damage, suspension = state.suspensionDamage ?? 0;
  shedParts(data, wingDamage);
  data.wingHalves.forEach((half, side) => {
    half.rotation.x = (side ? -1 : 1) * wingDamage * 0.5;
    half.position.y = -0.17 - wingDamage * 0.15;
  });
  data.paint.color.copy(data.baseColor).multiplyScalar(1 - damage * 0.5);
  const hanging = damage > 0.5;
  data.rearWing.rotation.x = hanging ? 0.55 : 0;
  data.rearWing.rotation.z = hanging ? 0.35 : 0;
  data.hubs[data.splayedWheel].rotation.x = suspension * 0.45;
  data.nose.position.x = 1.9 + data.noseOff * 2.5;
  data.nose.position.y = 0.05 + data.noseOff * 0.6;
  data.cover.rotation.z = data.coverOpen * 0.9;

  // The rear light: bright under braking, flashing in the rain so the cars behind can see it through the spray,
  // and a steady glow at night.
  const flashing = data.raining && Math.sin(time * 12) > 0;
  data.brakeLight.material.emissiveIntensity = state.brake > 0.05 ? 2.5 + state.brake * 3 : flashing ? 2.5 : data.lightsOn ? 0.8 : 0;
  data.headlightGlow.emissiveIntensity = data.lightsOn ? 3 : 0;
  // Exhaust flame: only on full throttle, bigger and hotter the more aggressive the engine mode.
  const engine = state.engine ?? 0.5;
  data.flame.visible = state.throttle > 0.9 && state.speed > 3;
  if (data.flame.visible) {
    const flicker = 0.8 + Math.sin(time * 70) * 0.2;
    data.flame.scale.set(0.6 + engine, (0.5 + engine * 1.2) * flicker, 0.6 + engine);
    data.flame.material.color.set(engine > 0.66 ? '#ff7a1a' : engine < 0.33 ? '#7ab8ff' : '#ffb347');
  }

  data.chevron.visible = yours;
  if (yours) {
    data.chevron.position.y = 3.2 + Math.sin(time * 3) * 0.35;
    data.chevron.rotation.y = time * 1.5;
  }
  data.marker.visible = selected;
  if (selected) data.marker.material.opacity = 0.55 + Math.sin(time * 5) * 0.3;
}

// Hides the parts a car this damaged has lost, noting where each one came off so it can be left on the track;
// a repair (damage back near zero) puts them all back.
function shedParts(data, damage) {
  for (const breakable of data.breakable) {
    const lost = damage >= breakable.at;
    if (lost && breakable.part.visible) {
      const point = new THREE.Vector3();
      breakable.part.getWorldPosition(point);
      data.justLost.push(point);
    }
    breakable.part.visible = !lost;
  }
}

// Where each tyre touches the ground, in world space: [front left, front right, rear left, rear right].
export function contactPatches(car) {
  car.updateMatrixWorld();
  return car.userData.hubs.map(hub => {
    const point = new THREE.Vector3();
    hub.getWorldPosition(point);
    point.y = 0.05;
    return point;
  });
}
