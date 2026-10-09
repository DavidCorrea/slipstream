// The replay's cameras: a short sequence of close shots, picked afresh for every replay, the way a broadcast cuts
// a crash together. The run-up from one angle (a chase camera, the onboard, low beside a wheel, a camera at the
// spot, the helicopter), the impact slowest from another (a long lens at the spot, a low orbit round it, the
// helicopter close), then the aftermath (the orbit, a camera looking back from the car, the chase, the helicopter),
// and sometimes the impact once more from yet another angle. Each part plays at its own slow-motion speed.
import * as THREE from 'three';
import { toWorld } from './scene.js';

const PARTS = [
  // name, which shots, the speed (times real), and how much of the replay it covers, around the impact.
  { name: 'approach', shots: ['chase', 'onboard', 'wheel', 'spot', 'helicopter'], speed: 0.6 },
  { name: 'impact', shots: ['spot', 'orbit', 'helicopter'], speed: 0.25 },
  { name: 'aftermath', shots: ['orbit', 'lookingBack', 'helicopter', 'chase'], speed: 0.5 },
];
const IMPACT = { before: 0.9, after: 1.3 };   // race seconds either side of the impact the slowest part covers
const AGAIN = { chance: 0.5, before: 1.4, after: 1.0 };
const LABELS = { chase: 'Chase cam', onboard: 'Onboard', wheel: 'Wheel cam', spot: '', helicopter: 'Helicopter', orbit: '', lookingBack: 'Rear cam' };

export function createReplayCamera({ aspect }) {
  const camera = new THREE.PerspectiveCamera(45, aspect, 0.3, 2400);
  const look = new THREE.Vector3();
  let parts = [], impactPoint = new THREE.Vector3(), crash = true;

  const random = (low, high) => low + Math.random() * (high - low);
  const choose = (options, not = []) => {
    const left = options.filter(option => !not.includes(option));
    return (left.length ? left : options)[Math.floor(Math.random() * (left.length || options.length))];
  };

  function carPose(frame, car) {
    const heading = frame.cars.heading[car];
    const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
    return { point: toWorld(frame.cars.x[car], frame.cars.y[car]), forward, left: new THREE.Vector3(-forward.z, 0, forward.x) };
  }

  // A shot's own randomness, fixed when it's chosen so it doesn't jitter frame to frame.
  function shot(kind) {
    return {
      kind, side: Math.random() < 0.5 ? 1 : -1, angle: Math.random() * Math.PI * 2, turn: (Math.random() < 0.5 ? 1 : -1) * random(0.25, 0.45),
      reach: random(12, 20), height: random(1.2, 3.5), lift: random(22, 34),
    };
  }

  return {
    camera,
    resize(newAspect) { camera.aspect = newAspect; camera.updateProjectionMatrix(); },

    // Plans a replay of `frames` around `impact` (race seconds), watching `cars` (the first one most). A `crash`
    // stays where it happened, so the cameras at the spot stay there; a replay you asked for has no spot, and they
    // go along with the car instead. Returns the parts in order: { from, to, speed, shot } in race seconds.
    plan(frames, impact, cars, impactFrame, { isCrash = true } = {}) {
      crash = isCrash;
      const start = frames[0].time, end = frames[frames.length - 1].time;
      const at = time => Math.min(end, Math.max(start, time));
      const points = cars.map(car => toWorld(impactFrame.cars.x[car], impactFrame.cars.y[car]));
      impactPoint = points.reduce((sum, point) => sum.add(point), new THREE.Vector3()).divideScalar(points.length);
      const used = [];
      // In a crash the cars are touching, and a camera on the side of one sits inside the other.
      const unsuitable = crash ? ['wheel'] : [];
      const pick = part => {
        const kind = choose(part.shots.filter(kind => !unsuitable.includes(kind)), used);
        used.push(kind);
        return shot(kind);
      };
      parts = [
        { from: start, to: at(impact - IMPACT.before), speed: PARTS[0].speed, shot: pick(PARTS[0]) },
        { from: at(impact - IMPACT.before), to: at(impact + IMPACT.after), speed: PARTS[1].speed, shot: pick(PARTS[1]) },
        { from: at(impact + IMPACT.after), to: end, speed: PARTS[2].speed, shot: pick(PARTS[2]) },
      ];
      if (Math.random() < AGAIN.chance) {
        parts.push({ from: at(impact - AGAIN.before), to: at(impact + AGAIN.after), speed: PARTS[1].speed, shot: pick(PARTS[1]), again: true });
      }
      return parts.filter(part => part.to > part.from);
    },

    // Aims the camera for `part` at this moment of the race, `seconds` of real time after the last; returns the
    // label to show.
    aim(part, frame, cars, seconds) {
      const { shot: chosen } = part;
      const subject = carPose(frame, cars[0]);
      const other = cars.length > 1 ? carPose(frame, cars[1]) : subject;
      if (!crash) impactPoint.copy(subject.point);
      switch (chosen.kind) {
        case 'chase':
          camera.position.copy(subject.point).addScaledVector(subject.forward, -7).y += 2.2;
          look.copy(subject.point).addScaledVector(subject.forward, 10).y += 0.6;
          camera.fov = 55;
          break;
        case 'onboard':
          camera.position.copy(subject.point).addScaledVector(subject.forward, -0.3).y += 1.45;
          look.copy(subject.point).addScaledVector(subject.forward, 25).y += 0.4;
          camera.fov = 72;
          break;
        case 'wheel':
          // Low on the side of the car, by the front wheel, looking down the road.
          camera.position.copy(subject.point).addScaledVector(subject.forward, 0.9).addScaledVector(subject.left, 1.25 * chosen.side).y += 0.45;
          look.copy(subject.point).addScaledVector(subject.forward, 22).y += 0.3;
          camera.fov = 70;
          break;
        case 'lookingBack':
          // On the car, looking back at whoever was behind it.
          camera.position.copy(subject.point).addScaledVector(subject.forward, 0.6).y += 1.2;
          look.copy(other === subject ? subject.point.clone().addScaledVector(subject.forward, -20) : other.point).y += 0.5;
          camera.fov = 60;
          break;
        case 'spot': {
          // Beside where it happened, low, on a long lens that keeps the cars filling the frame.
          camera.position.copy(impactPoint).add(new THREE.Vector3(Math.cos(chosen.angle) * chosen.reach, chosen.height, Math.sin(chosen.angle) * chosen.reach));
          look.copy(subject.point).lerp(other.point, 0.5).y += 0.6;
          camera.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(8 / camera.position.distanceTo(look))), 8, 60);
          break;
        }
        case 'orbit':
          // Round the spot at about head height, turning as the replay plays.
          chosen.angle += chosen.turn * seconds;
          camera.position.copy(impactPoint).add(new THREE.Vector3(Math.cos(chosen.angle) * chosen.reach * 0.7, chosen.height + 0.8, Math.sin(chosen.angle) * chosen.reach * 0.7));
          look.copy(impactPoint).lerp(subject.point, 0.5).y += 0.5;
          camera.fov = 50;
          break;
        case 'helicopter':
          chosen.angle += chosen.turn * seconds * 0.5;
          camera.position.copy(impactPoint).add(new THREE.Vector3(Math.cos(chosen.angle) * 16, chosen.lift, Math.sin(chosen.angle) * 16));
          look.copy(impactPoint).lerp(subject.point, 0.5);
          camera.fov = 38;
          break;
      }
      camera.updateProjectionMatrix();
      camera.lookAt(look);
      const label = LABELS[chosen.kind];
      return part.again ? `Again${label ? ` · ${label}` : ''}` : label;
    },

    get focus() { return impactPoint; },
  };
}
