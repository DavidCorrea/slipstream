// TV mode: a director cutting between cameras the way a race broadcast does. It follows the story (an overtake,
// contact, a stop, the closest fight when nothing is happening) and picks a shot for it: a trackside camera
// zooming in as the cars come past, the helicopter, a chase camera, the onboard T-cam, the pit lane, the start
// and the finish line. Shots are perspective, unlike the isometric camera the other modes use.
import * as THREE from 'three';
import { toWorld } from './scene.js';

const QUIET_SHOT_SECONDS = [5, 9];         // how long a shot lasts when nothing calls for a cut
const STORY_SECONDS = 7;                   // how long a big moment keeps the director's attention
const START_SHOT_RACE_SECONDS = 7;         // race time the start camera stays on before the field is gone
const ROTATION = ['trackside', 'helicopter', 'chase', 'trackside', 'onboard', 'helicopter'];
const STORY_SHOTS = {
  overtake: ['trackside', 'helicopter', 'chase'], battle: ['trackside', 'onboard', 'helicopter'],
  contact: ['helicopter', 'trackside'], offTrack: ['helicopter', 'trackside'], pitStop: ['pitlane'],
  fastestLap: ['onboard', 'chase'], finalLap: ['helicopter'], win: ['finish'], start: ['start'],
  rainStarts: ['onboard', 'chase'], rainStops: ['helicopter'],
  puncture: ['chase', 'trackside'], engineFailure: ['helicopter', 'trackside'], mistake: ['trackside', 'helicopter'], tow: ['chase', 'onboard'],
};
const LABELS = { trackside: '', helicopter: 'Helicopter', chase: '', onboard: 'Onboard', pitlane: 'Pit lane', start: 'Start', finish: 'Finish line' };

export function createDirector({ aspect }) {
  const camera = new THREE.PerspectiveCamera(40, aspect, 0.5, 2400);
  const sightLine = new THREE.Vector3(), target = new THREE.Vector3(), look = new THREE.Vector3();
  let race = null, circuit = null;
  let shot = null, story = null, rotation = 0;

  const random = (low, high) => low + Math.random() * (high - low);
  const choose = options => options[Math.floor(Math.random() * options.length)];

  function reset(intro, builtCircuit) {
    race = intro;
    circuit = builtCircuit;
    shot = null;
    story = null;
  }

  // Track geometry in sim coordinates: the point `distance` metres along the centre line, `lateral` to its left.
  function trackPoint(distance, lateral = 0) {
    const { points, normals, length } = race.track;
    const spacing = length / points.length;
    const wrapped = ((distance % length) + length) % length;
    const index = Math.floor(wrapped / spacing), next = (index + 1) % points.length, blend = wrapped / spacing - index;
    const [x0, y0] = points[index], [x1, y1] = points[next], [nx, ny] = normals[index];
    return { x: x0 + (x1 - x0) * blend + nx * lateral, y: y0 + (y1 - y0) * blend + ny * lateral, curvature: race.track.curvature[index] };
  }

  // Who to watch when nothing is happening: the closest fight on track, or the leader.
  function quietTarget(frame) {
    let best = { gap: Infinity, car: frame.order[0], other: undefined };
    for (let place = 1; place < frame.order.length; place++) {
      const ahead = frame.order[place - 1], behind = frame.order[place];
      if (frame.cars.finishTime[behind] !== null || frame.cars.pit[behind] >= 2) continue;
      const gap = frame.cars.progress[ahead] - frame.cars.progress[behind];
      if (gap < best.gap) best = { gap, car: behind, other: ahead };
    }
    return best.gap < 40 ? best : { car: frame.order[0] };
  }

  function cut(kind, car, other, frame, now, seconds) {
    shot = { kind, car, other, until: now + seconds * 1000 };
    if (kind === 'trackside') {
      // A camera beside the track some way ahead, on the outside of the bend if there is one.
      const ahead = (frame.cars.progress[car] % race.track.length) + random(70, 130);
      const bend = trackPoint(ahead).curvature;
      const side = Math.abs(bend) > 1 / 200 ? -Math.sign(bend) : choose([1, -1]);
      const spot = trackPoint(ahead, side * (race.track.width / 2 + random(12, 22)));
      shot.position = toWorld(spot.x, spot.y, random(3, 7));
    } else if (kind === 'helicopter') {
      shot.azimuth = Math.random() * Math.PI * 2;
      shot.height = random(40, 70);
    } else if (kind === 'start') {
      // From above the track ahead of the grid, looking back down it.
      const spot = trackPoint(45, 0);
      shot.position = toWorld(spot.x, spot.y, 9);
    } else if (kind === 'finish') {
      const spot = trackPoint(-6, race.track.width / 2 + 12);
      shot.position = toWorld(spot.x, spot.y, 3);
    }
  }

  function notice(moments, frame, now) {
    if (!race) return;
    const big = moments.filter(moment => STORY_SHOTS[moment.kind] && moment.priority >= 3).sort((a, b) => b.priority - a.priority)[0];
    if (!big) return;
    // Stay on a bigger story that is still playing out.
    if (story && story.until > now && story.priority > big.priority) return;
    story = { ...big, until: now + STORY_SECONDS * 1000 };
    cut(choose(STORY_SHOTS[big.kind]), big.car, big.other, frame, now, STORY_SECONDS);
  }

  function carPose(frame, car) {
    const heading = frame.cars.heading[car];
    return { point: toWorld(frame.cars.x[car], frame.cars.y[car]), forward: new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading)), speed: frame.cars.speed[car] };
  }

  // Aims the camera for this render frame; returns what the shot is looking at and the on-screen label.
  function update(frame, seconds, now) {
    if (!race) return null;
    if (!shot || now > shot.until || shotIsSpent(frame)) {
      const quiet = frame.time < START_SHOT_RACE_SECONDS ? { car: frame.order[0], kind: 'start' } : quietTarget(frame);
      const kind = quiet.kind ?? ROTATION[rotation++ % ROTATION.length];
      cut(kind, quiet.car, quiet.other, frame, now, random(...QUIET_SHOT_SECONDS));
    }
    const subject = carPose(frame, shot.car);
    target.copy(subject.point);
    if (shot.other !== undefined && shot.kind === 'helicopter') target.lerp(carPose(frame, shot.other).point, 0.5);

    switch (shot.kind) {
      case 'trackside':
      case 'start':
      case 'finish': {
        camera.position.copy(shot.position);
        look.copy(target).y += 0.6;
        // Zoom so the car stays about the same size on screen however far away it is: a long lens.
        const distance = camera.position.distanceTo(look);
        camera.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan((shot.kind === 'start' ? 30 : 9) / distance)), 6, 55);
        break;
      }
      case 'helicopter': {
        // Rides along with the cars at a fixed offset, slowly circling: easing towards them would leave it
        // hundreds of metres behind at high playback speeds.
        shot.azimuth += seconds * 0.12;
        camera.position.copy(target).add(new THREE.Vector3(Math.cos(shot.azimuth) * 80, shot.height, Math.sin(shot.azimuth) * 80));
        look.copy(target);
        camera.fov = 32;
        break;
      }
      case 'chase': {
        camera.position.copy(subject.point).addScaledVector(subject.forward, -9).y += 3.2;
        look.copy(subject.point).addScaledVector(subject.forward, 8).y += 0.8;
        camera.fov = 58;
        break;
      }
      case 'onboard': {
        // The T-cam above the driver's head, fixed to the car.
        camera.position.copy(subject.point).addScaledVector(subject.forward, -0.3).y += 1.45;
        look.copy(subject.point).addScaledVector(subject.forward, 25).y += 0.4;
        camera.fov = 72;
        break;
      }
      case 'pitlane': {
        const box = circuit.pits.boxes[shot.car];
        const towardsTrack = box.stop.clone().sub(box.home).setY(0).normalize();
        const along = new THREE.Vector3(Math.cos(box.heading), 0, -Math.sin(box.heading));
        camera.position.copy(box.stop).addScaledVector(towardsTrack, 7).addScaledVector(along, -9).y += 3;
        look.copy(target);
        camera.fov = 45;
        break;
      }
    }
    camera.updateProjectionMatrix();
    camera.lookAt(look);
    sightLine.copy(look).sub(camera.position);
    const label = LABELS[shot.kind];
    return { focus: target, distance: sightLine.length(), label: label === '' ? null : label, car: shot.car, kind: shot.kind };
  }

  // A trackside camera is done once the car has gone well past it, the start once the field has got away, and
  // the pit-lane one once the car has left. Playback can be sped up, so none of these go by the clock alone.
  function shotIsSpent(frame) {
    if (shot.kind === 'start') return frame.time > START_SHOT_RACE_SECONDS;
    if (shot.kind === 'trackside') {
      const car = toWorld(frame.cars.x[shot.car], frame.cars.y[shot.car]);
      const heading = frame.cars.heading[shot.car];
      const towardsCamera = shot.position.clone().sub(car).setY(0);
      const passed = towardsCamera.dot(new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading))) < 0;
      return passed && towardsCamera.length() > 70;
    }
    if (shot.kind === 'pitlane') return frame.cars.pit[shot.car] === 0;
    return false;
  }

  return {
    camera, reset, notice, update,
    resize(newAspect) { camera.aspect = newAspect; camera.updateProjectionMatrix(); },
  };
}
