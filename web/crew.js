// Pit crews: eight people per garage who wait out front, run to the car when it stops in its box, do the jobs
// the stop calls for on the stop's own timeline, and jog back once the lollipop says go.
//
// Timeline of a stop, in race seconds from the moment the car stops (jobs come from the server):
//   0.0-0.5      crew run to their places, jacks go in and lift the car
//   0.5-…        side by side: tyres (old wheels off, new ones on), fuel hose, wing, engine cover
//   then         repairs (a new nose), then brake pads
//   standing     lollipop up, car drops, crew run home
import * as THREE from 'three';
import { createWheel } from './cars.js';

const RUN = 0.5;
// Where each crew member works, in the car's frame: x forward, z to the car's right.
const POSTS = [
  { role: 'wheel', wheel: 0, x: 1.55, z: -1.75 }, { role: 'wheel', wheel: 1, x: 1.55, z: 1.75 },
  { role: 'wheel', wheel: 2, x: -1.45, z: -1.85 }, { role: 'wheel', wheel: 3, x: -1.45, z: 1.85 },
  { role: 'jack', x: 3.6, z: 0 }, { role: 'jack', x: -3.1, z: 0 },
  { role: 'fuel', x: -0.4, z: 1.7 }, { role: 'lollipop', x: 5.2, z: 0.4 },
];

function figure(color) {
  const person = new THREE.Group();
  const suit = new THREE.MeshStandardMaterial({ color, roughness: 0.7 });
  const dark = new THREE.MeshStandardMaterial({ color: '#22252b', roughness: 0.8 });
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.62, 0.3), suit);
  torso.position.y = 1.15;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 10, 8), new THREE.MeshStandardMaterial({ color: '#f2f2f2', roughness: 0.3 }));
  head.position.y = 1.62;
  const legs = [-0.11, 0.11].map(side => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.8, 0.16), dark);
    leg.geometry.translate(0, -0.4, 0);
    leg.position.set(0, 0.84, side);
    return leg;
  });
  const arms = [-0.27, 0.27].map(side => {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.55, 0.12), suit);
    arm.geometry.translate(0, -0.27, 0);
    arm.position.set(0, 1.42, side);
    return arm;
  });
  for (const part of [torso, head, ...legs, ...arms]) { part.castShadow = true; person.add(part); }
  person.userData = { legs, arms, torso };
  return person;
}

function lollipop() {
  const stick = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6, 6), new THREE.MeshStandardMaterial({ color: '#c8ccd4' }));
  pole.position.y = 0.8;
  const sign = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.04, 20), new THREE.MeshStandardMaterial({ color: '#e10600', emissive: '#550000' }));
  sign.rotation.x = Math.PI / 2;
  sign.position.y = 1.7;
  stick.add(pole, sign);
  stick.userData = { sign };
  return stick;
}

export function createCrews(scene, boxes, cars) {
  const group = new THREE.Group();
  scene.add(group);
  const crews = boxes.map(box => {
    const members = POSTS.map((post, index) => {
      const person = figure(cars[box.car].color);
      // Waiting in a loose line in front of the garage.
      const spread = (index - (POSTS.length - 1) / 2) * 1.1;
      const forward = new THREE.Vector3(Math.cos(box.heading), 0, -Math.sin(box.heading));
      person.position.copy(box.home).addScaledVector(forward, spread);
      person.userData.home = person.position.clone();
      person.userData.phase = Math.random() * Math.PI * 2;
      group.add(person);
      return { person, post };
    });
    const sign = lollipop();
    members.find(member => member.post.role === 'lollipop').person.add(sign);
    sign.position.set(0.3, 0.6, 0);
    const hose = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 1, 8), new THREE.MeshStandardMaterial({ color: '#111', roughness: 0.6 }));
    hose.visible = false;
    group.add(hose);
    const rig = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.6, 0.8), new THREE.MeshStandardMaterial({ color: '#30343c', roughness: 0.6 }));
    rig.position.copy(box.home).add(new THREE.Vector3(0, 0.8, 0));
    rig.castShadow = true;
    group.add(rig);
    return { box, members, sign, hose, rig, stop: null, loose: [] };
  });

  // Temporary parts in flight: old wheels rolling away, new ones arriving, a nose or a wing being carried.
  function looseWheel(crew, compound, from, to, start, duration, fade) {
    const wheel = createWheel(compound);
    group.add(wheel);
    crew.loose.push({ mesh: wheel, from, to, start, duration, fade });
  }

  return {
    // A car has just stopped in its box: `stop` is the server's pitStopped event, `time` the race time now.
    startStop(stop, time, car) {
      const crew = crews[stop.car];
      if (!crew) return;
      const parallel = Math.max(stop.jobs.tyres, stop.jobs.fuel, stop.jobs.wing, stop.jobs.engine);
      crew.stop = { ...stop, start: time, parallelEnd: RUN + parallel, repairEnd: RUN + parallel + stop.jobs.repair, wheelsOffAt: RUN + 0.25 };
      crew.stop.brakesEnd = crew.stop.repairEnd + stop.jobs.brakes;
      crew.stop.oldCompound = car.compound;
    },

    update(time, frame, carMeshes) {
      for (const crew of crews) {
        const { stop, members, sign, hose } = crew;
        const carMesh = carMeshes[crew.box.car];
        const elapsed = stop ? time - stop.start : Infinity;
        const working = stop && elapsed < stop.standing + 0.6;
        if (stop && elapsed > stop.standing + 3) crew.stop = null;

        // Car pose at the box, for placing people and parts around it.
        const origin = carMesh ? carMesh.position : crew.box.stop;
        const heading = carMesh ? carMesh.rotation.y : crew.box.heading;
        const local = (x, z, y = 0) => new THREE.Vector3(x, y, z).applyAxisAngle(new THREE.Vector3(0, 1, 0), heading).add(origin);

        members.forEach(({ person, post }) => {
          const data = person.userData;
          const target = working ? local(post.x, post.z) : data.home;
          const gap = target.clone().sub(person.position).setY(0);
          const moving = gap.length() > 0.05;
          // People cover the distance fast at first and ease in, which reads as a sprint and a stop.
          person.position.addScaledVector(gap, moving ? Math.min(1, 0.22) : 0);
          if (moving && gap.length() > 0.3) person.rotation.y = Math.atan2(-gap.z, gap.x);
          else if (working) person.rotation.y = Math.atan2(-(origin.z - person.position.z), origin.x - person.position.x);
          const stride = moving ? Math.sin(time * 18 + data.phase) * 0.7 : 0;
          data.legs[0].rotation.z = stride;
          data.legs[1].rotation.z = -stride;
          const busy = working && elapsed > RUN && elapsed < stop.standing;
          const effort = busy && post.role !== 'lollipop' ? Math.sin(time * 30 + data.phase) * 0.25 : 0;
          data.arms.forEach((arm, index) => { arm.rotation.z = (busy ? -1.1 : 0.15 * Math.sin(time * 2 + data.phase)) + (index ? effort : -effort); });
          // Wheel and jack crew crouch while they work.
          const crouch = busy && (post.role === 'wheel' || post.role === 'jack') ? 0.35 : 0;
          person.position.y += (-crouch - person.position.y) * 0.3;
          data.torso.rotation.z = crouch ? -0.5 : 0;
        });

        // Lollipop: red while they work, up and green when the car may go.
        const go = working && elapsed >= stop.standing - 0.2;
        sign.userData.sign.material.color.set(go ? '#19c94a' : '#e10600');
        sign.userData.sign.material.emissive.set(go ? '#0a5a22' : '#550000');
        sign.position.y = go ? 1.1 : 0.6;

        // Fuel hose from the rig to the car while fuelling.
        const fuelling = working && elapsed > RUN && elapsed < RUN + stop.jobs.fuel;
        hose.visible = !!fuelling;
        if (fuelling) {
          const from = crew.rig.position.clone().setY(1.1), to = local(-0.4, 0.6, 0.7);
          hose.position.copy(from).add(to).multiplyScalar(0.5);
          hose.scale.set(1, from.distanceTo(to), 1);
          hose.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.clone().sub(from).normalize());
        }

        if (!carMesh) continue;
        const parts = carMesh.userData;
        // Jacks lift the car while the crew works.
        parts.lift = working && elapsed > RUN * 0.6 && elapsed < stop.standing ? 0.12 : 0;

        // Tyres: the car's own wheels hide while old ones roll off and new ones go on.
        const tyreStart = RUN + 0.2, tyreEnd = RUN + stop?.jobs.tyres;
        const brakeStart = stop ? stop.repairEnd : 0;
        const wheelsOff = working && ((elapsed > tyreStart && elapsed < tyreEnd) || (stop.jobs.brakes && elapsed > brakeStart && elapsed < brakeStart + stop.jobs.brakes));
        parts.wheels.forEach(wheel => { wheel.visible = !wheelsOff; });
        if (working && !stop.wheelsAway && elapsed > tyreStart) {
          stop.wheelsAway = true;
          [[1.55, -0.85], [1.55, 0.85], [-1.45, -0.85], [-1.45, 0.85]].forEach(([x, z]) => {
            looseWheel(crew, stop.oldCompound, local(x, z, 0.36), local(x, z * 3.2, 0.36), time, 0.5, true);
          });
        }
        if (working && !stop.wheelsOn && elapsed > tyreEnd - 0.5) {
          stop.wheelsOn = true;
          [[1.55, -0.85], [1.55, 0.85], [-1.45, -0.85], [-1.45, 0.85]].forEach(([x, z]) => {
            looseWheel(crew, stop.compound, local(x, z * 3, 0.36), local(x, z, 0.36), time, 0.45, false);
          });
        }
        // Wing: lifted off and put back at its new angle. Engine cover: opened. Nose: swapped for a new one.
        const wingJob = working && stop.jobs.wing && elapsed > RUN && elapsed < RUN + stop.jobs.wing;
        parts.rearWingLift = wingJob ? Math.sin(Math.min(1, (elapsed - RUN) / stop.jobs.wing) * Math.PI) * 0.9 : 0;
        const engineJob = working && stop.jobs.engine && elapsed > RUN && elapsed < RUN + stop.jobs.engine;
        parts.coverOpen = engineJob ? Math.sin(Math.min(1, (elapsed - RUN) / stop.jobs.engine) * Math.PI) : 0;
        const noseJob = working && stop.jobs.repair && elapsed > stop.parallelEnd && elapsed < stop.repairEnd;
        parts.noseOff = noseJob ? Math.sin(Math.min(1, (elapsed - stop.parallelEnd) / stop.jobs.repair) * Math.PI) : 0;
      }

      // Loose wheels glide between their two spots, fading out if they're the old ones.
      for (const crew of crews) {
        crew.loose = crew.loose.filter(item => {
          const progress = (time - item.start) / item.duration;
          if (progress >= 1 || progress < -0.1) { group.remove(item.mesh); return false; }
          item.mesh.position.lerpVectors(item.from, item.to, Math.max(0, progress));
          item.mesh.rotation.y = crew.box.heading;
          item.mesh.rotation.x = item.fade ? progress * 1.2 : 0;
          return true;
        });
      }
    },

    dispose() {
      scene.remove(group);
      group.traverse(object => { object.geometry?.dispose(); object.material?.dispose?.(); });
    },
  };
}
