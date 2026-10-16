// The grid before the start, as the broadcast shows it: a helicopter sweeps over the grid under a title, a low
// camera walks down it from pole to the back, then the pole sitter waits for the lights. Augmented-reality
// graphics stand over every car (its place, name and tyre) and paint its grid place on the tarmac in front of it,
// the way broadcasts draw on the track. Any time, Skip (or Esc, Space, Enter, a tap) goes straight to the lights.
import * as THREE from 'three';
import { COMPOUND_COLORS } from './cars.js';
import { toWorld } from './scene.js';

const $ = id => document.getElementById(id);
const SWEEP_SECONDS = 3.5;
const POLE_SECONDS = 3.2;
// Each car gets this long on the walk, quicker in a big field so the walk stays short.
const SECONDS_PER_CAR = 0.8;
const LONGEST_WALK = 11;
// The walk's camera stands ahead of each car, over the other column, looking down across it.
const WALK = { ahead: 9, across: 1.6, height: 4.2, look: 0.9 };
// Tags fade out far away and when the camera is right on top of them.
const TAG = { height: 2.5, width: 3.3, near: 28, far: 70, tooClose: 7, close: 12 };
const PAINT = { ahead: 4.4, size: 3.2, lift: 0.04 };

const smooth = value => value * value * (3 - 2 * value);

function canvasTexture(width, height, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d'));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

// The floating tag over a car: a glass card with its place, a stripe of its colour, its name and number, its tyre.
function tagTexture(car, place, compound) {
  return canvasTexture(512, 168, context => {
    context.fillStyle = 'rgba(8, 11, 18, 0.78)';
    context.beginPath();
    context.roundRect(4, 4, 504, 132, 10);
    context.fill();
    context.strokeStyle = 'rgba(120, 220, 255, 0.55)';
    context.lineWidth = 3;
    context.stroke();
    context.fillStyle = car.color;
    context.fillRect(4, 4, 14, 132);
    context.fillStyle = '#ffffff';
    context.font = '900 74px "Titillium Web", sans-serif';
    context.textBaseline = 'middle';
    context.fillText(String(place), 34, 72);
    context.font = '700 44px "Titillium Web", sans-serif';
    context.fillText(car.name.toUpperCase(), 140, 56);
    context.font = '400 30px "JetBrains Mono", monospace';
    context.fillStyle = 'rgba(200, 225, 240, 0.85)';
    context.fillText(`#${car.number} · ${compound}`, 140, 104);
    context.strokeStyle = COMPOUND_COLORS[compound] ?? COMPOUND_COLORS.medium;
    context.lineWidth = 9;
    context.beginPath();
    context.arc(462, 70, 26, 0, Math.PI * 2);
    context.stroke();
    // A pointer down to the car.
    context.fillStyle = 'rgba(120, 220, 255, 0.55)';
    context.beginPath();
    context.moveTo(240, 136);
    context.lineTo(272, 136);
    context.lineTo(256, 164);
    context.fill();
  });
}

// The place painted on the tarmac: an outlined number in the car's colour, glowing.
function paintTexture(car, place) {
  return canvasTexture(256, 256, context => {
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.font = '900 200px "Titillium Web", sans-serif';
    context.shadowColor = car.color;
    context.shadowBlur = 24;
    context.lineWidth = 10;
    context.strokeStyle = car.color;
    context.strokeText(String(place), 128, 138);
    context.shadowBlur = 0;
    context.fillStyle = 'rgba(255, 255, 255, 0.35)';
    context.fillText(String(place), 128, 138);
  });
}

export function createGridIntro({ aspect }) {
  const camera = new THREE.PerspectiveCamera(42, aspect, 0.3, 2400);
  const focus = new THREE.Vector3();
  const overlay = $('grid-intro');
  let plan = null;

  function lowerThird(car, place, compound, caption) {
    $('grid-intro-third').innerHTML = `<span class="place">${place}</span><span class="stripe" style="background:${car.color}"></span>`
      + `<span class="who"><small>${caption}</small>${car.name}<em>#${car.number} · ${compound} tyres</em></span>`;
    $('grid-intro-third').classList.remove('hidden');
  }

  function dispose() {
    if (!plan) return;
    plan.group.removeFromParent();
    plan.group.traverse(object => {
      if (!object.material) return;
      object.material.map?.dispose();
      object.material.dispose();
      object.geometry?.dispose();
    });
    plan = null;
    overlay.classList.add('hidden');
  }

  return {
    camera,
    focus,
    get playing() { return plan !== null; },
    // The helicopter sees everything sharp; close up, the lens focuses on the car.
    get sharpEverywhere() { return plan?.shot === 'sweep'; },
    resize(newAspect) { camera.aspect = newAspect; camera.updateProjectionMatrix(); },

    // Plans the intro for a race on the grid: `intro` (the race), `grid` the cars' grid frame, `title` what the
    // opening caption says under "Starting grid".
    begin(scene, intro, grid, title) {
      dispose();
      const count = intro.cars.length;
      const cars = intro.cars.map((car, id) => {
        const heading = grid.heading[id];
        const forward = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
        const point = toWorld(grid.x[id], grid.y[id]);
        // The middle of the track beside the car: the camera walks down between the two columns.
        let nearest = 0, closest = Infinity;
        intro.track.points.forEach(([x, y], index) => {
          const distance = (x - grid.x[id]) ** 2 + (y - grid.y[id]) ** 2;
          if (distance < closest) { closest = distance; nearest = index; }
        });
        const middle = toWorld(...intro.track.points[nearest]);
        return { car, place: id + 1, compound: grid.compound[id], point, forward, middle };
      });
      const group = new THREE.Group();
      for (const entry of cars) {
        const tag = new THREE.Sprite(new THREE.SpriteMaterial({ map: tagTexture(entry.car, entry.place, entry.compound), transparent: true, depthWrite: false, toneMapped: false }));
        tag.scale.set(TAG.width, TAG.width * 168 / 512, 1);
        tag.position.copy(entry.point).add(new THREE.Vector3(0, TAG.height, 0));
        tag.renderOrder = 10;
        const paint = new THREE.Mesh(new THREE.PlaneGeometry(PAINT.size, PAINT.size), new THREE.MeshBasicMaterial({
          map: paintTexture(entry.car, entry.place), transparent: true, depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2,
        }));
        // Lying on the tarmac ahead of the nose, reading the right way up from behind the grid, where the
        // helicopter sees it.
        const holder = new THREE.Group();
        holder.position.copy(entry.point).addScaledVector(entry.forward, PAINT.ahead).setY(PAINT.lift);
        holder.rotation.y = Math.atan2(entry.forward.x, entry.forward.z) + Math.PI;
        paint.rotation.x = -Math.PI / 2;
        holder.add(paint);
        group.add(tag, holder);
        entry.tag = tag;
        entry.paint = paint;
      }
      scene.add(group);
      const centre = cars.reduce((sum, entry) => sum.add(entry.point), new THREE.Vector3()).divideScalar(count);
      const walk = Math.min(LONGEST_WALK, SECONDS_PER_CAR * Math.max(count - 1, 1));
      plan = { cars, group, centre, time: 0, shot: 'sweep', walk, length: SWEEP_SECONDS + walk + POLE_SECONDS, sweepSide: Math.random() < 0.5 ? 1 : -1 };
      $('grid-intro-title').innerHTML = `<b>Starting grid</b><span>${title}</span>`;
      $('grid-intro-title').classList.remove('hidden');
      $('grid-intro-third').classList.add('hidden');
      overlay.classList.remove('hidden');
    },

    // Moves the camera on; returns true once the intro has played to the end.
    update(seconds) {
      if (!plan) return true;
      plan.time += seconds;
      const { cars, centre, time } = plan;
      const pole = cars[0];
      const left = new THREE.Vector3(-pole.forward.z, 0, pole.forward.x);
      plan.shot = time < SWEEP_SECONDS ? 'sweep' : time < SWEEP_SECONDS + plan.walk ? 'walk' : 'pole';
      if (time < SWEEP_SECONDS) {
        // The helicopter: high behind and to one side, swinging round and coming down over the grid.
        const progress = smooth(time / SWEEP_SECONDS);
        const angle = (0.9 - 0.7 * progress) * plan.sweepSide;
        const away = new THREE.Vector3().addScaledVector(pole.forward, -Math.cos(angle)).addScaledVector(left, Math.sin(angle));
        camera.position.copy(centre).addScaledVector(away, 70 - 35 * progress).add(new THREE.Vector3(0, 55 - 30 * progress, 0));
        focus.copy(centre).addScaledVector(pole.forward, 6 * progress);
      } else if (time < SWEEP_SECONDS + plan.walk) {
        // The walk: low, in front of each car in turn, pausing on each one.
        $('grid-intro-title').classList.add('hidden');
        const along = ((time - SWEEP_SECONDS) / plan.walk) * (cars.length - 1);
        const index = Math.min(Math.floor(along), cars.length - 1);
        const next = cars[Math.min(index + 1, cars.length - 1)], current = cars[index];
        const blend = smooth(along - index);
        const eye = entry => entry.middle.clone().lerp(entry.point, -WALK.across).addScaledVector(entry.forward, WALK.ahead).setY(WALK.height);
        camera.position.lerpVectors(eye(current), eye(next), blend);
        focus.lerpVectors(current.point, next.point, blend).add(new THREE.Vector3(0, WALK.look, 0));
        const shown = cars[Math.round(along)];
        lowerThird(shown.car, shown.place, shown.compound, shown.place === 1 ? 'Pole position' : 'Starting');
      } else {
        // The pole sitter, from low in front, pushing in slowly as the lights wait.
        const progress = smooth(Math.min(1, (time - SWEEP_SECONDS - plan.walk) / POLE_SECONDS));
        camera.position.copy(pole.point).addScaledVector(pole.forward, 16 - 6 * progress).addScaledVector(left, 3).add(new THREE.Vector3(0, 1.1, 0));
        focus.copy(pole.point).add(new THREE.Vector3(0, 0.6, 0));
        lowerThird(pole.car, 1, pole.compound, 'Pole position');
      }
      camera.lookAt(focus);
      // The tags fade with distance, so the sweep isn't a wall of cards and the walk shows the near ones.
      for (const entry of cars) {
        const distance = camera.position.distanceTo(entry.point);
        const near = THREE.MathUtils.clamp((distance - TAG.tooClose) / (TAG.close - TAG.tooClose), 0, 1);
        const shown = near * (1 - THREE.MathUtils.clamp((distance - TAG.near) / (TAG.far - TAG.near), 0, 1));
        entry.tag.material.opacity = shown * Math.min(1, time / 0.8);
        entry.paint.material.opacity = Math.min(1, time / 1.2) * 0.9;
      }
      return time >= plan.length;
    },

    end: dispose,
  };
}
