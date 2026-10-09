// Everything about the circuit that doesn't move (and the crowd, which does), built from the track the server
// sends: tarmac, lines, curbs, gravel traps, the start gantry and grid, grandstands, tyre walls and trees.
import * as THREE from 'three';
import { toWorld } from './scene.js';
import { LOCATIONS, canvasTexture, trackside } from './scenery.js';

const CURB_CURVATURE = 1 / 85;      // corners tighter than this get curbs
const GRAVEL_CURVATURE = 1 / 50;    // and tighter than this a gravel trap on the outside
const TYRE_WALL_CURVATURE = 1 / 38;
const CUTAWAY_COSINE = Math.cos(THREE.MathUtils.degToRad(70));   // backdrop within 70° of the camera's side is hidden
const MOUTH_LENGTH = 20;            // metres over which the pit lane widens out of the track edge
// Heights of the flat layers, far enough apart that the depth buffer never mixes them up at a distance.
const LAYER = { verge: 0.03, gravel: 0.06, tarmac: 0.09, surface: 0.1, lines: 0.12, curbs: 0.14, paint: 0.13 };

// `locationName` picks the scenery (see scenery.js).
export function buildCircuit(scene, track, grid, seed, pitLane, cars, locationName) {
  const location = LOCATIONS[locationName];
  if (!location) throw new Error(`Unknown location ${locationName}`);
  const group = new THREE.Group();
  const points = track.points, normals = track.normals, curvature = track.curvature, half = track.width / 2;
  const random = seeded(seed);

  const centre = new THREE.Vector3();
  points.forEach(([x, y]) => centre.add(toWorld(x, y)));
  centre.divideScalar(points.length);
  const radius = Math.max(...points.map(([x, y]) => toWorld(x, y).distanceTo(centre)));

  group.add(ground(centre, location.ground));
  group.add(ribbon(points, normals, -half - 7, half + 7, LAYER.verge, material(location.verge, 0.95)));
  const clearOfLane = pitLaneClearance(track, pitLane);
  if (!location.walled) group.add(gravelTraps(points, normals, curvature, half, clearOfLane));
  const tarmac = ribbon(points, normals, -half, half, LAYER.tarmac, new THREE.MeshStandardMaterial({ map: asphaltTexture(), roughness: 0.92 }), 18);
  tarmac.receiveShadow = true;
  group.add(tarmac);
  const surface = trackSurface(points, normals, half, track.length);
  group.add(surface.mesh);
  for (const side of [1, -1]) {
    group.add(ribbon(points, normals, side > 0 ? half - 0.55 : -half + 0.25, side > 0 ? half - 0.25 : -half + 0.55, LAYER.lines, material('#f4f4f0', 0.6)));
  }
  group.add(curbs(points, normals, curvature, half, clearOfLane));
  group.add(startLine(points, normals, track.width));
  group.add(gridBoxes(grid));
  const gantry = startGantry(points, normals, half);
  group.add(gantry.group);
  const pits = pitArea(track, pitLane, cars);
  group.add(pits.group);
  const stands = grandstands(points, normals, half, pitLane.side, Math.abs(pitLane.offset) + 22, random);
  group.add(stands.group);
  if (!location.walled) group.add(tyreWalls(points, normals, curvature, half, clearOfLane));
  const coarse = points.filter((_, index) => index % 4 === 0).map(([x, y]) => toWorld(x, y));
  // Scenery keeps clear of the track, and well clear of the pits and grandstands behind them.
  const reserved = Array.from({ length: 12 }, (_, step) => {
    const [x, y] = pointAt(track, pitLane.entry + (pitLane.length * step) / 11, pitLane.offset + pitLane.side * 25);
    return toWorld(x, y);
  });
  const awayFromTrack = spot => Math.min(
    Math.sqrt(Math.min(...coarse.map(point => point.distanceToSquared(spot)))),
    Math.sqrt(Math.min(...reserved.map(point => point.distanceToSquared(spot)))) - 25,
  );
  // Spots along both sides of the track every `spacing` samples, `lateral` metres out, facing across it.
  const alongTrack = (spacing, lateral) => points.flatMap(([x, y], index) => {
    if (index % spacing || !clearOfLane(index, 1) || !clearOfLane(index, -1)) return [];
    const [nx, ny] = normals[index];
    return [1, -1].map(side => ({ spot: toWorld(x + nx * side * lateral, y + ny * side * lateral), angle: Math.atan2(-ny * side, -nx * side) }));
  });
  const scenery = location.scenery({ points, half, centre, radius, random, awayFromTrack, alongTrack });
  group.add(scenery);
  const backdrop = [];
  scenery.traverse(object => { if (object.userData.backdrop) backdrop.push(object); });
  const details = trackside({ points, normals, curvature, half, random, clearOfLane, location });
  group.add(details.group);
  scene.add(group);

  return {
    group, centre, radius, pits, tarmac: tarmac.material, setSurface: surface.show,
    update(time, wind = 0) {
      stands.update(time);
      gantry.update(time);
      details.update(time, wind);
    },
    // Hides the backdrop between the camera and what it's looking at, like the near wall of a diorama: hundreds
    // of metres tall, it otherwise hides the circuit whenever the camera looks across it.
    cutAway(cameraPosition, focus) {
      const towardCamera = new THREE.Vector2(cameraPosition.x - focus.x, cameraPosition.z - focus.z).normalize();
      const towardPiece = new THREE.Vector2();
      for (const piece of backdrop) {
        towardPiece.set(piece.position.x - focus.x, piece.position.z - focus.z).normalize();
        piece.visible = towardPiece.dot(towardCamera) < CUTAWAY_COSINE;
      }
    },
    dispose() {
      scene.remove(group);
      group.traverse(object => {
        object.geometry?.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach(item => { item?.map?.dispose(); item?.dispose?.(); });
      });
    },
  };
}

// The turn about the vertical axis that points a local axis ('x' or 'z') along the sim direction (dx, dy).
function facing(dx, dy, axis) {
  return axis === 'x' ? Math.atan2(dy, dx) : Math.atan2(dx, -dy);
}

function material(color, roughness = 0.8, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness, ...extra });
}

// A strip that follows the loop between two lateral offsets (left of the centre line is positive).
// `colors`, if given, is a function of the sample index returning a THREE.Color for that stretch.
function ribbon(points, normals, from, to, height, surface, uvLength = 0, colors = null, only = null) {
  const positions = [], uvs = [], vertexColors = [], indices = [];
  const count = points.length;
  let along = 0;
  for (let index = 0; index <= count; index++) {
    const [x, y] = points[index % count], [nx, ny] = normals[index % count];
    if (index > 0) along += Math.hypot(x - points[index - 1][0], y - points[index - 1][1]);
    for (const offset of [from, to]) {
      const point = toWorld(x + nx * offset, y + ny * offset, height);
      positions.push(point.x, point.y, point.z);
      uvs.push(offset === from ? 0 : 1, uvLength ? along / uvLength : 0);
      if (colors) { const color = colors(index % count); vertexColors.push(color.r, color.g, color.b); }
    }
  }
  for (let index = 0; index < count; index++) {
    if (only && !only(index)) continue;
    const a = index * 2, b = a + 1, c = a + 2, d = a + 3;
    indices.push(a, c, b, b, c, d);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  if (colors) geometry.setAttribute('color', new THREE.Float32BufferAttribute(vertexColors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, surface);
  mesh.receiveShadow = true;
  return mesh;
}

// What the race does to the tarmac (see slipstream/surface.py), drawn over it as a texture with a texel per cell:
// rubber darkening the racing line, marbles gathering off it, and in the wet the drier line cars have made.
function trackSurface(points, normals, half, length) {
  let texture = null;
  const material = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1 });
  const mesh = ribbon(points, normals, -half, half, LAYER.surface, material, length);
  mesh.receiveShadow = false;
  mesh.visible = false;   // nothing to show until the race sends its first map
  const decode = text => Uint8Array.from(atob(text), character => character.charCodeAt(0));
  return {
    mesh,
    show(map, wetness) {
      const rubber = decode(map.rubber), marbles = decode(map.marbles), dryness = decode(map.dryness);
      const cells = map.rows * map.lanes, pixels = new Uint8Array(cells * 4);
      const layers = [
        [rubber, [14, 14, 16], 0.4],
        [marbles, [74, 64, 52], 0.6],
        [dryness, [128, 128, 134], 0.55 * Math.min(1, wetness * 2)],
      ];
      for (let cell = 0; cell < cells; cell++) {
        // Each layer goes over the ones before it, like paint.
        let [red, green, blue, alpha] = [0, 0, 0, 0];
        for (const [values, color, strength] of layers) {
          const amount = (values[cell] / 255) * strength;
          if (!amount) continue;
          const kept = alpha * (1 - amount);
          alpha = amount + kept;
          red = (color[0] * amount + red * kept) / alpha;
          green = (color[1] * amount + green * kept) / alpha;
          blue = (color[2] * amount + blue * kept) / alpha;
        }
        pixels.set([red, green, blue, alpha * 255], cell * 4);
      }
      if (!texture || texture.image.width !== map.lanes || texture.image.height !== map.rows) {
        texture?.dispose();
        texture = new THREE.DataTexture(pixels, map.lanes, map.rows, THREE.RGBAFormat);
        texture.magFilter = texture.minFilter = THREE.LinearFilter;
        material.map = texture;
        material.needsUpdate = true;
      } else {
        texture.image.data.set(pixels);
      }
      texture.needsUpdate = true;
      mesh.visible = true;
    },
  };
}

function ground(centre, [base, stripes, [dark, light]]) {
  const texture = canvasTexture(256, (context, size) => {
    context.fillStyle = base;
    context.fillRect(0, 0, size, size);
    // Stripes (mowing, or wind across sand), then speckle.
    for (let stripe = 0; stripe < 8; stripe += 2) {
      context.fillStyle = stripes;
      context.fillRect(stripe * size / 8, 0, size / 8, size);
    }
    for (let dot = 0; dot < 2600; dot++) {
      context.fillStyle = `rgba(${Math.random() < 0.5 ? dark : light}, ${Math.random() * 0.18})`;
      context.fillRect(Math.random() * size, Math.random() * size, 2, 2);
    }
  });
  texture.repeat.set(90, 90);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(3600, 3600), new THREE.MeshStandardMaterial({ map: texture, roughness: 1 }));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(centre.x, 0, centre.z);
  mesh.receiveShadow = true;
  return mesh;
}

function asphaltTexture() {
  return canvasTexture(256, (context, size) => {
    context.fillStyle = '#3d4046';
    context.fillRect(0, 0, size, size);
    for (let dot = 0; dot < 9000; dot++) {
      const shade = 40 + Math.random() * 50;
      context.fillStyle = `rgba(${shade}, ${shade}, ${shade + 4}, 0.5)`;
      context.fillRect(Math.random() * size, Math.random() * size, 1.5, 1.5);
    }
    // A darker racing line down the middle, where the rubber goes down.
    const line = context.createLinearGradient(0, 0, size, 0);
    line.addColorStop(0.3, 'rgba(0, 0, 0, 0)');
    line.addColorStop(0.5, 'rgba(0, 0, 0, 0.12)');
    line.addColorStop(0.7, 'rgba(0, 0, 0, 0)');
    context.fillStyle = line;
    context.fillRect(0, 0, size, size);
  });
}

// Which samples are within `before` samples behind or `after` ahead of a corner tighter than `limit` turning
// the way `sign` says (0 for either way). A solid stretch, so features start and end cleanly.
function cornerStretch(curvature, limit, sign = 0, before = 6, after = 12) {
  const count = curvature.length;
  const tight = curvature.map(value => (sign === 0 ? Math.abs(value) : sign * value) > limit);
  return tight.map((_, index) => {
    for (let shift = -after; shift <= before; shift++) if (tight[(index - shift + count) % count]) return true;
    return false;
  });
}

function curbs(points, normals, curvature, half, clearOfLane) {
  const red = new THREE.Color('#d7263d'), white = new THREE.Color('#f5f5f5');
  const cornering = cornerStretch(curvature, CURB_CURVATURE, 0, 3, 3);
  const colors = index => (Math.floor(index / 1.5) % 2 ? red : white);
  const surface = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55 });
  const group = new THREE.Group();
  group.add(ribbon(points, normals, half, half + 1.4, LAYER.curbs, surface, 0, colors, index => cornering[index] && clearOfLane(index, 1)));
  group.add(ribbon(points, normals, -half - 1.4, -half, LAYER.curbs, surface, 0, colors, index => cornering[index] && clearOfLane(index, -1)));
  return group;
}

function gravelTraps(points, normals, curvature, half, clearOfLane) {
  // Gravel goes on the outside of tight corners, running on past the exit where a car that runs wide ends up:
  // the right of a left-hander and the left of a right-hander.
  const surface = material('#dcc794', 1);
  const group = new THREE.Group();
  const leftHanders = cornerStretch(curvature, GRAVEL_CURVATURE, 1, 4, 14);
  const rightHanders = cornerStretch(curvature, GRAVEL_CURVATURE, -1, 4, 14);
  group.add(ribbon(points, normals, -half - 18, -half - 1.4, LAYER.gravel, surface, 0, null, index => leftHanders[index] && clearOfLane(index, -1)));
  group.add(ribbon(points, normals, half + 1.4, half + 18, LAYER.gravel, surface, 0, null, index => rightHanders[index] && clearOfLane(index, 1)));
  return group;
}

function startLine(points, normals, width) {
  const texture = canvasTexture(128, (context, size) => {
    const cell = size / 8;
    for (let row = 0; row < 8; row++) for (let column = 0; column < 8; column++) {
      context.fillStyle = (row + column) % 2 ? '#111' : '#f8f8f8';
      context.fillRect(column * cell, row * cell, cell, cell);
    }
  });
  texture.repeat.set(width / 2.5, 0.5);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, 2.5), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.6 }));
  const [x, y] = points[0], [nx, ny] = normals[0];
  mesh.position.copy(toWorld(x, y, LAYER.paint));
  // Lay it flat, then turn its width to run across the track (along the normal).
  mesh.rotation.set(-Math.PI / 2, facing(nx, ny, 'x'), 0, 'YXZ');
  return mesh;
}

function gridBoxes(grid) {
  const group = new THREE.Group();
  const paint = material('#f4f4f0', 0.6);
  for (const { x, y, heading } of grid) {
    const box = new THREE.Group();
    // An open box drawn in three strips: across the front, and down each side.
    const front = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.02, 2.4), paint);
    front.position.x = 2.6;
    const left = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.02, 0.25), paint);
    left.position.set(1.5, 0, 1.2);
    const right = left.clone();
    right.position.z = -1.2;
    box.add(front, left, right);
    box.position.copy(toWorld(x, y, LAYER.paint));
    box.rotation.y = heading;
    group.add(box);
  }
  return group;
}

function startGantry(points, normals, half) {
  const group = new THREE.Group();
  const steel = material('#2b2f36', 0.5, { metalness: 0.6 });
  const span = half * 2 + 6;
  const beam = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.4, span), steel);
  beam.position.y = 7.5;
  beam.castShadow = true;
  group.add(beam);
  for (const side of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.8, 8.2, 0.8), steel);
    post.position.set(0, 4.1, side * span / 2);
    post.castShadow = true;
    group.add(post);
  }
  // Five lights that count down at the start and then turn green.
  const lights = [];
  for (let index = 0; index < 5; index++) {
    const light = new THREE.Mesh(new THREE.SphereGeometry(0.42, 12, 8), new THREE.MeshStandardMaterial({ color: '#330000', emissive: '#000000', emissiveIntensity: 2.5 }));
    light.position.set(0.65, 7.5, (index - 2) * 1.6);
    lights.push(light);
    group.add(light);
  }
  const banner = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1.1, span - 2), material('#e10600', 0.6));
  banner.position.set(-0.7, 7.5, 0);
  group.add(banner);
  const [x, y] = points[0], [nx, ny] = normals[0];
  group.position.copy(toWorld(x, y));
  // The beam spans the track along local z; the lights face back down the grid.
  group.rotation.y = facing(nx, ny, 'z');
  return {
    group,
    update(time) {
      lights.forEach((light, index) => {
        const lit = time < 0.6 + index * 0.25;
        const green = time >= 1.9 && time < 5;
        light.material.color.set(green ? '#0b3' : lit && time > 0.05 ? '#f22' : '#331111');
        light.material.emissive.set(green ? '#0f4' : lit && time > 0.05 ? '#f00' : '#000');
      });
    },
  };
}

function grandstands(points, normals, half, outside, distance, random) {
  // Along the start straight, behind the garages.
  const group = new THREE.Group();
  const concrete = material('#c9cbd1', 0.9), roof = material('#e10600', 0.5), seat = material('#2f4c8a', 0.7);
  const fans = [];
  const palette = ['#ffd23f', '#e63946', '#f1faee', '#457b9d', '#2a9d8f', '#f4a261', '#9b5de5', '#ffffff'].map(color => new THREE.Color(color));
  for (const offset of [-34, 0, 34]) {
    const index = (offset / 2 + points.length) % points.length | 0;
    const [x, y] = points[index], [nx, ny] = normals[index];
    const stand = new THREE.Group();
    for (let tier = 0; tier < 6; tier++) {
      const step = new THREE.Mesh(new THREE.BoxGeometry(30, 0.9, 2.2), tier % 2 ? concrete : seat);
      step.position.set(0, 0.45 + tier * 0.9, tier * 2.2);
      step.castShadow = step.receiveShadow = true;
      stand.add(step);
      for (let spot = 0; spot < 26; spot++) {
        if (random() < 0.25) continue;
        fans.push({ x: -14 + spot * 1.12 + random() * 0.3, y: 1.35 + tier * 0.9, z: tier * 2.2, phase: random() * Math.PI * 2, stand, color: palette[(random() * palette.length) | 0] });
      }
    }
    const back = new THREE.Mesh(new THREE.BoxGeometry(30, 7, 0.5), concrete);
    back.position.set(0, 3.5, 13.5);
    const cover = new THREE.Mesh(new THREE.BoxGeometry(31, 0.3, 15), roof);
    cover.position.set(0, 8.2, 6.3);
    cover.rotation.x = -0.12;
    for (const part of [back, cover]) { part.castShadow = part.receiveShadow = true; stand.add(part); }
    stand.position.copy(toWorld(x + nx * outside * distance, y + ny * outside * distance));
    // The stand's +z runs away from the track, so its tiers rise away from the cars.
    stand.rotation.y = facing(nx * outside, ny * outside, 'z');
    group.add(stand);
  }
  group.updateMatrixWorld(true);
  const crowd = new THREE.InstancedMesh(new THREE.BoxGeometry(0.55, 0.9, 0.45), new THREE.MeshStandardMaterial({ roughness: 0.8 }), fans.length);
  crowd.castShadow = true;
  const matrix = new THREE.Matrix4(), local = new THREE.Vector3();
  fans.forEach((fan, index) => crowd.setColorAt(index, fan.color));
  group.add(crowd);
  return {
    group,
    // The crowd never sits still.
    update(time) {
      fans.forEach((fan, index) => {
        local.set(fan.x, fan.y + Math.max(0, Math.sin(time * 4 + fan.phase)) * 0.18, fan.z);
        fan.stand.localToWorld(local);
        matrix.makeRotationY(fan.stand.rotation.y).setPosition(local);
        crowd.setMatrixAt(index, matrix);
      });
      crowd.instanceMatrix.needsUpdate = true;
    },
  };
}

function tyreWalls(points, normals, curvature, half, clearOfLane) {
  const spots = [];
  for (let index = 0; index < points.length; index += 2) {
    const bend = curvature[index];
    if (Math.abs(bend) < TYRE_WALL_CURVATURE) continue;
    const side = bend > 0 ? -1 : 1;
    if (!clearOfLane(index, side)) continue;
    const [x, y] = points[index], [nx, ny] = normals[index];
    const distance = half + 19;
    spots.push(toWorld(x + nx * side * distance, y + ny * side * distance));
  }
  const geometry = new THREE.TorusGeometry(0.42, 0.2, 6, 12);
  geometry.rotateX(Math.PI / 2);
  const walls = new THREE.InstancedMesh(geometry, material('#ffffff', 0.9), spots.length * 3);
  const matrix = new THREE.Matrix4();
  const colors = ['#1b1b1b', '#1b1b1b', '#e10600', '#f2f2f2'].map(color => new THREE.Color(color));
  spots.forEach((spot, index) => {
    for (let layer = 0; layer < 3; layer++) {
      matrix.makeTranslation(spot.x, 0.22 + layer * 0.4, spot.z);
      walls.setMatrixAt(index * 3 + layer, matrix);
      walls.setColorAt(index * 3 + layer, colors[(index + layer) % colors.length]);
    }
  });
  walls.castShadow = walls.receiveShadow = true;
  return walls;
}

function seeded(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

// ---- Pit lane --------------------------------------------------------------------------------------------

// A point `distance` metres along the centre line (wrapping), `lateral` metres to its left, in sim coordinates.
export function pointAt(track, distance, lateral = 0) {
  const count = track.points.length, spacing = track.length / count;
  const wrapped = ((distance % track.length) + track.length) % track.length;
  const index = Math.floor(wrapped / spacing), next = (index + 1) % count, blend = wrapped / spacing - index;
  const [x0, y0] = track.points[index], [x1, y1] = track.points[next];
  const [nx, ny] = track.normals[index];
  return [x0 + (x1 - x0) * blend + nx * lateral, y0 + (y1 - y0) * blend + ny * lateral];
}

// The lane's offset from the centre line `along` metres in, easing off the track edge and back, as the sim does.
export function laneLateral(lane, along, half) {
  const ease = value => value * value * (3 - 2 * value);
  const into = Math.min(1, Math.max(0, along / lane.ramp)), out = Math.min(1, Math.max(0, (lane.length - along) / lane.ramp));
  const edge = lane.side * half;
  return edge + (lane.offset - edge) * ease(Math.min(into, out));
}

// Whether the track sample at `index` is clear of the pit lane on `side` (1 left, -1 right). The lane runs where
// kerbs, gravel and tyre walls would otherwise go when the start straight sits next to a corner, and a lane
// through a gravel trap looks broken. A few samples of margin cover where the lane's ends fan out.
function pitLaneClearance(track, lane) {
  const count = track.points.length, spacing = track.length / count, margin = 4;
  const first = Math.floor(lane.entry / spacing) - margin, span = Math.ceil(lane.length / spacing) + 2 * margin;
  return (index, side) => side !== lane.side || ((index - first) % count + count) % count > span;
}

function pitArea(track, lane, cars) {
  const group = new THREE.Group();
  const half = track.width / 2;
  const at = (along, lateral) => { const [x, y] = pointAt(track, lane.entry + along, lateral); return toWorld(x, y); };
  const headingAt = along => {
    const [x0, y0] = pointAt(track, lane.entry + along), [x1, y1] = pointAt(track, lane.entry + along + 1);
    return Math.atan2(y1 - y0, x1 - x0);
  };

  // Lane surface and its edge lines, built as strips that follow the lane's own offset. `from` and `to` count
  // outwards from the lane's centre. Nothing is drawn inside the track edge, so where the lane leaves and rejoins
  // it peels off the racing surface instead of painting over it.
  const strip = (from, to, height, surface) => {
    const positions = [], indices = [];
    const steps = Math.ceil(lane.length / 2);
    for (let step = 0; step <= steps; step++) {
      const along = Math.min(lane.length, step * 2), outwards = lane.side * laneLateral(lane, along, half);
      // The lane opens out of the track edge from nothing, rather than starting as a blunt full-width end.
      const opening = Math.min(1, along / MOUTH_LENGTH, (lane.length - along) / MOUTH_LENGTH);
      for (const offset of [from, to]) {
        const point = at(along, lane.side * Math.max(half, outwards + offset * opening));
        positions.push(point.x, height, point.z);
      }
      if (step < steps) indices.push(step * 2, step * 2 + 2, step * 2 + 1, step * 2 + 1, step * 2 + 2, step * 2 + 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, surface);
    mesh.material.side = THREE.DoubleSide;
    mesh.receiveShadow = true;
    return mesh;
  };
  group.add(strip(-3.6, 3.6, 0.085, new THREE.MeshStandardMaterial({ color: '#4a4d54', roughness: 0.9 })));
  for (const edge of [-3.6, 3.3]) group.add(strip(edge, edge + 0.3, 0.12, new THREE.MeshStandardMaterial({ color: '#f4f4f0', roughness: 0.6 })));
  // A dashed line between the fast lane (nearest the track) and the working lane along the garages.
  const dash = new THREE.MeshStandardMaterial({ color: '#f4f4f0', roughness: 0.6 });
  for (let along = lane.ramp; along < lane.length - lane.ramp; along += 4) {
    const mark = new THREE.Mesh(new THREE.PlaneGeometry(2, 0.14), dash);
    mark.rotation.set(-Math.PI / 2, headingAt(along + 1), 0, 'YXZ');
    mark.position.copy(at(along + 1, lane.offset)).setY(0.121);
    group.add(mark);
  }

  // Pit wall with a catch fence, between the track and the lane, along the straight part of the lane.
  const wallFrom = lane.ramp, wallTo = lane.length - lane.ramp;
  const concrete = new THREE.MeshStandardMaterial({ color: '#d9dbe0', roughness: 0.9 });
  const sponsor = new THREE.MeshStandardMaterial({ color: '#e10600', roughness: 0.6 });
  const post = new THREE.MeshStandardMaterial({ color: '#6c7280', roughness: 0.5, metalness: 0.6 });
  for (let along = wallFrom; along < wallTo; along += 6) {
    const length = Math.min(6, wallTo - along), middle = along + length / 2;
    const segment = new THREE.Group();
    const wall = new THREE.Mesh(new THREE.BoxGeometry(length, 1.1, 0.5), concrete);
    wall.position.y = 0.55;
    const band = new THREE.Mesh(new THREE.BoxGeometry(length, 0.25, 0.52), sponsor);
    band.position.y = 0.85;
    const pole = new THREE.Mesh(new THREE.BoxGeometry(0.08, 2.4, 0.08), post);
    pole.position.set(-length / 2, 2.3, 0);
    const fence = new THREE.Mesh(new THREE.PlaneGeometry(length, 2.2), new THREE.MeshStandardMaterial({ color: '#9aa3b1', transparent: true, opacity: 0.28, side: THREE.DoubleSide }));
    fence.position.y = 2.2;
    for (const part of [wall, band, pole]) { part.castShadow = part.receiveShadow = true; }
    segment.add(wall, band, pole, fence);
    segment.position.copy(at(middle, lane.side * (half + 3.4)));
    segment.rotation.y = headingAt(middle);
    group.add(segment);
  }

  // A garage and a painted box per car, in its colour; the crew waits in front of the garage.
  const boxes = lane.boxes.slice(0, cars.length).map((along, car) => {
    const color = new THREE.Color(cars[car].color);
    const heading = headingAt(along);
    const garage = new THREE.Group();
    const shell = new THREE.MeshStandardMaterial({ color: '#eceef2', roughness: 0.8 });
    const inside = new THREE.MeshStandardMaterial({ color: '#2a2e36', roughness: 0.9 });
    const team = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
    const wallPart = (width, height, depth, x, y, z, material) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
      mesh.position.set(x, y, z);
      mesh.castShadow = mesh.receiveShadow = true;
      garage.add(mesh);
    };
    // Local z points away from the lane, so the open front faces the cars.
    wallPart(11, 4.2, 0.4, 0, 2.1, 7, shell);           // back
    wallPart(0.4, 4.2, 7, -5.5, 2.1, 3.5, shell);       // sides
    wallPart(0.4, 4.2, 7, 5.5, 2.1, 3.5, shell);
    wallPart(11.4, 0.4, 7.4, 0, 4.3, 3.5, shell);       // roof
    wallPart(11.4, 0.9, 0.3, 0, 3.75, 0, team);         // team-coloured fascia
    wallPart(10.6, 0.05, 6.8, 0, 0.05, 3.5, inside);    // floor
    // The box is on the working lane, where the sim stops the car (see slipstream/pit.py).
    const [laneX, laneY] = pointAt(track, lane.entry + along, lane.workingLane ?? lane.offset);
    garage.position.copy(toWorld(...pointAt(track, lane.entry + along, lane.offset + lane.side * 7)));
    // Turn local z to point away from the lane (the lane side's normal): that works out to the heading, plus a
    // half turn when the lane is on the left.
    garage.rotation.y = lane.side > 0 ? heading + Math.PI : heading;
    group.add(garage);
    const paint = new THREE.Mesh(new THREE.PlaneGeometry(6, 3.6), new THREE.MeshStandardMaterial({ color, roughness: 0.7, transparent: true, opacity: 0.55 }));
    paint.rotation.set(-Math.PI / 2, heading, 0, 'YXZ');
    paint.position.copy(toWorld(laneX, laneY, 0.1));
    group.add(paint);
    return {
      car, along, heading,
      // Where the crew stands while waiting: between the garage and the lane.
      home: toWorld(...pointAt(track, lane.entry + along, lane.offset + lane.side * 4.5)),
      stop: toWorld(laneX, laneY),
    };
  });
  return { group, boxes };
}
