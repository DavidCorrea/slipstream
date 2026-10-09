// Where a circuit is: the ground, the sky and light, and everything around the track that isn't racing. Each circuit
// gets a location from its seed (or the one you pick): countryside, desert, alpine, coast, city, or the city at
// night. None of it touches the race itself; the sim only knows tarmac and grass.
import * as THREE from 'three';
import { toWorld } from './scene.js';

export const LOCATIONS = {
  countryside: {
    label: 'Countryside', ground: ['#7fae55', 'rgba(255,255,255,0.045)', ['40, 80, 20', '190, 220, 140']], verge: '#6e9a47',
    sky: '#9cc7e8', fog: '#b9d6ea', sun: ['#fff3dc', 2.3], hemisphere: ['#dcefff', '#5b7a3a', 1.1],
    scenery: countryside,
  },
  desert: {
    label: 'Desert', ground: ['#d8b67a', 'rgba(120,70,20,0.05)', ['150, 100, 50', '240, 215, 160']], verge: '#c9a46a',
    sky: '#a9c9e2', fog: '#e6d2b0', sun: ['#ffe6b8', 2.7], hemisphere: ['#fff2dc', '#b08a55', 1.2],
    scenery: desert,
  },
  alpine: {
    label: 'Alpine', ground: ['#6f9e58', 'rgba(255,255,255,0.03)', ['30, 70, 30', '170, 205, 150']], verge: '#5f8f4c',
    sky: '#8ec1ec', fog: '#c9dff0', sun: ['#f2f6ff', 2.2], hemisphere: ['#e4f0ff', '#4f6d47', 1.1],
    scenery: alpine,
  },
  coast: {
    label: 'Coast', ground: ['#8fbb5e', 'rgba(255,255,255,0.04)', ['50, 90, 30', '200, 225, 150']], verge: '#7aa851',
    sky: '#87c4f0', fog: '#cfe6f5', sun: ['#fff0d0', 2.5], hemisphere: ['#e6f4ff', '#6c8a4a', 1.15],
    scenery: coast,
  },
  city: {
    label: 'City', ground: ['#8a8d93', 'rgba(255,255,255,0.03)', ['60, 62, 66', '170, 172, 178']], verge: '#9a9da3',
    sky: '#a7c3dc', fog: '#c3cdd6', sun: ['#fff5e6', 2.2], hemisphere: ['#e8eef5', '#6f6f72', 1.1],
    scenery: city, walled: true,
  },
  night: {
    label: 'City at night', ground: ['#3c3f45', 'rgba(255,255,255,0.02)', ['20, 22, 26', '80, 84, 92']], verge: '#4a4d54',
    sky: '#0d1426', fog: '#141c30', sun: ['#9fb4ff', 0.5], hemisphere: ['#6a7aaa', '#2c2c32', 1.15],
    scenery: (context) => city({ ...context, night: true }), walled: true, night: true,
  },
};

export function locationFor(seed) {
  const names = Object.keys(LOCATIONS);
  return names[Math.abs(Math.floor(seed)) % names.length];
}

// ---- Shared helpers --------------------------------------------------------------------------------------

const material = (color, roughness = 0.85, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, ...extra });

// Places copies of one geometry with one matrix each (position, turn, size), coloured per copy if `colors` is given.
function instances(geometry, surface, placements, colors = null) {
  const mesh = new THREE.InstancedMesh(geometry, surface, Math.max(placements.length, 1));
  mesh.count = placements.length;
  const matrix = new THREE.Matrix4(), turn = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  placements.forEach(({ spot, angle = 0, scale = 1 }, index) => {
    turn.setFromAxisAngle(up, angle);
    matrix.compose(spot, turn, typeof scale === 'number' ? new THREE.Vector3(scale, scale, scale) : scale);
    mesh.setMatrixAt(index, matrix);
    if (colors) mesh.setColorAt(index, colors[index]);
  });
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

// Spots scattered around the circuit, at least `clear` metres from the track and inside `reach` of its middle.
function scatter({ centre, radius, random, awayFromTrack }, count, clear, reach = radius + 320, attempts = count * 6) {
  const spots = [];
  for (let attempt = 0; attempt < attempts && spots.length < count; attempt++) {
    const spot = new THREE.Vector3(centre.x + (random() * 2 - 1) * reach, 0, centre.z + (random() * 2 - 1) * reach);
    if (awayFromTrack(spot) > clear) spots.push(spot);
  }
  return spots;
}

// A ring of big shapes far out on the horizon: mountains, mesas or a skyline. `make` builds one at the origin, and
// it goes `clearance` metres beyond the circuit's reach measured from its foot, however wide that is: measured
// from its middle, the widest mountains reached over the track. Each is marked as backdrop, so the pieces between
// the camera and the circuit can be cut away (see circuit.js).
function horizon({ centre, radius, random }, count, clearance, make) {
  const group = new THREE.Group();
  const size = new THREE.Vector3();
  for (let index = 0; index < count; index++) {
    const angle = (index / count) * Math.PI * 2 + random() * 0.2;
    const piece = make(random);
    new THREE.Box3().setFromObject(piece).getSize(size);
    const distance = radius + clearance + Math.max(size.x, size.z) / 2;
    piece.position.x = centre.x + Math.cos(angle) * distance;
    piece.position.z = centre.z + Math.sin(angle) * distance;
    piece.userData.backdrop = true;
    group.add(piece);
  }
  return group;
}

function trees(context, count, { pine = 0.35, palette = [0.27, 0.06] } = {}) {
  const { random } = context;
  const spots = scatter(context, count, context.half + 30);
  const trunk = new THREE.CylinderGeometry(0.25, 0.35, 2.4, 6).translate(0, 1.2, 0);
  const cone = new THREE.ConeGeometry(2.2, 6.5, 7).translate(0, 5.2, 0);
  const crown = new THREE.IcosahedronGeometry(2.6, 0).translate(0, 4.3, 0);
  const placed = spots.map(spot => ({ spot, angle: random() * Math.PI, scale: 0.7 + random() * 0.9, pine: random() < pine }));
  const group = new THREE.Group();
  group.add(instances(trunk, material('#6b4a2b', 0.9), placed));
  group.add(instances(cone, material('#2f6b3a', 0.85, { flatShading: true }), placed.filter(tree => tree.pine)));
  const rounds = placed.filter(tree => !tree.pine);
  group.add(instances(crown, material('#ffffff', 0.85, { flatShading: true }), rounds,
    rounds.map(() => new THREE.Color().setHSL(palette[0] + random() * palette[1], 0.45, 0.32 + random() * 0.1))));
  return group;
}

// ---- Locations --------------------------------------------------------------------------------------------

function countryside(context) {
  const group = trees(context, 900);
  // Hay bales dotted about the fields.
  const bales = scatter(context, 60, context.half + 40).map(spot => ({ spot: spot.setY(0.7), angle: context.random() * Math.PI }));
  group.add(instances(new THREE.CylinderGeometry(0.75, 0.75, 1.3, 12).rotateZ(Math.PI / 2), material('#d9b45a', 0.95), bales));
  return group;
}

function desert(context) {
  const { random } = context;
  const group = new THREE.Group();
  // Boulders and cacti, and flat-topped mesas on the horizon.
  const rocks = scatter(context, 260, context.half + 25).map(spot => ({
    spot, angle: random() * Math.PI, scale: new THREE.Vector3(1 + random() * 3, 0.6 + random() * 1.6, 1 + random() * 3),
  }));
  group.add(instances(new THREE.DodecahedronGeometry(1, 0), material('#b0703c', 0.95, { flatShading: true }), rocks,
    rocks.map(() => new THREE.Color().setHSL(0.06 + random() * 0.03, 0.5, 0.36 + random() * 0.12))));
  const cacti = scatter(context, 200, context.half + 25).map(spot => ({ spot, angle: random() * Math.PI, scale: 0.8 + random() * 0.8 }));
  const stem = new THREE.CylinderGeometry(0.35, 0.4, 4, 8).translate(0, 2, 0);
  const arm = new THREE.CylinderGeometry(0.25, 0.25, 1.6, 7).translate(0.9, 2.6, 0);
  const elbow = new THREE.CylinderGeometry(0.25, 0.25, 0.9, 7).rotateZ(Math.PI / 2).translate(0.5, 1.9, 0);
  const cactus = material('#4d7a3a', 0.8, { flatShading: true });
  for (const piece of [stem, arm, elbow]) group.add(instances(piece, cactus, cacti));
  group.add(horizon(context, 16, 150, random => {
    const height = 50 + random() * 110, width = 140 + random() * 240;
    const mesa = new THREE.Mesh(new THREE.CylinderGeometry(width * 0.42, width * 0.55, height, 9), material('#b8693b', 1, { flatShading: true }));
    mesa.position.y = height / 2;
    return mesa;
  }));
  return group;
}

function alpine(context) {
  const { random } = context;
  const group = trees(context, 1400, { pine: 0.85, palette: [0.3, 0.04] });
  // Snow-capped mountains all round, and a lake.
  group.add(horizon(context, 18, 150, random => {
    const height = 260 + random() * 300, width = 280 + random() * 220;
    const mountain = new THREE.Group();
    const rock = new THREE.Mesh(new THREE.ConeGeometry(width, height, 7), material('#6d7480', 1, { flatShading: true }));
    rock.position.y = height / 2;
    const snow = new THREE.Mesh(new THREE.ConeGeometry(width * 0.36, height * 0.36, 7), material('#f4f7fb', 0.7, { flatShading: true }));
    snow.position.y = height * 0.82;
    mountain.add(rock, snow);
    mountain.rotation.y = random() * Math.PI;
    return mountain;
  }));
  const lake = scatter(context, 1, context.half + 120, context.radius + 260, 400)[0];
  if (lake) {
    const water = new THREE.Mesh(new THREE.CircleGeometry(70 + random() * 40, 40), material('#3d7fb3', 0.15, { metalness: 0.2 }));
    water.rotation.x = -Math.PI / 2;
    water.position.copy(lake).setY(0.05);
    group.add(water);
  }
  return group;
}

function coast(context) {
  const { random, centre, radius } = context;
  const group = new THREE.Group();
  // The sea fills one side of the world beyond a beach; palms line the land side.
  const direction = random() * Math.PI * 2;
  const outward = new THREE.Vector3(Math.cos(direction), 0, Math.sin(direction));
  const shore = centre.clone().addScaledVector(outward, radius + 90);
  const sea = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), material('#2f86c4', 0.2, { metalness: 0.15 }));
  sea.rotation.x = -Math.PI / 2;
  sea.position.copy(shore).addScaledVector(outward, 2000).setY(0.06);
  sea.rotation.z = -direction;
  const beach = new THREE.Mesh(new THREE.PlaneGeometry(60, 4000), material('#e8d39b', 1));
  beach.rotation.x = -Math.PI / 2;
  beach.rotation.z = -direction;
  beach.position.copy(shore).addScaledVector(outward, 20).setY(0.05);
  group.add(sea, beach);
  const inland = spot => spot.clone().sub(shore).dot(outward) < -10;
  const palms = scatter(context, 500, context.half + 20).filter(inland).map(spot => ({ spot, angle: random() * Math.PI * 2, scale: 0.8 + random() * 0.6 }));
  const trunk = new THREE.CylinderGeometry(0.22, 0.35, 7, 6).translate(0, 3.5, 0).rotateZ(0.12);
  group.add(instances(trunk, material('#8a6a45', 0.9), palms));
  const frond = new THREE.ConeGeometry(0.5, 4.2, 4).rotateZ(Math.PI / 2 + 0.35).translate(2, 7, 0);
  const leaves = material('#3f8a3a', 0.8, { flatShading: true });
  for (let turn = 0; turn < 6; turn++) {
    group.add(instances(frond.clone().rotateY((turn / 6) * Math.PI * 2), leaves, palms));
  }
  // A marina: boats moored off the beach.
  const boats = Array.from({ length: 24 }, (_, index) => ({
    spot: shore.clone().addScaledVector(outward, 50 + (index % 3) * 14).addScaledVector(new THREE.Vector3(-outward.z, 0, outward.x), (index - 12) * 9).setY(0.6),
    angle: -direction, scale: new THREE.Vector3(1, 1, 1),
  }));
  group.add(instances(new THREE.BoxGeometry(6, 1.2, 2.2), material('#f4f4f2', 0.5), boats));
  group.add(instances(new THREE.CylinderGeometry(0.08, 0.08, 7, 5).translate(0, 4, 0), material('#d0d4da', 0.4), boats));
  return group;
}

// A street circuit: buildings close up behind the walls, a skyline beyond; lit windows and street lamps at night.
function city(context) {
  const { random, night } = context;
  const group = new THREE.Group();
  const windows = windowTexture(night);
  const facade = new THREE.MeshStandardMaterial({ map: windows, roughness: 0.7, emissive: night ? '#ffffff' : '#000000', emissiveMap: night ? windows : null, emissiveIntensity: night ? 0.9 : 0 });
  // Low blocks along the track, so the racing stays in view from above; towers further back.
  const blocks = scatter(context, 320, context.half + 16, context.radius + 260).map(spot => {
    const back = Math.min(context.awayFromTrack(spot) / 120, 1);
    const height = 6 + random() * 10 + back * random() ** 2 * 80, width = 10 + random() * 16;
    return { spot: spot.setY(height / 2), angle: random() * Math.PI, scale: new THREE.Vector3(width, height, 10 + random() * 16) };
  });
  group.add(instances(new THREE.BoxGeometry(1, 1, 1), facade, blocks,
    blocks.map(() => new THREE.Color().setHSL(0.58 + random() * 0.1, 0.08 + random() * 0.1, night ? 0.5 : 0.62 + random() * 0.2))));
  group.add(horizon(context, 40, 500, random => {
    const height = 60 + random() * 180;
    const tower = new THREE.Mesh(new THREE.BoxGeometry(30 + random() * 40, height, 30 + random() * 40), facade);
    tower.position.y = height / 2;
    return tower;
  }));
  // Street lamps along the track, glowing at night.
  const lamps = context.alongTrack(night ? 16 : 28, context.half + 4).map(({ spot, angle }) => ({ spot, angle }));
  group.add(instances(new THREE.CylinderGeometry(0.1, 0.14, 8, 6).translate(0, 4, 0), material('#4a4e57', 0.5, { metalness: 0.5 }), lamps));
  const glow = new THREE.MeshStandardMaterial({ color: '#fff6d8', emissive: '#ffe9b0', emissiveIntensity: night ? 3 : 0.2 });
  group.add(instances(new THREE.BoxGeometry(1.2, 0.25, 0.5).translate(0.5, 8, 0), glow, lamps));
  if (night) group.add(lightPools(lamps));
  return group;
}

// At night each lamp throws a pool of light on the tarmac: a soft additive disc, so dozens cost nothing.
function lightPools(lamps) {
  const texture = canvasTexture(64, (paint, size) => {
    const gradient = paint.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, 'rgba(255,236,190,0.85)');
    gradient.addColorStop(1, 'rgba(255,236,190,0)');
    paint.fillStyle = gradient;
    paint.fillRect(0, 0, size, size);
  });
  const pool = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const discs = lamps.map(({ spot, angle }) => ({ spot: spot.clone().add(new THREE.Vector3(Math.cos(angle) * 3, 0.16, -Math.sin(angle) * 3)), angle: 0, scale: 1 }));
  const mesh = instances(new THREE.PlaneGeometry(30, 30).rotateX(-Math.PI / 2), pool, discs);
  mesh.castShadow = mesh.receiveShadow = false;
  return mesh;
}

function windowTexture(night) {
  const texture = canvasTexture(128, (paint, size) => {
    paint.fillStyle = night ? '#101420' : '#c8ccd3';
    paint.fillRect(0, 0, size, size);
    for (let row = 0; row < 8; row++) for (let column = 0; column < 4; column++) {
      const lit = Math.random() < 0.55;
      paint.fillStyle = night ? (lit ? '#ffd98a' : '#1a2030') : (lit ? '#7b93ad' : '#5f7690');
      paint.fillRect(column * 32 + 6, row * 16 + 4, 20, 9);
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 4);
  return texture;
}

export function canvasTexture(size, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  draw(canvas.getContext('2d'), size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

// ---- Trackside, everywhere -------------------------------------------------------------------------------

const SPONSORS = [
  ['SLIPSTREAM', '#e10600', '#ffffff'], ['NORTHWIND', '#0b3d91', '#ffffff'], ['APEX OIL', '#ffd23f', '#111111'],
  ['VOLTA', '#111111', '#3ddc84'], ['KESTREL', '#ffffff', '#e10600'], ['MERIDIAN', '#2a9d8f', '#ffffff'],
  ['HALCYON', '#9b5de5', '#ffffff'], ['TORQUE', '#f4a261', '#111111'],
];

function sponsorTexture([name, background, ink]) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 96;
  const paint = canvas.getContext('2d');
  paint.fillStyle = background;
  paint.fillRect(0, 0, 512, 96);
  paint.fillStyle = ink;
  paint.font = '900 64px "Titillium Web", sans-serif';
  paint.textAlign = 'center';
  paint.textBaseline = 'middle';
  paint.fillText(name, 256, 52);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Sponsor boards along the straights, marshal posts with waving flags, braking boards before the big stops,
// TV camera towers on the outside of corners, a sponsor bridge over a straight, and armco on the outside of the
// fast bits (walls all round on a street circuit). Returns the group and what moves each frame.
export function trackside({ points, normals, curvature, half, random, clearOfLane, location }) {
  const group = new THREE.Group();
  const count = points.length;
  const at = (index, lateral, height = 0) => {
    const [x, y] = points[index % count], [nx, ny] = normals[index % count];
    return toWorld(x + nx * lateral, y + ny * lateral, height);
  };
  const heading = index => {
    const [x0, y0] = points[index % count], [x1, y1] = points[(index + 1) % count];
    return Math.atan2(y1 - y0, x1 - x0);
  };
  const straight = index => Math.abs(curvature[index % count]) < 1 / 300;

  // Sponsor boards: every so often along straights, on the outside.
  const boardMaterials = SPONSORS.map(sponsor => new THREE.MeshStandardMaterial({ map: sponsorTexture(sponsor), roughness: 0.6 }));
  for (let index = 0; index < count; index += 14) {
    if (!straight(index)) continue;
    for (const side of [1, -1]) {
      if (!clearOfLane(index, side) || random() < 0.4) continue;
      const board = new THREE.Mesh(new THREE.PlaneGeometry(9, 1.6), boardMaterials[Math.floor(random() * boardMaterials.length)]);
      board.position.copy(at(index, side * (half + (location.walled ? 1.5 : 9)), 1.3));
      board.rotation.y = heading(index) + (side > 0 ? Math.PI : 0);
      group.add(board);
    }
  }

  // Braking boards (3, 2, 1) before tight corners, on the outside of the approach.
  const boardFace = number => new THREE.MeshStandardMaterial({ map: sponsorTexture([String(number), '#ffffff', '#111111']), roughness: 0.6 });
  const faces = [boardFace(1), boardFace(2), boardFace(3)];
  for (let index = 0; index < count; index++) {
    const ahead = (index + 25) % count;
    if (!(Math.abs(curvature[ahead]) > 1 / 40 && Math.abs(curvature[index]) < 1 / 200)) continue;
    if (Math.abs(curvature[(index + count - 1) % count]) < 1 / 200 && Math.abs(curvature[(ahead + count - 1) % count]) > 1 / 40) continue;
    const side = curvature[ahead] > 0 ? -1 : 1;
    if (!clearOfLane(index, side)) continue;
    [3, 2, 1].forEach((number, step) => {
      const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), faces[number - 1]);
      sign.position.copy(at(index + step * 5, side * (half + 3), 1.2));
      sign.rotation.y = heading(index) - Math.PI / 2;
      group.add(sign);
    });
    index += 30;
  }

  // Marshal posts at corners: a hut, a marshal in orange, and a waving flag.
  const flags = [];
  const flagCloth = new THREE.MeshStandardMaterial({ color: '#ffd23f', side: THREE.DoubleSide, roughness: 0.8 });
  for (let index = 0; index < count; index += 20) {
    if (Math.abs(curvature[index]) < 1 / 70) continue;
    const side = curvature[index] > 0 ? -1 : 1;
    if (!clearOfLane(index, side)) continue;
    const post = new THREE.Group();
    const hut = new THREE.Mesh(new THREE.BoxGeometry(2, 2.2, 2), material('#f2f2f2', 0.8));
    hut.position.y = 1.1;
    const marshal = new THREE.Mesh(new THREE.CapsuleGeometry(0.25, 0.9, 3, 6), material('#ff7a1a', 0.7));
    marshal.position.set(0, 0.8, -1.8);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6, 4), material('#cccccc', 0.4));
    pole.position.set(0.35, 1.7, -1.8);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.5), flagCloth);
    flag.position.set(0.7, 2.3, -1.8);
    post.add(hut, marshal, pole, flag);
    post.position.copy(at(index, side * (half + 14)));
    post.rotation.y = heading(index) + (side > 0 ? Math.PI : 0);
    group.add(post);
    flags.push({ flag, phase: random() * Math.PI * 2 });
    for (const part of [hut, marshal]) part.castShadow = true;
  }

  // TV camera towers on the outside of a few corners.
  for (let index = 7; index < count; index += 55) {
    if (Math.abs(curvature[index]) < 1 / 90) continue;
    const side = curvature[index] > 0 ? -1 : 1;
    if (!clearOfLane(index, side)) continue;
    const tower = new THREE.Group();
    for (const [x, z] of [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8]]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.12, 7, 0.12), material('#9aa0aa', 0.5, { metalness: 0.5 }));
      leg.position.set(x, 3.5, z);
      tower.add(leg);
    }
    const deck = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.2, 2.2), material('#6c7280', 0.6));
    deck.position.y = 7;
    const camera = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.5, 0.5), material('#1c1f24', 0.4));
    camera.position.set(0, 7.5, 0);
    const operator = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.7, 3, 6), material('#2f4c8a', 0.7));
    operator.position.set(0.6, 7.6, 0.4);
    tower.add(deck, camera, operator);
    tower.position.copy(at(index, side * (half + 22)));
    tower.rotation.y = heading(index);
    group.add(tower);
  }

  // A sponsor bridge over the longest straight away from the start.
  let best = -1, run = 0, bestRun = 0;
  for (let index = Math.floor(count * 0.25); index < Math.floor(count * 0.8); index++) {
    run = straight(index) ? run + 1 : 0;
    if (run > bestRun) { bestRun = run; best = index - Math.floor(run / 2); }
  }
  if (bestRun > 20) {
    const bridge = new THREE.Group();
    const span = half * 2 + 8;
    const deck = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.6, span), boardMaterials[0]);
    deck.position.y = 6.8;
    for (const end of [-1, 1]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(1.2, 7, 1.2), material('#d9dbe0', 0.8));
      leg.position.set(0, 3.5, end * span / 2);
      bridge.add(leg);
    }
    bridge.add(deck);
    bridge.position.copy(at(best, 0));
    bridge.rotation.y = heading(best);
    bridge.traverse(part => { part.castShadow = true; });
    group.add(bridge);
  }

  // Barriers: armco along the outside of straights, or concrete walls on both sides of a street circuit.
  const barrier = new THREE.Group();
  const wallMaterial = location.walled ? material('#c9ccd2', 0.9) : material('#b7bcc4', 0.35, { metalness: 0.6 });
  const offset = location.walled ? half + 3.2 : half + 12;
  const height = location.walled ? 1.2 : 0.7;
  for (const side of [1, -1]) {
    const positions = [], indices = [];
    let segment = 0;
    for (let index = 0; index <= count; index++) {
      const keep = (location.walled || straight(index)) && clearOfLane(index % count, side);
      const bottom = at(index, side * offset, 0), top = at(index, side * offset, height);
      positions.push(bottom.x, bottom.y, bottom.z, top.x, top.y, top.z);
      if (index > 0 && keep && segment) indices.push((index - 1) * 2, index * 2, (index - 1) * 2 + 1, (index - 1) * 2 + 1, index * 2, index * 2 + 1);
      segment = keep ? 1 : 0;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const wall = new THREE.Mesh(geometry, wallMaterial);
    wall.material.side = THREE.DoubleSide;
    wall.castShadow = wall.receiveShadow = true;
    barrier.add(wall);
  }
  group.add(barrier);

  return {
    group,
    // Flags flap, harder in a stronger wind.
    update(time, wind = 0) {
      for (const { flag, phase } of flags) {
        flag.rotation.y = Math.sin(time * (3 + wind * 0.4) + phase) * (0.25 + wind * 0.04);
        flag.rotation.z = Math.sin(time * 5 + phase) * 0.08;
      }
    },
  };
}
