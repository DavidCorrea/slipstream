// Where a circuit is: the ground, the sky and light, and everything around the track. What stands where comes from
// the sim (slipstream/scenery.py), which places every prop and lets the cars hit them; this draws that list, so
// what you see is what the cars can hit. A breakable prop (a board, a sign, a lamp, a cactus, a bale) disappears
// when a car breaks it: each one registers how to hide it.
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
    scenery: context => city({ ...context, night: true }), walled: true, night: true,
  },
};

// ---- Shared helpers --------------------------------------------------------------------------------------

const material = (color, roughness = 0.85, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, ...extra });
const spot = (prop, height = 0) => toWorld(prop.x, prop.y, height);
const ofKind = (props, kind) => props.filter(prop => prop.kind === kind);

// Places copies of one geometry with one matrix each (position, turn, size), coloured per copy if `colors` is given.
// With `breakable`, each copy's prop can be hidden when it breaks (see context.breakable).
function instances(geometry, surface, placements, colors = null, breakable = null) {
  const mesh = new THREE.InstancedMesh(geometry, surface, Math.max(placements.length, 1));
  mesh.count = placements.length;
  const matrix = new THREE.Matrix4(), turn = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  placements.forEach(({ spot: where, angle = 0, scale = 1, id }, index) => {
    turn.setFromAxisAngle(up, angle);
    matrix.compose(where, turn, typeof scale === 'number' ? new THREE.Vector3(scale, scale, scale) : scale);
    mesh.setMatrixAt(index, matrix);
    if (colors) mesh.setColorAt(index, colors[index]);
    if (breakable && id !== undefined) breakable(id, () => hideInstance(mesh, index));
  });
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

function hideInstance(mesh, index) {
  const matrix = new THREE.Matrix4();
  mesh.getMatrixAt(index, matrix);
  matrix.scale(new THREE.Vector3(0, 0, 0));
  mesh.setMatrixAt(index, matrix);
  mesh.instanceMatrix.needsUpdate = true;
}

// A backdrop piece (a mountain, a mesa, a tower): far beyond the circuit, and cut away when it's between the camera
// and the race (see circuit.js).
function backdrop(piece, prop) {
  piece.position.x = spot(prop).x;
  piece.position.z = spot(prop).z;
  piece.userData.backdrop = true;
  return piece;
}

function trees(context, kind = 'tree', palette = [0.27, 0.06]) {
  const { random } = context;
  const placed = ofKind(context.props, kind).map(prop => ({ spot: spot(prop), angle: prop.angle, scale: prop.scale, pine: prop.pine }));
  const trunk = new THREE.CylinderGeometry(0.25, 0.35, 2.4, 6).translate(0, 1.2, 0);
  const cone = new THREE.ConeGeometry(2.2, 6.5, 7).translate(0, 5.2, 0);
  const crown = new THREE.IcosahedronGeometry(2.6, 0).translate(0, 4.3, 0);
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
  const group = trees(context);
  // Hay bales dotted about the fields.
  const bales = ofKind(context.props, 'bale').map(prop => ({ spot: spot(prop, 0.7), angle: prop.angle, id: prop.id }));
  group.add(instances(new THREE.CylinderGeometry(0.75, 0.75, 1.3, 12).rotateZ(Math.PI / 2), material('#d9b45a', 0.95), bales, null, context.breakable));
  return group;
}

function desert(context) {
  const { random, props } = context;
  const group = new THREE.Group();
  // Boulders and cacti, and flat-topped mesas on the horizon.
  const rocks = ofKind(props, 'rock').map(prop => ({ spot: spot(prop), angle: prop.angle, scale: new THREE.Vector3(...prop.size) }));
  group.add(instances(new THREE.DodecahedronGeometry(1, 0), material('#b0703c', 0.95, { flatShading: true }), rocks,
    rocks.map(() => new THREE.Color().setHSL(0.06 + random() * 0.03, 0.5, 0.36 + random() * 0.12))));
  const cacti = ofKind(props, 'cactus').map(prop => ({ spot: spot(prop), angle: prop.angle, scale: prop.scale, id: prop.id }));
  const stem = new THREE.CylinderGeometry(0.35, 0.4, 4, 8).translate(0, 2, 0);
  const arm = new THREE.CylinderGeometry(0.25, 0.25, 1.6, 7).translate(0.9, 2.6, 0);
  const elbow = new THREE.CylinderGeometry(0.25, 0.25, 0.9, 7).rotateZ(Math.PI / 2).translate(0.5, 1.9, 0);
  const cactus = material('#4d7a3a', 0.8, { flatShading: true });
  for (const piece of [stem, arm, elbow]) group.add(instances(piece, cactus, cacti, null, context.breakable));
  for (const prop of ofKind(props, 'mesa')) {
    const mesa = new THREE.Mesh(new THREE.CylinderGeometry(prop.width * 0.42, prop.width * 0.55, prop.height, 9), material('#b8693b', 1, { flatShading: true }));
    mesa.position.y = prop.height / 2;
    group.add(backdrop(mesa, prop));
  }
  return group;
}

function alpine(context) {
  const { props } = context;
  const group = trees(context, 'tree', [0.3, 0.04]);
  // Snow-capped mountains all round, and a lake.
  for (const prop of ofKind(props, 'mountain')) {
    const mountain = new THREE.Group();
    const rock = new THREE.Mesh(new THREE.ConeGeometry(prop.width, prop.height, 7), material('#6d7480', 1, { flatShading: true }));
    rock.position.y = prop.height / 2;
    const snow = new THREE.Mesh(new THREE.ConeGeometry(prop.width * 0.36, prop.height * 0.36, 7), material('#f4f7fb', 0.7, { flatShading: true }));
    snow.position.y = prop.height * 0.82;
    mountain.add(rock, snow);
    mountain.rotation.y = prop.turn;
    group.add(backdrop(mountain, prop));
  }
  for (const prop of ofKind(props, 'lake')) {
    const water = new THREE.Mesh(new THREE.CircleGeometry(prop.size, 40), material('#3d7fb3', 0.15, { metalness: 0.2 }));
    water.rotation.x = -Math.PI / 2;
    water.position.copy(spot(prop, 0.05));
    group.add(water);
  }
  return group;
}

function coast(context) {
  const { props } = context;
  const group = new THREE.Group();
  // The sea fills one side of the world beyond a beach; palms line the land side, boats are moored off it.
  for (const prop of ofKind(props, 'sea')) {
    const outward = new THREE.Vector3(Math.cos(prop.direction), 0, Math.sin(prop.direction));
    const shore = spot(prop);
    const sea = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), material('#2f86c4', 0.2, { metalness: 0.15 }));
    sea.rotation.x = -Math.PI / 2;
    sea.position.copy(shore).addScaledVector(outward, 2000).setY(0.06);
    sea.rotation.z = -prop.direction;
    const beach = new THREE.Mesh(new THREE.PlaneGeometry(60, 4000), material('#e8d39b', 1));
    beach.rotation.x = -Math.PI / 2;
    beach.rotation.z = -prop.direction;
    beach.position.copy(shore).addScaledVector(outward, 20).setY(0.05);
    group.add(sea, beach);
  }
  const palms = ofKind(props, 'palm').map(prop => ({ spot: spot(prop), angle: prop.angle, scale: prop.scale }));
  const trunk = new THREE.CylinderGeometry(0.22, 0.35, 7, 6).translate(0, 3.5, 0).rotateZ(0.12);
  group.add(instances(trunk, material('#8a6a45', 0.9), palms));
  const frond = new THREE.ConeGeometry(0.5, 4.2, 4).rotateZ(Math.PI / 2 + 0.35).translate(2, 7, 0);
  const leaves = material('#3f8a3a', 0.8, { flatShading: true });
  for (let turn = 0; turn < 6; turn++) group.add(instances(frond.clone().rotateY((turn / 6) * Math.PI * 2), leaves, palms));
  const boats = ofKind(props, 'boat').map(prop => ({ spot: spot(prop, 0.6), angle: prop.angle }));
  group.add(instances(new THREE.BoxGeometry(6, 1.2, 2.2), material('#f4f4f2', 0.5), boats));
  group.add(instances(new THREE.CylinderGeometry(0.08, 0.08, 7, 5).translate(0, 4, 0), material('#d0d4da', 0.4), boats));
  return group;
}

// A street circuit: buildings close up behind the walls, a skyline beyond; lit windows and street lamps at night.
function city(context) {
  const { props, night } = context;
  const group = new THREE.Group();
  const windows = windowTexture(night);
  const facade = new THREE.MeshStandardMaterial({ map: windows, roughness: 0.7, emissive: night ? '#ffffff' : '#000000', emissiveMap: night ? windows : null, emissiveIntensity: night ? 0.9 : 0 });
  const blocks = ofKind(props, 'block').map(prop => ({ spot: spot(prop, prop.height / 2), angle: prop.angle, scale: new THREE.Vector3(prop.width, prop.height, prop.depth), shade: prop.shade }));
  group.add(instances(new THREE.BoxGeometry(1, 1, 1), facade, blocks,
    blocks.map(block => new THREE.Color().setHSL(0.58 + block.shade * 0.1, 0.08 + block.shade * 0.1, night ? 0.5 : 0.62 + block.shade * 0.2))));
  for (const prop of ofKind(props, 'skyline')) {
    const tower = new THREE.Mesh(new THREE.BoxGeometry(prop.width, prop.height, prop.depth), facade);
    tower.position.y = prop.height / 2;
    group.add(backdrop(tower, prop));
  }
  // Street lamps along the track, glowing at night; each one breaks, its pool of light with it.
  const lamps = ofKind(props, 'lamp').map(prop => ({ spot: spot(prop), angle: prop.angle, id: prop.id }));
  group.add(instances(new THREE.CylinderGeometry(0.1, 0.14, 8, 6).translate(0, 4, 0), material('#4a4e57', 0.5, { metalness: 0.5 }), lamps, null, context.breakable));
  const glow = new THREE.MeshStandardMaterial({ color: '#fff6d8', emissive: '#ffe9b0', emissiveIntensity: night ? 3 : 0.2 });
  group.add(instances(new THREE.BoxGeometry(1.2, 0.25, 0.5).translate(0.5, 8, 0), glow, lamps, null, context.breakable));
  if (night) group.add(lightPools(lamps, context.breakable));
  return group;
}

// At night each lamp throws a pool of light on the tarmac: a soft additive disc, so dozens cost nothing.
function lightPools(lamps, breakable) {
  const texture = canvasTexture(64, (paint, size) => {
    const gradient = paint.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, 'rgba(255,236,190,0.85)');
    gradient.addColorStop(1, 'rgba(255,236,190,0)');
    paint.fillStyle = gradient;
    paint.fillRect(0, 0, size, size);
  });
  const pool = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const discs = lamps.map(({ spot: where, angle, id }) => ({ spot: where.clone().add(new THREE.Vector3(Math.cos(angle) * 3, 0.16, -Math.sin(angle) * 3)), angle: 0, scale: 1, id }));
  const mesh = instances(new THREE.PlaneGeometry(30, 30).rotateX(-Math.PI / 2), pool, discs, null, breakable);
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

// Sponsor boards along the straights, braking boards before the big stops, marshal posts with waving flags, TV
// camera towers on the outside of corners, a sponsor bridge over a straight, armco on the outside of the fast bits
// (walls all round on a street circuit), and tyre walls on the outside of the tightest corners: all where the sim
// put them. Returns the group and what moves each frame.
export function trackside({ props, walled, breakable }) {
  const group = new THREE.Group();
  const boardMaterials = SPONSORS.map(sponsor => new THREE.MeshStandardMaterial({ map: sponsorTexture(sponsor), roughness: 0.6 }));
  // A breakable prop drawn as its own object: hidden whole when it breaks.
  const own = (object, prop) => {
    breakable(prop.id, () => { object.visible = false; });
    group.add(object);
  };

  for (const prop of ofKind(props, 'board')) {
    const board = new THREE.Mesh(new THREE.PlaneGeometry(9, 1.6), boardMaterials[prop.sponsor % boardMaterials.length]);
    board.position.copy(spot(prop, 1.3));
    board.rotation.y = prop.angle;
    own(board, prop);
  }

  const boardFace = number => new THREE.MeshStandardMaterial({ map: sponsorTexture([String(number), '#ffffff', '#111111']), roughness: 0.6 });
  const faces = [boardFace(1), boardFace(2), boardFace(3)];
  for (const prop of ofKind(props, 'marker')) {
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), faces[prop.number - 1]);
    sign.position.copy(spot(prop, 1.2));
    sign.rotation.y = prop.angle;
    own(sign, prop);
  }

  // Marshal posts: a hut, a marshal in orange, and a waving flag.
  const flags = [];
  const flagCloth = new THREE.MeshStandardMaterial({ color: '#ffd23f', side: THREE.DoubleSide, roughness: 0.8 });
  for (const prop of ofKind(props, 'marshal')) {
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
    post.position.copy(spot(prop));
    post.rotation.y = prop.angle;
    group.add(post);
    flags.push({ flag, phase: prop.phase });
    for (const part of [hut, marshal]) part.castShadow = true;
  }

  for (const prop of ofKind(props, 'camera')) {
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
    tower.position.copy(spot(prop));
    tower.rotation.y = prop.angle;
    group.add(tower);
  }

  // The sponsor bridge: the sim places its two legs, either side of the track; the deck spans between them.
  const legs = ofKind(props, 'bridge');
  if (legs.length === 2) {
    const [first, second] = legs.map(leg => spot(leg));
    const bridge = new THREE.Group();
    const span = first.distanceTo(second);
    const deck = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.6, span), boardMaterials[0]);
    deck.position.y = 6.8;
    for (const end of [-1, 1]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(1.2, 7, 1.2), material('#d9dbe0', 0.8));
      leg.position.set(0, 3.5, end * span / 2);
      bridge.add(leg);
    }
    bridge.add(deck);
    bridge.position.copy(first.clone().add(second).multiplyScalar(0.5));
    bridge.rotation.y = legs[0].angle;
    bridge.traverse(part => { part.castShadow = true; });
    group.add(bridge);
  }

  // Barriers: the sim's chain of short pieces, armco or concrete.
  const pieces = ofKind(props, walled ? 'wall' : 'armco');
  const height = walled ? 1.2 : 0.7;
  const barrier = instances(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0),
    walled ? material('#c9ccd2', 0.9) : material('#b7bcc4', 0.35, { metalness: 0.6 }),
    pieces.map(prop => ({ spot: spot(prop), angle: prop.angle, scale: new THREE.Vector3(prop.halfLength * 2, height, prop.halfWidth * 2) })));
  group.add(barrier);

  // Tyre walls: three tyres high, in the stripes the sim gave each stack.
  const stacks = ofKind(props, 'tyres');
  const tyre = new THREE.TorusGeometry(0.42, 0.2, 6, 12).rotateX(Math.PI / 2);
  const tyres = new THREE.InstancedMesh(tyre, material('#ffffff', 0.9), Math.max(stacks.length * 3, 1));
  tyres.count = stacks.length * 3;
  const colors = ['#1b1b1b', '#1b1b1b', '#e10600', '#f2f2f2'].map(color => new THREE.Color(color));
  const matrix = new THREE.Matrix4();
  stacks.forEach((prop, index) => {
    const where = spot(prop);
    for (let layer = 0; layer < 3; layer++) {
      matrix.makeTranslation(where.x, 0.22 + layer * 0.4, where.z);
      tyres.setMatrixAt(index * 3 + layer, matrix);
      tyres.setColorAt(index * 3 + layer, colors[(prop.shade + layer) % colors.length]);
    }
  });
  tyres.castShadow = tyres.receiveShadow = true;
  group.add(tyres);

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
