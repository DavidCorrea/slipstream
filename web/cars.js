// The cars: a low-poly open-wheeler per driver, posed every frame from the sim, with every part showing the
// car's setup and condition.
//   shape       a tub and nose that taper to the tip, sidepods that narrow towards the back, an engine cover that
//               slopes down into a shark fin, a halo round the cockpit, and wishbones out to every wheel
//   paint       a clear-coated gloss over the livery colour; carbon parts matte, with a fine weave
//   livery      the race number on the nose and sidepods, a sponsor's name on the sidepods, engine cover and
//               rear wing endplates
//   driver      a helmet in the livery with a dark visor; the head leans against the corner, tips forward under
//               braking and looks into the turn, and the steering wheel turns with the steering
//   wheels      rounded tyres that spin with speed, fronts steer; the sidewall stripe is the compound's colour,
//               greying as it wears, and the tread goes from matte black to a shinier, greyer grained look
//   brakes      discs glow orange under hard braking and cool off after
//   body        leans in corners, dips under braking, and sits lower on a full tank
//   rear wing   steeper the more wing it runs, with an extra flap at high settings
//   damage      the front wing droops, then breaks in two: one half goes at 15% damage and the other at 35%;
//               past 50% the rear wing hangs off its pillar, past 65% a front wheel splays out, and the paint
//               darkens throughout. Lost halves of wing are left on the track.
//   exhaust     a flame on full throttle, bigger and hotter in the more powerful engine modes
// The crew can lift the car on jacks, take the rear wing off, open the engine cover and swap the nose.
//
// Twenty cars are drawn every frame, three times over (the picture, the sun's shadow and the shading where things
// meet), so each car is cheap to draw: the parts that move together are merged into one mesh per material, every
// car shares the same geometry, a car too small on screen to show its details is drawn as a simple two-mesh car
// instead (setCarDetail), and that simple car is also what casts the detailed one's shadow: at the shadow map's
// resolution the two look the same.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { SHADOW_ONLY, toWorld } from './scene.js';
import { SPONSORS } from './scenery.js';

const WHEEL_RADIUS = 0.36;
const WHEELBASE = 3.0;
const TRACK_WIDTH = 1.7;
// Where the tyres touch: the tarmac's top (see circuit.js), or the grass and gravel a little below it.
export const ROAD_HEIGHT = 0.1;
const OFF_ROAD_HEIGHT = 0.05;
const RIDE_HEIGHT = 0.3;            // the body's floor plane above the ground
export const COMPOUND_COLORS = { soft: '#e63946', medium: '#ffd23f', hard: '#f2f2f2', intermediate: '#3ddc84', wet: '#2f8cff' };
const WORN = new THREE.Color('#6b6b6b');
const FRESH_TREAD = new THREE.Color('#141414'), WORN_TREAD = new THREE.Color('#34312e');
const TREAD_ROUGHNESS = { fresh: 0.88, worn: 0.5 };
// How the driver's head moves: rolled against the corner and pitched under braking, more than the body (a neck
// isn't a suspension), and turned towards where the car is steering.
const HEAD = { roll: 3.5, pitch: 5, look: 0.55 };
const STEERING_LOCK = 1.6;          // steering wheel turn for full steering, radians

// ---- Shapes --------------------------------------------------------------------------------------------------

// The car's cross sections from nose to tail, in the body's frame: [x, half width, bottom, top]. Each is a
// rectangle with its corners cut, so the panels between them come out as flat facets, like the rest of the scene.
const NOSE = [[2.75, 0.07, -0.02, 0.06], [2.2, 0.13, -0.04, 0.14], [1.4, 0.24, -0.08, 0.24]];
const TUB = [[1.4, 0.24, -0.08, 0.24], [0.9, 0.36, -0.1, 0.34], [0, 0.42, -0.1, 0.36], [-0.6, 0.4, -0.1, 0.32],
             [-1.6, 0.22, -0.06, 0.2], [-2.05, 0.15, -0.04, 0.14]];
// Wide at the radiator inlets, narrowing towards the rear wheels (the "coke bottle").
const SIDEPODS = [[0.78, 0.66, -0.08, 0.17], [0.4, 0.74, -0.1, 0.22], [-0.5, 0.62, -0.1, 0.2], [-1.2, 0.38, -0.08, 0.14],
                  [-1.6, 0.25, -0.06, 0.1]];
// From the airbox over the driver's head down to the gearbox; hinged at COVER_HINGE so the crew can open it.
const COVER = [[0.08, 0.15, 0.3, 0.64], [-0.2, 0.2, 0.3, 0.6], [-0.9, 0.24, 0.28, 0.42], [-1.6, 0.12, 0.16, 0.24]];
const COVER_HINGE = new THREE.Vector3(-0.55, 0.56, 0);
const NOSE_MOUNT = new THREE.Vector3(1.9, 0.05, 0);
const REAR_WING_MOUNT = new THREE.Vector3(-1.9, 0.75, 0);
const CHAMFER = 0.35;

// A shape through cross sections, as flat-shaded triangles. Shifted by `origin`, for parts on their own hinge.
function loft(sections, origin = new THREE.Vector3()) {
  const rings = sections.map(([x, half, bottom, top]) => {
    const cut = CHAMFER * Math.min(half, (top - bottom) / 2);
    return [[half, bottom + cut], [half, top - cut], [half - cut, top], [cut - half, top], [-half, top - cut],
            [-half, bottom + cut], [cut - half, bottom], [half - cut, bottom]]
      .map(([z, y]) => new THREE.Vector3(x, y, z).sub(origin));
  });
  const positions = [];
  const triangle = (a, b, c) => positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
  for (let ring = 0; ring + 1 < rings.length; ring++) {
    const [front, back] = [rings[ring], rings[ring + 1]];
    for (let corner = 0; corner < 8; corner++) {
      const next = (corner + 1) % 8;
      triangle(front[corner], back[corner], back[next]);
      triangle(front[corner], back[next], front[next]);
    }
  }
  // Close both ends.
  for (const [ring, flip] of [[rings[0], false], [rings[rings.length - 1], true]]) {
    for (let corner = 1; corner + 1 < 8; corner++) {
      if (flip) triangle(ring[0], ring[corner + 1], ring[corner]);
      else triangle(ring[0], ring[corner], ring[corner + 1]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(positions.length / 3 * 2), 2));
  geometry.computeVertexNormals();
  return geometry;
}

// The width, bottom and top of a shape at `x`, and how fast its width and top change along it.
function sectionAt(sections, x) {
  for (let index = 0; index + 1 < sections.length; index++) {
    const [x0, half0, bottom0, top0] = sections[index], [x1, half1, bottom1, top1] = sections[index + 1];
    if (x > x0 || x < x1) continue;
    const along = (x0 - x) / (x0 - x1);
    return { half: half0 + (half1 - half0) * along, bottom: bottom0 + (bottom1 - bottom0) * along, top: top0 + (top1 - top0) * along,
             widening: (half1 - half0) / (x1 - x0), rising: (top1 - top0) / (x1 - x0) };
  }
  throw new Error(`x ${x} is outside the shape`);
}

// A geometry moved into place: position, then rotation (Euler XYZ) and scale.
function placed(geometry, [x, y, z], rotation = [0, 0, 0], scale = [1, 1, 1]) {
  const matrix = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation)),
                                             new THREE.Vector3(...scale));
  return (geometry.index ? geometry.toNonIndexed() : geometry.clone()).applyMatrix4(matrix);
}

// Several placed pieces as one geometry, so they draw as one.
function merged(pieces) {
  const geometry = mergeGeometries(pieces.map(piece => {
    for (const name of Object.keys(piece.attributes)) if (!['position', 'normal', 'uv', 'color'].includes(name)) piece.deleteAttribute(name);
    return piece;
  }));
  if (!geometry) throw new Error('Car pieces could not be merged: their attributes differ');
  return geometry;
}

// A rod between two points, for wishbones and arms.
function rod(from, to, radius) {
  const length = from.distanceTo(to);
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 5).toNonIndexed();
  geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.clone().sub(from).normalize()));
  return geometry.translate(...from.clone().add(to).multiplyScalar(0.5).toArray());
}

// A decal plane lying on the side of a shape at `x`, turned to follow its taper.
function onSide(sections, x, side, [width, height]) {
  const at = sectionAt(sections, x);
  const facing = Math.atan2(-at.widening, side);
  return placed(new THREE.PlaneGeometry(width, height), [x, (at.bottom + at.top) / 2, side * (at.half + 0.006)], [0, facing, 0]);
}

// A decal plane lying on top of a shape at `x`, tilted to follow its slope, turned to read along the car.
function onTop(sections, x, z, [width, height], turned = false) {
  const at = sectionAt(sections, x);
  const geometry = new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);
  if (turned) geometry.rotateY(-Math.PI / 2);
  geometry.rotateZ(Math.atan(at.rising));
  return placed(geometry, [x, at.top + 0.006, z]);
}

// Every car has the same shape, so the geometry is built once and shared.
let shapes = null;
function carShapes() {
  if (shapes) return shapes;
  const box = (size, at, rotation) => placed(new THREE.BoxGeometry(...size), at, rotation);
  const sides = [-1, 1];

  // The halo: a hoop round the driver's head on a pillar in front of the cockpit.
  const hoop = new THREE.CatmullRomCurve3([[0.02, 0.4, 0.33], [0.4, 0.6, 0.3], [0.7, 0.65, 0], [0.4, 0.6, -0.3], [0.02, 0.4, -0.33]]
    .map(point => new THREE.Vector3(...point)));
  const halo = [new THREE.TubeGeometry(hoop, 16, 0.03, 5).toNonIndexed(), rod(new THREE.Vector3(0.95, 0.33, 0), new THREE.Vector3(0.7, 0.65, 0), 0.028)];

  // A shark fin along the top of the engine cover, its foot following the cover's slope.
  const finShape = new THREE.Shape([[-0.15, 0.6], [-0.9, 0.41], [-1.45, 0.26], [-1.45, 0.4], [-0.3, 0.63]].map(([x, y]) => new THREE.Vector2(x, y)));
  const fin = new THREE.ExtrudeGeometry(finShape, { depth: 0.02, bevelEnabled: false }).translate(-COVER_HINGE.x, -COVER_HINGE.y, -0.01);

  shapes = {
    bodyPaint: merged([loft(TUB), loft(SIDEPODS), ...sides.map(side => box([0.07, 0.06, 0.14], [0.74, 0.57, side * 0.47]))]), // and mirrors
    bodyCarbon: merged([
      box([3.9, 0.04, 1.3], [0, -0.135, 0]),                                    // floor, just under the bodywork
      box([2.6, 0.03, 0.3], [-0.1, -0.165, 0]),                                 // plank
      ...[-0.4, 0, 0.4].map(fin => box([0.6, 0.18, 0.03], [-1.75, -0.06, fin])), // diffuser fins
      box([0.08, 0.5, 0.08], [-1.65, 0.45, 0]),                                 // rear wing pillar
      ...halo,
      ...sides.map(side => rod(new THREE.Vector3(0.62, 0.3, side * 0.36), new THREE.Vector3(0.74, 0.55, side * 0.46), 0.015)), // mirror stalks
    ]),
    // The dark of the cockpit opening and the radiator inlets.
    bodyInterior: merged([
      placed(new THREE.PlaneGeometry(0.85, 0.5).rotateX(-Math.PI / 2), [0.35, 0.365, 0]),
      ...sides.map(side => placed(new THREE.PlaneGeometry(0.24, 0.16), [0.785, 0.045, side * 0.52], [0, Math.PI / 2, 0])),
      box([0.16, 0.1, 0.12], [-0.05, 0.6, 0]),                                  // airbox intake over the driver
    ]),
    cover: loft(COVER, COVER_HINGE),
    coverFin: fin,
    nose: loft(NOSE, NOSE_MOUNT),
    // Each half of the front wing: main plane and endplate (the flap is its own, as it moves with the setting).
    wingHalf: sides.map(side => merged([box([0.45, 0.05, 1.02], [0, 0, 0]), box([0.55, 0.24, 0.04], [0, 0.09, side * 0.5])])),
    frontFlap: new THREE.BoxGeometry(0.3, 0.04, 0.95),
    mainPlane: new THREE.BoxGeometry(0.5, 0.07, 1.55),
    rearFlap: new THREE.BoxGeometry(0.32, 0.05, 1.5),
    endplates: merged(sides.map(side => box([0.75, 0.6, 0.05], [0.05, -0.25, side * 0.78]))),
    brakeLight: new THREE.BoxGeometry(0.06, 0.1, 0.22),
    headlights: merged(sides.map(side => box([0.05, 0.06, 0.1], [2.62, 0.05, side * 0.13]))),
    flame: new THREE.ConeGeometry(0.12, 0.7, 8),
    // Upper and lower wishbones from the body to each hub, in the car's frame (they don't lean with the body).
    wishbones: merged([WHEELBASE / 2 + 0.05, -WHEELBASE / 2 + 0.05].flatMap(x => sides.flatMap(side => [[-0.1, 0], [0.1, 0.17]].flatMap(([atHub, atBody]) => {
      const hub = new THREE.Vector3(x, WHEEL_RADIUS + atHub, side * (TRACK_WIDTH / 2 - 0.2));
      return [-0.3, 0.3].map(spread => rod(new THREE.Vector3(x + spread, RIDE_HEIGHT + atBody, side * 0.2), hub, 0.018));
    })))),
    // The driver: shoulders and arms reaching to the wheel, and a helmet with a visor and a stripe.
    suit: merged([box([0.22, 0.12, 0.44], [0.2, 0.33, 0]), ...sides.map(side => rod(new THREE.Vector3(0.24, 0.36, side * 0.19), new THREE.Vector3(0.6, 0.42, side * 0.11), 0.035))]),
    helmet: new THREE.SphereGeometry(0.16, 12, 9).scale(1.12, 1, 1).translate(0, 0.1, 0).toNonIndexed(),
    visor: new THREE.SphereGeometry(0.163, 10, 4, Math.PI - 0.95, 1.9, 1.15, 0.38).scale(1.12, 1, 1).translate(0, 0.1, 0).toNonIndexed(),
    helmetStripe: new THREE.SphereGeometry(0.163, 12, 2, 0, Math.PI * 2, 0.55, 0.18).scale(1.12, 1, 1).translate(0, 0.1, 0).toNonIndexed(),
    steeringWheel: merged([box([0.03, 0.1, 0.26], [0, 0, 0]), ...sides.map(side => box([0.04, 0.12, 0.04], [0, 0, side * 0.13]))]),
    // The simple car, far away: one mesh in the paint and one for everything dark.
    simplePaint: merged([loft(TUB), loft(NOSE), loft(COVER), loft(SIDEPODS),
                         box([0.45, 0.05, 2.04], [2.55, -0.12, 0]), ...sides.map(side => box([0.75, 0.6, 0.05], [-1.85, 0.5, side * 0.78]))]).translate(0, RIDE_HEIGHT, 0),
    simpleDark: merged([box([3.9, 0.04, 1.3], [0, RIDE_HEIGHT - 0.135, 0]), box([0.5, 0.07, 1.55], [-1.9, RIDE_HEIGHT + 0.75, 0]),
      ...[WHEELBASE / 2 + 0.05, -WHEELBASE / 2 + 0.05].flatMap(x => sides.map(side =>
        placed(new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.38, 10), [x, WHEEL_RADIUS, side * TRACK_WIDTH / 2], [Math.PI / 2, 0, 0])))]),
  };
  return shapes;
}

// A wheel in two meshes, the rubber and the metal, each piece told apart by where it reads its texture (`u`) and,
// for the metal, its colour per vertex: one draw call each instead of five.
function tinted(geometry, u, color = null) {
  const count = geometry.attributes.position.count;
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(count * 2).map((_, index) => index % 2 ? 0.5 : u), 2));
  const rgb = new THREE.Color(color ?? '#ffffff');
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(count * 3).map((_, index) => [rgb.r, rgb.g, rgb.b][index % 3]), 3));
  return geometry;
}

// The tyre's section is flat across the tread with rounded shoulders, revolved round its axle. `side` is which side
// of the car the wheel is on, so the brake disc sits inboard.
let wheelShapes = null;
function wheelShape(front, side) {
  wheelShapes ??= {};
  const key = `${front ? 'front' : 'rear'}${side}`;
  if (wheelShapes[key]) return wheelShapes[key];
  const width = front ? 0.34 : 0.42, half = width / 2, shoulder = 0.06;
  const profile = [[0.22, -half], [0.31, -half], [WHEEL_RADIUS - 0.015, -half + 0.02], [WHEEL_RADIUS, -half + shoulder],
                   [WHEEL_RADIUS, half - shoulder], [WHEEL_RADIUS - 0.015, half - 0.02], [0.31, half], [0.22, half]]
    .map(([radius, along]) => new THREE.Vector2(radius, along));
  const faces = [-1, 1];
  const spokes = faces.flatMap(face => [0, 1, 2, 3, 4].map(spoke =>
    placed(new THREE.BoxGeometry(0.2, 0.035, 0.02), [0, 0, face * (half - 0.01)], [0, 0, spoke * Math.PI * 2 / 5]).translate(
      Math.cos(spoke * Math.PI * 2 / 5) * 0.1, Math.sin(spoke * Math.PI * 2 / 5) * 0.1, 0)));
  wheelShapes[key] = {
    // The tread reads the left of the tyre texture, the compound's stripe round each sidewall the right.
    rubber: merged([tinted(new THREE.LatheGeometry(profile, 20).rotateX(Math.PI / 2).toNonIndexed(), 0.25),
                    ...faces.map(face => tinted(placed(new THREE.TorusGeometry(WHEEL_RADIUS * 0.8, 0.03, 5, 24), [0, 0, face * (half + 0.003)]), 0.75))]),
    // The rim and its nuts, five dark spokes on each face so the spin reads at a glance, and the brake disc inboard,
    // the only part that glows (round, it looks the same spinning with the wheel).
    metal: merged([tinted(placed(new THREE.CylinderGeometry(0.225, 0.225, width - 0.02, 12), [0, 0, 0], [Math.PI / 2, 0, 0]), 0.25, '#b9bec7'),
                   ...faces.map(face => tinted(placed(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 6), [0, 0, face * half], [Math.PI / 2, 0, 0]), 0.25, '#b9bec7')),
                   ...spokes.map(spoke => tinted(spoke, 0.25, '#2b2b2b')),
                   tinted(placed(new THREE.CylinderGeometry(WHEEL_RADIUS * 0.5, WHEEL_RADIUS * 0.5, 0.05, 14), [0, 0, -side * 0.24], [Math.PI / 2, 0, 0]), 0.75, '#555555')]),
  };
  return wheelShapes[key];
}

// ---- Surfaces ------------------------------------------------------------------------------------------------

// A fine twill weave for the carbon parts, shared by every car.
let carbonWeave = null;
function weave() {
  if (carbonWeave) return carbonWeave;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 16;
  const paint = canvas.getContext('2d');
  for (let row = 0; row < 16; row += 2) for (let column = 0; column < 16; column += 2) {
    paint.fillStyle = ((row + column) / 2) % 4 < 2 ? '#26292e' : '#16181c';
    paint.fillRect(column, row, 2, 2);
  }
  carbonWeave = new THREE.CanvasTexture(canvas);
  carbonWeave.wrapS = carbonWeave.wrapT = THREE.RepeatWrapping;
  carbonWeave.repeat.set(6, 6);
  carbonWeave.magFilter = THREE.NearestFilter;
  carbonWeave.colorSpace = THREE.SRGBColorSpace;
  return carbonWeave;
}

// A texture of a row of flat colours, one per texel, for parts told apart by where they read it.
function pixels(colors) {
  const texture = new THREE.DataTexture(new Uint8Array(colors.flatMap(rgb => [...rgb, 255])), colors.length, 1);
  texture.magFilter = texture.minFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

// What every car has in common, made once.
let common = null;
function commonSurfaces() {
  common ??= {
    carbon: new THREE.MeshStandardMaterial({ map: weave(), roughness: 0.72, metalness: 0.1 }),
    // The cockpit's dark and the driver's suit, coloured per vertex: matte.
    cockpit: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85 }),
    // The helmet in the team's colour, its stripe and its visor, coloured per vertex under a gloss coat.
    helmet: new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.2, clearcoat: 1, clearcoatRoughness: 0.06 }),
    // Black on the left, white on the right: only the brake disc, on the right, glows.
    glowMask: pixels([[0, 0, 0], [255, 255, 255]]),
    dark: new THREE.MeshStandardMaterial({ color: '#1a1c20', roughness: 0.7 }),
    footprint: new THREE.MeshBasicMaterial({ map: footprintTexture(), transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4 }),
  };
  return common;
}

// The shade right under a car: soft round the body, darkest where the tyres press on the road. The sun's shadow
// can't draw this (it's too fine for its map), and without it a car looks as if it hovers.
function footprintTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const paint = canvas.getContext('2d');
  const soft = (x, y, radiusX, radiusY, darkness) => {
    const gradient = paint.createRadialGradient(0, 0, 0, 0, 0, 1);
    gradient.addColorStop(0, `rgba(0,0,0,${darkness})`);
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    paint.save();
    paint.translate(x, y);
    paint.scale(radiusX, radiusY);
    paint.fillStyle = gradient;
    paint.fillRect(-1, -1, 2, 2);
    paint.restore();
  };
  soft(128, 64, 120, 46, 0.42);
  // The tyres, at their places on the 6 m by 3 m footprint (see createCar).
  for (const x of [128 + (WHEELBASE / 2 + 0.05) / 6 * 256, 128 - (WHEELBASE / 2 - 0.05) / 6 * 256]) {
    for (const y of [64 - TRACK_WIDTH / 2 / 3 * 128, 64 + TRACK_WIDTH / 2 / 3 * 128]) soft(x, y, 16, 12, 0.75);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Which sponsor a car carries: always the same for the same driver.
function sponsorOf(name) {
  let hash = 0;
  for (const character of name) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return SPONSORS[hash % SPONSORS.length];
}

// A car's decals in one texture, so they share a material: the sponsor's name in the livery's accent colour along
// the top half, on a clear background so the paint shows round it, and the race number in a square of the paint's
// colour below. LIVERY says where each is, as [u from, v from, u to, v to].
const LIVERY = { sponsor: [0, 0.5, 1, 1], number: [0, 0, 0.25, 0.5] };
function livery(sponsorName, number, color, accent) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const paint = canvas.getContext('2d');
  paint.fillStyle = accent;
  paint.font = '900 44px "Titillium Web", sans-serif';
  paint.textAlign = 'center';
  paint.textBaseline = 'middle';
  paint.fillText(sponsorName, 128, 36, 244);
  if (number !== null) {
    paint.fillStyle = color;
    paint.fillRect(0, 64, 64, 64);
    paint.fillStyle = accent;
    paint.font = '900 46px "Titillium Web", sans-serif';
    paint.fillText(String(number), 32, 100);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return new THREE.MeshStandardMaterial({ map: texture, transparent: true, alphaTest: 0.3, roughness: 0.3, depthWrite: false,
                                          polygonOffset: true, polygonOffsetFactor: -2 });
}

// A decal plane reading its part of the livery texture.
function inRegion(geometry, [uFrom, vFrom, uTo, vTo]) {
  const uv = geometry.attributes.uv;
  for (let vertex = 0; vertex < uv.count; vertex++) uv.setXY(vertex, uFrom + uv.getX(vertex) * (uTo - uFrom), vFrom + uv.getY(vertex) * (vTo - vFrom));
  return geometry;
}

// The livery's second colour: white on dark paint, near-black on light, so the stripes always read.
function accentFor(color) {
  const { l } = new THREE.Color(color).getHSL({});
  return l < 0.5 ? '#f4f4f2' : '#16181d';
}

// ---- Building ------------------------------------------------------------------------------------------------

function mesh(geometry, material, parent, { shadow = false, at = null } = {}) {
  const part = new THREE.Mesh(geometry, material);
  part.castShadow = shadow;
  if (at) part.position.copy(at);
  parent.add(part);
  return part;
}

// One wheel: the rubber (tread and the compound's stripe) and the metal (rim, spokes and brake disc). Also used for
// the loose wheels in a pit stop. A car's four wheels share their two materials, passed in as `surfaces`.
export function createWheel(compound = 'medium', front = true, surfaces = null, side = 1) {
  const wheel = new THREE.Group();
  const shape = wheelShape(front, side);
  const { rubber, metal } = surfaces ?? wheelSurfaces();
  if (!surfaces) paintTyres(rubber.map, compound, 0);
  mesh(shape.rubber, rubber, wheel, { shadow: !surfaces });
  mesh(shape.metal, metal, wheel);
  return wheel;
}

// A car's wheel materials. The tyre's colours (tread, stripe) are a two-texel texture, repainted as the tyres wear
// or change compound; the metal's brake disc glows through the glow mask.
function wheelSurfaces() {
  const tyreColours = pixels([[0, 0, 0], [0, 0, 0]]);
  tyreColours.colorSpace = THREE.SRGBColorSpace;
  return {
    rubber: new THREE.MeshStandardMaterial({ map: tyreColours, roughness: TREAD_ROUGHNESS.fresh }),
    metal: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.7, emissive: '#ff5a00', emissiveMap: commonSurfaces().glowMask, emissiveIntensity: 0 }),
  };
}

// The tread goes from black to a greyer grain as it wears, and the stripe greys from the compound's colour.
function paintTyres(texture, compound, wear) {
  const tread = FRESH_TREAD.clone().lerp(WORN_TREAD, wear), stripe = new THREE.Color(COMPOUND_COLORS[compound] ?? COMPOUND_COLORS.medium).lerp(WORN, wear * 0.8);
  [tread, stripe].forEach((color, index) => {
    const { r, g, b } = color.getRGB({}, THREE.SRGBColorSpace);
    texture.image.data.set([r, g, b].map(value => Math.round(value * 255)).concat(255), index * 4);
  });
  texture.needsUpdate = true;
}

export function createCar(color, name, number = null) {
  const shape = carShapes();
  const surfaces = commonSurfaces();
  const car = new THREE.Group();
  car.name = name;
  const baseColor = new THREE.Color(color);
  const accentColor = accentFor(color);
  // Paint under a clear coat: the colour stays satin, the coat on top catches the sky's reflections sharply.
  const paint = new THREE.MeshPhysicalMaterial({ color, roughness: 0.45, metalness: 0.15, clearcoat: 0.7, clearcoatRoughness: 0.1, envMapIntensity: 0.7 });

  // The detailed car: the body leans and pitches on its own, while the wheels stay planted.
  const detailed = new THREE.Group();
  car.add(detailed);
  const body = new THREE.Group();
  body.position.y = RIDE_HEIGHT;
  detailed.add(body);
  mesh(shape.bodyPaint, paint, body);
  mesh(shape.bodyCarbon, surfaces.carbon, body);
  // The driver, sitting in the cockpit, wears the team's colour (which doesn't darken with the car's damage): the
  // head turns and leans on its neck, the steering wheel on its column.
  mesh(merged([tinted(shape.bodyInterior.clone(), 0.5, '#0d0e11'), tinted(shape.suit.clone(), 0.5, color)]), surfaces.cockpit, body);
  const head = new THREE.Group();
  head.position.set(0.3, 0.47, 0);
  body.add(head);
  mesh(merged([tinted(shape.helmet.clone(), 0.5, color), tinted(shape.visor.clone(), 0.5, '#0b0c10'), tinted(shape.helmetStripe.clone(), 0.5, accentColor)]),
       surfaces.helmet, head);
  const steeringWheel = mesh(shape.steeringWheel, surfaces.carbon, body, { at: new THREE.Vector3(0.64, 0.44, 0) });

  // Engine cover, hinged at its front so the crew can open it.
  const cover = new THREE.Group();
  cover.position.copy(COVER_HINGE);
  body.add(cover);
  mesh(shape.cover, paint, cover);
  mesh(shape.coverFin, paint, cover);

  // Nose and front wing: one assembly that droops with damage and comes off for repairs.
  const nose = new THREE.Group();
  nose.position.copy(NOSE_MOUNT);
  body.add(nose);
  mesh(shape.nose, paint, nose);
  // The front wing is two halves (plane, flap and endplate each), so a hit can break one off and leave the other.
  const wingHalves = [-1, 1].map((side, index) => {
    const half = new THREE.Group();
    half.position.set(0.65, -0.17, side * 0.52);
    nose.add(half);
    mesh(shape.wingHalf[index], paint, half);
    return half;
  });
  const frontFlaps = wingHalves.map(half => mesh(shape.frontFlap, surfaces.carbon, half, { at: new THREE.Vector3(-0.2, 0.07, 0) }));
  const firstToGo = Math.random() < 0.5 ? 0 : 1;

  // Rear wing: main plane and flap tilt with the wing setting; the whole assembly lifts off in a stop.
  const rearWing = new THREE.Group();
  rearWing.position.copy(REAR_WING_MOUNT);
  body.add(rearWing);
  const mainPlane = mesh(shape.mainPlane, surfaces.carbon, rearWing);
  const flap = mesh(shape.rearFlap, paint, rearWing, { at: new THREE.Vector3(-0.12, 0.17, 0) });
  mesh(shape.endplates, paint, rearWing);

  // Livery, from one texture per car (see livery): the number on the nose and both sidepods, the sponsor along the
  // sidepods, the cover and the endplates.
  const [sponsorName] = sponsorOf(name);
  const decals = livery(sponsorName, number, color, accentColor);
  const sides = [-1, 1];
  const sponsor = geometry => inRegion(geometry, LIVERY.sponsor), raceNumber = geometry => inRegion(geometry, LIVERY.number);
  mesh(merged([...sides.map(side => sponsor(onTop(SIDEPODS, -0.2, side * 0.5, [0.7, 0.17]))),
               ...(number === null ? [] : sides.map(side => raceNumber(onSide(SIDEPODS, -0.15, side, [0.2, 0.2]))))]), decals, body);
  mesh(merged(sides.map(side => sponsor(onSide(COVER, -0.55, side, [0.75, 0.18]).translate(-COVER_HINGE.x, -COVER_HINGE.y, 0)))), decals, cover);
  mesh(merged(sides.map(side => sponsor(placed(new THREE.PlaneGeometry(0.7, 0.18), [0.05, -0.2, side * 0.81], [0, side > 0 ? 0 : Math.PI, 0])))), decals, rearWing);
  if (number !== null) mesh(raceNumber(onTop(NOSE, 1.75, 0, [0.3, 0.3], true).translate(-NOSE_MOUNT.x, -NOSE_MOUNT.y, 0)), decals, nose);

  const brakeLight = mesh(shape.brakeLight, new THREE.MeshStandardMaterial({ color: '#400', emissive: '#ff1a1a', emissiveIntensity: 0 }), body,
                          { at: new THREE.Vector3(-2.07, 0.05, 0) });
  // Headlights for racing at night, set into the nose.
  const headlightGlow = new THREE.MeshStandardMaterial({ color: '#333', emissive: '#fff4d6', emissiveIntensity: 0 });
  mesh(shape.headlights, headlightGlow, body);
  const flame = mesh(shape.flame, new THREE.MeshBasicMaterial({ color: '#ffb347', transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }),
                     body, { at: new THREE.Vector3(-2.0, 0.2, 0) });
  flame.rotation.z = Math.PI / 2;
  flame.visible = false;

  // The wishbones hang off the chassis, which the jacks lift with the body.
  const chassis = new THREE.Group();
  detailed.add(chassis);
  mesh(shape.wishbones, surfaces.carbon, chassis);

  const wheelParts = wheelSurfaces();
  const wheels = [], hubs = [], steering = [];
  for (const [x, front] of [[WHEELBASE / 2 + 0.05, true], [-WHEELBASE / 2 + 0.05, false]]) {
    for (const side of sides) {
      const hub = new THREE.Group();
      hub.position.set(x, WHEEL_RADIUS, side * TRACK_WIDTH / 2);
      const wheel = createWheel('medium', front, wheelParts, side);
      hub.add(wheel);
      detailed.add(hub);
      wheels.push(wheel);
      hubs.push(hub);
      if (front) steering.push(hub);
    }
  }

  // The simple car, for when it's too small on screen for any of that to show, and always for the car's shadow.
  const simple = new THREE.Group();
  car.add(simple);
  const simpleParts = [mesh(shape.simplePaint, paint, simple, { shadow: true }), mesh(shape.simpleDark, surfaces.dark, simple, { shadow: true })];

  // The shade under the car, so it sits on the ground whatever the sun's shadow does.
  const footprint = mesh(new THREE.PlaneGeometry(6, 3).rotateX(-Math.PI / 2), surfaces.footprint, car, { at: new THREE.Vector3(0, 0.035, 0) });
  footprint.renderOrder = -1;

  // A gold chevron hovering over the car the camera follows, so you can find it in the pack.
  const chevron = mesh(new THREE.ConeGeometry(0.7, 1.3, 4), new THREE.MeshStandardMaterial({ color: '#ffd23f', emissive: '#ffb000', emissiveIntensity: 0.6, roughness: 0.4 }), car);
  chevron.rotation.x = Math.PI;
  chevron.visible = false;

  car.userData = {
    detailed, simpleParts, body, chassis, head, steeringWheel, wheels, hubs, steering, wheelParts, tyresPainted: null, brakeLight, headlightGlow, flame,
    chevron, cover, nose, wingHalves, frontFlaps, rearWing, mainPlane, flap,
    // Set from outside each frame: lights on (night) and a wet track (the rear light flashes in the rain).
    lightsOn: false, raining: false,
    // Parts that come off as damage builds, in the order they go, with the damage that takes each one.
    breakable: [{ part: wingHalves[firstToGo], at: 0.15 }, { part: wingHalves[1 - firstToGo], at: 0.35 }],
    splayedWheel: Math.random() < 0.5 ? 0 : 1,
    justLost: [],
    paint, baseColor, compound: 'medium',
    spin: 0, lean: 0, pitch: 0, lastSpeed: 0, brakeHeat: 0, ground: ROAD_HEIGHT,
    // Set by the pit crew while it works on the car.
    lift: 0, rearWingLift: 0, coverOpen: 0, noseOff: 0,
  };
  setCarDetail(car, true);
  return car;
}

// Draws a car in full, or as the simple car when it's too small on screen for its details to show. Either way the
// simple car casts the shadow; drawn in full, it's only in the shadow.
// What every car shares (the shapes, the common surfaces), which a car going away must leave for the others.
function sharedResources() {
  const shared = new Set([carbonWeave]);
  for (const geometry of Object.values(shapes ?? {}).flat()) shared.add(geometry);
  for (const parts of Object.values(wheelShapes ?? {})) for (const geometry of Object.values(parts)) shared.add(geometry);
  for (const surface of Object.values(common ?? {})) shared.add(surface).add(surface.map);
  return shared;
}

// Frees what a car made for itself (its paint, livery, tyre colours, lights, merged pieces) once it's gone: the GPU
// doesn't let them go by itself, and every race left a field's worth behind.
export function disposeCar(car) {
  const shared = sharedResources();
  car.removeFromParent();
  car.traverse(object => {
    if (object.geometry && !shared.has(object.geometry)) object.geometry.dispose();
    const materials = [object.material].flat().filter(Boolean);
    for (const material of materials) {
      if (shared.has(material)) continue;
      for (const texture of [material.map, material.emissiveMap]) if (texture && !shared.has(texture)) texture.dispose();
      material.dispose();
    }
  });
}

export function setCarDetail(car, detailed) {
  car.userData.detailed.visible = detailed;
  for (const part of car.userData.simpleParts) part.layers.set(detailed ? SHADOW_ONLY : 0);
}

// Poses a car from interpolated sim values for this render frame. `seconds` is the real time since the last
// render frame and `simSeconds` how much race time it covered, so wheels spin at the race's pace.
export function poseCar(car, state, seconds, simSeconds, followed, time) {
  const data = car.userData;
  // On the tarmac the tyres sit on its top; off it they settle a little into the grass or gravel.
  data.ground += ((state.offTrack ? OFF_ROAD_HEIGHT : ROAD_HEIGHT) - data.ground) * (1 - Math.exp(-seconds * 10));
  car.position.copy(toWorld(state.x, state.y)).setY(data.ground);
  car.rotation.y = state.heading;

  data.chevron.visible = followed;
  if (followed) {
    data.chevron.position.y = 3.2 + Math.sin(time * 3) * 0.35;
    data.chevron.rotation.y = time * 1.5;
  }
  // Damage darkens the paint, which the simple car shares, so it's set even when only that one shows.
  data.paint.color.copy(data.baseColor).multiplyScalar(1 - (state.damage ?? 0) * 0.5);
  if (!data.detailed.visible) return;

  data.spin -= state.speed * simSeconds / WHEEL_RADIUS;
  for (const wheel of data.wheels) wheel.rotation.z = data.spin;
  for (const hub of data.steering) hub.rotation.y = state.steer * 0.32;

  // Tyres: compound colour, greying as they wear, and the tread grains from matte black to a greyer sheen.
  if (state.compound) data.compound = state.compound;
  const wear = Math.min(1, state.tyreWear ?? 0);
  // Repainted only when they've changed enough to see.
  const tyres = `${data.compound}|${wear.toFixed(2)}`;
  if (tyres !== data.tyresPainted) {
    paintTyres(data.wheelParts.rubber.map, data.compound, wear);
    data.wheelParts.rubber.roughness = TREAD_ROUGHNESS.fresh + (TREAD_ROUGHNESS.worn - TREAD_ROUGHNESS.fresh) * wear;
    data.tyresPainted = tyres;
  }

  // Brake discs glow with the race's own brake temperature: dull red past about 400°C, bright orange near 1000°C.
  const heat = Math.min(1, Math.max(0, ((state.brakeTemp ?? 0.1) - 0.3) / 0.6));
  data.brakeHeat += (heat - data.brakeHeat) * (1 - Math.exp(-seconds * 6));
  data.wheelParts.metal.emissiveIntensity = data.brakeHeat * 3;
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
  data.body.position.y = RIDE_HEIGHT - (state.fuel ?? 0) * 0.05 + data.lift;
  data.chassis.position.y = data.lift;
  for (const hub of data.hubs) hub.position.y = WHEEL_RADIUS + data.lift;

  // The driver: the head pushed the way the body leans and forward under braking, looking into the turn.
  data.head.rotation.set(data.lean * HEAD.roll, state.steer * HEAD.look, data.pitch * HEAD.pitch, 'YXZ');
  data.steeringWheel.rotation.x = -state.steer * STEERING_LOCK;

  // Wing: steeper planes for more wing, the extra flap only on higher settings; lifted off by the crew.
  const wing = state.wing ?? 0.5;
  data.mainPlane.rotation.z = -0.05 - wing * 0.3;
  data.flap.rotation.z = -0.2 - wing * 0.55;
  data.flap.visible = wing > 0.25;
  data.rearWing.position.y = REAR_WING_MOUNT.y + data.rearWingLift;
  for (const flap of data.frontFlaps) flap.rotation.z = -wing * 0.4;

  // Damage: the front wing droops, then breaks; the rear wing hangs off, a wheel splays. The nose comes away
  // while the crew swaps it.
  // Where the damage is shows where the car was hit: the front wing from hits to the nose, a splayed wheel from
  // hits to the side, and the rest (darker, rear wing hanging) from everything.
  const damage = state.damage ?? 0, wingDamage = state.wingDamage ?? damage, suspension = state.suspensionDamage ?? 0;
  shedParts(data, wingDamage);
  data.wingHalves.forEach((half, side) => {
    half.rotation.x = (side ? -1 : 1) * wingDamage * 0.5;
    half.position.y = -0.17 - wingDamage * 0.15;
  });
  const hanging = damage > 0.5;
  data.rearWing.rotation.x = hanging ? 0.55 : 0;
  data.rearWing.rotation.z = hanging ? 0.35 : 0;
  data.hubs[data.splayedWheel].rotation.x = suspension * 0.45;
  data.nose.position.x = NOSE_MOUNT.x + data.noseOff * 2.5;
  data.nose.position.y = NOSE_MOUNT.y + data.noseOff * 0.6;
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
    point.y = car.position.y;
    return point;
  });
}
