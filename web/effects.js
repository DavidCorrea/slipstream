// Things that come and go: tyre smoke, dust off the grass, sparks when cars touch, confetti at the flag; and the
// race's history, which stays on the circuit until the next one: skid marks where cars slid, lock-up streaks
// where they stood on the brakes, ruts where they ran onto the grass or gravel, and the bits that came off in
// crashes. Particles are one pool per blending mode, updated on the CPU.
import * as THREE from 'three';

const vertexShader = `
  attribute float size;
  attribute float alpha;
  attribute vec3 tint;
  uniform float pixelsPerMetre;
  varying float vAlpha;
  varying vec3 vTint;
  void main() {
    vAlpha = alpha;
    vTint = tint;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(1.0, size * pixelsPerMetre);
  }
`;
const fragmentShader = `
  varying float vAlpha;
  varying vec3 vTint;
  uniform float softness;
  void main() {
    float distance = length(gl_PointCoord - 0.5);
    float edge = 1.0 - smoothstep(0.5 - softness, 0.5, distance);
    if (edge <= 0.0) discard;
    gl_FragColor = vec4(vTint, vAlpha * edge);
  }
`;

class ParticlePool {
  constructor(scene, capacity, blending, softness) {
    this.capacity = capacity;
    this.next = 0;
    this.hadAlive = false;
    this.particles = Array.from({ length: capacity }, () => ({ life: 0, age: 0, position: new THREE.Vector3(), velocity: new THREE.Vector3() }));
    const geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(capacity * 3);
    this.sizes = new Float32Array(capacity);
    this.alphas = new Float32Array(capacity);
    this.tints = new Float32Array(capacity * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('alpha', new THREE.BufferAttribute(this.alphas, 1).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('tint', new THREE.BufferAttribute(this.tints, 3).setUsage(THREE.DynamicDrawUsage));
    this.material = new THREE.ShaderMaterial({
      vertexShader, fragmentShader, transparent: true, depthWrite: false, blending,
      uniforms: { pixelsPerMetre: { value: 4 }, softness: { value: softness } },
    });
    this.points = new THREE.Points(geometry, this.material);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }

  emit({ position, velocity, color, size = 1, grow = 0, life = 1, gravity = 0, drag = 0, opacity = 1 }) {
    const particle = this.particles[this.next];
    this.next = (this.next + 1) % this.capacity;
    Object.assign(particle, { age: 0, life, size, grow, gravity, drag, opacity, color: color.clone() });
    particle.position.copy(position);
    particle.velocity.copy(velocity);
  }

  update(seconds, pixelsPerMetre) {
    this.material.uniforms.pixelsPerMetre.value = pixelsPerMetre;
    // A pool that was empty last frame and still is has nothing new to send to the GPU.
    let anyAlive = false;
    this.particles.forEach((particle, index) => {
      if (particle.age >= particle.life) { this.alphas[index] = 0; return; }
      anyAlive = true;
      particle.age += seconds;
      particle.velocity.y -= particle.gravity * seconds;
      particle.velocity.multiplyScalar(Math.exp(-particle.drag * seconds));
      particle.position.addScaledVector(particle.velocity, seconds);
      if (particle.position.y < 0.05) { particle.position.y = 0.05; particle.velocity.y *= -0.3; }
      const fade = 1 - particle.age / particle.life;
      particle.position.toArray(this.positions, index * 3);
      this.sizes[index] = particle.size + particle.grow * particle.age;
      this.alphas[index] = particle.opacity * fade * Math.min(1, particle.age * 12);
      particle.color.toArray(this.tints, index * 3);
    });
    if (!anyAlive && !this.hadAlive) return;
    this.hadAlive = anyAlive;
    for (const name of ['position', 'size', 'alpha', 'tint']) this.points.geometry.attributes[name].needsUpdate = true;
  }

  dispose() {
    this.points.parent?.remove(this.points);
    this.points.geometry.dispose();
    this.material.dispose();
  }
}

// Dark quads laid down behind tyres, oldest overwritten first.
class SkidMarks {
  constructor(scene, { capacity = 6000, color = '#111', opacity = 0.42, height = 0.115 } = {}) {
    this.height = height;
    this.capacity = capacity;
    this.next = 0;
    this.count = 0;
    this.positions = new Float32Array(capacity * 4 * 3);
    const indices = new Uint32Array(capacity * 6);
    for (let quad = 0; quad < capacity; quad++) {
      const base = quad * 4;
      indices.set([base, base + 1, base + 2, base + 1, base + 3, base + 2], quad * 6);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
    }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    scene.add(this.mesh);
  }

  add(from, to, width = 0.32) {
    const along = new THREE.Vector3().subVectors(to, from);
    if (along.lengthSq() < 0.01 || along.lengthSq() > 64) return;
    const across = new THREE.Vector3(-along.z, 0, along.x).normalize().multiplyScalar(width / 2);
    const corners = [from.clone().add(across), from.clone().sub(across), to.clone().add(across), to.clone().sub(across)];
    corners.forEach((corner, index) => this.positions.set([corner.x, this.height, corner.z], (this.next * 4 + index) * 3));
    this.next = (this.next + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.setDrawRange(0, this.count * 6);
  }

  dispose() {
    this.mesh.parent?.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

// Bits of bodywork from crashes: they fly, tumble, land and stay where they fell for the rest of the race.
class Debris {
  constructor(scene, capacity = 500) {
    this.capacity = capacity;
    this.next = 0;
    this.count = 0;
    this.pieces = Array.from({ length: capacity }, () => ({
      position: new THREE.Vector3(), velocity: new THREE.Vector3(), spin: new THREE.Vector3(), rotation: new THREE.Euler(), scale: new THREE.Vector3(), flying: false,
    }));
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.2 }), capacity);
    this.mesh.count = 0;
    this.mesh.castShadow = true;
    this.mesh.frustumCulled = false;
    this.matrix = new THREE.Matrix4();
    this.quaternion = new THREE.Quaternion();
    this.keys = new Map();
    scene.add(this.mesh);
  }

  // Takes away every keyed piece not in `keep` (it was run over): shrunk to nothing where it lay.
  keepOnly(keep) {
    for (const [key, index] of this.keys) {
      if (keep.has(key)) continue;
      this.pieces[index].scale.setScalar(0);
      this.pieces[index].flying = false;
      this.place(index);
      this.keys.delete(key);
      this.pieces[index].key = null;
    }
  }

  // `key` names a piece the race knows about, so it can be taken away again when a car runs over it.
  add(position, velocity, color, size, key = null) {
    const index = this.next, piece = this.pieces[index];
    if (piece.key) this.keys.delete(piece.key);
    piece.key = key;
    if (key) this.keys.set(key, index);
    this.next = (this.next + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
    piece.position.copy(position);
    piece.velocity.copy(velocity);
    piece.spin.set(Math.random() * 12 - 6, Math.random() * 12 - 6, Math.random() * 12 - 6);
    piece.rotation.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
    piece.scale.set(size * (0.6 + Math.random()), size * 0.25, size * (0.4 + Math.random() * 0.8));
    piece.flying = true;
    this.mesh.setColorAt(index, color);
    this.mesh.instanceColor.needsUpdate = true;
    this.mesh.count = this.count;
    this.place(index);
  }

  place(index) {
    const piece = this.pieces[index];
    this.quaternion.setFromEuler(piece.rotation);
    this.matrix.compose(piece.position, this.quaternion, piece.scale);
    this.mesh.setMatrixAt(index, this.matrix);
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  update(seconds) {
    this.pieces.forEach((piece, index) => {
      if (!piece.flying) return;
      piece.velocity.y -= 14 * seconds;
      piece.position.addScaledVector(piece.velocity, seconds);
      piece.rotation.x += piece.spin.x * seconds;
      piece.rotation.y += piece.spin.y * seconds;
      piece.rotation.z += piece.spin.z * seconds;
      const resting = piece.scale.y / 2 + 0.1;
      if (piece.position.y <= resting) {
        // Bounce once or twice, skid to a stop, then lie flat where it fell.
        piece.position.y = resting;
        piece.velocity.y *= -0.3;
        piece.velocity.x *= 0.6;
        piece.velocity.z *= 0.6;
        piece.spin.multiplyScalar(0.5);
        if (piece.velocity.lengthSq() < 0.5) {
          piece.flying = false;
          piece.rotation.set(0, piece.rotation.y, 0);
        }
      }
      this.place(index);
    });
  }

  clear() {
    this.keys.clear();
    this.count = this.next = 0;
    this.mesh.count = 0;
    this.pieces.forEach(piece => { piece.flying = false; });
  }

  dispose() {
    this.mesh.parent?.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

const SMOKE = new THREE.Color('#e6e6e6'), DUST = new THREE.Color('#b8956a'), GRASS = new THREE.Color('#6c9b3f');
const DAMAGE_SMOKE = new THREE.Color('#3a3a3a'), SPRAY = new THREE.Color('#d9e2ea'), CARBON = new THREE.Color('#1c1f24');
const WIND = new THREE.Color('#dbeeff');
const SPARK_IMPULSE = 0.6;   // m/s of closing speed a touch needs to throw sparks
// Braking this hard at this speed leaves streaks. (Trained drivers rarely press past 0.85, so a higher mark was
// never reached.)
const LOCKUP_BRAKE = 0.55, LOCKUP_SPEED = 15;
const SPARK = [new THREE.Color('#ffd23f'), new THREE.Color('#ff8a1a'), new THREE.Color('#fff2b0')];
const CONFETTI = ['#e63946', '#ffd23f', '#2a9d8f', '#457b9d', '#f4a261', '#9b5de5', '#ffffff'].map(color => new THREE.Color(color));

export function createEffects(scene) {
  const soft = new ParticlePool(scene, 2500, THREE.NormalBlending, 0.35);
  const bright = new ParticlePool(scene, 1500, THREE.AdditiveBlending, 0.2);
  const marks = new SkidMarks(scene);
  // Ruts sit just above the grass and gravel, under the tarmac, so they only show off the track.
  const ruts = new SkidMarks(scene, { capacity: 5000, color: '#3b2c17', opacity: 0.55, height: 0.075 });
  const debris = new Debris(scene);
  const lastPatch = new Map(), lastBrakePatch = new Map(), lastRut = new Map();
  const jitter = (amount) => (Math.random() * 2 - 1) * amount;

  return {
    // Called every render frame for each car with its wheels' ground positions and what it's doing.
    car(id, patches, state, velocity, seconds, wetness = 0) {
      const rear = patches.slice(2);
      const previous = lastPatch.get(id);
      if (state.sliding && state.speed > 6 && previous) rear.forEach((patch, wheel) => marks.add(previous[wheel], patch));
      lastPatch.set(id, state.sliding ? rear.map(patch => patch.clone()) : null);

      // Lock-ups: all four tyres, faint, wherever a car stands on the brakes at speed.
      const lockingUp = state.brake > LOCKUP_BRAKE && state.speed > LOCKUP_SPEED && !state.offTrack;
      const previousBrake = lastBrakePatch.get(id);
      if (lockingUp && previousBrake) patches.forEach((patch, wheel) => marks.add(previousBrake[wheel], patch, 0.24));
      lastBrakePatch.set(id, lockingUp ? patches.map(patch => patch.clone()) : null);

      // Ruts torn into the grass and gravel by a car off the track.
      const rutting = state.offTrack && state.speed > 3 && !state.pit;
      const previousRut = lastRut.get(id);
      if (rutting && previousRut) patches.forEach((patch, wheel) => ruts.add(previousRut[wheel], patch, 0.4));
      lastRut.set(id, rutting ? patches.map(patch => patch.clone()) : null);

      const rate = seconds * 60;
      if (state.sliding && state.speed > 6) {
        for (const patch of rear) if (Math.random() < 0.7 * rate) {
          soft.emit({
            position: patch.clone().add(new THREE.Vector3(jitter(0.3), 0.3, jitter(0.3))),
            velocity: velocity.clone().multiplyScalar(0.15).add(new THREE.Vector3(jitter(1), 0.8 + Math.random(), jitter(1))),
            color: SMOKE, size: 1.0, grow: 3.2, life: 1.4 + Math.random(), drag: 1.5, opacity: 0.55,
          });
        }
      }
      // On a wet track every car throws up a plume of spray behind it, more the faster and the wetter.
      if (wetness > 0.15 && state.speed > 12 && !state.pit) {
        const amount = wetness * Math.min(state.speed, 60) / 60;
        for (const patch of rear) if (Math.random() < amount * 1.2 * rate) {
          soft.emit({
            position: patch.clone().add(new THREE.Vector3(jitter(0.4), 0.4, jitter(0.4))),
            velocity: velocity.clone().multiplyScalar(0.35).add(new THREE.Vector3(jitter(1.5), 1 + Math.random() * 1.5, jitter(1.5))),
            color: SPRAY, size: 1.2, grow: 4.5, life: 0.7 + Math.random() * 0.4, drag: 2.2, opacity: 0.32,
          });
        }
      }
      // In another car's tow the air ahead streams past in lines.
      if ((state.draft ?? 0) > 0.3 && state.speed > 30) {
        for (let streak = 0; streak < 2; streak++) if (Math.random() < state.draft * 0.9 * rate) {
          const ahead = velocity.clone().normalize();
          const side = new THREE.Vector3(-ahead.z, 0, ahead.x).multiplyScalar(jitter(1.4));
          bright.emit({
            position: patches[0].clone().add(patches[1]).multiplyScalar(0.5).addScaledVector(ahead, 3 + Math.random() * 8).add(side).setY(0.6 + Math.random() * 0.8),
            velocity: velocity.clone().multiplyScalar(-0.15),
            color: WIND, size: 0.12, life: 0.25, opacity: 0.45 * state.draft,
          });
        }
      }
      // A puncture: the rim grinds on the tarmac.
      if (state.punctured && state.speed > 8 && Math.random() < 0.5 * rate) {
        bright.emit({
          position: patches[3].clone().add(new THREE.Vector3(0, 0.15, 0)),
          velocity: velocity.clone().multiplyScalar(-0.2).add(new THREE.Vector3(jitter(2), 1 + Math.random() * 2, jitter(2))),
          color: SPARK[0], size: 0.16, life: 0.3, gravity: 18, drag: 2,
        });
      }
      // A blown engine: flames at first, then smoke for as long as the car sits there.
      if (state.retired) {
        const middle = patches[2].clone().add(patches[3]).multiplyScalar(0.5);
        if (Math.random() < 0.8 * rate) {
          soft.emit({
            position: middle.clone().add(new THREE.Vector3(jitter(0.4), 1, jitter(0.4))),
            velocity: new THREE.Vector3(jitter(0.6), 2 + Math.random(), jitter(0.6)),
            color: DAMAGE_SMOKE, size: 1.2, grow: 3.5, life: 2.5, drag: 0.6, opacity: 0.55,
          });
        }
      }
      // A badly damaged car trails dark smoke from its bodywork.
      if (state.damage > 0.35 && Math.random() < (state.damage - 0.3) * 0.8 * rate) {
        const middle = patches[0].clone().add(patches[3]).multiplyScalar(0.5);
        soft.emit({
          position: middle.add(new THREE.Vector3(jitter(0.4), 0.8, jitter(0.4))),
          velocity: velocity.clone().multiplyScalar(0.3).add(new THREE.Vector3(jitter(0.5), 1.2, jitter(0.5))),
          color: DAMAGE_SMOKE, size: 0.8, grow: 2.5, life: 1.2, drag: 1.2, opacity: 0.45,
        });
      }
      if (state.offTrack && state.speed > 4 && !state.pit) {
        for (const patch of patches) if (Math.random() < 0.35 * rate) {
          soft.emit({
            position: patch.clone().add(new THREE.Vector3(0, 0.2, 0)),
            velocity: velocity.clone().multiplyScalar(0.25).add(new THREE.Vector3(jitter(1.5), 1.2 + Math.random() * 1.5, jitter(1.5))),
            color: Math.random() < 0.6 ? DUST : GRASS, size: 0.7, grow: 2.2, life: 0.9 + Math.random() * 0.5, gravity: 2, drag: 1.2, opacity: 0.7,
          });
        }
      }
    },

    // Debris the race left on the track (see race.py): each piece hops from where the hit happened to where the
    // race says it landed. `pieces` are [x, y] in world coordinates; `from` is the hit.
    scatter(from, pieces, colors) {
      pieces.forEach(([x, z, key], index) => {
        const landing = new THREE.Vector3(x, 0, z);
        const flight = landing.clone().sub(from).setY(0);
        debris.add(from.clone().setY(0.6), flight.multiplyScalar(1.4).add(new THREE.Vector3(0, 5, 0)),
          index % 3 ? CARBON : new THREE.Color(colors[index % colors.length]), 0.35, key);
      });
    },

    // Every piece of debris the race still has; the rest were run over.
    keepDebris(keys) {
      debris.keepOnly(keys);
    },

    // A blown engine's first moment: a burst of flame.
    blowUp(point) {
      for (let flame = 0; flame < 30; flame++) {
        bright.emit({
          position: point.clone().add(new THREE.Vector3(jitter(0.5), 0.8, jitter(0.5))),
          velocity: new THREE.Vector3(jitter(3), 3 + Math.random() * 4, jitter(3)),
          color: SPARK[flame % 2 ? 1 : 0], size: 0.5, grow: 1, life: 0.6 + Math.random() * 0.4, drag: 2,
        });
      }
    },

    // Bits flying off a car: carbon and its own paint. Also used when a damaged car loses a part.
    shed(point, velocity, color, pieces = 3, size = 0.35) {
      const paint = new THREE.Color(color);
      for (let piece = 0; piece < pieces; piece++) {
        debris.add(point.clone().add(new THREE.Vector3(jitter(0.5), 0.5, jitter(0.5))),
          velocity.clone().multiplyScalar(0.5).add(new THREE.Vector3(jitter(5), 3 + Math.random() * 4, jitter(5))),
          piece % 3 ? CARBON : paint, size);
      }
    },

    contact(point, impulse) {
      // Cars nose to tail in a pack touch nearly every tick; only a real knock throws sparks.
      if (impulse < SPARK_IMPULSE) return;
      const sparks = Math.min(40, 8 + impulse * 4);
      for (let spark = 0; spark < sparks; spark++) {
        const angle = Math.random() * Math.PI * 2, speed = 6 + Math.random() * 12;
        bright.emit({
          position: point.clone().add(new THREE.Vector3(0, 0.4, 0)),
          velocity: new THREE.Vector3(Math.cos(angle) * speed, 2 + Math.random() * 5, Math.sin(angle) * speed),
          color: SPARK[spark % SPARK.length], size: 0.22, life: 0.35 + Math.random() * 0.35, gravity: 22, drag: 2.5,
        });
      }
      soft.emit({ position: point.clone().add(new THREE.Vector3(0, 0.5, 0)), velocity: new THREE.Vector3(0, 1, 0), color: SMOKE, size: 1.5, grow: 3, life: 0.9, opacity: 0.5 });
    },

    // Wheelspin smoke as a car leaves its pit box.
    puff(points) {
      for (const point of points) for (let wisp = 0; wisp < 6; wisp++) {
        soft.emit({
          position: point.clone().add(new THREE.Vector3(jitter(0.3), 0.3, jitter(0.3))),
          velocity: new THREE.Vector3(jitter(1.5), 1 + Math.random(), jitter(1.5)),
          color: SMOKE, size: 0.9, grow: 2.6, life: 1.1 + Math.random() * 0.5, drag: 1.5, opacity: 0.5,
        });
      }
    },

    celebrate(point, amount = 160) {
      for (let piece = 0; piece < amount; piece++) {
        const angle = Math.random() * Math.PI * 2, speed = 3 + Math.random() * 9;
        soft.emit({
          position: point.clone().add(new THREE.Vector3(jitter(4), 8 + Math.random() * 2, jitter(4))),
          velocity: new THREE.Vector3(Math.cos(angle) * speed, 6 + Math.random() * 8, Math.sin(angle) * speed),
          color: CONFETTI[piece % CONFETTI.length], size: 0.45, life: 2.6 + Math.random() * 1.5, gravity: 7, drag: 1.4,
        });
      }
    },

    update(seconds, pixelsPerMetre) {
      soft.update(seconds, pixelsPerMetre);
      bright.update(seconds, pixelsPerMetre);
      debris.update(seconds);
    },

    dispose() {
      soft.dispose();
      bright.dispose();
      marks.dispose();
      ruts.dispose();
      debris.dispose();
    },

    clearMarks() {
      for (const layer of [marks, ruts]) {
        layer.count = 0;
        layer.mesh.geometry.setDrawRange(0, 0);
      }
      debris.clear();
      lastPatch.clear();
      lastBrakePatch.clear();
      lastRut.clear();
    },
  };
}
