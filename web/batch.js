// Static scenery drawn as few meshes as possible. Every mesh is a draw call, twice over with the sun's shadow,
// and a draw call costs the browser far more than its triangles: twenty garages of six walls each, a pit wall in
// six-metre pieces and sixty grid-box strips were more draw calls than all the cars. Merging what never moves into
// one mesh per look (material, shadows) draws the same picture in a handful of calls.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// What a material looks like apart from its colour: meshes that differ only in colour merge into one, the colour
// carried per vertex instead.
function look(material) {
  return [material.type, material.roughness, material.metalness, material.map?.uuid, material.transparent, material.opacity,
          material.side, material.emissive?.getHexString(), material.emissiveIntensity, material.depthWrite, material.flatShading].join('|');
}

// Merges every static mesh under `group` into one mesh per look and shadow setting, in place. A mesh that changes
// later (a light that changes colour, a flag that flaps, a prop that can break) is marked `userData.changes` and
// left alone, with everything under it; so are instanced meshes, ones already coloured per vertex, and ones with a
// shader of their own.
export function mergeStatic(group) {
  group.updateMatrixWorld(true);
  const toGroup = group.matrixWorld.clone().invert();
  const batches = new Map();
  const merged = [];
  const visit = object => {
    if (object.userData.changes) return;
    if (object.isMesh && !object.isInstancedMesh && !Array.isArray(object.material) && !object.material.vertexColors
        && object.material.onBeforeCompile === THREE.Material.prototype.onBeforeCompile && object.visible) {
      const key = `${look(object.material)}|${object.castShadow}|${object.receiveShadow}`;
      if (!batches.has(key)) batches.set(key, { material: object.material, castShadow: object.castShadow, receiveShadow: object.receiveShadow, pieces: [] });
      batches.get(key).pieces.push(piece(object, toGroup));
      merged.push(object);
    }
    for (const child of [...object.children]) visit(child);
  };
  visit(group);
  for (const object of merged) object.removeFromParent();

  for (const { material, castShadow, receiveShadow, pieces } of batches.values()) {
    const geometry = mergeGeometries(pieces);
    if (!geometry) throw new Error(`Static meshes with the look ${look(material)} could not be merged`);
    const shared = material.clone();
    shared.color.set('#ffffff');
    shared.vertexColors = true;
    const mesh = new THREE.Mesh(geometry, shared);
    mesh.castShadow = castShadow;
    mesh.receiveShadow = receiveShadow;
    group.add(mesh);
  }
  return group;
}

// A mesh's geometry in the group's frame, with its material's colour on every vertex and only the attributes all
// pieces share.
function piece(mesh, toGroup) {
  const geometry = (mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry.clone())
    .applyMatrix4(new THREE.Matrix4().multiplyMatrices(toGroup, mesh.matrixWorld));
  for (const name of Object.keys(geometry.attributes)) if (!['position', 'normal', 'uv'].includes(name)) geometry.deleteAttribute(name);
  const count = geometry.attributes.position.count;
  if (!geometry.attributes.normal) geometry.computeVertexNormals();
  if (!geometry.attributes.uv) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(count * 2), 2));
  const color = mesh.material.color ?? new THREE.Color('#ffffff');
  const colors = new Float32Array(count * 3);
  for (let vertex = 0; vertex < count; vertex++) colors.set([color.r, color.g, color.b], vertex * 3);
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}
