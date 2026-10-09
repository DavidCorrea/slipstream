// The fastest lap so far, driven again by a see-through car in step with the selected car's current lap, so you
// can see where it gains and loses against the best anyone has done.
import * as THREE from 'three';
import { createCar } from './cars.js';
import { toWorld } from './scene.js';

export function createGhost(scene) {
  let mesh = null, trace = null, enabled = false;

  function build(color) {
    if (mesh) scene.remove(mesh);
    mesh = createCar(color, 'ghost');
    mesh.traverse(object => {
      if (!object.isMesh) return;
      object.castShadow = false;
      object.material = object.material.clone();
      object.material.transparent = true;
      object.material.opacity = 0.32;
      object.material.depthWrite = false;
    });
    mesh.userData.flame.visible = false;
    mesh.visible = false;
    scene.add(mesh);
  }

  // Where the trace was `seconds` into its lap, between its two nearest points.
  function poseAt(seconds) {
    const points = trace.points;
    let low = 0, high = points.length - 1;
    if (seconds <= points[0][0] || seconds >= points[high][0]) return null;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (points[middle][0] <= seconds) low = middle; else high = middle;
    }
    const [fromTime, fromX, fromY, fromHeading] = points[low], [toTime, toX, toY, toHeading] = points[high];
    const blend = (seconds - fromTime) / (toTime - fromTime);
    const turn = Math.atan2(Math.sin(toHeading - fromHeading), Math.cos(toHeading - fromHeading));
    return { x: fromX + (toX - fromX) * blend, y: fromY + (toY - fromY) * blend, heading: fromHeading + turn * blend };
  }

  return {
    get trace() { return trace; },
    get enabled() { return enabled; },
    setEnabled(value) { enabled = value; },
    reset() {
      trace = null;
      if (mesh) mesh.visible = false;
    },
    setTrace(newTrace, color) {
      trace = newTrace;
      build(color);
    },
    // `lapSeconds` is how far the selected car is into its lap, or null when it isn't on a timed lap.
    update(lapSeconds) {
      if (!mesh) return;
      const pose = enabled && trace && lapSeconds !== null ? poseAt(lapSeconds) : null;
      mesh.visible = !!pose;
      if (!pose) return;
      mesh.position.copy(toWorld(pose.x, pose.y));
      mesh.rotation.y = pose.heading;
    },
    dispose() {
      if (mesh) scene.remove(mesh);
      mesh = null;
    },
  };
}
