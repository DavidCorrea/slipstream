// The driver's eyes: a camera inside the selected driver's helmet, on their head as cars.js poses it (rolled
// against the corner, pitched under braking, turned towards where the car is steering). From there the halo's
// pillar splits the view, the steering wheel and the nose sit below it, and the mirrors and wheels are out to
// the sides, as a driver sees them. The road shakes it a little, more the faster the car goes.
import * as THREE from 'three';

// In the head's frame (x forward, y up): just behind the visor, at eye height inside the helmet. Higher puts the
// eyes level with the halo's hoop, which then hides the road.
const EYE = new THREE.Vector3(0.08, 0.12, 0);
const FOV = { slow: 62, fast: 56, at: 80 };   // degrees; the view narrows a little with speed, as focus does
// How hard the road shakes the view at FOV.at metres a second and over: radians, and the shake's rates in hertz.
const SHAKE = { amount: 0.0035, rates: [13, 21, 34] };

// The camera looks down its -z, the head forward along its +x.
const FACING = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);

export function createDriverCamera({ aspect }) {
  // Near enough to keep the steering wheel, a third of a metre in front of the eyes.
  const camera = new THREE.PerspectiveCamera(FOV.slow, aspect, 0.05, 2400);
  const head = new THREE.Quaternion(), shake = new THREE.Euler();

  return {
    camera,
    resize(newAspect) { camera.aspect = newAspect; camera.updateProjectionMatrix(); },

    // Puts the camera in `car`'s helmet, once the car has been posed for this frame.
    aim(car, speed, time) {
      car.updateMatrixWorld(true);
      const helmet = car.userData.head;
      camera.position.copy(EYE).applyMatrix4(helmet.matrixWorld);
      helmet.getWorldQuaternion(head);
      const rough = SHAKE.amount * Math.min(1, Math.abs(speed) / FOV.at);
      const [pitch, yaw, roll] = SHAKE.rates.map((rate, index) => Math.sin(time * rate * Math.PI * 2 + index * 1.7) * rough);
      camera.quaternion.copy(head).multiply(FACING).multiply(new THREE.Quaternion().setFromEuler(shake.set(pitch, yaw, roll)));
      const fov = THREE.MathUtils.lerp(FOV.slow, FOV.fast, Math.min(1, Math.abs(speed) / FOV.at));
      if (Math.abs(camera.fov - fov) > 0.05) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
      }
    },
  };
}
