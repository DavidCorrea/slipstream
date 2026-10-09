// The pixel style: the same 3D scene drawn the way a 16-bit racer looked, but with modern lighting underneath.
// The scene renders into a small image (one game pixel for every few screen pixels), its colours snap to a
// short palette with a little ordered dithering, edges where the depth jumps get dark outlines, and the image is
// blown up with hard pixel edges. Drawing a quarter of the pixels is also much lighter on the GPU.
import * as THREE from 'three';

const SCREEN_PIXELS_PER_PIXEL = 4;
const LEVELS = 11.0;         // shades per colour channel
// How much the ordered dither nudges a colour, in palette steps: enough to soften the bands of a gradient, too
// little to checker a flat field of grass or tarmac.
const DITHER = 0.35;
const OUTLINE_DEPTH = 0.012; // relative jump in distance that counts as an edge

const vertexShader = `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;
const fragmentShader = `
  uniform sampler2D colour;
  uniform sampler2D depth;
  uniform vec2 size;
  uniform float near;
  uniform float far;
  uniform bool perspective;
  varying vec2 vUv;

  float distanceAt(vec2 uv) {
    float z = texture2D(depth, uv).x;
    if (!perspective) return near + z * (far - near);
    return (near * far) / (far - z * (far - near));
  }

  // A 4x4 Bayer matrix: the ordered dither 16-bit games used to fake more colours than they had.
  float bayer(vec2 pixel) {
    int x = int(mod(pixel.x, 4.0)), y = int(mod(pixel.y, 4.0));
    int index = x + y * 4;
    float values[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return values[index] / 16.0 - 0.5;
  }

  void main() {
    vec2 texel = 1.0 / size;
    vec3 colour = texture2D(colour, vUv).rgb;
    gl_FragColor = vec4(colour, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>

    float here = distanceAt(vUv);
    float nearest = min(min(distanceAt(vUv + vec2(texel.x, 0.0)), distanceAt(vUv - vec2(texel.x, 0.0))),
                        min(distanceAt(vUv + vec2(0.0, texel.y)), distanceAt(vUv - vec2(0.0, texel.y))));
    // Only the far side of an edge darkens, so outlines are one pixel wide and sit around the nearer object.
    float edge = step(${OUTLINE_DEPTH} * here, here - nearest);

    vec3 shaded = gl_FragColor.rgb + bayer(gl_FragCoord.xy / ${SCREEN_PIXELS_PER_PIXEL.toFixed(1)}) * ${DITHER.toFixed(2)} / ${LEVELS.toFixed(1)};
    shaded = floor(shaded * ${LEVELS.toFixed(1)} + 0.5) / ${LEVELS.toFixed(1)};
    gl_FragColor = vec4(mix(shaded, vec3(0.06, 0.05, 0.09), edge * 0.85), 1.0);
  }
`;

export function createPixelRenderer(renderer) {
  const depthTexture = new THREE.DepthTexture(1, 1);
  const target = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthTexture, type: THREE.HalfFloatType,
  });
  const material = new THREE.ShaderMaterial({
    vertexShader, fragmentShader, depthTest: false, depthWrite: false,
    uniforms: {
      colour: { value: target.texture }, depth: { value: depthTexture }, size: { value: new THREE.Vector2(1, 1) },
      near: { value: 1 }, far: { value: 1000 }, perspective: { value: false },
    },
  });
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  screen.frustumCulled = false;
  const screenScene = new THREE.Scene();
  screenScene.add(screen);
  const screenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  function resize() {
    const width = Math.max(1, Math.round(window.innerWidth / SCREEN_PIXELS_PER_PIXEL));
    const height = Math.max(1, Math.round(window.innerHeight / SCREEN_PIXELS_PER_PIXEL));
    target.setSize(width, height);
    material.uniforms.size.value.set(width, height);
  }
  window.addEventListener('resize', resize);
  resize();

  return {
    // Height of the small image, for anything sized in pixels (the particles).
    get height() { return target.height; },
    render(scene, camera) {
      material.uniforms.near.value = camera.near;
      material.uniforms.far.value = camera.far;
      material.uniforms.perspective.value = !!camera.isPerspectiveCamera;
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      renderer.render(screenScene, screenCamera);
    },
  };
}
