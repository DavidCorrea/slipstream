// A debug readout of what the GPU is asked to do: frame rate, draw calls, triangles, and the geometries, textures
// and shader programs alive. Hidden until ` (backquote) or ?stats in the address; the memory counts are what show
// a leak, climbing race after race when something isn't disposed.
const $ = id => document.getElementById(id);
const SHOWN_EVERY_MS = 500;

export function createStats(renderer) {
  const panel = $('stats');
  // A frame is many renders (the shadow map, every post-processing pass), so the counts are kept across all of
  // them and cleared once a frame instead of after each one.
  renderer.info.autoReset = false;
  let shown = new URLSearchParams(location.search).has('stats');
  let frames = 0, busiest = 0, since = performance.now(), frameStart = 0;

  function toggle() {
    shown = !shown;
    panel.classList.toggle('hidden', !shown);
  }
  panel.classList.toggle('hidden', !shown);
  window.addEventListener('keydown', event => {
    if (event.key !== '`' || event.target.matches('input, select, textarea')) return;
    toggle();
  });

  return {
    beginFrame(now) {
      renderer.info.reset();
      frameStart = now;
    },

    endFrame(now) {
      frames += 1;
      busiest = Math.max(busiest, performance.now() - frameStart);
      if (!shown || now - since < SHOWN_EVERY_MS) return;
      const { render, memory, programs } = renderer.info;
      const perSecond = (frames * 1000) / (now - since);
      panel.textContent = [
        `${perSecond.toFixed(0)} fps · slowest ${busiest.toFixed(1)} ms`,
        `${render.calls} draw calls · ${(render.triangles / 1000).toFixed(0)}k triangles`,
        `${memory.geometries} geometries · ${memory.textures} textures · ${programs?.length ?? 0} programs`,
      ].join('\n');
      frames = 0;
      busiest = 0;
      since = now;
    },
  };
}
