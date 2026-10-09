// The loading screen for the published site, while the worker (engine-worker.js) downloads Python, numpy, the
// race code and the drivers, then starts the simulation. The bar fills by bytes received, each stage weighted by
// how much it downloads; a stage whose size was a guess holds just short of done until it really is.
const $ = id => document.getElementById(id);

const STAGES = [
  { name: 'python', label: 'Python, compiled for your browser' },
  { name: 'numpy', label: 'numpy, for the physics' },
  { name: 'files', label: 'The race code and the trained drivers' },
  { name: 'starting', label: 'Starting the race' },
];
const STARTING_SHARE = 0.04;   // of the bar, for the moment between the last download and the grid
const MEGABYTE = 1e6;

export function createLoadingScreen() {
  const progress = new Map(STAGES.map(({ name }) => [name, { received: 0, expected: null, done: false }]));
  let current = null;
  $('loading-steps').replaceChildren(...STAGES.map(({ name, label }) => {
    const item = document.createElement('li');
    item.dataset.stage = name;
    item.innerHTML = '<span class="mark"></span><span class="label"></span><span class="amount"></span>';
    item.querySelector('.label').textContent = label;
    return item;
  }));

  function render() {
    const downloads = STAGES.filter(({ name }) => name !== 'starting').map(({ name }) => progress.get(name));
    const total = downloads.reduce((sum, stage) => sum + (stage.expected ?? 0), 0);
    const fetched = downloads.reduce((sum, stage) => sum + shareDone(stage) * (stage.expected ?? 0), 0);
    const fraction = (total ? (fetched / total) * (1 - STARTING_SHARE) : 0) + (progress.get('starting').done ? STARTING_SHARE : 0);
    $('loading-bar').style.width = `${(fraction * 100).toFixed(1)}%`;
    $('loading-percent').textContent = `${Math.floor(fraction * 100)}%`;
    const megabytes = downloads.reduce((sum, stage) => sum + Math.min(stage.received, stage.expected ?? stage.received), 0) / MEGABYTE;
    $('loading-bytes').textContent = total ? `${megabytes.toFixed(1)} of ${(total / MEGABYTE).toFixed(1)} MB` : '';
    for (const { name } of STAGES) {
      const stage = progress.get(name), item = document.querySelector(`#loading-steps [data-stage="${name}"]`);
      const active = !stage.done && name === current;
      item.classList.toggle('done', stage.done);
      item.classList.toggle('active', active);
      item.querySelector('.amount').textContent = stage.expected
        ? `${(Math.min(stage.received, stage.expected) / MEGABYTE).toFixed(1)} / ${(stage.expected / MEGABYTE).toFixed(1)} MB` : '';
    }
  }

  return {
    show() {
      document.body.classList.add('loading');
      $('loading').classList.remove('hidden');
      render();
    },

    // From the worker, before anything downloads: how much each stage will fetch.
    plan(expected) {
      for (const [name, bytes] of Object.entries(expected)) progress.get(name).expected = bytes;
      render();
    },

    // From the worker: how much of a stage has arrived. A stage starting means the ones before it are done.
    progress({ stage: name, received, expected }) {
      for (const { name: earlier } of STAGES) {
        if (earlier === name) break;
        progress.get(earlier).done = true;
      }
      current = name;
      Object.assign(progress.get(name), { received, expected: expected || progress.get(name).expected });
      render();
    },

    ready() {
      for (const stage of progress.values()) stage.done = true;
      render();
    },

    fail(message) {
      $('loading-step').textContent = "The race couldn't start";
      $('loading-error').textContent = message;
      $('loading-error').classList.remove('hidden');
      $('loading-retry').classList.remove('hidden');
      $('loading').classList.add('failed');
    },

    hide() {
      document.body.classList.remove('loading');
      $('loading').classList.add('hidden');
    },
  };
}

// How much of a stage is done, from 0 to 1. Sizes known in advance can be a little off (a newer file, a proxy),
// so a stage never shows complete until the worker has moved past it.
function shareDone(stage) {
  if (stage.done) return 1;
  if (!stage.expected) return 0;
  return Math.min(stage.received / stage.expected, 0.98);
}
