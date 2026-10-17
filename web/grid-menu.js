// The grid menu's shape: three tabs (the race, the garage, your predictions), the format as buttons, and only the
// options the chosen format uses (a timed race has a race time, not laps). Every race setting is remembered in this
// browser, so after the first race it's just Start. Changing one starts the race again on the same circuit
// (`onChange`), as the server only sets a race up on the grid.
const $ = id => document.getElementById(id);
const STORAGE_KEY = 'slipstream.race';
// The settings kept between visits, by the id of the control that holds each.
const REMEMBERED = ['format', 'minutes', 'laps', 'cars', 'weather', 'location'];

// What each format does, under the buttons.
const FORMAT_NOTES = {
  standard: 'The laps you set, from the grid as drawn.',
  reverse: 'The grid turned round: the front row starts at the back.',
  sprint: 'A third of the laps, flat out.',
  elimination: 'From lap 2, the last car is out each time the leader completes a lap, until three are left.',
  endurance: 'Twice the laps and two mandatory stops: 30 s for every finisher short of them.',
  timed: 'Against the clock: when the time is up, the leader’s next lap is the last.',
};

export function createGridMenu({ onChange }) {
  restore();

  function showTab(name) {
    document.querySelectorAll('.grid-tabs [data-tab]').forEach(tab => {
      const chosen = tab.dataset.tab === name;
      tab.classList.toggle('active', chosen);
      tab.setAttribute('aria-selected', String(chosen));
    });
    document.querySelectorAll('.grid-tab').forEach(panel => panel.classList.toggle('hidden', panel.dataset.tab !== name));
  }
  document.querySelectorAll('.grid-tabs [data-tab]').forEach(tab => tab.addEventListener('click', () => showTab(tab.dataset.tab)));
  // Anything marked data-open-tab (your card's Edit) opens that tab.
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-open-tab]');
    if (link) showTab(link.dataset.openTab);
  });

  function showFormat() {
    const format = $('format').value;
    document.querySelectorAll('#format-buttons [data-format]').forEach(button => button.classList.toggle('active', button.dataset.format === format));
    $('format-note').textContent = FORMAT_NOTES[format];
    $('minutes-field').classList.toggle('hidden', format !== 'timed');
    $('laps-field').classList.toggle('hidden', format === 'timed');
    // A sprint and an endurance race are measured against a full race; the note says what this one runs.
    $('laps-label').textContent = format === 'sprint' || format === 'endurance' ? 'Laps of a full race' : 'Laps';
  }
  document.querySelectorAll('#format-buttons [data-format]').forEach(button => button.addEventListener('click', () => {
    if ($('format').value === button.dataset.format) return;
    $('format').value = button.dataset.format;
    showFormat();
    changed();
  }));

  function changed() {
    save();
    onChange();
  }
  for (const id of ['minutes', 'laps', 'cars', 'weather', 'location']) $(id).addEventListener('change', changed);
  showFormat();
  showTab('race');

  return {
    // Back on the grid for a new race: the race tab first, with this race's length in the format's note.
    show(intro) {
      showTab('race');
      const format = intro.format?.kind ?? 'standard';
      const length = format === 'timed' ? '' : ` This race: ${intro.laps} lap${intro.laps === 1 ? '' : 's'}.`;
      $('format-note').textContent = FORMAT_NOTES[format] + (format === 'standard' ? '' : length);
    },
    showTab,
  };
}

function restore() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
  } catch {
    // Unreadable or blocked storage keeps the defaults.
  }
  for (const id of REMEMBERED) {
    const value = saved?.[id];
    if (value === undefined || value === null) continue;
    const control = $(id);
    // Only a value the control offers (a removed option, or a number out of range, keeps the default).
    if (control.tagName === 'SELECT' && ![...control.options].some(option => option.value === String(value))) continue;
    if (control.type === 'number' && !(Number(value) >= Number(control.min) && Number(value) <= Number(control.max))) continue;
    control.value = String(value);
  }
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(REMEMBERED.map(id => [id, $(id).value]))));
  } catch {
    // Private windows and blocked storage just don't remember it.
  }
}
