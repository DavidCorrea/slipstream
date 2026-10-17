// Predictions: before the lights, pick the podium and two bonus calls (who sets the fastest lap, who's first out
// of the race, or nobody). They lock when the race starts and are scored in the race report: POINTS.exact for each
// podium place called right, POINTS.podium for a podium driver in the wrong place, POINTS.bonus for each bonus
// call, all doubled in a sprint. Your running total, races predicted and best race stay in this browser.
const $ = id => document.getElementById(id);
const POINTS = { exact: 10, podium: 4, bonus: 5 };
const SPRINT_FACTOR = 2;
const STORAGE_KEY = 'slipstream.predictions';
const NOBODY = 'nobody';
const PICKS = ['p1', 'p2', 'p3', 'fastest', 'firstOut'];

export function createPredictions() {
  let race = null, locked = null, firstOut = null;
  const record = load();

  function options(select, withNobody) {
    const chosen = select.value;
    select.replaceChildren(new Option('–', ''), ...(withNobody ? [new Option('Nobody', NOBODY)] : []),
      ...race.cars.map(car => new Option(car.name, car.key)));
    select.value = [...select.options].some(option => option.value === chosen) ? chosen : '';
  }

  function showRecord() {
    $('prediction-record').textContent = record.races
      ? `${record.total} pts from ${record.races} race${record.races === 1 ? '' : 's'} · best ${record.best}`
      : 'Pick before the lights; scored in the race report.';
  }

  function picks() {
    return Object.fromEntries(PICKS.map(pick => [pick, $(`predict-${pick}`).value || null]));
  }

  return {
    // A new race on the grid: the choices are its drivers (anything picked for someone not racing is cleared).
    reset(intro) {
      race = intro;
      locked = null;
      firstOut = null;
      for (const pick of PICKS) options($(`predict-${pick}`), pick === 'firstOut');
      showRecord();
    },

    // The lights are about to go out.
    lock() {
      const chosen = picks();
      locked = Object.values(chosen).some(Boolean) ? chosen : null;
    },

    // Every live frame: who retires first.
    watch(frame) {
      if (!locked || firstOut) return;
      const out = frame.cars.retired?.findIndex(Boolean);
      if (out >= 0) firstOut = race.cars[out].key;
    },

    // The report's predictions block, scoring `rows` (the result, see report.js) and keeping the record.
    score(rows, fastestCar) {
      if (!locked) return '';
      const key = car => race.cars[car].key;
      const name = pickKey => (pickKey === NOBODY ? 'Nobody' : race.cars.find(car => car.key === pickKey)?.name ?? '–');
      const podium = rows.filter(row => !row.out).slice(0, 3).map(row => key(row.car));
      const factor = race.format?.kind === 'sprint' ? SPRINT_FACTOR : 1;
      const lines = [];
      let points = 0;
      ['p1', 'p2', 'p3'].forEach((pick, index) => {
        const chosen = locked[pick];
        if (!chosen) return;
        const earned = podium[index] === chosen ? POINTS.exact : podium.includes(chosen) ? POINTS.podium : 0;
        points += earned;
        lines.push([`P${index + 1}`, name(chosen), earned]);
      });
      const bonuses = [['fastest', 'Fastest lap', fastestCar === null ? null : key(fastestCar)], ['firstOut', 'First out', firstOut ?? NOBODY]];
      for (const [pick, label, actual] of bonuses) {
        if (!locked[pick]) continue;
        const earned = locked[pick] === actual ? POINTS.bonus : 0;
        points += earned;
        lines.push([label, name(locked[pick]), earned]);
      }
      points *= factor;
      record.total += points;
      record.races += 1;
      record.best = Math.max(record.best, points);
      save(record);
      locked = null;
      const items = lines.map(([label, pick, earned]) => `<li class="${earned ? 'right' : 'wrong'}"><small>${label}</small>${escape(pick)}`
        + `<b>${earned ? `+${earned * factor}` : '0'}</b></li>`).join('');
      return `<div class="report-predictions"><h3>Your predictions${factor > 1 ? ' · sprint, double points' : ''}</h3><ol>${items}</ol>`
        + `<p><b>${points} pts</b> · ${record.total} pts from ${record.races} race${record.races === 1 ? '' : 's'} · best ${record.best}</p></div>`;
    },
  };
}

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Number.isFinite(saved.total)) return saved;
  } catch {
    // Unreadable or blocked storage starts a fresh record.
  }
  return { total: 0, races: 0, best: 0 };
}

function save(record) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
  } catch {
    // Private windows and blocked storage just don't keep the record.
  }
}

function escape(text) {
  return String(text).replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
}
