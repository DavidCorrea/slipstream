// Race control messages, the way the broadcast shows them: one at a time across the top of the screen for a few
// seconds, in race control's own words. Penalties, retirements and flags live here rather than on the car's card.
const $ = id => document.getElementById(id);
const SHOWN_MS = 4500;
// A busy moment (a pile-up) would otherwise keep the screen full for a minute; the feed still has them all.
const MOST_WAITING = 3;

// The stewards' reasons (slipstream/stewards.py) as race control words them.
const OFFENCES = {
  'track limits': 'Track limits',
  collision: 'Causing a collision',
  reversing: 'Reversing on track',
  'pit speeding': 'Speeding in the pit lane',
  'pit line': 'Crossing the pit lane line',
  'jump start': 'False start',
  'yellow flag': 'Overtaking under yellow flags',
  'blue flags': 'Ignoring blue flags',
};

// Why a car is out (slipstream/race.py RETIREMENTS), as race control words it.
export const RETIREMENT_WORDS = {
  barrier: 'Accident',
  'barrier after contact': 'Collision damage',
  stuck: 'Car stopped',
  engine: 'Engine failure',
};

export function createRaceControl() {
  const board = $('race-control');
  let race = null, waiting = [], timer = null;

  function message(moment) {
    const name = race.cars[moment.car]?.name.toUpperCase();
    switch (moment.kind) {
      case 'penalty': return [`${moment.seconds} second time penalty`, `${name} · ${OFFENCES[moment.reason] ?? moment.reason}`];
      case 'crash': return ['Retired', `${name} · ${RETIREMENT_WORDS[moment.why] ?? 'Out of the race'}`];
      case 'engineFailure': return ['Retired', `${name} · Engine failure`];
      case 'yellow': return ['Yellow flag', 'Incident on track · no overtaking'];
      default: return null;
    }
  }

  function showNext() {
    timer = null;
    const next = waiting.shift();
    if (!next) { board.classList.add('hidden'); return; }
    const [title, detail] = next;
    board.innerHTML = `<b>Race control</b><span><strong>${title}</strong>${detail}</span>`;
    board.classList.remove('hidden');
    // Restart the animation for each message.
    board.style.animation = 'none';
    void board.offsetWidth;
    board.style.animation = '';
    timer = setTimeout(showNext, SHOWN_MS);
  }

  return {
    reset(intro) {
      race = intro;
      waiting = [];
      clearTimeout(timer);
      timer = null;
      board.classList.add('hidden');
    },
    add(moments) {
      if (!race) return;
      for (const moment of moments) {
        const said = message(moment);
        if (said) waiting.push(said);
      }
      waiting = waiting.slice(-MOST_WAITING);
      if (!timer && waiting.length) showNext();
    },
  };
}
