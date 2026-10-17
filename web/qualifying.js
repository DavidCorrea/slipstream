// The qualifying results, as the broadcast shows them: the field from the back up to pole, one row at a time, each
// with its best lap and the gap to pole, pole's time in purple. Start race (or Esc, Enter, Space) goes on to the start.
import { formatLap } from './feed.js';

const $ = id => document.getElementById(id);
const ROW_MS = 260;

export function createQualifyingBoard({ onDone }) {
  const board = $('qualifying');
  let timers = [], open = false;

  function close() {
    if (!open) return;
    open = false;
    timers.forEach(clearTimeout);
    timers = [];
    board.classList.add('hidden');
    onDone();
  }
  $('qualifying-go').addEventListener('click', close);
  window.addEventListener('keydown', event => {
    if (!open || !['Escape', 'Enter', ' '].includes(event.key)) return;
    event.preventDefault();
    close();
  });

  return {
    get open() { return open; },

    // Shows the results of `intro.qualifying` for the race `intro` describes.
    show(intro) {
      const byKey = new Map(intro.cars.map(car => [car.key, car]));
      const pole = intro.qualifying[0]?.time;
      const rows = intro.qualifying.map(({ key, time }, place) => {
        const car = byKey.get(key);
        const row = document.createElement('li');
        const gap = time === null ? 'No time' : place === 0 ? 'Pole' : `+${(time - pole).toFixed(3)}`;
        row.innerHTML = `<span class="place">${place + 1}</span><span class="chip" style="background:${car.color}"></span>`
          + `<span class="name"></span><span class="time${place === 0 ? ' pole' : ''}">${time === null ? '–' : formatLap(time)}</span>`
          + `<span class="gap">${gap}</span>`;
        row.querySelector('.name').textContent = car.name;
        row.classList.toggle('yours', key === intro.cars[intro.yours].key);
        return row;
      });
      $('qualifying-list').replaceChildren(...rows);
      const reversed = intro.format?.kind === 'reverse';
      $('qualifying-note').textContent = reversed ? 'Reverse grid: the race starts in the opposite order.' : '';
      board.classList.remove('hidden');
      open = true;
      // From the back of the field to pole.
      timers = rows.map((row, index) => setTimeout(() => row.classList.add('shown'), (rows.length - 1 - index) * ROW_MS));
    },

    close,
  };
}
