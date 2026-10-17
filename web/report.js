// The post-race report: the result, headlines (winner, fastest lap, driver of the day, how you did) and a lap chart
// of every car's place lap by lap. It's kept from the live frames as the race runs (replays don't count) and shown
// once the race is over; the race stays as it ended until you pick the next one.
//
// The lap chart is an emphasis chart: every car is a thin grey line and yours is in its colour, since twenty
// colours can't be told apart and the colours are the teams' own (you can paint them). Hovering a line or a row
// of the results shows that car in its colour, with its place lap by lap; the results table is the chart's table.
import { formatLap } from './feed.js';
import { RETIREMENT_WORDS } from './race-control.js';

const $ = id => document.getElementById(id);
const SHOW_AFTER_MS = 3500;          // the winner crosses the line and celebrates first
const CHART = { width: 560, row: 22, top: 14, bottom: 30, left: 34, right: 96 };
const SVG = 'http://www.w3.org/2000/svg';

export function createReport({ onNext, onAgain, extras = () => '' }) {
  const panel = $('report');
  let race = null, places = [], grid = [], fastest = null, shown = false, timer = null, last = null;

  $('report-next').addEventListener('click', () => { hide(); onNext(); });
  $('report-again').addEventListener('click', () => { hide(); onAgain(); });
  $('report-close').addEventListener('click', hide);

  function hide() {
    clearTimeout(timer);
    panel.classList.add('hidden');
  }

  // Each car's place at the end of every lap it completed, from the grid (lap 0) on.
  function record(frame) {
    if (!race || shown) return;
    last = frame;
    for (const car of frame.events.laps ?? []) {
      places[car].push({ lap: frame.cars.lap[car], place: frame.order.indexOf(car) + 1 });
    }
    if (frame.done) {
      shown = true;
      timer = setTimeout(() => show(frame), SHOW_AFTER_MS);
    }
  }

  function results(frame) {
    const { cars, order } = frame;
    const winner = order[0];
    return order.map((car, index) => {
      const finished = cars.finishTime[car] !== null;
      const lapsDown = cars.lap[winner] - cars.lap[car];
      let time;
      if (cars.retired[car]) time = RETIREMENT_WORDS[cars.retiredBy?.[car]] ?? 'Retired';
      else if (index === 0 && finished) time = formatLap(cars.finishTime[car]);
      else if (finished) time = `+${(cars.finishTime[car] - cars.finishTime[winner]).toFixed(3)}`;
      else time = lapsDown > 0 ? `+${lapsDown} lap${lapsDown === 1 ? '' : 's'}` : 'Running';
      return { car, place: index + 1, start: grid.indexOf(car) + 1, time, out: !!cars.retired[car], stops: cars.stops[car],
               best: cars.bestLap?.[car] ?? null, penalty: cars.penalty?.[car] ?? 0 };
    });
  }

  function tile(label, value, detail = '') {
    return `<div class="tile"><small>${label}</small><b>${value}</b><span>${detail}</span></div>`;
  }

  function headlines(rows, yours) {
    const name = car => escape(race.cars[car].name);
    const winner = rows[0];
    const gained = rows.filter(row => !row.out).map(row => ({ ...row, gain: row.start - row.place }))
      .sort((a, b) => b.gain - a.gain || a.place - b.place)[0];
    const mine = rows.find(row => row.car === yours);
    const change = mine.start - mine.place;
    // Nobody took the flag when every car is out (in a race that froze, say): there's no winner to name.
    const winnerTile = winner.out ? tile('Winner', 'No finishers', 'every car was out') : tile('Winner', name(winner.car), winner.time);
    return winnerTile
      + tile('Fastest lap', fastest ? name(fastest.car) : '–', fastest ? formatLap(fastest.time) : '')
      + tile('Driver of the day', gained ? name(gained.car) : '–', gained ? `P${gained.start} → P${gained.place}` : '')
      + tile('You', mine.out ? 'Out' : `P${mine.place}`, `${name(yours)} · from P${mine.start}${change ? ` (${change > 0 ? '+' : ''}${change})` : ''}`);
  }

  function table(rows, yours) {
    const body = rows.map(row => {
      const change = row.start - row.place;
      const arrow = row.out || !change ? '' : `<em class="${change > 0 ? 'up' : 'down'}">${change > 0 ? '▲' : '▼'}${Math.abs(change)}</em>`;
      const best = row.best === null ? '–' : formatLap(row.best);
      return `<tr data-car="${row.car}" class="${row.car === yours ? 'yours' : ''}${row.out ? ' out' : ''}">`
        + `<td class="place">${row.out ? '–' : row.place}</td>`
        + `<td><span class="chip" style="background:${race.cars[row.car].color}"></span>${escape(race.cars[row.car].name)}</td>`
        + `<td class="mono">${row.time}</td><td class="mono">P${row.start} ${arrow}</td><td class="mono">${row.stops}</td>`
        + `<td class="mono${fastest?.car === row.car ? ' fastest' : ''}">${best}</td><td class="mono">${row.penalty ? `${row.penalty} s` : ''}</td></tr>`;
    }).join('');
    return '<table><thead><tr><th>Pos</th><th>Driver</th><th>Time</th><th>Grid</th><th>Stops</th><th>Best lap</th><th>Penalty</th></tr></thead>'
      + `<tbody>${body}</tbody></table>`;
  }

  function element(name, attributes = {}, text = '') {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text) node.textContent = text;
    return node;
  }

  function chart(yours) {
    const count = race.cars.length;
    const laps = Math.max(1, ...places.map(points => points.at(-1)?.lap ?? 0));
    const height = CHART.top + CHART.row * (count - 1) + CHART.bottom;
    const x = lap => CHART.left + (lap / laps) * (CHART.width - CHART.left - CHART.right);
    const y = place => CHART.top + (place - 1) * CHART.row;
    const svg = element('svg', { viewBox: `0 0 ${CHART.width} ${height}`, role: 'img', 'aria-label': 'Places lap by lap' });
    const grid = element('g', { class: 'grid' });
    for (let place = 1; place <= count; place++) {
      grid.append(element('line', { x1: CHART.left, x2: x(laps), y1: y(place), y2: y(place) }));
      grid.append(element('text', { x: CHART.left - 8, y: y(place) + 4, 'text-anchor': 'end' }, `P${place}`));
    }
    const step = Math.max(1, Math.ceil(laps / 10));
    for (let lap = 0; lap <= laps; lap += step) grid.append(element('text', { x: x(lap), y: height - 10, 'text-anchor': 'middle' }, lap === 0 ? 'Grid' : String(lap)));
    svg.append(grid);
    const lines = element('g', { class: 'lines' });
    const label = element('text', { class: 'end-label' });
    const tooltip = $('report-tooltip');
    places.forEach((points, car) => {
      const path = points.map(point => `${x(point.lap)},${y(point.place)}`).join(' ');
      const group = element('g', { class: `car${car === yours ? ' yours' : ''}`, 'data-car': car, style: `--car:${race.cars[car].color}` });
      group.append(element('polyline', { points: path, class: 'mark' }));
      group.append(element('polyline', { points: path, class: 'hit' }));
      const end = points.at(-1);
      group.append(element('circle', { cx: x(end.lap), cy: y(end.place), r: 4, class: 'end' }));
      lines.append(group);
    });
    svg.append(lines, label);

    const highlight = car => {
      lines.querySelectorAll('.car').forEach(group => group.classList.toggle('hover', Number(group.dataset.car) === car));
      panel.querySelectorAll('tbody tr').forEach(row => row.classList.toggle('hover', Number(row.dataset.car) === car));
      const shownCar = car ?? yours;
      const end = places[shownCar].at(-1);
      label.setAttribute('x', x(end.lap) + 9);
      label.setAttribute('y', y(end.place) + 4);
      label.textContent = race.cars[shownCar].name;
      if (car === null) tooltip.classList.add('hidden');
    };
    lines.addEventListener('pointermove', event => {
      const group = event.target.closest('.car');
      if (!group) return;
      const car = Number(group.dataset.car);
      highlight(car);
      // The lap nearest the pointer, for the tooltip.
      const box = svg.getBoundingClientRect();
      const lap = Math.round(((event.clientX - box.left) / box.width * CHART.width - CHART.left) / (x(laps) - CHART.left) * laps);
      const point = places[car].reduce((best, candidate) => (Math.abs(candidate.lap - lap) < Math.abs(best.lap - lap) ? candidate : best));
      tooltip.textContent = `${race.cars[car].name} · ${point.lap === 0 ? 'Grid' : `Lap ${point.lap}`} · P${point.place}`;
      tooltip.style.left = `${event.clientX + 12}px`;
      tooltip.style.top = `${event.clientY - 28}px`;
      tooltip.classList.remove('hidden');
    });
    lines.addEventListener('pointerleave', () => highlight(null));
    highlight(null);
    return { svg, highlight };
  }

  function show(frame) {
    const yours = race.yours;
    const rows = results(frame);
    // The race's fastest lap: the best of everyone's best.
    const timed = rows.filter(row => row.best !== null);
    fastest = timed.length ? timed.reduce((best, row) => (row.best < best.best ? row : best)) : null;
    if (fastest) fastest = { car: fastest.car, time: fastest.best };
    const format = $('format').selectedOptions[0]?.text ?? '';
    $('report-title').textContent = `${race.format?.kind && race.format.kind !== 'standard' ? `${format} · ` : ''}${frame.laps ?? race.laps} laps · ${race.locationLabel ?? ''}`;
    $('report-headlines').innerHTML = headlines(rows, yours);
    $('report-extras').innerHTML = extras(rows, yours, fastest?.car ?? null);
    $('report-results').innerHTML = table(rows, yours);
    const { svg, highlight } = chart(yours);
    $('report-chart').replaceChildren(svg);
    panel.querySelectorAll('tbody tr').forEach(row => {
      row.addEventListener('pointerenter', () => highlight(Number(row.dataset.car)));
      row.addEventListener('pointerleave', () => highlight(null));
    });
    panel.querySelector('.legend .key.yours').style.background = race.cars[yours].color;
    panel.classList.remove('hidden');
  }

  return {
    reset(intro, yours, locationLabel) {
      hide();
      race = intro;
      race.yours = yours;
      race.locationLabel = locationLabel;
      shown = false;
      fastest = null;
      last = null;
      grid = intro.cars.map((_, car) => car);
      places = intro.cars.map((_, car) => [{ lap: 0, place: car + 1 }]);
    },
    setYours(car) { if (race) race.yours = car; },
    record,
    get lastFrame() { return last; },
  };
}

function escape(text) {
  return String(text).replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
}
