// The overlay: standings tower, lap counter and clock, the selected car's onboard graphic and lap timing, and
// banners. Like the TV broadcast, the card shows what a viewer sees over an onboard camera; the car's condition and
// temperatures are the team's, behind the pit wall button.
import { formatLap } from './feed.js';
import { RETIREMENT_WORDS } from './race-control.js';

const $ = id => document.getElementById(id);
// Each compound's working window in the race's 0-1 temperature (see slipstream/car.py TYRE_WINDOW, ±0.06).
const TYRE_WINDOWS = { soft: [0.49, 0.61], medium: [0.54, 0.66], hard: [0.59, 0.71], intermediate: [0.36, 0.48], wet: [0.26, 0.38] };

const PIT_WALL_SETTING = 'slipstream.pitWall';
// Below these the pit wall bars turn red: worth a stop, or a worry.
const LOW = { tyres: 0.25, fuel: 0.15, brakes: 0.25 };
const HEAVY_DAMAGE = 0.4;

// A temperature in degrees, coloured and worded by where it sits against the range it works best in.
function showTemperature(id, value, [low, high], coldest, hottest) {
  const element = $(id);
  if (value === undefined) { element.textContent = '–'; return; }
  const state = value < low ? 'cold' : value > high + 0.15 ? 'overheating' : value > high ? 'hot' : 'ok';
  element.innerHTML = `${Math.round(coldest + value * (hottest - coldest))}°<small>${state}</small>`;
  element.className = state;
}

function showBar(id, share, warn) {
  $(id).style.width = `${share * 100}%`;
  $(`${id}-value`).textContent = `${Math.round(share * 100)}%`;
  $(id).parentElement.parentElement.classList.toggle('low', warn);
}

function remembered(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private windows and blocked storage just don't remember it.
  }
}

function formatDelta(seconds) {
  return `${seconds < 0 ? '−' : '+'}${Math.abs(seconds).toFixed(3)}`;
}

export function createHud({ onSelect }) {
  const tower = $('tower');
  let race = null, rows = [], lastOrder = [], changed = new Map(), bannerTimer = null;
  // Each car's sector times on its current lap ({ time, rating } per sector), and who holds the fastest lap.
  let sectors = [], fastestCar = null;
  // Each car's sector times on its best lap, and its time against them at the last sector it closed: the delta
  // the onboard graphic shows next to the lap clock.
  let bestSplits = [], deltas = [];

  const pitWall = $('pit-wall'), pitWallToggle = $('pit-wall-toggle');
  function showPitWall(open) {
    pitWall.classList.toggle('hidden', !open);
    pitWallToggle.setAttribute('aria-expanded', String(open));
    pitWallToggle.classList.toggle('active', open);
  }
  showPitWall(remembered(PIT_WALL_SETTING) === 'open');
  pitWallToggle.addEventListener('click', () => {
    const open = pitWall.classList.contains('hidden');
    showPitWall(open);
    remember(PIT_WALL_SETTING, open ? 'open' : 'closed');
  });

  function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, '0')}`;
  }

  return {
    setRace(intro) {
      race = intro;
      tower.replaceChildren();
      rows = intro.cars.map((car, id) => {
        const row = document.createElement('li');
        row.innerHTML = `<span class="place"></span><span class="chip" style="background:${car.color}"></span><span class="name"><span class="driver">${car.name}</span><span class="yours"></span><span class="pitting"></span><span class="penalty-tag"></span><span class="fastest" title="Fastest lap">⏱</span></span><span class="tyre-cell"><span class="tyre"></span><small class="tyre-age"></small></span><span class="gap"></span>`;
        row.addEventListener('click', () => onSelect(id));
        return row;
      });
      lastOrder = [];
      changed.clear();
      sectors = intro.cars.map(() => []);
      bestSplits = intro.cars.map(() => null);
      deltas = intro.cars.map(() => null);
      fastestCar = null;
      $('seed').textContent = `circuit #${intro.seed}`;
    },

    recolor(id) {
      rows[id].querySelector('.chip').style.background = race.cars[id].color;
    },

    rename(id) {
      rows[id].querySelector('.driver').textContent = race.cars[id].name;
    },

    update(frame, selected, yours, now) {
      if (!race) return;
      const cars = frame.cars, order = frame.order;
      const leader = order[0];
      const leaderLap = Math.min(cars.lap[leader] + 1, race.laps);
      $('lap-counter').textContent = frame.done ? 'Finished' : `Lap ${leaderLap}/${race.laps}`;
      $('race-clock').textContent = formatTime(frame.time);
      const { wetness, rain, temperature = 0.5, wind = [0, 0] } = frame.weather;
      const sky = rain > 0.6 ? 'Heavy rain' : rain > 0 ? 'Rain' : wetness > 0.05 ? 'Drying' : 'Dry';
      // The air's 0-1 temperature as degrees: a cold day about 8°C, a hot one about 35°C.
      const day = `${Math.round(8 + temperature * 27)}°C · wind ${Math.round(Math.hypot(wind[0], wind[1]))} m/s`;
      $('weather-text').textContent = `${wetness > 0.05 ? `${sky} · track ${Math.round(wetness * 100)}% wet` : sky} · ${day}`;
      $('wet-bar').style.width = `${wetness * 100}%`;

      const winnerTime = cars.finishTime[leader];
      order.forEach((id, place) => {
        const row = rows[id];
        const before = lastOrder.indexOf(id);
        if (before >= 0 && before !== place) changed.set(id, { gained: place < before, until: now + 2500 });
        const change = changed.get(id);
        row.classList.toggle('gained', !!change && change.gained && change.until > now);
        row.classList.toggle('lost', !!change && !change.gained && change.until > now);
        row.classList.toggle('selected', id === selected);
        row.classList.toggle('finished', cars.finishTime[id] !== null);
        row.querySelector('.place').textContent = place + 1;
        row.querySelector('.yours').textContent = id === yours ? '★' : '';
        row.classList.toggle('fastest-lap', id === fastestCar);
        row.querySelector('.tyre').className = `tyre ${cars.compound[id]}`;
        row.querySelector('.tyre-age').textContent = cars.tyreLaps?.[id] ?? '';
        // A finished car's gap already has its penalty in it.
        const penaltyTag = row.querySelector('.penalty-tag');
        const owed = cars.finishTime[id] === null && !cars.retired?.[id] ? cars.penalty?.[id] : 0;
        penaltyTag.textContent = owed ? `+${owed}s` : '';
        penaltyTag.style.display = owed ? '' : 'none';
        const pitting = row.querySelector('.pitting');
        pitting.textContent = cars.pit[id] >= 2 ? 'PIT' : cars.stops[id] ? `${cars.stops[id]}×` : '';
        pitting.classList.toggle('in-lane', cars.pit[id] >= 2);
        pitting.style.display = pitting.textContent ? '' : 'none';
        let gap;
        row.classList.toggle('retired', !!cars.retired?.[id]);
        if (cars.retired?.[id]) gap = 'OUT';
        else if (cars.finishTime[id] !== null) gap = place === 0 ? 'WIN' : `+${(cars.finishTime[id] - winnerTime).toFixed(1)}s`;
        else if (place === 0) gap = 'Leader';
        else {
          const behind = cars.progress[leader] - cars.progress[id];
          const length = race.track.length;
          gap = behind > length ? `+${Math.floor(behind / length)} lap` : `+${(behind / Math.max(cars.speed[leader], 15)).toFixed(1)}s`;
        }
        row.querySelector('.gap').textContent = gap;
        if (tower.children[place] !== row) tower.insertBefore(row, tower.children[place] || null);
      });
      lastOrder = order.slice();

      const panel = $('telemetry');
      if (selected === null || selected === undefined) { panel.classList.add('hidden'); return; }
      panel.classList.remove('hidden');
      const car = race.cars[selected];
      $('tele-swatch').style.background = car.color;
      $('tele-name').textContent = car.name;
      $('tele-position').textContent = cars.retired?.[selected] ? 'OUT' : `P${order.indexOf(selected) + 1}`;
      // A car out of the race says why and when, and its onboard graphic stops.
      const out = !!cars.retired?.[selected];
      panel.classList.toggle('retired', out);
      $('tele-out').classList.toggle('hidden', !out);
      if (out) $('tele-out').textContent = `Out · ${RETIREMENT_WORDS[cars.retiredBy?.[selected]] ?? 'Retired'} · lap ${cars.lap[selected] + 1}`;
      // Backing up shows its speed too; the Reverse flag says which way.
      $('tele-speed').textContent = out ? 0 : Math.round(Math.abs(cars.speed[selected]) * 3.6);
      $('tele-throttle').style.height = `${out ? 0 : cars.throttle[selected] * 100}%`;
      $('tele-brake').style.height = `${out ? 0 : cars.brake[selected] * 100}%`;
      $('tele-steer').style.left = `${50 + (out ? 0 : cars.steer[selected]) * -46}%`;
      const compound = cars.compound[selected];
      const tyreLaps = cars.tyreLaps?.[selected];
      $('tele-tyre').className = `tyre ${compound}`;
      $('tele-tyre-text').textContent = `${compound[0].toUpperCase()}${compound.slice(1)}${tyreLaps === undefined ? '' : tyreLaps === 0 ? ' · new' : ` · ${tyreLaps} lap${tyreLaps === 1 ? '' : 's'} old`}`;

      const tyresLeft = 1 - cars.tyreWear[selected], brakesLeft = 1 - cars.brakeWear[selected];
      showBar('tele-tyres', tyresLeft, tyresLeft < LOW.tyres);
      showBar('tele-fuel', cars.fuel[selected], cars.fuel[selected] < LOW.fuel);
      showBar('tele-brakes', brakesLeft, brakesLeft < LOW.brakes);
      showBar('tele-damage', cars.damage[selected], cars.damage[selected] > HEAVY_DAMAGE);
      const engineMode = cars.engine[selected] > 0.66 ? 'push' : cars.engine[selected] < 0.33 ? 'lean' : 'standard';
      const balance = cars.balance?.[selected] ?? 0;
      const handling = balance > 0.2 ? ' · oversteers' : balance < -0.2 ? ' · understeers' : '';
      $('tele-setup').textContent = `wing ${Math.round(cars.wing[selected] * 100)}% · ${engineMode} engine · ${cars.stops[selected]} stop${cars.stops[selected] === 1 ? '' : 's'}${handling}`;
      showTemperature('temp-tyres', cars.tyreTemp?.[selected], TYRE_WINDOWS[compound], 20, 140);
      showTemperature('temp-brakes', cars.brakeTemp?.[selected], [0.3, 0.85], 100, 1100);
      showTemperature('temp-engine', cars.engineTemp?.[selected], [0.35, 0.8], 70, 130);

      // What the car is doing now; penalties and retirements are race control's (race-control.js) and the tower's.
      const flags = [];
      if (!out) {
        if (cars.punctured?.[selected]) flags.push('<span class="warning">Puncture</span>');
        if (frame.marshals?.blue?.[selected]) flags.push('<span class="tow">Blue flag</span>');
        const yellowHere = (frame.marshals?.yellow ?? []).some(([start, end]) => {
          const length = race.track.length;
          return (((cars.progress[selected] % length) - start) % length + length) % length <= end - start;
        });
        if (yellowHere) flags.push('<span class="done">Yellow flag</span>');
        if (cars.pit[selected] === 1) flags.push('<span class="done">Box this lap</span>');
        if (cars.pit[selected] === 2) flags.push('<span class="grass">Pit lane</span>');
        if (cars.pit[selected] === 3) flags.push(`<span class="slide">Stopped ${cars.serviceLeft[selected].toFixed(1)} s</span>`);
        if (cars.reversing?.[selected]) flags.push('<span class="slide">Reverse</span>');
        if (cars.sliding[selected]) flags.push('<span class="slide">Sliding</span>');
        if (cars.offTrack[selected] && !cars.pit[selected]) flags.push('<span class="grass">On the grass</span>');
        if (cars.draft?.[selected] > 0.3) flags.push('<span class="tow">Tow</span>');
        if (cars.dirtyAir?.[selected] > 0.3) flags.push('<span class="slide">Dirty air</span>');
      }
      if (cars.finishTime[selected] !== null) flags.push(`<span class="done">Finished ${formatTime(cars.finishTime[selected])}</span>`);
      $('tele-flags').innerHTML = flags.join('');

      const lapStart = cars.lapStart?.[selected];
      const running = out || lapStart === null || lapStart === undefined || cars.finishTime[selected] !== null ? null : frame.time - lapStart;
      $('lap-running').textContent = running === null ? '–:–' : formatLap(running).slice(0, -2);
      const delta = running === null ? null : deltas[selected];
      $('lap-delta').textContent = delta === null || delta === undefined ? '' : formatDelta(delta);
      $('lap-delta').className = `delta ${delta < 0 ? 'faster' : 'slower'}`;
      $('lap-last').textContent = cars.lastLap?.[selected] ? formatLap(cars.lastLap[selected]) : '–';
      $('lap-best').textContent = cars.bestLap?.[selected] ? formatLap(cars.bestLap[selected]) : '–';
      const done = sectors[selected] ?? [];
      document.querySelectorAll('#sectors span').forEach((chip, index) => {
        const sector = done[index];
        chip.className = sector ? sector.rating : '';
        chip.textContent = sector ? sector.time.toFixed(1) : `S${index + 1}`;
      });
    },

    // Timing events from the server: a sector closes, a lap closes, maybe a new fastest lap.
    timing(events) {
      for (const event of events) {
        if (event.kind === 'lap') {
          if (event.rating === 'overall') fastestCar = event.car;
          // A personal best becomes what the delta counts against, once this lap's own delta is shown.
          const lap = sectors[event.car];
          if (event.rating !== 'none' && lap.length === 3) bestSplits[event.car] = lap.map(sector => sector.time);
        }
        if (event.kind !== 'sector') continue;
        // The first sector of a lap clears the last lap's.
        if (event.sector === 0) sectors[event.car] = [];
        sectors[event.car][event.sector] = { time: event.time, rating: event.rating };
        const best = bestSplits[event.car];
        const sum = times => times.slice(0, event.sector + 1).reduce((total, time) => total + time, 0);
        deltas[event.car] = best ? sum(sectors[event.car].map(sector => sector.time)) - sum(best) : null;
      }
    },

    banner(title, subtitle = '') {
      const banner = $('banner');
      banner.innerHTML = `${title}${subtitle ? `<small>${subtitle}</small>` : ''}`;
      banner.classList.remove('hidden');
      // Restart the animation even when a banner is already showing.
      banner.style.animation = 'none';
      void banner.offsetWidth;
      banner.style.animation = '';
      clearTimeout(bannerTimer);
      bannerTimer = setTimeout(() => banner.classList.add('hidden'), 2700);
    },

    status(message) {
      const status = $('status');
      status.textContent = message || '';
      status.classList.toggle('hidden', !message);
    },

    formatTime,
  };
}
