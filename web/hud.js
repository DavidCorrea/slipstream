// The overlay: standings tower, lap counter and clock, the selected car's telemetry and lap timing, and banners.
import { formatLap } from './feed.js';

const $ = id => document.getElementById(id);
// Each compound's working window in the race's 0-1 temperature (see slipstream/car.py TYRE_WINDOW, ±0.06).
const TYRE_WINDOWS = { soft: [0.49, 0.61], medium: [0.54, 0.66], hard: [0.59, 0.71], intermediate: [0.36, 0.48], wet: [0.26, 0.38] };

// A temperature in degrees, coloured by where it sits against the range it works best in.
function showTemperature(id, value, [low, high], coldest, hottest) {
  const element = $(id);
  if (value === undefined) { element.textContent = '–'; return; }
  element.textContent = `${Math.round(coldest + value * (hottest - coldest))}°`;
  element.className = value < low ? 'cold' : value > high + 0.15 ? 'overheating' : value > high ? 'hot' : '';
}

export function createHud({ onSelect }) {
  const tower = $('tower');
  let race = null, rows = [], lastOrder = [], changed = new Map(), bannerTimer = null;
  // Each car's sector times on its current lap ({ time, rating } per sector), and who holds the fastest lap.
  let sectors = [], fastestCar = null;

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
        row.innerHTML = `<span class="place"></span><span class="chip" style="background:${car.color}"></span><span class="name">${car.name}<span class="yours"></span><span class="pitting"></span><span class="fastest" title="Fastest lap">⏱</span></span><span class="tyre"></span><span class="gap"></span>`;
        row.addEventListener('click', () => onSelect(id));
        return row;
      });
      lastOrder = [];
      changed.clear();
      sectors = intro.cars.map(() => []);
      fastestCar = null;
      $('seed').textContent = `circuit #${intro.seed}`;
    },

    rename(id) {
      rows[id].querySelector('.name').firstChild.textContent = race.cars[id].name;
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
      $('tele-position').textContent = `P${order.indexOf(selected) + 1}`;
      $('tele-speed').textContent = Math.round(Math.max(0, cars.speed[selected]) * 3.6);
      $('tele-throttle').style.height = `${cars.throttle[selected] * 100}%`;
      $('tele-brake').style.height = `${cars.brake[selected] * 100}%`;
      $('tele-tyres').style.width = `${(1 - cars.tyreWear[selected]) * 100}%`;
      $('tele-fuel').style.width = `${cars.fuel[selected] * 100}%`;
      $('tele-damage').style.width = `${cars.damage[selected] * 100}%`;
      $('tele-brakes').style.width = `${(1 - cars.brakeWear[selected]) * 100}%`;
      const engineMode = cars.engine[selected] > 0.66 ? 'push' : cars.engine[selected] < 0.33 ? 'lean' : 'standard';
      const balance = cars.balance?.[selected] ?? 0;
      const handling = balance > 0.2 ? ' · oversteers' : balance < -0.2 ? ' · understeers' : '';
      $('tele-setup').textContent = `${cars.compound[selected]} tyres · wing ${Math.round(cars.wing[selected] * 100)}% · ${engineMode} engine · ${cars.stops[selected]} stop${cars.stops[selected] === 1 ? '' : 's'}${handling}`;
      showTemperature('temp-tyres', cars.tyreTemp?.[selected], TYRE_WINDOWS[cars.compound[selected]], 20, 140);
      showTemperature('temp-brakes', cars.brakeTemp?.[selected], [0.3, 0.85], 100, 1100);
      showTemperature('temp-engine', cars.engineTemp?.[selected], [0.35, 0.8], 70, 130);
      $('tele-steer').style.left = `${50 + cars.steer[selected] * -46}%`;
      const flags = [];
      if (cars.sliding[selected]) flags.push('<span class="slide">Sliding</span>');
      if (cars.offTrack[selected] && !cars.pit[selected]) flags.push('<span class="grass">On the grass</span>');
      if (cars.pit[selected] === 1) flags.push('<span class="done">Pitting this lap</span>');
      if (cars.pit[selected] === 2) flags.push('<span class="grass">Pit lane</span>');
      if (cars.pit[selected] === 3) flags.push(`<span class="slide">Stopped ${cars.serviceLeft[selected].toFixed(1)} s</span>`);
      if (cars.finishTime[selected] !== null) flags.push(`<span class="done">Finished ${formatTime(cars.finishTime[selected])}</span>`);
      if (cars.draft?.[selected] > 0.3) flags.push('<span class="tow">Tow</span>');
      if (cars.dirtyAir?.[selected] > 0.3) flags.push('<span class="slide">Dirty air</span>');
      if (cars.punctured?.[selected]) flags.push('<span class="warning">Puncture</span>');
      if (cars.retired?.[selected]) flags.push('<span class="warning">Engine failure</span>');
      $('tele-flags').innerHTML = flags.join('');

      const lapStart = cars.lapStart?.[selected];
      const running = lapStart === null || lapStart === undefined || cars.finishTime[selected] !== null ? null : frame.time - lapStart;
      $('lap-running').textContent = running === null ? '–:–' : formatLap(running).slice(0, -2);
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
        if (event.kind === 'lap' && event.rating === 'overall') fastestCar = event.car;
        if (event.kind !== 'sector') continue;
        // The first sector of a lap clears the last lap's.
        if (event.sector === 0) sectors[event.car] = [];
        sectors[event.car][event.sector] = { time: event.time, rating: event.rating };
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
