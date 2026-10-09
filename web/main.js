// Connects to the server, keeps the last two frames and draws smoothly between them, and wires up the controls.
import * as THREE from 'three';
import { buildCircuit } from './circuit.js';
import { contactPatches, createCar, poseCar } from './cars.js';
import { createCrews } from './crew.js';
import { createEffects } from './effects.js';
import { createBroadcaster } from './broadcaster.js';
import { createDirector } from './director.js';
import { createFeed } from './feed.js';
import { createGhost } from './ghost.js';
import { createHud } from './hud.js';
import { createRaceEvents } from './race-events.js';
import { createRacerPanel } from './racer.js';
import { createStage, toWorld } from './scene.js';
import { createLoadingScreen } from './loading.js';
import { createSpeech } from './speech.js';
import { LOCATIONS, locationFor } from './scenery.js';
import { createWeather } from './weather.js';

const SPEEDS = [0.5, 1, 2, 4, 8];
const PAUSE_ICON = '<svg viewBox="0 0 12 12"><rect x="1.5" y="1" width="3" height="10"/><rect x="7.5" y="1" width="3" height="10"/></svg>';
const PLAY_ICON = '<svg viewBox="0 0 12 12"><path d="M2 1l9 5-9 5z"/></svg>';
// In TV mode the controls step aside after this long without the mouse moving, like a broadcast.
const IDLE_MS = 2500;
const STYLE_KEY = 'slipstream.style';
const FRAME_SECONDS = 1 / 20;
const NUMERIC = ['x', 'y', 'speed', 'steer', 'throttle', 'brake'];
const $ = id => document.getElementById(id);

const stage = createStage($('scene'));
const hud = createHud({ onSelect: id => select(id) });
const racer = createRacerPanel({
  send: message => send(message),
  onYours: car => { state.yours = car; broadcaster.setYours(car); select(car); },
  onFollow: () => select(state.yours),
  onRename: id => hud.rename(id),
});
const raycaster = new THREE.Raycaster();
const loading = createLoadingScreen();
$('loading-retry').addEventListener('click', () => location.reload());
const weather = createWeather(stage);
const raceEvents = createRaceEvents();
const feed = createFeed({ onSelect: id => select(id) });
const ghost = createGhost(stage.scene);
const director = createDirector({ aspect: window.innerWidth / window.innerHeight });
window.addEventListener('resize', () => director.resize(window.innerWidth / window.innerHeight));
const speech = createSpeech({ onProgress: showVoicesProgress });
const broadcaster = createBroadcaster({ onStatus: text => hud.status(text), speech });

const state = {
  socket: null, worker: null, race: null, circuit: null, effects: null, crews: null, cars: [],
  previous: null, current: null, arrived: 0,
  selected: 0, yours: 0, cameraMode: 'follow', shot: null,
  speed: 1, paused: false, announcedLap: 0, celebrated: false, brainLabels: new Map(),
};

// ---- Connection: the local server, or the simulation in a web worker on the published site --------------

function connect() {
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  state.socket = socket;
  socket.addEventListener('open', () => { hud.status(''); startRace(); });
  socket.addEventListener('message', event => receive(JSON.parse(event.data)));
  socket.addEventListener('close', () => {
    hud.status('Lost the server. Retrying…');
    setTimeout(connect, 1500);
  });
}

// With no server behind the page (the published site), the race code runs in a worker (engine-worker.js).
function startWorker() {
  loading.show();
  const worker = new Worker(new URL('./engine-worker.js', import.meta.url), { type: 'module' });
  state.worker = worker;
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'plan') {
      loading.plan(data.expected);
    } else if (data.type === 'progress') {
      loading.progress(data);
    } else if (data.type === 'ready') {
      loading.ready();
      showBrains(data.brains);
      startRace();
    } else if (data.type === 'failed') {
      if (state.race) hud.status(`The simulation stopped: ${data.message}`);
      else loading.fail(data.message);
    } else {
      receive(JSON.parse(data.text));
    }
  });
}

function send(message) {
  if (state.worker) return state.worker.postMessage({ text: JSON.stringify(message) });
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(message));
}

function startRace(sameCircuit = false) {
  send({
    type: 'start', brain: $('brain').value || 'scripted', pitwall: $('pitwall').value || 'driver', weather: $('weather').value,
    cars: Number($('cars').value), laps: Number($('laps').value),
    seed: sameCircuit && state.race ? state.race.seed : undefined,
  });
}

function receive(message) {
  if (message.type === 'race') return setUpRace(message);
  if (message.type === 'frame') return receiveFrame(message);
  if (message.type === 'error') hud.status(message.message);
}

// ---- Race set-up and frames ----------------------------------------------------------------------------

function setUpRace(intro) {
  loading.hide();
  state.circuit?.dispose();
  state.effects?.dispose();
  state.crews?.dispose();
  state.cars.forEach(car => stage.scene.remove(car));
  state.race = intro;
  const chosen = $('location').value;
  const locationName = chosen === 'random' ? locationFor(intro.seed) : chosen;
  state.location = LOCATIONS[locationName];
  state.circuit = buildCircuit(stage.scene, intro.track, intro.grid, intro.seed, intro.pitLane, intro.cars, locationName);
  stage.setLocation(state.location);
  weather.setLocation(state.location);
  state.crews = createCrews(stage.scene, state.circuit.pits.boxes, intro.cars);
  weather.setTarmac(state.circuit.tarmac);
  state.effects = createEffects(stage.scene);
  state.cars = intro.cars.map(({ color, name, number }) => {
    const car = createCar(color, name, number);
    stage.scene.add(car);
    return car;
  });
  const grid = {
    x: intro.grid.map(slot => slot.x), y: intro.grid.map(slot => slot.y), heading: intro.grid.map(slot => slot.heading),
    speed: intro.cars.map(() => 0), steer: intro.cars.map(() => 0), throttle: intro.cars.map(() => 0), brake: intro.cars.map(() => 0),
    sliding: intro.cars.map(() => 0), offTrack: intro.cars.map(() => 0), tyreWear: intro.cars.map(() => 0), fuel: intro.cars.map(() => 1),
    lap: intro.cars.map(() => 0), progress: intro.cars.map(() => 0), finishTime: intro.cars.map(() => null),
    compound: intro.cars.map(() => 'medium'), wing: intro.cars.map(() => 0.5), engine: intro.cars.map(() => 0.5),
    damage: intro.cars.map(() => 0), brakeWear: intro.cars.map(() => 0), pit: intro.cars.map(() => 0),
    stops: intro.cars.map(() => 0), serviceLeft: intro.cars.map(() => 0),
    lapStart: intro.cars.map(() => null), lastLap: intro.cars.map(() => null), bestLap: intro.cars.map(() => null),
    tyreTemp: intro.cars.map(() => 0.45), brakeTemp: intro.cars.map(() => 0.1), engineTemp: intro.cars.map(() => 0.4),
    punctured: intro.cars.map(() => 0), retired: intro.cars.map(() => 0), wingDamage: intro.cars.map(() => 0),
    suspensionDamage: intro.cars.map(() => 0), draft: intro.cars.map(() => 0), dirtyAir: intro.cars.map(() => 0),
    balance: intro.cars.map(() => 0),
  };
  state.previous = state.current = {
    time: 0, done: false, cars: grid, order: intro.cars.map((_, id) => id),
    yourPit: { strategy: intro.strategy, boxCalled: false }, weather: { wetness: 0, rain: 0 }, debris: [], surface: null,
    events: { contacts: [], laps: [], finished: [], pitEntered: [], pitStopped: [], pitReleased: [], pitExited: [], timing: [], fastestLap: null,
              punctures: [], engineFailures: [], mistakes: [], debris: [] },
  };
  state.arrived = performance.now();
  state.yours = intro.yours;
  // The grid is drawn afresh each race, so the car to follow at the start is yours.
  state.selected = intro.yours;
  racer.setRace(intro);
  racer.setStats(intro);
  state.announcedLap = 0;
  state.celebrated = false;
  hud.setRace(intro);
  $('seed').textContent += ` · ${state.location.label}`;
  raceEvents.reset(intro);
  feed.reset(intro);
  ghost.reset();
  director.reset(intro, state.circuit);
  broadcaster.reset(intro, intro.yours);
  const first = intro.grid[state.selected];
  stage.view.focus.copy(toWorld(first.x, first.y));
  stage.view.setMode(state.cameraMode, state.circuit);
  showGrid(intro);
}

// Every race waits on the grid (the server holds it there) until you've picked your car and set it up.
function showGrid(intro) {
  const weatherChoice = $('weather').selectedOptions[0].text.toLowerCase();
  $('grid-info').textContent = `${state.location.label} · ${intro.laps} laps of a ${Math.round(intro.track.length)} m circuit`
    + ` · weather ${weatherChoice} · ${intro.cars.length} cars`;
  openDrawer(null);
  $('grid-menu').classList.remove('hidden');
  document.body.classList.add('on-grid');
}

function lightsOut() {
  send({ type: 'go' });
  $('grid-menu').classList.add('hidden');
  document.body.classList.remove('on-grid');
  const intro = state.race;
  const strategy = state.brainLabels.get(intro.pitwall) ?? intro.pitwall;
  hud.banner(state.brainLabels.get(intro.brain) ?? intro.brain, `${intro.laps} laps · ${Math.round(intro.track.length)} m circuit · pit wall: ${strategy}`);
}

function receiveFrame(frame) {
  if (!state.race) return;
  state.previous = interpolated(performance.now());
  state.current = frame;
  state.arrived = performance.now();

  for (const [x, y, impulse] of frame.events.contacts) state.effects.contact(toWorld(x, y), impulse);
  // Debris is the race's own (punctures come from it): each new piece flies from the hardest hit to where it landed,
  // in the colours of the cars involved, and disappears when a car runs over it.
  if (frame.events.debris.length) {
    const [hitX, hitY] = frame.events.contacts.reduce((hardest, contact) => (contact[2] > hardest[2] ? contact : hardest), [...frame.events.debris[0], 0]);
    const colors = nearestCars(frame.cars, hitX, hitY, 2).map(car => state.race.cars[car].color);
    state.effects.scatter(toWorld(hitX, hitY), frame.events.debris.map(([x, y]) => [x, -y, debrisKey(x, y)]), colors);
  }
  state.effects.keepDebris(new Set(frame.debris.map(([x, y]) => debrisKey(x, y))));
  for (const car of frame.events.engineFailures) state.effects.blowUp(toWorld(frame.cars.x[car], frame.cars.y[car]));
  if (frame.surface) state.circuit.setSurface(frame.surface, frame.weather.wetness);
  hud.timing(frame.events.timing);
  const fastest = frame.events.fastestLap;
  if (fastest) ghost.setTrace(fastest, state.race.cars[fastest.car].color);
  const moments = raceEvents.read(frame);
  feed.add(moments);
  director.notice(moments, frame, performance.now());
  broadcaster.notice(moments);
  for (const stop of frame.events.pitStopped) {
    state.crews.startStop(stop, frame.time, { compound: frame.cars.compound[stop.car] });
    if (stop.car === state.yours || stop.car === state.selected) {
      const jobs = Object.entries(stop.jobs).filter(([name, seconds]) => seconds > 0 && name !== 'tyres').map(([name]) => name);
      hud.banner(`${state.race.cars[stop.car].name} pits`, `${stop.compound} tyres${jobs.length ? ' · ' + jobs.join(' · ') : ''} · ${stop.standing.toFixed(1)} s`);
    }
  }
  for (const car of frame.events.pitReleased) {
    const mesh = state.cars[car];
    if (mesh) state.effects.puff(contactPatches(mesh).slice(2));
  }
  const leader = frame.order[0];
  const leaderLap = frame.cars.lap[leader];
  if (leaderLap > state.announcedLap && leaderLap < state.race.laps) {
    state.announcedLap = leaderLap;
    hud.banner(leaderLap === state.race.laps - 1 ? 'Final lap' : `Lap ${leaderLap + 1}`, `${state.race.cars[leader].name} leads`);
  }
  if (frame.events.finished.length && !state.celebrated) {
    state.celebrated = true;
    const winner = frame.order[0];
    hud.banner(`${state.race.cars[winner].name} wins`, hud.formatTime(frame.cars.finishTime[winner]));
    const [x, y] = state.race.track.points[0];
    state.effects.celebrate(toWorld(x, y));
  }
}

// Every car's state, blended between the last two frames by how far we are into the current one.
function interpolated(now) {
  const { previous, current } = state;
  if (!current) return null;
  const blend = Math.min(1, (now - state.arrived) / (FRAME_SECONDS * 1000));
  const cars = { ...current.cars };
  for (const key of NUMERIC) cars[key] = current.cars[key].map((value, id) => previous.cars[key][id] + (value - previous.cars[key][id]) * blend);
  cars.heading = current.cars.heading.map((value, id) => {
    const from = previous.cars.heading[id];
    const turn = Math.atan2(Math.sin(value - from), Math.cos(value - from));
    return from + turn * blend;
  });
  return { ...current, time: previous.time + (current.time - previous.time) * blend, cars };
}

// ---- Rendering -----------------------------------------------------------------------------------------

// 120 Hz displays would otherwise draw every frame (and its shadow pass) twice as often as anyone can tell
// apart. A paused race only needs enough frames to keep the camera smooth.
const RACING_FRAME_MS = 1000 / 60;
const PAUSED_FRAME_MS = 1000 / 30;
// requestAnimationFrame timing jitters by a millisecond or two; without slack a 60 Hz display would skip frames.
const FRAME_SLACK_MS = 2;

let lastRender = performance.now();
function render(now) {
  requestAnimationFrame(render);
  const frameMs = state.paused || !state.race ? PAUSED_FRAME_MS : RACING_FRAME_MS;
  if (now - lastRender < frameMs - FRAME_SLACK_MS) return;
  const seconds = Math.min(0.1, (now - lastRender) / 1000);
  lastRender = now;
  const frame = state.race && interpolated(now);
  if (frame) {
    const simRate = state.paused ? 0 : Math.max(0, state.current.time - state.previous.time) / FRAME_SECONDS;
    state.cars.forEach((car, id) => {
      const pose = carState(frame.cars, id);
      // A broadcast doesn't circle anyone, so TV mode drops the selection ring.
      const ringed = id === state.selected && state.cameraMode !== 'tv';
      car.userData.lightsOn = !!state.location?.night;
      car.userData.raining = frame.weather.wetness > 0.3;
      poseCar(car, pose, seconds, seconds * simRate, ringed, id === state.yours, now / 1000);
      const velocity = carVelocity(pose);
      for (const point of car.userData.justLost.splice(0)) state.effects.shed(point, velocity, state.race.cars[id].color, 3, 0.45);
      if (simRate > 0) state.effects.car(id, contactPatches(car), pose, velocity, seconds * Math.min(simRate, 2), frame.weather.wetness);
    });
    state.circuit.update(frame.time, Math.hypot(...(frame.weather.wind ?? [0, 0])));
    state.crews.update(frame.time, frame, state.cars);
    racer.update(frame);
    const height = stage.pixelsTall();
    let pixelsPerMetre = height / (2 * stage.view.zoom);
    state.shot = state.cameraMode === 'tv' ? director.update(frame, seconds, now) : null;
    if (state.shot) {
      // The isometric rig still follows the shot's subject, because the sun's shadow box follows the rig.
      stage.view.update(seconds, state.shot.focus, 0);
      pixelsPerMetre = height / (2 * Math.tan(THREE.MathUtils.degToRad(director.camera.fov) / 2) * state.shot.distance);
    } else {
      aimCamera(frame, seconds);
    }
    showShot(state.shot, frame);
    weather.update(frame.weather.wetness, frame.weather.rain, state.shot ? state.shot.focus : stage.view.focus, seconds, now / 1000, frame.weather.wind);
    const lapStart = frame.cars.lapStart[state.selected];
    ghost.update(lapStart === null || frame.cars.finishTime[state.selected] !== null ? null : frame.time - lapStart);
    hud.update(frame, state.selected, state.yours, now);
    broadcaster.update(commentaryContext(frame));
    state.effects.update(seconds * Math.max(simRate, 0.25), pixelsPerMetre);
  }
  state.circuit?.cutAway(activeCamera().position, stage.view.focus);
  stage.render(activeCamera());
}

function activeCamera() {
  return state.shot ? director.camera : stage.camera;
}

// The TV graphic in the corner: which camera, and whose car on the onboard.
function showShot(shot, frame) {
  const tag = $('tv-tag');
  if (!shot || !shot.label) { tag.classList.add('hidden'); return; }
  tag.classList.remove('hidden');
  const car = state.race.cars[shot.car];
  const speed = shot.kind === 'onboard' ? ` · ${Math.round(Math.max(0, frame.cars.speed[shot.car]) * 3.6)} km/h` : '';
  tag.innerHTML = `<b>${shot.label}</b>${shot.kind === 'onboard' || shot.kind === 'pitlane' ? `<span class="chip" style="background:${car.color}"></span>${car.name}${speed}` : ''}`;
}

// What the commentator may say when it's quiet: the order, the gap at the front, the lap.
function commentaryContext(frame) {
  const { cars, order } = frame;
  const name = car => state.race.cars[car].name;
  const gap = car => Math.max(0, (cars.progress[order[0]] - cars.progress[car]) / Math.max(cars.speed[car], 15));
  return {
    racing: frame.time > 3 && !frame.done && cars.finishTime[order[0]] === null,
    lap: Math.min(cars.lap[order[0]] + 1, state.race.laps), laps: state.race.laps,
    leader: name(order[0]), second: order.length > 1 ? name(order[1]) : '', leadGap: order.length > 1 ? gap(order[1]).toFixed(1) : '0',
    order: order.slice(0, 6).map((car, place) => `P${place + 1} ${name(car)}${place ? ` (+${gap(car).toFixed(1)}s)` : ''} on ${cars.compound[car]} tyres, ${cars.stops[car]} stops`).join('; '),
  };
}

function debrisKey(x, y) {
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function nearestCars(cars, x, y, count) {
  return cars.x.map((carX, id) => [Math.hypot(carX - x, cars.y[id] - y), id]).sort((a, b) => a[0] - b[0]).slice(0, count).map(([, id]) => id);
}

function carVelocity(pose) {
  return new THREE.Vector3(Math.cos(pose.heading), 0, -Math.sin(pose.heading)).multiplyScalar(pose.speed);
}

function carState(cars, id) {
  const value = {};
  for (const key of Object.keys(cars)) value[key] = cars[key][id];
  return value;
}

function aimCamera(frame, seconds) {
  const target = state.selected;
  const point = toWorld(frame.cars.x[target], frame.cars.y[target]);
  // Look a little ahead of the car, so there's more road in front than behind.
  const heading = frame.cars.heading[target];
  point.add(new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading)).multiplyScalar(Math.min(frame.cars.speed[target], 60) * 0.25));
  stage.view.update(seconds, point, frame.cars.speed[target]);
}

// ---- Controls ------------------------------------------------------------------------------------------

function select(id) {
  state.selected = id;
  if (state.cameraMode !== 'follow') setCamera('follow');
}

function setCamera(mode) {
  state.cameraMode = mode;
  noticeActivity();
  stage.view.setMode(mode, state.circuit);
  document.querySelectorAll('#camera [data-camera]').forEach(button => button.classList.toggle('active', button.dataset.camera === mode));
}

function setSpeed(speed) {
  state.speed = speed;
  send({ type: 'speed', value: speed });
  document.querySelectorAll('#speeds button').forEach(button => button.classList.toggle('active', Number(button.dataset.speed) === speed));
}

function setPaused(paused) {
  state.paused = paused;
  send({ type: 'pause', value: paused });
  $('pause').innerHTML = paused ? PLAY_ICON : PAUSE_ICON;
  $('pause').title = paused ? 'Resume (Space)' : 'Pause (Space)';
  $('pause').setAttribute('aria-label', paused ? 'Resume' : 'Pause');
}

// The server lists every snapshot as training saves it; the published site has the ones published with it.
async function serverBrains() {
  try {
    const response = await fetch('api/brains');
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

function showBrains({ drivers, pitwalls }) {
  state.brainLabels = new Map([...drivers, ...pitwalls].map(brain => [brain.id, brain.label]));
  // Default to the newest snapshot of any run, since watching training is the point.
  fillPicker($('brain'), drivers, drivers[1]?.id ?? 'scripted');
  fillPicker($('pitwall'), pitwalls, pitwalls[2]?.id ?? 'driver');
}

function fillPicker(picker, choices, fallback) {
  const wanted = picker.value;
  picker.replaceChildren(...choices.map(choice => new Option(choice.label, choice.id)));
  picker.value = choices.some(choice => choice.id === wanted) ? wanted : fallback;
}

SPEEDS.forEach(speed => {
  const button = document.createElement('button');
  button.textContent = `${speed}×`;
  button.dataset.speed = speed;
  button.addEventListener('click', () => setSpeed(speed));
  $('speeds').append(button);
});
$('new-race').addEventListener('click', () => startRace());
$('new-circuit').addEventListener('click', () => startRace());
$('replay').addEventListener('click', () => startRace(true));
$('go').addEventListener('click', lightsOut);
$('pause').addEventListener('click', () => setPaused(!state.paused));
$('brain').addEventListener('change', () => startRace(true));
$('location').addEventListener('change', () => startRace(true));
$('pitwall').addEventListener('change', () => startRace(true));
document.querySelectorAll('#camera [data-camera]').forEach(button => button.addEventListener('click', () => setCamera(button.dataset.camera)));

window.addEventListener('keydown', event => {
  if (event.key === 'Escape') return openDrawer(null);
  if (event.target.matches('input, select')) return;
  if (document.body.classList.contains('on-grid')) return;
  if (event.key === 'q' || event.key === 'Q') stage.view.rotate(-1);
  else if (event.key === 'e' || event.key === 'E') stage.view.rotate(1);
  else if (event.key === ' ') { event.preventDefault(); setPaused(!state.paused); }
  else if (event.key === 'f') setCamera('follow');
  else if (event.key === 't') setCamera('tv');
  else if (event.key === 'o') setCamera('overview');
  else if (event.key === 'g') setGhost(!ghost.enabled);
  else if (event.key === 'p') setStyle(stage.style === 'pixel' ? 'classic' : 'pixel');
  else if (/^[1-8]$/.test(event.key) && state.race && Number(event.key) <= state.race.cars.length) select(Number(event.key) - 1);
});
$('scene').addEventListener('wheel', event => { event.preventDefault(); stage.view.zoomBy(Math.exp(event.deltaY * 0.0012)); }, { passive: false });
$('scene').addEventListener('click', event => {
  const pointer = new THREE.Vector2((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, activeCamera());
  const hit = raycaster.intersectObjects(state.cars, true)[0];
  if (!hit) return;
  let object = hit.object;
  while (object && !state.cars.includes(object)) object = object.parent;
  if (object) select(state.cars.indexOf(object));
});

function setGhost(enabled) {
  ghost.setEnabled(enabled);
  $('ghost').classList.toggle('active', enabled);
}

function setStyle(style) {
  stage.setStyle(style);
  document.querySelectorAll('[data-style]').forEach(button => button.classList.toggle('active', button.dataset.style === style));
  try { localStorage.setItem(STYLE_KEY, style); } catch { /* blocked storage just forgets the choice */ }
}

function savedStyle() {
  try { return localStorage.getItem(STYLE_KEY) === 'pixel' ? 'pixel' : 'classic'; } catch { return 'classic'; }
}

function setCommentary(mode) {
  broadcaster.setMode(mode);
  $('commentary').value = mode;
}

// The pickers fill again when voices arrive: the browser loads its own late (Chrome only after asking), and the
// natural ones once they've downloaded.
function fillVoices() {
  const { natural, browser } = speech.options();
  const group = (label, options) => {
    const element = document.createElement('optgroup');
    element.label = label;
    element.append(...options.map(({ id, label: name }) => new Option(name, id)));
    return element;
  };
  for (const role of speech.roles) {
    const picker = $(`voice-${role}`);
    picker.replaceChildren(new Option('Automatic', ''), ...(natural.length ? [group('Natural voices', natural)] : []),
      ...(browser.length ? [group("Your browser's voices", browser)] : []));
    const wanted = speech.chosenVoice(role);
    picker.value = [...natural, ...browser].some(({ id }) => id === wanted) ? wanted : '';
  }
}

function showVoicesProgress({ state, text }) {
  $('voices-status').textContent = text;
  $('voices-status').classList.remove('hidden');
  $('load-voices').disabled = state === 'loading' || state === 'ready';
  $('load-voices').textContent = state === 'ready' ? 'Natural voices on' : 'Load natural voices';
  if (state === 'ready') {
    fillVoices();
    setTimeout(() => $('voices-status').classList.add('hidden'), 3000);
  }
}

for (const role of speech.roles) $(`voice-${role}`).addEventListener('change', event => speech.setVoice(role, event.target.value));
document.querySelectorAll('[data-preview]').forEach(button => button.addEventListener('click', () => speech.preview(button.dataset.preview)));
$('load-voices').addEventListener('click', () => speech.loadNatural());
if ('speechSynthesis' in window) speechSynthesis.addEventListener('voiceschanged', fillVoices);
fillVoices();
// Loaded on an earlier visit, so it comes from the browser's cache this time.
if (speech.wantsNatural()) speech.loadNatural();

$('ghost').addEventListener('click', () => setGhost(!ghost.enabled));

// Drawers: race setup and your racer, one open at a time, closed with their × or Escape.
function openDrawer(id) {
  document.querySelectorAll('.drawer').forEach(drawer => drawer.classList.toggle('hidden', drawer.id !== id));
  document.querySelectorAll('[data-drawer]').forEach(button => button.classList.toggle('open', button.dataset.drawer === id));
}
document.querySelectorAll('[data-drawer]').forEach(button => button.addEventListener('click', () => {
  openDrawer($(button.dataset.drawer).classList.contains('hidden') ? button.dataset.drawer : null);
}));
document.querySelectorAll('.drawer .close').forEach(button => button.addEventListener('click', () => openDrawer(null)));

let idleTimer = null;
function noticeActivity() {
  document.body.classList.remove('watching');
  clearTimeout(idleTimer);
  if (state.cameraMode === 'tv') idleTimer = setTimeout(() => document.body.classList.add('watching'), IDLE_MS);
}
window.addEventListener('pointermove', noticeActivity);
$('commentary').addEventListener('change', event => setCommentary(event.target.value));
$('load-commentator').addEventListener('click', async () => {
  $('load-commentator').disabled = true;
  await broadcaster.loadModel();
  $('load-commentator').disabled = broadcaster.modelState === 'ready';
  $('load-commentator').textContent = broadcaster.modelState === 'ready' ? 'AI commentator on' : 'Load AI commentator';
  if (broadcaster.modelState === 'ready') setTimeout(() => hud.status(''), 3000);
});

$('pause').innerHTML = PAUSE_ICON;
document.querySelectorAll('[data-style]').forEach(button => button.addEventListener('click', () => setStyle(button.dataset.style)));
setStyle(savedStyle());
setSpeed(1);
setCamera('follow');
setGhost(false);
setCommentary(broadcaster.mode);
// It was loaded on an earlier visit, so it comes from the browser's cache this time.
if (broadcaster.wantsModel()) $('load-commentator').click();
serverBrains().then(brains => {
  if (!brains) return startWorker();
  // The local server is quick to start a race: no downloads to wait for.
  loading.hide();
  showBrains(brains);
  connect();
  setInterval(() => serverBrains().then(listed => listed && showBrains(listed)), 30000);
});
requestAnimationFrame(render);
