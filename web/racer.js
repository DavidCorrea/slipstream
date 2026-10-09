// The grid menu and your pit wall. On the grid: the field as cards, which car is yours, and sliders for whichever
// driver you're setting up and their car (sent to the server as you drag, a few times a second at most). During
// the race: your pit wall, with the strategy, the stop plan and the box call.
const $ = id => document.getElementById(id);

const PERSONALITY = [
  { name: 'aggression', label: 'Aggression', low: 'Avoids contact', high: 'Leans on others' },
  { name: 'risk', label: 'Risk', low: 'Keeps a margin', high: 'On the limit' },
  { name: 'overtaking', label: 'Overtaking', low: 'Holds station', high: 'Hunts places' },
  { name: 'conservation', label: 'Conservation', low: 'Pushes', high: 'Saves tyres and fuel' },
  { name: 'consistency', label: 'Consistency', low: 'Erratic', high: 'Metronomic' },
  { name: 'stamina', label: 'Stamina', low: 'Tires early', high: 'Fresh to the flag' },
];
const CAR = [
  { name: 'top_speed', label: 'Top speed', unit: value => `${Math.round(value * 3.6)} km/h` },
  { name: 'acceleration', label: 'Acceleration', unit: value => `${value.toFixed(1)} m/s²` },
  { name: 'braking', label: 'Braking', unit: value => `${value.toFixed(0)} m/s²` },
  { name: 'grip', label: 'Grip', unit: value => `${value.toFixed(2)} g` },
  // Handling balance has no physical unit: the slider runs from understeer through neutral to oversteer.
  { name: 'balance', label: 'Handling', low: 'Understeer', high: 'Oversteer',
    describe: slider => (slider < 0.4 ? 'understeer' : slider > 0.6 ? 'oversteer' : 'neutral') },
];
const PLAN = [
  { name: 'fuel', label: 'Fuel to', unit: (value, intro) => `${Math.round(value * intro.fuelCapacity)} kg` },
  { name: 'wing', label: 'Wing', unit: value => `${Math.round(value * 100)}%`, low: 'Top speed', high: 'Cornering grip' },
  { name: 'engine', label: 'Engine', unit: value => (value > 0.66 ? 'push' : value < 0.33 ? 'lean' : 'standard'), low: 'Saves fuel', high: 'More power' },
];
const SEND_EVERY = 120;   // milliseconds
// How a driver's character reads on their card: the traits furthest from the middle, in words.
const CHARACTER = {
  aggression: ['gentle', 'aggressive'], risk: ['cautious', 'on the limit'], overtaking: ['patient', 'a hunter'],
  conservation: ['pushes', 'saves tyres'], consistency: ['erratic', 'metronomic'], stamina: ['tires early', 'tireless'],
};
const STANDS_OUT = 0.2;
// The server's rules for a name (cast.tidy_name), checked here too so a refused name never shows anywhere.
const NAME = /^\p{L}[\p{L} .'-]*$/u;
const NAME_LENGTH = 16;

function nameProblem(name, others) {
  if (!name) return 'A name cannot be empty.';
  if (name.length > NAME_LENGTH) return `A name can be at most ${NAME_LENGTH} characters.`;
  if (!NAME.test(name)) return "A name is letters, spaces and . ' - starting with a letter.";
  const holder = others.find(other => other.toLowerCase() === name.toLowerCase());
  if (holder) return `${holder} already has that name.`;
  return null;
}

function character(stats) {
  const marked = Object.keys(CHARACTER).map(name => ({ name, lean: stats[name] - 0.5 }))
    .filter(({ lean }) => Math.abs(lean) >= STANDS_OUT)
    .sort((first, second) => Math.abs(second.lean) - Math.abs(first.lean)).slice(0, 2);
  if (!marked.length) return 'all-rounder';
  return marked.map(({ name, lean }) => CHARACTER[name][lean > 0 ? 1 : 0]).join(' · ');
}

export function createRacerPanel({ send, onYours, onFollow, onRename }) {
  const inputs = new Map(), planInputs = new Map();
  let ranges = {}, pending = {}, timer = null, current = null, plan = {}, boxCalled = false, editing = 0;

  // The field in grid order, as cards in their colours: yours is marked, and the one being set up is outlined.
  function showCars() {
    $('car-picker').replaceChildren(...current.cars.map((car, id) => {
      const card = document.createElement('li');
      const isYours = id === current.yours;
      card.innerHTML = `<span class="livery" style="background:${car.color}"></span><span class="slot">P${id + 1}</span>`
        + `<span class="who"><span class="name"><span></span><small>#${car.number}</small></span>`
        + `<span class="character">${car.edited ? '<span class="edited">edited · </span>' : ''}${character(current.stats[id])}</span></span>`
        + (isYours ? '<span class="yours-tag">★ You</span>' : '<button class="secondary drive">Drive</button>');
      card.querySelector('.name span').textContent = car.name;
      card.classList.toggle('chosen', isYours);
      card.classList.toggle('editing', id === editing);
      card.addEventListener('click', () => edit(id));
      card.querySelector('.drive')?.addEventListener('click', event => {
        event.stopPropagation();
        send({ type: 'yours', car: id });
        current.yours = id;
        edit(id);
        onYours(id);
      });
      return card;
    }));
  }

  // Shows one driver's sliders; changes still waiting go to the driver they were meant for first.
  function edit(id) {
    flush();
    editing = id;
    const name = current.cars[id].name;
    $('driver-name').value = name;
    showNameProblem(null);
    $('driver-heading').textContent = id === current.yours ? 'Your driver' : `${name}, the driver`;
    $('car-heading').textContent = id === current.yours ? 'Your car' : `${name}'s car`;
    for (const [stat, { input }] of inputs) {
      input.value = current.stats[id][stat];
      show(stat);
    }
    showCars();
  }

  function slider(container, { name, label, low, high }) {
    const row = document.createElement('label');
    row.className = 'slider';
    row.innerHTML = `<span>${label}</span><span class="value"></span><input type="range" min="0" max="1" step="0.01" value="0.5">`
      + (low ? `<span class="ends"><span>${low}</span><span>${high}</span></span>` : '');
    const input = row.querySelector('input');
    input.addEventListener('input', () => {
      show(name);
      pending[name] = Number(input.value);
      current.stats[editing][name] = Number(input.value);
      if (!current.cars[editing].edited) {
        current.cars[editing].edited = true;
        showCars();
      }
      if (!timer) timer = setTimeout(flush, SEND_EVERY);
    });
    container.append(row);
    inputs.set(name, { input, value: row.querySelector('.value') });
  }

  function show(name) {
    const { input, value } = inputs.get(name);
    const slider = Number(input.value);
    const car = CAR.find(item => item.name === name);
    if (car?.describe) {
      value.textContent = car.describe(slider);
    } else if (car && ranges[name]) {
      const [low, high] = ranges[name];
      value.textContent = car.unit(low + (high - low) * slider);
    } else {
      value.textContent = `${Math.round(slider * 100)}%`;
    }
  }

  function flush() {
    clearTimeout(timer);
    timer = null;
    if (Object.keys(pending).length) send({ type: 'stats', car: editing, stats: pending });
    pending = {};
  }

  PERSONALITY.forEach(item => slider($('personality-sliders'), item));
  CAR.forEach(item => slider($('car-sliders'), item));

  // ---- Pit wall --------------------------------------------------------------------------------------
  const sendPlan = changes => { Object.assign(plan, changes); send({ type: 'pit', plan: changes }); showPlan(); };
  function showPlan() {
    document.querySelectorAll('.compounds button').forEach(button => button.classList.toggle('active', button.dataset.compound === plan.compound));
    for (const item of PLAN) {
      const { input, value } = planInputs.get(item.name);
      input.value = plan[item.name];
      value.textContent = item.unit(Number(input.value), current ?? { fuelCapacity: 60 });
    }
    $('plan-repair').checked = !!plan.repair;
    $('plan-brakes').checked = !!plan.brakes;
  }
  for (const item of PLAN) {
    const row = document.createElement('label');
    row.className = 'slider';
    row.innerHTML = `<span>${item.label}</span><span class="value"></span><input type="range" min="0" max="1" step="0.01">`
      + (item.low ? `<span class="ends"><span>${item.low}</span><span>${item.high}</span></span>` : '');
    const input = row.querySelector('input');
    input.addEventListener('change', () => sendPlan({ [item.name]: Number(input.value) }));
    input.addEventListener('input', () => { row.querySelector('.value').textContent = item.unit(Number(input.value), current ?? { fuelCapacity: 60 }); });
    $('plan-sliders').append(row);
    planInputs.set(item.name, { input, value: row.querySelector('.value') });
  }
  document.querySelectorAll('.compounds button').forEach(button => button.addEventListener('click', () => sendPlan({ compound: button.dataset.compound })));
  $('plan-repair').addEventListener('change', event => sendPlan({ repair: event.target.checked }));
  $('plan-brakes').addEventListener('change', event => sendPlan({ brakes: event.target.checked }));
  document.querySelectorAll('[data-strategy]').forEach(button => button.addEventListener('click', () => {
    send({ type: 'pit', strategy: button.dataset.strategy });
    showStrategy(button.dataset.strategy);
  }));
  $('box').addEventListener('click', () => send({ type: 'pit', box: !boxCalled }));
  function showStrategy(strategy) {
    document.querySelectorAll('[data-strategy]').forEach(button => button.classList.toggle('active', button.dataset.strategy === strategy));
    $('pit-plan').classList.toggle('disabled', strategy !== 'mine');
  }

  function showNameProblem(problem) {
    $('name-problem').textContent = problem ?? '';
    $('name-problem').classList.toggle('hidden', !problem);
  }
  const typedName = () => $('driver-name').value.trim().replace(/\s+/g, ' ');
  const otherNames = () => Object.entries(current.castNames).filter(([key]) => key !== current.cars[editing].key).map(([, name]) => name);
  $('driver-name').addEventListener('input', () => showNameProblem(nameProblem(typedName(), otherNames())));
  $('driver-name').addEventListener('change', () => {
    const name = typedName();
    if (nameProblem(name, otherNames()) || name === current.cars[editing].name) return;
    send({ type: 'rename', car: editing, name });
    current.cars[editing].name = name;
    current.castNames[current.cars[editing].key] = name;
    edit(editing);
    onRename(editing);
  });

  $('reset-stats').addEventListener('click', () => {
    pending = {};
    send({ type: 'reset', car: editing });
    current.stats[editing] = { ...current.cars[editing].usual };
    current.cars[editing].edited = false;
    edit(editing);
  });
  $('follow-yours').addEventListener('click', () => onFollow());

  return {
    // Every frame: the box button and the status line follow what your car is doing.
    update(frame) {
      if (!current) return;
      const car = current.yours;
      boxCalled = frame.yourPit?.boxCalled ?? false;
      const box = $('box');
      box.classList.toggle('called', boxCalled);
      box.textContent = boxCalled ? 'Called in · cancel' : 'Box this lap';
      const pit = frame.cars.pit[car];
      const lines = {
        0: frame.yourPit?.strategy === 'ai' ? 'The AI decides when to stop.' : boxCalled ? '' : 'Racing. Set a plan, then call it in.',
        1: 'Pitting at the end of this lap.',
        2: 'In the pit lane, at the speed limit.',
        3: `Stopped: ${frame.cars.serviceLeft[car].toFixed(1)} s to go.`,
      };
      $('pit-status').textContent = `${lines[pit] ?? ''}${frame.cars.stops[car] ? ` ${frame.cars.stops[car]} stop${frame.cars.stops[car] === 1 ? '' : 's'} so far.` : ''}`;
    },

    setRace(race) {
      current = race;
      plan = { ...race.myPlan };
      showPlan();
      showStrategy(race.strategy);
    },

    setStats(intro) {
      ranges = intro.specRanges;
      pending = {};
      edit(intro.yours);
      // Personality sliders only mean something to brains that were trained with personalities.
      const reads = intro.readsPersonality;
      const note = intro.brain === 'scripted'
        ? 'The scripted driver has no personality: only the car sliders change how it drives.'
        : reads ? '' : 'This brain was trained before personalities existed, so only the car sliders change how it drives.';
      $('racer-note').textContent = note;
      $('racer-note').classList.toggle('hidden', !note);
      for (const { name } of PERSONALITY) inputs.get(name).input.disabled = !reads;
    },
  };
}
