// The commentator. Picks the moment most worth talking about, says one line about it as a caption and, if you
// want, out loud with the browser's own voice. Fills quiet spells with a word about the order or the strategy.
// Your car's team radio is spoken too, the driver and the race engineer each in their own voice; every voice can
// be chosen from the ones the browser has.
//
// Lines come from templates until you load the AI commentator: a small language model (Llama 3.2 1B, about
// 0.9 GB, downloaded once and cached by the browser) that runs on your GPU in a web worker. It only gets the facts
// of the moment, so it can't make up who's leading. If it's slow or fails, the template line goes out instead.
import { formatLap } from './feed.js';
import { pick } from './race-events.js';

const WEBLLM = 'https://esm.run/@mlc-ai/web-llm@0.2.85';
const MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const MODEL_WITHOUT_F16 = 'Llama-3.2-1B-Instruct-q4f32_1-MLC';
const MODEL_WAIT_MS = 3500;              // a line later than this is old news
const STALE_MS = 5000;                   // a moment nobody got round to within this long goes unsaid
// How each voice speaks: a commentator a touch quick, a driver quicker and lower (mid-corner, under strain), the
// engineer calm.
const DELIVERY = { commentator: { rate: 1.08, pitch: 1 }, driver: { rate: 1.15, pitch: 0.85 }, engineer: { rate: 1.02, pitch: 1 } };
const SAMPLES = {
  commentator: "And it's lights out, and away they go!",
  driver: 'The rear is gone, I have no grip at all.',
  engineer: 'Box this lap, box this lap. Softs are ready.',
};
const QUIET_MS = 14000;                  // silence this long gets filled
const PAUSE_BETWEEN_MS = 600;
const SETTINGS_KEY = 'slipstream.commentary';

const SYSTEM_PROMPT = `You are the lead commentator on a TV broadcast of a motor race.
Turn the fact you are given into one line of live commentary: at most 18 words, excited British broadcast style, present tense.
Use only the names and numbers in the fact. Do not add positions, gaps, laps or events that are not in it.
Refer to drivers by name or as "they", never as he or she.

Fact: Okafor has just overtaken Sato and is now P3.
Line: Okafor dives down the inside and Sato can do nothing about it, up into P3 goes Okafor!

Fact: Nothing dramatic is happening. Vega leads Lindqvist by 3.1 seconds on lap 4 of 8.
Line: Vega looking very comfortable out front, 3.1 seconds clear of Lindqvist.`;
// A small model copies names from its examples, and drivers can be renamed, so these count as drivers to check
// for in every line, racing or not.
const EXAMPLE_NAMES = ['Okafor', 'Sato', 'Vega', 'Lindqvist'];

const TEMPLATES = {
  start: ['Lights out and away we go!', 'And it is lights out! {car} gets away cleanly from the front.', 'Green light, here we go, {laps} laps of racing ahead!'],
  overtake: ['{car} goes past {other}, that is P{place}!', 'Brilliant move from {car}, through on {other} for P{place}!', '{car} makes it stick on {other}. Up to P{place}.', 'And {other} has nothing to answer {car} with. P{place} changes hands.'],
  battle: ['{car} is right on the gearbox of {other}, fighting for P{place}.', 'Look at this, {car} all over the back of {other}!', 'The gap between {other} and {car} is nothing at all.'],
  contact: ['Contact! {car} and {other} touch!', 'Oh, they have banged wheels, {car} and {other}!', 'A clash between {car} and {other}, any damage there?'],
  offTrack: ['{car} is off! Into the run-off area.', 'A big moment for {car}, wide and onto the grass!', '{car} gets it all wrong there and runs off the road.'],
  damage: ['{car} has damage, that front wing looks broken.', 'Trouble for {car}, the car is damaged.'],
  tyresGone: ['{car} is complaining about the tyres, those {compound}s are finished.', 'The tyres are going away on the {car} car.'],
  lowFuel: ['{car} is running very low on fuel now.', 'Fuel is getting critical for {car}.'],
  pitCall: ['{car} is being called in this lap.', 'The pit wall wants {car} in, box this lap.'],
  pitStop: ['{car} in the pits, {compound} tyres, {standing} seconds stationary.', 'Stop for {car}: on go the {compound}s, {standing} seconds.', 'A {standing} second stop for {car}, out on {compound} tyres.'],
  pitExit: ['{car} rejoins in P{place}.', 'Out comes {car}, back on track in P{place}.'],
  fastestLap: ['Fastest lap of the race for {car}, a {lapTime}!', 'Purple for {car}, a {lapTime} is the new fastest lap.'],
  finalLap: ['Final lap! {car} leads, {other} is chasing.', 'The white flag is out, one lap to go and {car} is in front.'],
  win: ['{car} wins it! What a drive!', 'And {car} takes the chequered flag, victory!', 'It is {car}! {car} wins the race!'],
  finish: ['{car} crosses the line in P{place}.', 'P{place} at the flag for {car}.'],
  rainStarts: ['And here comes the rain! This changes everything.', 'Rain is falling! Who gambles on intermediates first?', 'The heavens open, it is raining on the circuit!'],
  rainStops: ['The rain has stopped. How quickly will this track dry?', 'No more rain, and the clock starts on a drying track.'],
  puncture: ['Oh, a puncture for {car}! That tyre is shredding.', 'Disaster for {car}, a puncture, and a long way back to the pits.', '{car} has picked up a puncture from the debris!'],
  engineFailure: ['Smoke! {car} has a failure, the engine has gone!', 'And that is the end of the race for {car}, engine failure.', 'Heartbreak for {car}, the engine lets go.'],
  mistake: ['A big moment for {car}, a mistake there!', '{car} gets it wrong, that will cost time.', 'Oh, a lock-up from {car}!'],
  tow: ['{car} is right in the slipstream of {other}, getting a big tow.', 'Look at the tow {car} is getting from {other}!', '{car} uses the slipstream, closing on {other}.'],
  quiet: ['{leader} leads by {leadGap} seconds on lap {lap} of {laps}.', 'Lap {lap} of {laps}, and {leader} is still out in front.', '{leader} controls it from the front, {second} is the nearest challenger.'],
};

export function createBroadcaster({ onStatus }) {
  const caption = document.getElementById('caption');
  let mode = loadSetting('mode', 'captions');
  let race = null, yours = null, pending = null, speakingUntil = 0, lastSpoke = 0, busy = false, captionTimer = null;
  let engine = null, modelState = 'off';
  // role -> the chosen voice's voiceURI; a role missing from it gets the automatic choice.
  let chosenVoices = loadSetting('voices', {});
  const recent = [];

  function setMode(value) {
    mode = value;
    saveSetting('mode', value);
    if (value !== 'voice') speechSynthesis?.cancel();
    if (value === 'off') caption.classList.add('hidden');
  }

  function reset(intro, yourCar) {
    race = intro;
    yours = yourCar;
    pending = null;
    speechSynthesis?.cancel();
    lastSpoke = performance.now();
    caption.classList.add('hidden');
  }

  // Keeps the best unsaid moment: the higher priority, or the newer of two equally big ones.
  function notice(moments) {
    if (mode === 'off' || !race) return;
    const now = performance.now();
    for (const moment of moments) {
      if (moment.radio && moment.radio.car === yours && mode === 'voice') speakRadio(moment.radio);
      if (!TEMPLATES[moment.kind] || moment.priority < 2) continue;
      if (!pending || moment.priority >= pending.moment.priority) pending = { moment, at: now };
    }
  }

  // Called every render frame with the facts the commentator may use when it's quiet.
  function update(context) {
    if (mode === 'off' || !race || busy) return;
    const now = performance.now();
    if (now < speakingUntil + PAUSE_BETWEEN_MS) return;
    if (pending && now - pending.at > STALE_MS) pending = null;
    if (pending) {
      const { moment } = pending;
      pending = null;
      say(moment, context);
    } else if (now - lastSpoke > QUIET_MS && context.racing) {
      say({ kind: 'quiet', priority: 1 }, context);
    }
  }

  async function say(moment, context) {
    const facts = describe(moment, context);
    const fallback = fill(pickFresh(TEMPLATES[moment.kind]), facts);
    busy = true;
    let line = fallback;
    if (engine && modelState === 'ready') {
      line = (await writeLine(moment, facts)) ?? fallback;
    }
    busy = false;
    show(line);
  }

  async function writeLine(moment, facts) {
    const fact = narrate(moment, facts);
    const reply = engine.chat.completions.create({
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: `Fact: ${fact}\nLine:` }],
      max_tokens: 40, temperature: 0.6,
    });
    const timeout = new Promise(resolve => setTimeout(() => resolve(null), MODEL_WAIT_MS));
    try {
      const answer = await Promise.race([reply, timeout]);
      if (!answer) {
        engine.interruptGenerate();
        return null;
      }
      const choice = answer.choices[0];
      // A line cut off at the token limit is a run-on; better the template than half a sentence.
      if (!choice || choice.finish_reason === 'length') return null;
      const line = tidy(choice.message?.content ?? '');
      const drivers = [...new Set([...race.cars.map(car => car.name), ...EXAMPLE_NAMES])];
      return line && sticksToFacts(line, fact, facts.car, drivers) ? line : null;
    } catch (error) {
      // A failed generation leaves the engine usable; this line goes out from the template instead.
      console.warn('Commentator could not write a line:', error);
      return null;
    }
  }

  function show(line) {
    if (!line) return;
    recent.push(line);
    if (recent.length > 12) recent.shift();
    lastSpoke = performance.now();
    caption.innerHTML = `<b>Commentary</b><span></span>`;
    caption.querySelector('span').textContent = line;
    caption.classList.remove('hidden');
    caption.style.animation = 'none';
    void caption.offsetWidth;
    caption.style.animation = '';
    const readingTime = 1500 + line.split(' ').length * 330;
    speakingUntil = performance.now() + readingTime;
    if (mode === 'voice' && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(line);
      voiceAs(utterance, 'commentator');
      utterance.onend = () => { speakingUntil = Math.min(speakingUntil, performance.now()); };
      speakingUntil = performance.now() + readingTime * 2;
      speechSynthesis.speak(utterance);
    }
    clearTimeout(captionTimer);
    captionTimer = setTimeout(() => caption.classList.add('hidden'), readingTime + 1200);
  }

  function speakRadio({ speaker, text }) {
    if (!('speechSynthesis' in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    voiceAs(utterance, speaker);
    speechSynthesis.speak(utterance);
  }

  function voiceAs(utterance, role) {
    utterance.voice = voiceFor(role, chosenVoices[role]);
    utterance.rate = DELIVERY[role].rate;
    utterance.pitch = DELIVERY[role].pitch;
  }

  function setVoice(role, id) {
    chosenVoices = { ...chosenVoices, [role]: id };
    if (!id) delete chosenVoices[role];
    saveSetting('voices', chosenVoices);
  }

  // Says a sample line in a role's voice, whatever the commentary mode, so you can hear it before choosing it.
  function previewVoice(role) {
    if (!('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(SAMPLES[role]);
    voiceAs(utterance, role);
    speechSynthesis.speak(utterance);
  }

  function pickFresh(lines) {
    const unused = lines.filter(line => !recent.includes(line));
    return pick(unused.length ? unused : lines);
  }

  async function loadModel() {
    if (modelState === 'loading' || modelState === 'ready') return;
    if (!navigator.gpu) {
      modelState = 'failed';
      onStatus('This browser has no WebGPU, so the AI commentator can\'t run here. Using template lines.');
      return;
    }
    modelState = 'loading';
    saveSetting('ai', true);
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const model = adapter?.features.has('shader-f16') ? MODEL : MODEL_WITHOUT_F16;
      const { CreateWebWorkerMLCEngine } = await import(WEBLLM);
      const worker = new Worker(new URL('./commentator-worker.js', import.meta.url), { type: 'module' });
      engine = await CreateWebWorkerMLCEngine(worker, model, {
        initProgressCallback: progress => onStatus(`AI commentator: ${progress.text}`),
      });
      modelState = 'ready';
      onStatus('AI commentator ready.');
    } catch (error) {
      modelState = 'failed';
      engine = null;
      saveSetting('ai', false);
      onStatus(`The AI commentator couldn't start (${error.message}). Using template lines.`);
    }
  }

  return {
    get mode() { return mode; },
    get modelState() { return modelState; },
    wantsModel: () => loadSetting('ai', false),
    setMode, reset, notice, update, loadModel, setVoice, previewVoice,
    chosenVoice: role => chosenVoices[role] ?? '',
    setYours(car) { yours = car; },
  };

  // The names and numbers a line about this moment may use.
  function describe(moment, context) {
    const name = car => (car === undefined ? '' : race.cars[car].name);
    return {
      car: name(moment.car), other: name(moment.other), place: moment.place ?? '', compound: moment.compound ?? '',
      standing: moment.standing?.toFixed(1) ?? '', lapTime: moment.lapTime ? formatLap(moment.lapTime) : '',
      laps: context.laps, lap: context.lap, leader: context.leader, second: context.second, leadGap: context.leadGap,
    };
  }
}

// The moment in plain words, for the model.
function narrate(moment, facts) {
  switch (moment.kind) {
    case 'start': return 'The race has just started.';
    case 'overtake': return `${facts.car} has just overtaken ${facts.other} and is now P${facts.place}.`;
    case 'battle': return `${facts.car} is less than a second behind ${facts.other}, fighting for P${facts.place}.`;
    case 'contact': return `${facts.car} and ${facts.other} have just made contact.`;
    case 'offTrack': return `${facts.car} has just run off the track.`;
    case 'damage': return `${facts.car} has a damaged car.`;
    case 'tyresGone': return `${facts.car}'s ${facts.compound} tyres are worn out.`;
    case 'lowFuel': return `${facts.car} is almost out of fuel.`;
    case 'pitCall': return `${facts.car} has been told to pit this lap.`;
    case 'pitStop': return `${facts.car} is making a pit stop: ${facts.compound} tyres, ${facts.standing} seconds stationary.`;
    case 'pitExit': return `${facts.car} has left the pits and is P${facts.place}.`;
    case 'fastestLap': return `${facts.car} has set the fastest lap of the race, ${facts.lapTime}.`;
    case 'finalLap': return `The final lap has started. ${facts.car} leads ${facts.other}.`;
    case 'puncture': return `${facts.car} has just picked up a puncture and is in P${facts.place}.`;
    case 'engineFailure': return `${facts.car}'s engine has just failed and ${facts.car} is out of the race.`;
    case 'mistake': return `${facts.car} has just made a driving mistake in P${facts.place}.`;
    case 'tow': return `${facts.car} is in the slipstream of ${facts.other}, gaining on the straight.`;
    case 'rainStarts': return 'It has just started raining on the circuit, and the cars are on the tyres they started with.';
    case 'rainStops': return 'The rain has just stopped and the track will start to dry.';
    case 'win': return `${facts.car} has just won the race.`;
    case 'finish': return `${facts.car} has finished in P${facts.place}.`;
    default: return `Nothing dramatic is happening. ${facts.leader} leads ${facts.second} by ${facts.leadGap} seconds on lap ${facts.lap} of ${facts.laps}.`;
  }
}

// A small model sometimes swaps who is ahead or invents a number. A line passes only if every driver and
// number it mentions is in the fact, and it names the driver the moment is about.
function sticksToFacts(line, fact, subject, drivers) {
  const named = drivers.filter(name => mentions(line, name));
  if (named.some(name => !mentions(fact, name))) return false;
  if (subject && mentions(fact, subject) && !mentions(line, subject)) return false;
  // The drivers are AIs with no stated gender, so the commentator shouldn't guess one.
  if (/\b(he|she|him|her|his|hers|himself|herself)\b/i.test(line)) return false;
  const numbers = line.match(/\d+(\.\d+)?/g) ?? [];
  return numbers.every(number => fact.includes(number));
}

// Whether `text` names `name` as a whole word: a driver called Al isn't mentioned by "all", nor Sato by "Satoshi".
function mentions(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?<![\\p{L}])${escaped}(?![\\p{L}])`, 'u').test(text);
}

function fill(template, facts) {
  return template.replace(/\{(\w+)\}/g, (_, key) => facts[key] ?? '');
}

// Keeps the first line or sentence of what the model wrote, without quotes or a "Commentator:" prefix.
function tidy(text) {
  const line = text.split('\n').map(part => part.trim()).find(Boolean) ?? '';
  const cleaned = line.replace(/^(commentary( line)?|commentator)\s*:\s*/i, '').replace(/^["'“”]+|["'“”]+$/g, '').trim();
  return cleaned.length >= 8 ? cleaned : null;
}

// The browser's English voices (every line is in English, and other voices mangle it), as { id, label }.
export function voiceOptions() {
  if (!('speechSynthesis' in window)) return [];
  return speechSynthesis.getVoices().filter(option => option.lang.startsWith('en'))
    .sort((first, second) => first.lang.localeCompare(second.lang) || first.name.localeCompare(second.name))
    .map(option => ({ id: option.voiceURI, label: `${option.name} · ${option.lang}` }));
}

// The chosen voice if the browser still has it. Otherwise: a British voice for the commentator if there is one,
// and two different other English voices for the driver and the engineer, so the radio is a conversation.
function voiceFor(role, chosen) {
  const all = speechSynthesis.getVoices();
  const picked = chosen && all.find(option => option.voiceURI === chosen);
  if (picked) return picked;
  const voices = all.filter(option => option.lang.startsWith('en'));
  const british = voices.filter(option => option.lang === 'en-GB');
  const others = voices.filter(option => option.lang !== 'en-GB');
  if (role === 'commentator') return british[0] ?? voices[0] ?? null;
  const pool = others.length ? others : voices;
  return (role === 'engineer' ? pool[1] : pool[0]) ?? pool[0] ?? null;
}

function loadSetting(name, fallback) {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
    return name in saved ? saved[name] : fallback;
  } catch {
    return fallback;
  }
}

function saveSetting(name, value) {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...saved, [name]: value }));
  } catch {
    // Private windows and blocked storage just don't remember the setting.
  }
}
