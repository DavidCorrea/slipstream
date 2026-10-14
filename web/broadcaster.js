// The commentator. Picks the moment most worth talking about, says one line about it as a caption and, if you
// want, out loud with the browser's own voice. Fills quiet spells with a word about the order or the strategy.
// Your car's team radio is spoken too; how every line sounds (which voice, the radio) is speech.js's job.
//
// Lines come from templates until you load the AI commentator: a small language model (Llama 3.2 1B, about
// 0.9 GB, downloaded once and cached by the browser) that runs on your GPU in a web worker. It only gets the facts
// of the moment, so it can't make up who's leading. If it's slow or fails, the template line goes out instead.
import { formatLap } from './feed.js';
import { REPLAY_LINES, TEMPLATES } from './commentary-lines.js';
import { pick } from './race-events.js';
import { loadSetting, saveSetting } from './settings.js';

const WEBLLM = 'https://esm.run/@mlc-ai/web-llm@0.2.85';
const MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const MODEL_WITHOUT_F16 = 'Llama-3.2-1B-Instruct-q4f32_1-MLC';
const MODEL_WAIT_MS = 3500;              // a line later than this is old news
const STALE_MS = 5000;                   // a moment nobody got round to within this long goes unsaid
const CAPTION_WAIT_MS = 3000;            // a caption waits this long at most for its voice to start
const QUIET_MS = 14000;                  // silence this long gets filled
const PAUSE_BETWEEN_MS = 600;

const SYSTEM_PROMPT = `You are the lead commentator on a TV broadcast of a motor race.
Turn the fact you are given into one line of live commentary: one sentence of at most 14 words, excited British
broadcast style, present tense.
Use only the names and numbers in the fact. Do not add positions, gaps, laps or events that are not in it.
Refer to drivers by name or as "they", never as he or she.`;
// The examples the model is shown before each fact: lines for the same kind of moment, written about these
// drivers, so it sees the shape of a good line for exactly this moment.
const EXAMPLE_FACTS = {
  car: 'Okafor', other: 'Sato', place: 3, compound: 'medium', standing: '2.4', lapTime: formatLap(92.418),
  lap: 4, laps: 8, lapsToGo: 5, leader: 'Vega', second: 'Lindqvist', leadGap: '3.1',
};
const EXAMPLES_SHOWN = 2;
// A small model copies names from its examples, and drivers can be renamed, so these count as drivers to check
// for in every line, racing or not.
const EXAMPLE_NAMES = ['Okafor', 'Sato', 'Vega', 'Lindqvist'];

export function createBroadcaster({ onStatus, speech }) {
  const caption = document.getElementById('caption');
  let mode = loadSetting('mode', 'captions');
  let race = null, yours = null, pending = null, speakingUntil = 0, lastSpoke = 0, busy = false, captionTimer = null;
  let engine = null, modelState = 'off';
  const recent = [];

  function setMode(value) {
    mode = value;
    saveSetting('mode', value);
    if (value !== 'voice') speech.cancel();
    if (value === 'off') caption.classList.add('hidden');
  }

  function reset(intro, yourCar) {
    race = intro;
    yours = yourCar;
    pending = null;
    speech.cancel();
    lastSpoke = performance.now();
    caption.classList.add('hidden');
  }

  // Keeps the best unsaid moment: the higher priority, or the newer of two equally big ones.
  function notice(moments) {
    if (mode === 'off' || !race) return;
    const now = performance.now();
    for (const moment of moments) {
      if (moment.radio && moment.radio.car === yours && mode === 'voice') speech.say(moment.radio.text, moment.radio.speaker);
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
    const fallback = pickFresh(TEMPLATES[moment.kind], facts);
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
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...examples(moment.kind), { role: 'user', content: `Fact: ${fact}` }],
      max_tokens: 32, temperature: 0.7,
    });
    const timeout = new Promise(resolve => setTimeout(() => resolve(null), MODEL_WAIT_MS));
    try {
      const answer = await Promise.race([reply, timeout]);
      if (!answer) {
        engine.interruptGenerate();
        return null;
      }
      const choice = answer.choices[0];
      if (!choice) return null;
      // A line cut off at the token limit ran on; its first sentence is often a good line on its own.
      const text = choice.message?.content ?? '';
      const line = choice.finish_reason === 'length' ? firstSentence(tidy(text)) : tidy(text);
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
    const readingTime = 1500 + line.split(' ').length * 330;
    speakingUntil = performance.now() + readingTime;
    if (mode !== 'voice') return showCaption(line, readingTime);
    // The caption comes up as the voice starts, so the two arrive together (a natural voice takes a moment to
    // generate the first words). Speech that never starts still gets its caption once it gives up.
    // A line cut short by a new race (speech.cancel) never gets a caption.
    // And speech running late never holds a caption back for more than CAPTION_WAIT_MS.
    let captioned = false;
    const forRace = race;
    const caption = () => {
      if (captioned || race !== forRace) return;
      captioned = true;
      showCaption(line, readingTime);
    };
    setTimeout(caption, CAPTION_WAIT_MS);
    speakingUntil = performance.now() + readingTime * 3;
    speech.say(line, 'commentator', { onStart: caption }).then(() => {
      if (race !== forRace) return;
      caption();
      speakingUntil = Math.min(speakingUntil, performance.now());
    });
  }

  function showCaption(line, readingTime) {
    caption.innerHTML = `<b>Commentary</b><span></span>`;
    caption.querySelector('span').textContent = line;
    caption.classList.remove('hidden');
    caption.style.animation = 'none';
    void caption.offsetWidth;
    caption.style.animation = '';
    clearTimeout(captionTimer);
    captionTimer = setTimeout(() => caption.classList.add('hidden'), readingTime + 1200);
  }

  // A line not said lately, filled in with `facts`.
  function pickFresh(lines, facts) {
    const filled = lines.map(line => fill(line, facts));
    const unused = filled.filter(line => !recent.includes(line));
    return pick(unused.length ? unused : filled);
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
    setMode, reset, notice, update, loadModel,
    setYours(car) { yours = car; },
    // Opens a replay with a line about the cars in it and what it shows (a `kind` of replay.js), ahead of anything
    // waiting to be said about the race.
    replay(names, kind) {
      if (mode === 'off' || !race) return;
      pending = null;
      const lines = REPLAY_LINES[kind] ?? REPLAY_LINES[names.length > 1 ? 'two' : 'one'];
      show(pickFresh(lines, { car: names[0], other: names[1] }));
    },
  };

  // The names and numbers a line about this moment may use.
  function describe(moment, context) {
    const name = car => (car === undefined ? '' : race.cars[car].name);
    return {
      car: name(moment.car), other: name(moment.other), place: moment.place ?? '', compound: moment.compound ?? '',
      standing: moment.standing?.toFixed(1) ?? '', seconds: moment.seconds ?? '', reason: moment.reason ?? '', lapTime: moment.lapTime ? formatLap(moment.lapTime) : '',
      laps: context.laps, lap: context.lap, lapsToGo: Math.max(context.laps - context.lap + 1, 1),
      leader: context.leader, second: context.second, leadGap: context.leadGap,
    };
  }
}

// The moment in plain words, for the model.
function narrate(moment, facts) {
  switch (moment.kind) {
    case 'start': return facts.car ? `The race has just started, and ${facts.car} leads into the first corner.` : 'The race has just started.';
    case 'overtake': return `${facts.car} has just overtaken ${facts.other} and is now P${facts.place}.`;
    case 'battle': return `${facts.car} is less than a second behind ${facts.other}, fighting for P${facts.place}.`;
    case 'contact': return `${facts.car} and ${facts.other} have just made contact.`;
    case 'offTrack': return `${facts.car} has just run off the track.`;
    case 'yellow': return 'The marshals are waving yellow flags after an incident, so no overtaking there.';
    case 'penalty': return `The stewards have given ${facts.car} a ${facts.seconds}-second time penalty for ${facts.reason}.`;
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
    case 'crash': return `${facts.car} has crashed into a barrier and is out of the race.`;
    case 'barrier': return `${facts.car} has hit a barrier but is still going.`;
    case 'mistake': return `${facts.car} has just made a driving mistake in P${facts.place}.`;
    case 'tow': return `${facts.car} is in the slipstream of ${facts.other}, gaining on the straight.`;
    case 'rainStarts': return 'It has just started raining on the circuit, and the cars are on the tyres they started with.';
    case 'rainStops': return 'The rain has just stopped and the track will start to dry.';
    case 'win': return `${facts.car} has just won the race.`;
    case 'finish': return `${facts.car} has finished in P${facts.place}.`;
    default: return `Nothing dramatic is happening. ${facts.leader} leads ${facts.second} by ${facts.leadGap} seconds on lap ${facts.lap} of ${facts.laps}, ${facts.lapsToGo} laps to go.`;
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
  if (!numbers.every(number => fact.includes(number))) return false;
  // Positions said in words, and who leads, are claims too.
  if (ORDINALS.test(line) && !ORDINALS.test(fact)) return false;
  return !LEADING.test(line) || LEADING.test(fact) || /\bP1\b/.test(fact);
}
const ORDINALS = /\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|last)\b/i;
const LEADING = /\b(lead|leads|leader|leading|led|front|win|wins|won|winner|victory)\b/i;

// Whether `text` names `name` as a whole word: a driver called Al isn't mentioned by "all", nor Sato by "Satoshi".
function mentions(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?<![\\p{L}])${escaped}(?![\\p{L}])`, 'u').test(text);
}

// Example turns for this kind of moment: template lines about the example drivers, only those that stick to the
// example fact themselves, so the examples never show the model adding something.
function examples(kind) {
  const fact = narrate({ kind }, EXAMPLE_FACTS);
  const fitting = (TEMPLATES[kind] ?? TEMPLATES.quiet).map(template => fill(template, EXAMPLE_FACTS))
    .filter(line => sticksToFacts(line, fact, EXAMPLE_FACTS.car, EXAMPLE_NAMES));
  const chosen = [...fitting].sort(() => Math.random() - 0.5).slice(0, EXAMPLES_SHOWN);
  return chosen.flatMap(line => [{ role: 'user', content: `Fact: ${fact}` }, { role: 'assistant', content: line }]);
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

// The first whole sentence of `line`, or nothing if it never finishes one. A full stop inside a number (3.1) doesn't end it.
function firstSentence(line) {
  const sentence = line?.match(/^.*?[.!?](?=\s|$)/)?.[0];
  return sentence && sentence.length >= 8 ? sentence : null;
}
