// Lines spoken aloud: the commentator's, and your driver's and race engineer's on team radio. The voices are either
// natural (Kokoro on the graphics card, see voice-worker.js, which has to be accepted first: it's a big download)
// or the browser's own, and each role picks one of that kind. Lines take turns rather than talk over each other,
// and team radio sounds like a radio: squeezed into the telephone band, a little overdriven, with a click as the
// button's pressed.
//
// Lines known ahead of time (team radio, given to createSpeech) are generated in quiet moments, ready at once.
import { loadSetting, saveSetting } from './settings.js';

const ROLES = ['commentator', 'driver', 'engineer'];
// How each voice speaks: a commentator a touch quick, a driver quicker and lower (mid-corner, under strain), the
// engineer calm. Natural voices take the speed only.
const DELIVERY = {
  commentator: { rate: 1.08, pitch: 1, volume: 1 },
  driver: { rate: 1.15, pitch: 0.85, volume: 0.7 },
  engineer: { rate: 1.02, pitch: 1, volume: 0.7 },
};
const SAMPLES = {
  commentator: "And it's lights out, and away they go!",
  driver: 'The rear is gone, I have no grip at all.',
  engineer: 'Box this lap, box this lap. Softs are ready.',
};
// Kokoro's best-rated voices, two of each accent and gender; a British commentator, an American driver and a
// British engineer unless chosen otherwise.
const NATURAL_VOICES = {
  bm_george: 'George · British man', bm_fable: 'Fable · British man', bf_emma: 'Emma · British woman', bf_isabella: 'Isabella · British woman',
  am_michael: 'Michael · American man', am_fenrir: 'Fenrir · American man', af_heart: 'Heart · American woman', af_bella: 'Bella · American woman',
};
const NATURAL_DEFAULTS = { commentator: 'bm_george', driver: 'am_michael', engineer: 'bf_emma' };
// Whether natural voices were offered and what was said: 'accepted', 'declined', or not asked yet.
const CONSENT = 'naturalVoicesOffer';
// The overdrive that makes the radio sound like a radio also makes it louder, so it's turned down after it, to sit
// under the commentator (`volume`); browser voices on the radio are turned down too (DELIVERY).
const RADIO = { low: 320, high: 3200, drive: 18, click: 0.04, volume: 0.55 };
const SAFETY_SECONDS = { base: 3, perWord: 0.6 };   // a browser voice that never says it finished is given up on
const GENERATE_LIMIT_MS = 12000;   // a line taking longer than this is given up on, so the ones behind it go on

export function createSpeech({ onProgress, knownLines = [] }) {
  // 'natural' or 'browser', and per kind, role -> chosen voice (a Kokoro voice or a browser voiceURI); a role with
  // none gets the default.
  let kind = loadSetting('voiceKind', 'natural');
  let chosen = loadSetting('voiceChoices', { natural: {}, browser: {} });
  const natural = { state: 'off', worker: null, requests: new Map(), next: 0, downloads: new Map() };
  // voice + text -> a line generated ahead of time, as a promise of its audio.
  const prepared = new Map();
  const playing = new Set();
  let audio = null, queue = Promise.resolve(), turn = 0;
  // Natural voices need a graphics card the browser can use. Having WebGPU isn't enough (a browser can have it
  // and no adapter), so it's asked for one, once.
  let gpu = null;
  const gpuChecked = (navigator.gpu ? navigator.gpu.requestAdapter().then(Boolean, () => false) : Promise.resolve(false))
    .then(found => { gpu = found; });

  // ---- Choosing a voice ----------------------------------------------------------------------------------

  function naturalUsable() {
    return gpu === true && loadSetting(CONSENT, null) === 'accepted' && natural.state !== 'failed';
  }

  function shownKind() {
    return naturalUsable() ? kind : 'browser';
  }

  // Natural voices until they're ready (or when the browser's are chosen) fall back to the browser's.
  function voiceFor(role) {
    if (kind === 'natural' && natural.state === 'ready') return { natural: chosen.natural[role] ?? NATURAL_DEFAULTS[role] };
    const browserVoice = browserVoices().find(option => option.voiceURI === chosen.browser[role]);
    return { browser: browserVoice ?? automaticBrowserVoice(role) };
  }

  // ---- Speaking --------------------------------------------------------------------------------------------

  // Queues a line; resolves once it has been spoken (or dropped by cancel). `onStart` is called as the first sound
  // plays. A natural voice starts generating the line straight away, while the ones before it are still being
  // spoken, so only the speaking waits its turn.
  function say(text, role, { onStart } = {}) {
    const myTurn = turn;
    const voice = voiceFor(role);
    const generated = voice.natural ? audioFor(text, voice.natural, role, true) : null;
    queue = queue.then(() => (myTurn === turn ? speak(text, role, voice, generated, onStart, myTurn) : null))
      .catch(error => console.warn('Could not speak a line:', error));
    return queue;
  }

  function cancel() {
    turn++;
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    for (const source of playing) source.stop();
    playing.clear();
  }

  function speak(text, role, voice, generated, onStart, myTurn) {
    if (!generated) return speakInBrowser(text, voice.browser, role, onStart);
    return playLine(generated, role !== 'commentator', onStart, myTurn);
  }

  // The line's audio, prepared ahead of time when it's a known line, or generated now. A whole line at once: on
  // the GPU that takes about half a second, and spoken in one go it keeps one rise and fall.
  function audioFor(text, voice, role, urgent) {
    const key = `${voice}|${text}`;
    if (prepared.has(key)) return prepared.get(key);
    const generated = generate(text, voice, DELIVERY[role].rate, urgent);
    generated.catch(() => {});
    return generated;
  }

  function speakInBrowser(text, voice, role, onStart) {
    if (!('speechSynthesis' in window)) {
      onStart?.();
      return Promise.resolve();
    }
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.rate = DELIVERY[role].rate;
      utterance.pitch = DELIVERY[role].pitch;
      utterance.volume = DELIVERY[role].volume;
      utterance.onstart = () => onStart?.();
      utterance.onend = utterance.onerror = () => resolve();
      // Some browsers drop the end event now and then; a line can't hold up the ones behind it forever.
      setTimeout(resolve, (SAFETY_SECONDS.base + text.split(' ').length * SAFETY_SECONDS.perWord) * 1000);
      speechSynthesis.speak(utterance);
    });
  }

  async function playLine(generated, overRadio, onStart, myTurn) {
    const { samples, rate } = await generated;
    if (myTurn !== turn) return;
    const context = audioContext();
    const startsAt = context.currentTime + (overRadio ? RADIO.click : 0);
    if (overRadio) click(context);
    setTimeout(() => onStart?.(), (startsAt - context.currentTime) * 1000);
    await schedule(context, samples, rate, overRadio, startsAt);
  }

  function schedule(context, samples, rate, overRadio, startsAt) {
    const buffer = context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = new AudioBufferSourceNode(context, { buffer });
    (overRadio ? radio(context, source) : source).connect(context.destination);
    playing.add(source);
    return new Promise(resolve => {
      source.onended = () => { playing.delete(source); resolve(); };
      source.start(startsAt);
    });
  }

  function radio(context, source) {
    const low = new BiquadFilterNode(context, { type: 'highpass', frequency: RADIO.low });
    const high = new BiquadFilterNode(context, { type: 'lowpass', frequency: RADIO.high });
    const overdrive = new WaveShaperNode(context, { curve: overdriveCurve(RADIO.drive) });
    const volume = new GainNode(context, { gain: RADIO.volume });
    source.connect(low).connect(high).connect(overdrive).connect(volume);
    return volume;
  }

  // The click of the radio opening: a few milliseconds of fading noise.
  function click(context) {
    const length = Math.round(context.sampleRate * RADIO.click);
    const noise = context.createBuffer(1, length, context.sampleRate);
    const data = noise.getChannelData(0);
    for (let index = 0; index < length; index++) data[index] = (Math.random() * 2 - 1) * 0.25 * RADIO.volume * (1 - index / length);
    const source = new AudioBufferSourceNode(context, { buffer: noise });
    source.connect(new BiquadFilterNode(context, { type: 'bandpass', frequency: 1800 })).connect(context.destination);
    source.start();
  }

  function audioContext() {
    audio ??= new AudioContext();
    if (audio.state === 'suspended') audio.resume();
    return audio;
  }

  // ---- Natural voices ------------------------------------------------------------------------------------

  function loadNatural() {
    if (natural.state === 'loading' || natural.state === 'ready') return;
    natural.state = 'loading';
    // Started from a click (the offer, Settings), it's that click which lets the page play sound; started on a
    // later visit, the sound waits for the first click anywhere, which the grid's Start race always is.
    audioContext();
    natural.worker = new Worker(new URL('./voice-worker.js', import.meta.url), { type: 'module' });
    natural.worker.addEventListener('message', ({ data }) => receive(data));
    // A worker that dies (out of memory, say) sends no message about it.
    natural.worker.addEventListener('error', event => receive({ type: 'failed', message: event.message || 'the voice worker stopped' }));
    natural.worker.postMessage({ type: 'load' });
    onProgress({ state: 'loading', text: 'Natural voices: starting…' });
  }

  function receive(data) {
    if (data.type === 'progress') {
      natural.downloads.set(data.file, { loaded: data.loaded, total: data.total });
      const files = [...natural.downloads.values()];
      const loaded = files.reduce((sum, file) => sum + file.loaded, 0), total = files.reduce((sum, file) => sum + file.total, 0);
      onProgress({ state: 'loading', text: `Natural voices: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB` });
    } else if (data.type === 'ready') {
      natural.state = 'ready';
      onProgress({ state: 'ready', text: 'Natural voices ready.' });
      prepareKnownLines();
    } else if (data.id !== undefined) {
      const request = natural.requests.get(data.id);
      natural.requests.delete(data.id);
      if (data.type === 'spoken') request?.resolve(data);
      else request?.reject(new Error(data.message));
    } else if (data.type === 'failed') {
      natural.state = 'failed';
      natural.worker.terminate();
      for (const request of natural.requests.values()) request.reject(new Error(data.message));
      natural.requests.clear();
      onProgress({ state: 'failed', text: `Natural voices couldn't load (${data.message}). Using the browser's voices.` });
    }
  }

  // `urgent` lines go ahead of the ones being prepared in the background.
  function generate(text, voice, speed, urgent) {
    const id = natural.next++;
    return new Promise((resolve, reject) => {
      natural.requests.set(id, { resolve, reject });
      natural.worker.postMessage({ type: 'speak', id, text, voice, speed, urgent });
      // Live lines only: background ones may rightly wait behind the race's commentary.
      if (urgent) setTimeout(() => {
        if (!natural.requests.delete(id)) return;
        reject(new Error(`gave up on "${text}" after ${GENERATE_LIMIT_MS / 1000} s`));
      }, GENERATE_LIMIT_MS);
    });
  }

  // Generates the known lines in each role's natural voice, in the background, ready to play at once.
  function prepareKnownLines() {
    if (natural.state !== 'ready') return;
    for (const { speaker, text } of knownLines) {
      const voice = voiceFor(speaker);
      const key = `${voice.natural}|${text}`;
      if (!voice.natural || prepared.has(key)) continue;
      prepared.set(key, audioFor(text, voice.natural, speaker, false));
    }
  }

  return {
    roles: ROLES,
    say, cancel,
    get naturalState() { return natural.state; },
    gpuChecked,
    // Whether natural voices could run here (known once gpuChecked has settled), and whether they can be chosen.
    naturalPossible: () => gpu === true,
    naturalUsable,
    get consent() { return loadSetting(CONSENT, null); },
    accept() {
      saveSetting(CONSENT, 'accepted');
      loadNatural();
    },
    decline: () => saveSetting(CONSENT, 'declined'),
    // Loads them on a later visit, from the browser's cache, if they were accepted before.
    resume() {
      gpuChecked.then(() => { if (gpu && loadSetting(CONSENT, null) === 'accepted') loadNatural(); });
    },

    // The kind the pickers show and set: the chosen one, unless natural voices can't be had here.
    get kind() { return shownKind(); },
    setKind(value) {
      kind = value;
      saveSetting('voiceKind', value);
      prepareKnownLines();
    },
    // The voices a role can pick, of the kind shown.
    options() {
      if (shownKind() === 'natural') return Object.entries(NATURAL_VOICES).map(([id, label]) => ({ id, label }));
      return browserVoices().map(voice => ({ id: voice.voiceURI, label: `${voice.name} · ${voice.lang}` }));
    },
    chosenVoice: role => chosen[shownKind()][role] ?? '',
    setVoice(role, id) {
      const of = shownKind();
      chosen = { ...chosen, [of]: { ...chosen[of], [role]: id } };
      if (!id) delete chosen[of][role];
      saveSetting('voiceChoices', chosen);
      prepareKnownLines();
    },
    // Says a sample line in a role's voice, whatever the commentary mode, so you can hear it before choosing it.
    preview(role) {
      cancel();
      say(SAMPLES[role], role);
    },
  };
}

// The browser's English voices (every line is in English, and other voices mangle it), in a steady order.
function browserVoices() {
  if (!('speechSynthesis' in window)) return [];
  return speechSynthesis.getVoices().filter(option => option.lang.startsWith('en'))
    .sort((first, second) => first.lang.localeCompare(second.lang) || first.name.localeCompare(second.name));
}

// Without natural voices: a British voice for the commentator if there is one, and two different other English
// voices for the driver and the engineer, so the radio is a conversation.
function automaticBrowserVoice(role) {
  const voices = browserVoices();
  const british = voices.filter(option => option.lang === 'en-GB');
  const others = voices.filter(option => option.lang !== 'en-GB');
  if (role === 'commentator') return british[0] ?? voices[0] ?? null;
  const pool = others.length ? others : voices;
  return (role === 'engineer' ? pool[1] : pool[0]) ?? pool[0] ?? null;
}

function overdriveCurve(amount) {
  const curve = new Float32Array(256);
  for (let index = 0; index < curve.length; index++) {
    const input = (index / (curve.length - 1)) * 2 - 1;
    curve[index] = ((1 + amount) * input) / (1 + amount * Math.abs(input));
  }
  return curve;
}
