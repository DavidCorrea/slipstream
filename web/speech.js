// Lines spoken aloud: the commentator's, and your driver's and race engineer's on team radio. Each role speaks in
// the voice chosen for it, either one of the browser's own or a natural voice (Kokoro, see voice-worker.js) once
// those are loaded. Lines take turns rather than talk over each other, and team radio sounds like a radio:
// squeezed into the telephone band, a little overdriven, with a click as the button's pressed.
import { loadSetting, saveSetting } from './settings.js';

const ROLES = ['commentator', 'driver', 'engineer'];
// How each voice speaks: a commentator a touch quick, a driver quicker and lower (mid-corner, under strain), the
// engineer calm. Natural voices take the speed only.
const DELIVERY = { commentator: { rate: 1.08, pitch: 1 }, driver: { rate: 1.15, pitch: 0.85 }, engineer: { rate: 1.02, pitch: 1 } };
const SAMPLES = {
  commentator: "And it's lights out, and away they go!",
  driver: 'The rear is gone, I have no grip at all.',
  engineer: 'Box this lap, box this lap. Softs are ready.',
};
const NATURAL = 'natural:';
// A British commentator, an American driver and a British engineer, among Kokoro's better voices.
const NATURAL_DEFAULTS = { commentator: 'bm_george', driver: 'am_michael', engineer: 'bf_emma' };
const RADIO = { low: 320, high: 3200, drive: 18, click: 0.04 };
const SAFETY_SECONDS = { base: 3, perWord: 0.6 };   // a browser voice that never says it finished is given up on

export function createSpeech({ onProgress }) {
  // role -> chosen voice: a browser voiceURI, or NATURAL + a Kokoro voice; a role missing from it gets the automatic one.
  let chosen = loadSetting('voices', {});
  const natural = { state: 'off', voices: [], worker: null, requests: new Map(), next: 0, downloads: new Map() };
  let audio = null, playing = null, queue = Promise.resolve(), turn = 0;

  // ---- Choosing a voice ----------------------------------------------------------------------------------

  function voiceFor(role) {
    const choice = chosen[role];
    if (choice?.startsWith(NATURAL) && natural.state === 'ready') return { natural: choice.slice(NATURAL.length) };
    const browserVoice = choice && browserVoices().find(option => option.voiceURI === choice);
    if (browserVoice) return { browser: browserVoice };
    if (natural.state === 'ready') return { natural: NATURAL_DEFAULTS[role] };
    return { browser: automaticBrowserVoice(role) };
  }

  // ---- Speaking --------------------------------------------------------------------------------------------

  // Queues a line; resolves once it has been spoken (or dropped by cancel). A natural voice starts generating it
  // straight away, while the lines before it are still being spoken, so only the speaking waits its turn.
  function say(text, role) {
    const myTurn = turn;
    const voice = voiceFor(role);
    const generated = voice.natural ? generate(text, voice.natural, DELIVERY[role].rate) : null;
    generated?.catch(() => {});
    queue = queue.then(() => (myTurn === turn ? speak(text, role, voice, generated, myTurn) : null))
      .catch(error => console.warn('Could not speak a line:', error));
    return queue;
  }

  function cancel() {
    turn++;
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    playing?.stop();
  }

  async function speak(text, role, voice, generated, myTurn) {
    if (!generated) return speakInBrowser(text, voice.browser, role);
    const { samples, rate } = await generated;
    if (myTurn !== turn) return;
    await play(samples, rate, role !== 'commentator');
  }

  function speakInBrowser(text, voice, role) {
    if (!('speechSynthesis' in window)) return Promise.resolve();
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.rate = DELIVERY[role].rate;
      utterance.pitch = DELIVERY[role].pitch;
      utterance.onend = utterance.onerror = () => resolve();
      // Some browsers drop the end event now and then; a line can't hold up the ones behind it forever.
      setTimeout(resolve, (SAFETY_SECONDS.base + text.split(' ').length * SAFETY_SECONDS.perWord) * 1000);
      speechSynthesis.speak(utterance);
    });
  }

  function play(samples, rate, overRadio) {
    const context = audioContext();
    const buffer = context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    let output = source;
    if (overRadio) {
      output = radio(context, source);
      click(context);
    }
    output.connect(context.destination);
    return new Promise(resolve => {
      source.onended = () => { playing = null; resolve(); };
      playing = source;
      source.start(context.currentTime + (overRadio ? RADIO.click : 0));
    });
  }

  function radio(context, source) {
    const low = new BiquadFilterNode(context, { type: 'highpass', frequency: RADIO.low });
    const high = new BiquadFilterNode(context, { type: 'lowpass', frequency: RADIO.high });
    const overdrive = new WaveShaperNode(context, { curve: overdriveCurve(RADIO.drive) });
    source.connect(low).connect(high).connect(overdrive);
    return overdrive;
  }

  // The click of the radio opening: a few milliseconds of fading noise.
  function click(context) {
    const length = Math.round(context.sampleRate * RADIO.click);
    const noise = context.createBuffer(1, length, context.sampleRate);
    const data = noise.getChannelData(0);
    for (let index = 0; index < length; index++) data[index] = (Math.random() * 2 - 1) * 0.25 * (1 - index / length);
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
    // Started by a click, which is what lets the page play sound later.
    audioContext();
    natural.worker = new Worker(new URL('./voice-worker.js', import.meta.url), { type: 'module' });
    natural.worker.addEventListener('message', ({ data }) => receive(data));
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
      natural.voices = data.voices;
      saveSetting('naturalVoices', true);
      onProgress({ state: 'ready', text: 'Natural voices ready.' });
    } else if (data.id !== undefined) {
      const request = natural.requests.get(data.id);
      natural.requests.delete(data.id);
      if (data.type === 'spoken') request?.resolve(data);
      else request?.reject(new Error(data.message));
    } else if (data.type === 'failed') {
      natural.state = 'failed';
      natural.worker.terminate();
      saveSetting('naturalVoices', false);
      onProgress({ state: 'failed', text: `Natural voices couldn't load (${data.message}). Using the browser's voices.` });
    }
  }

  function generate(text, voice, speed) {
    const id = natural.next++;
    return new Promise((resolve, reject) => {
      natural.requests.set(id, { resolve, reject });
      natural.worker.postMessage({ type: 'speak', id, text, voice, speed });
    });
  }

  return {
    roles: ROLES,
    say, cancel, loadNatural,
    get naturalState() { return natural.state; },
    wantsNatural: () => loadSetting('naturalVoices', false),

    // What the voice pickers offer: the natural voices once loaded, then the browser's English ones.
    options() {
      const naturalOptions = natural.voices.map(voice => ({ id: NATURAL + voice.id, label: naturalLabel(voice) }));
      const browserOptions = browserVoices().map(voice => ({ id: voice.voiceURI, label: `${voice.name} · ${voice.lang}` }));
      return { natural: naturalOptions, browser: browserOptions };
    },
    chosenVoice: role => chosen[role] ?? '',
    setVoice(role, id) {
      chosen = { ...chosen, [role]: id };
      if (!id) delete chosen[role];
      saveSetting('voices', chosen);
    },
    // Says a sample line in a role's voice, whatever the commentary mode, so you can hear it before choosing it.
    preview(role) {
      cancel();
      say(SAMPLES[role], role);
    },
  };
}

function naturalLabel({ name, language, gender, grade }) {
  const accent = language.toLowerCase() === 'en-gb' ? 'British' : 'American';
  return `${name} · ${accent} ${gender.toLowerCase().startsWith('f') ? 'woman' : 'man'} (natural, grade ${grade})`;
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
