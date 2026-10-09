// Natural voices: Kokoro, a small open speech model (82M parameters, Apache-licensed), run in this worker with
// kokoro-js, so speaking a line never stalls the race. It runs on the graphics card (WebGPU), where it speaks a
// line in about a tenth of its length: on the CPU the same line took longer to generate than to say, too late for
// a race. On the GPU only the full-precision model (325 MB, cached by the browser after the first time) works:
// the half-precision one is half the size but gave broken audio for some voices.
//
// Messages in: { type: 'load' }, then { type: 'speak', id, text, voice, speed, urgent } for each piece of a line.
// Pieces are generated one at a time, the urgent ones (being spoken now) before those prepared for later.
// Messages out: 'progress' while loading, 'ready', 'spoken' with each piece's audio, or 'failed'.
import { KokoroTTS } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js';

const MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';
// Background pieces wait this long after the last urgent one: a piece can't be interrupted once started, and
// lines come in bursts (a pass, then the radio), so this keeps the background out of a burst's way.
const BACKGROUND_PAUSE_MS = 3000;
let speaker = null, working = false, lastUrgent = 0;
const urgent = [], later = [];

async function load() {
  // Downloads arrive file by file; the page adds them up for one progress figure.
  speaker = await KokoroTTS.from_pretrained(MODEL, {
    dtype: 'fp32', device: 'webgpu',
    progress_callback: progress => {
      if (progress.status === 'progress') self.postMessage({ type: 'progress', file: progress.file, loaded: progress.loaded, total: progress.total });
    },
  });
  // The first line generated takes several times longer than the rest (the model warming up); better here, while
  // loading, than on the first thing that happens in the race.
  await speaker.generate('Ready.', { voice: 'af_heart' });
  self.postMessage({ type: 'ready' });
}

async function work() {
  if (working) return;
  working = true;
  while (urgent.length || later.length) {
    // Let messages in between pieces: generating doesn't give way by itself, and without this an urgent line sent
    // during the background work waited for all of it.
    await new Promise(resolve => setTimeout(resolve, 0));
    const quietFor = performance.now() - lastUrgent;
    if (!urgent.length && quietFor < BACKGROUND_PAUSE_MS) {
      await new Promise(resolve => setTimeout(resolve, BACKGROUND_PAUSE_MS - quietFor));
      continue;
    }
    const { id, text, voice, speed } = urgent.shift() ?? later.shift();
    try {
      const audio = await speaker.generate(text, { voice, speed });
      const samples = audio.audio;
      self.postMessage({ type: 'spoken', id, samples, rate: audio.sampling_rate }, [samples.buffer]);
    } catch (error) {
      self.postMessage({ type: 'failed', id, message: String(error.message ?? error) });
    }
  }
  working = false;
}

self.addEventListener('message', ({ data }) => {
  if (data.type === 'load') {
    load().catch(error => self.postMessage({ type: 'failed', message: String(error.message ?? error) }));
    return;
  }
  if (data.urgent) lastUrgent = performance.now();
  (data.urgent ? urgent : later).push(data);
  work();
});
