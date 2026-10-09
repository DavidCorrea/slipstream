// Natural voices: Kokoro, a small open speech model (82M parameters, Apache-licensed), run in this worker with
// kokoro-js, so speaking a line never stalls the race. It's the 8-bit model (92 MB, cached by the browser after
// the first time) on the CPU: the GPU build wants the full-precision one, 325 MB, too much for a page.
//
// Messages in: { type: 'load' }, then { type: 'speak', id, text, voice, speed } for each line.
// Messages out: 'progress' while loading, 'ready' with the voices, 'spoken' with each line's audio, or 'failed'.
import { KokoroTTS } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js';

const MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const ENGLISH = /^en/;
let speaker = null;

async function load() {
  // Downloads arrive file by file; the page adds them up for one progress figure.
  speaker = await KokoroTTS.from_pretrained(MODEL, {
    dtype: 'q8', device: 'wasm',
    progress_callback: progress => {
      if (progress.status === 'progress') self.postMessage({ type: 'progress', file: progress.file, loaded: progress.loaded, total: progress.total });
    },
  });
  const voices = Object.entries(speaker.voices)
    .filter(([, voice]) => ENGLISH.test(voice.language))
    .map(([id, voice]) => ({ id, name: voice.name, language: voice.language, gender: voice.gender, grade: voice.overallGrade }));
  self.postMessage({ type: 'ready', voices });
}

async function speak({ id, text, voice, speed }) {
  const audio = await speaker.generate(text, { voice, speed });
  const samples = audio.audio;
  self.postMessage({ type: 'spoken', id, samples, rate: audio.sampling_rate }, [samples.buffer]);
}

self.addEventListener('message', ({ data }) => {
  const work = data.type === 'load' ? load() : speak(data);
  work.catch(error => self.postMessage({ type: 'failed', id: data.id, message: String(error.message ?? error) }));
});
