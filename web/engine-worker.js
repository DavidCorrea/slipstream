// The simulation for the published site, where no server runs it: Pyodide (Python and numpy compiled to
// WebAssembly) runs slipstream's own race code (browser.py) in this worker, so the page stays smooth. The page
// posts messages as it would to the server; races play at 20 frames a second and every frame comes back as the
// JSON text the server would have sent.
//
// While it starts, it reports how far each stage has got (for the loading screen, loading.js): every download
// in this worker, Pyodide's own included, is counted as it arrives, against what each stage is known to fetch.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

const FRAME_MS = 50;
const REPORT_MS = 100;
const ROOT = new URL('../', self.location.href);
// What Pyodide v314.0.7 fetches, unpacked: the interpreter (pyodide.asm.wasm), the standard library and the
// package list for Python, and numpy's wheel. They change with the Pyodide version above.
const PYTHON_BYTES = 9598218 + 2545637 + 119077;
const NUMPY_BYTES = 2960568;
const waiting = [];
let races = null, busy = false;

// ---- Counting downloads -------------------------------------------------------------------------------------

let received = 0, lastReport = 0, stage = null;
const fetchQuietly = self.fetch.bind(self);
self.fetch = async (...request) => {
  const response = await fetchQuietly(...request);
  if (!response.body) return response;
  const counted = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      report();
      controller.enqueue(chunk);
    },
  }));
  return new Response(counted, { status: response.status, statusText: response.statusText, headers: response.headers });
};

function report(force = false) {
  const now = performance.now();
  if (!force && now - lastReport < REPORT_MS) return;
  lastReport = now;
  self.postMessage({ type: 'progress', stage: stage.name, received: received - stage.from, expected: stage.expected });
}

function begin(name, expected) {
  if (stage) report(true);
  stage = { name, expected, from: received };
  report(true);
}

// ---- Starting up ---------------------------------------------------------------------------------------------

async function download(path, fetcher = self.fetch) {
  let response;
  try {
    response = await fetcher(new URL(path, ROOT));
  } catch (error) {
    throw new Error(`Could not download ${path}: ${error.message}. Check your connection and try again.`);
  }
  if (!response.ok) throw new Error(`Could not download ${path}: the site answered ${response.status}.`);
  return response;
}

async function fetchBytes(path) {
  return new Uint8Array(await (await download(path)).arrayBuffer());
}

async function start() {
  // The list of files comes first (it's tiny), so every stage's size is known before any of them starts and
  // the bar only ever moves forward.
  const files = await (await download('web/engine-files.json', fetchQuietly)).json();
  self.postMessage({ type: 'plan', expected: { python: PYTHON_BYTES, numpy: NUMPY_BYTES, files: files.bytes } });
  begin('python', PYTHON_BYTES);
  const pyodide = await loadPyodide();
  begin('numpy', NUMPY_BYTES);
  await pyodide.loadPackage('numpy');
  begin('files', files.bytes);
  for (const path of [...files.modules, ...files.brains]) {
    pyodide.FS.mkdirTree(path.split('/').slice(0, -1).join('/'));
    pyodide.FS.writeFile(path, await fetchBytes(path));
  }
  begin('starting', 0);
  races = pyodide.runPython('import sys; sys.path.insert(0, "."); from slipstream.browser import BrowserRaces; BrowserRaces()');
  self.postMessage({ type: 'ready', brains: JSON.parse(races.brains()) });
  setInterval(play, FRAME_MS);
}

// One frame: what the page asked for since the last one, then the race moves on and what it sent goes back.
async function play() {
  if (busy) return;
  busy = true;
  try {
    while (waiting.length) await races.receive(waiting.shift());
    await races.tick();
    const sent = races.collect();
    for (const text of sent.toJs()) self.postMessage({ type: 'message', text });
    sent.destroy();
  } catch (error) {
    self.postMessage({ type: 'failed', message: String(error.message ?? error) });
  } finally {
    busy = false;
  }
}

self.addEventListener('message', event => waiting.push(event.data.text));
start().catch(error => self.postMessage({ type: 'failed', message: String(error.message ?? error) }));
