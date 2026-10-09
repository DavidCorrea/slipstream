// The simulation for the published site, where no server runs it: Pyodide (Python and numpy compiled to
// WebAssembly) runs slipstream's own race code (browser.py) in this worker, so the page stays smooth. The page
// posts messages as it would to the server; races play at 20 frames a second and every frame comes back as the
// JSON text the server would have sent.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

const FRAME_MS = 50;
const ROOT = new URL('../', self.location.href);
const waiting = [];
let races = null, busy = false;

async function fetchBytes(path) {
  const response = await fetch(new URL(path, ROOT));
  if (!response.ok) throw new Error(`could not download ${path} (${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}

async function start() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage('numpy');
  const files = JSON.parse(new TextDecoder().decode(await fetchBytes('web/engine-files.json')));
  for (const path of [...files.modules, ...files.brains]) {
    const folder = path.split('/').slice(0, -1).join('/');
    pyodide.FS.mkdirTree(folder);
    pyodide.FS.writeFile(path, await fetchBytes(path));
  }
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
