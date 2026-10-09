// Runs the commentator's language model off the main thread, so writing a line never stutters the race.
import { WebWorkerMLCEngineHandler } from 'https://esm.run/@mlc-ai/web-llm@0.2.85';

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = message => handler.onmessage(message);
