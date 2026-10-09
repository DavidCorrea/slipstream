"""Races played inside the browser, for a page with no server behind it (web/engine-worker.js runs this with
Pyodide in a web worker).

It's the same Viewer the server gives each tab (viewer.py). Instead of a websocket, messages arrive as the JSON
text the page posted, and what the Viewer sends waits here until the worker collects it after each tick. The
brains are the networks published with the site (`brains/`, exported to run with numpy, see numpy_network.py).
"""
import json
from pathlib import Path

from .session import list_brains, list_pitwalls
from .viewer import Viewer

PUBLISHED = Path('brains')


class Outbox:
    """Stands in for the websocket: keeps what the Viewer sends until it's collected."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(json.dumps(message))


class BrowserRaces:
    def __init__(self, runs=PUBLISHED):
        self.runs = runs
        self.outbox = Outbox()
        self.viewer = Viewer(self.outbox, runs=runs)

    async def receive(self, text):
        await self.viewer.receive(text)

    async def tick(self):
        await self.viewer.tick()

    def collect(self):
        """Everything sent since the last call, as JSON text, oldest first."""
        messages, self.outbox.messages = self.outbox.messages, []
        return messages

    def brains(self):
        return json.dumps({'drivers': list_brains(self.runs), 'pitwalls': list_pitwalls(self.runs)})
