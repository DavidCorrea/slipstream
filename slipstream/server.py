"""The viewer's server: serves the page, lists the brains, and plays races to each open tab over a websocket.

    .venv/bin/python -m slipstream.server            # then open http://127.0.0.1:8765

Each tab gets its own race. The race runs in real time (or faster) and a frame goes out 20 times a second.
The tab sends back what it wants: a new race with a given brain, a playback speed, pause.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from .car import DT
from .session import DRIVER_DECIDES, SCRIPTED, RaceSession, list_brains, list_pitwalls

WEB = Path(__file__).resolve().parent.parent / 'web'
FRAMES_PER_SECOND = 20
SPEEDS = (0.5, 1, 2, 4, 8)
NEXT_RACE_AFTER = 6.0      # seconds the finished race stays on screen before the next one starts
LIMITS = {'cars': (2, 8), 'laps': (1, 10)}

log = logging.getLogger('slipstream.server')


async def index(request):
    return web.FileResponse(WEB / 'index.html')


async def brains(request):
    return web.json_response({'drivers': list_brains(), 'pitwalls': list_pitwalls()})


class Viewer:
    """One open tab: its race and its playback settings."""

    def __init__(self, socket):
        self.socket = socket
        self.session = None
        # `weather` is a forecast name, or 'random' to let each circuit's seed decide.
        self.settings = {'brain': SCRIPTED, 'pitwall': DRIVER_DECIDES, 'cars': 6, 'laps': 8, 'weather': 'random'}
        # Like a championship, your driver, every driver you've edited and the names you've given them carry over
        # from race to race (by each driver's key, since the grid is drawn afresh each time).
        self.yours = None
        self.edited = {}
        self.names = {}
        self.pit = None
        self.speed = 1.0
        self.paused = False
        self.ticks_owed = 0.0
        self.finished_for = 0.0
        # A new race waits on the grid, so you can pick your car and set it up, until you send 'go'.
        self.on_grid = False

    async def start(self, brain=None, seed=None, cars=None, laps=None, pitwall=None, weather=None):
        if brain is not None:
            self.settings['brain'] = str(brain)
        if weather is not None:
            self.settings['weather'] = str(weather)
        if pitwall is not None:
            self.settings['pitwall'] = str(pitwall)
        for key, value in (('cars', cars), ('laps', laps)):
            if value is not None:
                low, high = LIMITS[key]
                self.settings[key] = min(high, max(low, int(value)))
        try:
            self.session = RaceSession(self.settings['brain'], seed=None if seed is None else int(seed),
                                       cars=self.settings['cars'], laps=self.settings['laps'],
                                       yours=self.yours, edited=self.edited, names=self.names, pitwall_id=self.settings['pitwall'],
                                       forecast=None if self.settings['weather'] == 'random' else self.settings['weather'])
            if self.pit:
                self.session.set_pit(self.pit['strategy'], self.pit['plan'])
        except ValueError as error:
            await self.socket.send_json({'type': 'error', 'message': str(error)})
            return
        self.ticks_owed = 0.0
        self.finished_for = 0.0
        self.on_grid = True
        await self.socket.send_json(self.session.intro())

    async def handle(self, message):
        kind = message.get('type')
        if kind == 'start':
            await self.start(message.get('brain'), message.get('seed'), message.get('cars'), message.get('laps'), message.get('pitwall'),
                             message.get('weather'))
        elif kind == 'go':
            self.on_grid = False
        elif kind == 'speed' and message.get('value') in SPEEDS:
            self.speed = float(message['value'])
        elif kind == 'pause':
            self.paused = bool(message.get('value'))
        elif kind == 'stats' and self.session:
            car = self.session.yours if message.get('car') is None else message['car']
            self.session.set_stats(message.get('stats') or {}, car)
            self.edited[self.session.lineup[car].key] = dict(self.session.stats[car])
        elif kind == 'reset' and self.session:
            car = message.get('car')
            self.session.reset_stats(car)
            self.edited.pop(self.session.lineup[car].key, None)
        elif kind == 'rename' and self.session:
            car = message.get('car')
            self.session.rename(car, message.get('name'))
            key = self.session.lineup[car].key
            self.names[key] = self.session.names[key]
        elif kind == 'pit' and self.session:
            self.session.set_pit(message.get('strategy'), message.get('plan'), message.get('box'))
            self.pit = {'strategy': self.session.strategy, 'plan': dict(self.session.my_plan)}
        elif kind == 'yours' and self.session:
            self.session.set_yours(message.get('car'))
            self.yours = self.session.lineup[self.session.yours].key

    async def tick(self):
        if self.session is None or self.paused or self.on_grid:
            return
        if self.session.race.done:
            self.finished_for += 1 / FRAMES_PER_SECOND
            if self.finished_for >= NEXT_RACE_AFTER:
                await self.start()
            return
        self.ticks_owed += self.speed / (FRAMES_PER_SECOND * DT)
        ticks = int(self.ticks_owed)
        self.ticks_owed -= ticks
        if ticks:
            await self.socket.send_json(self.session.advance(ticks))


async def race_socket(request):
    socket = web.WebSocketResponse(heartbeat=20)
    await socket.prepare(request)
    viewer = Viewer(socket)

    async def play():
        while not socket.closed:
            started = asyncio.get_running_loop().time()
            await viewer.tick()
            await asyncio.sleep(max(0.0, 1 / FRAMES_PER_SECOND - (asyncio.get_running_loop().time() - started)))

    player = asyncio.create_task(play())
    try:
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                await viewer.handle(json.loads(message.data))
            except (ValueError, TypeError) as error:
                await socket.send_json({'type': 'error', 'message': f'Could not use message {message.data[:200]!r}: {error}'})
    finally:
        player.cancel()
    return socket


@web.middleware
async def always_fresh(request, handler):
    """Makes browsers check for a newer page and scripts on every load. Without it they reuse cached copies by
    guesswork, and a refresh after the viewer changes can keep running the old scripts."""
    response = await handler(request)
    if request.path == '/' or request.path.startswith('/web/'):
        response.headers['Cache-Control'] = 'no-cache'
    return response


def create_app():
    app = web.Application(middlewares=[always_fresh])
    app.add_routes([
        web.get('/', index),
        web.get('/api/brains', brains),
        web.get('/ws', race_socket),
        web.static('/web', WEB),
    ])
    return app


def main():
    parser = argparse.ArgumentParser(description='Slipstream viewer')
    parser.add_argument('--port', type=int, default=8765)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f'Slipstream viewer at http://127.0.0.1:{arguments.port}')
    web.run_app(create_app(), host='127.0.0.1', port=arguments.port, print=None)


if __name__ == '__main__':
    main()
