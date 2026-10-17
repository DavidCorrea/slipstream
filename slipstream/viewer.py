"""One open tab's race: what it asked for, and the frames that go back to it.

The tab sends messages (a new race, a playback speed, pause, its racer's stats, its pit calls, its radio orders) and gets back a
description of each race and a frame 20 times a second. Nothing here knows how they travel: the server sends
them over a websocket (server.py), and in a browser they go between the page and a web worker (browser.py).
"""
import json

from .car import DT
from .formats import FORMATS, REVERSE, STANDARD, TIMED_MINUTES
from .session import DRIVER_DECIDES, RUNS, SCRIPTED, RaceSession

FRAMES_PER_SECOND = 20
# Times real time. The page's buttons say half these (its "1×" is twice real time), since our races are slower than
# a Formula 1 race (less grip, shorter circuits) and at real time they looked it.
SPEEDS = (1, 2, 4, 8, 16)
# Up to a Formula 1 grid, and races about as long as one (its laps are longer than ours).
LIMITS = {'cars': (2, 20), 'laps': (1, 70)}


class Viewer:
    """One open tab: its race and its playback settings."""

    def __init__(self, socket, runs=RUNS):
        """`socket` is anything with an async send_json; `runs` is where the brains are."""
        self.socket = socket
        self.runs = runs
        self.session = None
        # `weather` is a forecast name, or 'random' to let each circuit's seed decide.
        self.settings = {'brain': SCRIPTED, 'pitwall': DRIVER_DECIDES, 'cars': 6, 'laps': 8, 'weather': 'random', 'location': 'random',
                         'format': STANDARD, 'minutes': TIMED_MINUTES[0]}
        # Like a championship, your driver, every driver you've edited and the names and colours you've given them carry over
        # from race to race (by each driver's key, since the grid is drawn afresh each time).
        self.yours = None
        self.edited = {}
        self.names = {}
        self.colors = {}
        self.pit = None
        self.speed = 1.0
        self.paused = False
        self.ticks_owed = 0.0
        # A new race waits on the grid, so you can pick your car and set it up, until you send 'go'.
        self.on_grid = False

    async def start(self, brain=None, seed=None, cars=None, laps=None, pitwall=None, weather=None, location=None, race_format=None, minutes=None):
        if brain is not None:
            self.settings['brain'] = str(brain)
        if weather is not None:
            self.settings['weather'] = str(weather)
        if location is not None:
            self.settings['location'] = str(location)
        if pitwall is not None:
            self.settings['pitwall'] = str(pitwall)
        if race_format is not None:
            if race_format not in FORMATS:
                raise ValueError(f'Format must be one of {", ".join(FORMATS)}, got {race_format!r}')
            self.settings['format'] = race_format
        if minutes is not None:
            if minutes not in TIMED_MINUTES:
                raise ValueError(f'A timed race lasts one of {", ".join(map(str, TIMED_MINUTES))} minutes, got {minutes!r}')
            self.settings['minutes'] = minutes
        for key, value in (('cars', cars), ('laps', laps)):
            if value is not None:
                low, high = LIMITS[key]
                self.settings[key] = min(high, max(low, int(value)))
        try:
            self.session = RaceSession(self.settings['brain'], seed=None if seed is None else int(seed),
                                       cars=self.settings['cars'], laps=self.settings['laps'], runs=self.runs,
                                       yours=self.yours, edited=self.edited, names=self.names, colors=self.colors, pitwall_id=self.settings['pitwall'],
                                       forecast=None if self.settings['weather'] == 'random' else self.settings['weather'],
                                       location=None if self.settings['location'] == 'random' else self.settings['location'],
                                       race_format=self.settings['format'], minutes=self.settings['minutes'])
            if self.pit:
                self.session.set_pit(self.pit['strategy'], self.pit['plan'])
        except ValueError as error:
            await self.socket.send_json({'type': 'error', 'message': str(error)})
            return
        self.ticks_owed = 0.0
        self.on_grid = True
        await self.socket.send_json(self.session.intro())

    async def qualify(self):
        """Qualifying, then the race again on the grid it set (turned round for a reverse grid), with the times."""
        results = self.session.qualify()
        order = [key for key, _ in results]
        self.session = self.session.regrid(order[::-1] if self.settings['format'] == REVERSE else order)
        if self.pit:
            self.session.set_pit(self.pit['strategy'], self.pit['plan'])
        intro = self.session.intro()
        intro['qualifying'] = [{'key': key, 'time': time} for key, time in results]
        await self.socket.send_json(intro)

    async def receive(self, text):
        """A message from the tab as sent, answering one it can't use with an error rather than failing."""
        try:
            await self.handle(json.loads(text))
        except (ValueError, TypeError) as error:
            await self.socket.send_json({'type': 'error', 'message': f'Could not use message {text[:200]!r}: {error}'})

    async def handle(self, message):
        kind = message.get('type')
        if kind == 'start':
            await self.start(message.get('brain'), message.get('seed'), message.get('cars'), message.get('laps'), message.get('pitwall'),
                             message.get('weather'), message.get('location'), message.get('format'), message.get('minutes'))
        elif kind == 'go':
            self.on_grid = False
        elif kind == 'qualify' and self.session and self.on_grid:
            await self.qualify()
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
        elif kind == 'recolor' and self.session:
            car = message.get('car')
            self.session.recolor(car, message.get('color'))
            key = self.session.lineup[car].key
            self.colors[key] = self.session.colors[key]
        elif kind == 'pit' and self.session:
            self.session.set_pit(message.get('strategy'), message.get('plan'), message.get('box'), message.get('stayOut'))
            self.pit = {'strategy': self.session.strategy, 'plan': dict(self.session.my_plan)}
        elif kind == 'radio' and self.session:
            self.session.set_radio(message.get('pace'), message.get('approach'))
        elif kind == 'yours' and self.session:
            self.session.set_yours(message.get('car'))
            self.yours = self.session.lineup[self.session.yours].key

    async def tick(self):
        if self.session is None or self.paused or self.on_grid:
            return
        # A finished race stays as it ended, for the report, until the tab starts the next one.
        if self.session.race.done:
            return
        self.ticks_owed += self.speed / (FRAMES_PER_SECOND * DT)
        ticks = int(self.ticks_owed)
        self.ticks_owed -= ticks
        if ticks:
            await self.socket.send_json(self.session.advance(ticks))
