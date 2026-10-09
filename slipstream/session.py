"""A race played for watching: who drives, and what the viewer needs to draw it.

A brain is the scripted driver or any saved network, from any run and any snapshot, so the viewer can show
how a run drove at every stage of its training. The field comes from the cast (cast.py): the same drivers every
race, each with their own character and car, on a grid drawn afresh each time. On the grid any driver's traits and
car can be changed; one car is yours, and its pit strategy is either the AI's or yours: a plan you set and a stop
you call. A session plays one race and turns it into messages: one
describing the circuit and the cars, then a frame per update with every car's state and what happened since
the last frame (contacts, laps, finishes), so effects aren't lost when frames skip ticks at higher speeds.
"""
import base64
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cast
from . import personality as traits
from .car import BALANCE_RANGE, COMPOUNDS, DT, FUEL_CAPACITY, SPEC_DEFAULTS, SPEC_SPREAD, CarSpecs
from .drivers import DECISION_TICKS
from .field import field_controls, makes_pit_calls
from .strategy import scripted_strategy
from .observe import OBSERVATION_NAMES
from .race import Race
from .scenery import location_for, place
from .brains import NetworkDriver, load_network
from .human import Driving
from .surface import CELL_LENGTH
from .timing import Timing
from .track import generate_track
from .weather import FORECASTS, Weather

RUNS = Path('runs')
# Saved networks: Stable-Baselines3's own files, or ones exported to run with numpy alone (numpy_network.py).
NETWORK_FILES = ('.zip', '.npz')
SCRIPTED = 'scripted'
STATS = (*traits.TRAITS, *SPEC_DEFAULTS, 'balance')


DRIVER_DECIDES = 'driver'
SCRIPTED_STRATEGIST = 'scripted'


def is_pitwall_run(run):
    """Pit-wall runs say which driver they learned with (see train_pitwall.py); driver runs don't."""
    return (run / 'driver.json').exists()


def _snapshots(runs, pitwalls, unit, scale):
    """Each matching run's snapshots, newest first, the run that saved one most recently first."""
    snapshots = {run.parent: sorted((path for path in run.glob('step-*') if path.suffix in NETWORK_FILES), reverse=True)
                 for run in runs.glob('*/checkpoints') if is_pitwall_run(run.parent) == pitwalls}
    listed = []
    for run in sorted(snapshots, key=lambda run: max((path.stat().st_mtime for path in snapshots[run]), default=0), reverse=True):
        for checkpoint in snapshots[run]:
            step = int(checkpoint.stem.split('-')[1])
            listed.append({'id': f'{run.name}/{checkpoint.stem}', 'label': f'{run.name} · {step / scale:.1f}{unit}', 'run': run.name, 'step': step})
    return listed


def list_brains(runs=RUNS):
    """Every driver the viewer can pick: the scripted driver, then each driver run's snapshots, newest first.
    The run that saved a snapshot most recently comes first, so the default is whatever is training now."""
    return [{'id': SCRIPTED, 'label': 'Scripted driver', 'run': None, 'step': None}] + _snapshots(runs, False, 'M decisions', 1e6)


def list_pitwalls(runs=RUNS):
    """Every pit wall the viewer can pick: the driver's own calls, the scripted strategist, then each pit-wall
    run's snapshots (in thousands of pit decisions), newest first."""
    return ([{'id': DRIVER_DECIDES, 'label': 'Driver decides', 'run': None, 'step': None},
             {'id': SCRIPTED_STRATEGIST, 'label': 'Scripted strategist', 'run': None, 'step': None}]
            + _snapshots(runs, True, 'k pit decisions', 1e3))


def brain_path(brain_id, runs=RUNS):
    """The checkpoint file for a brain id like 'main/step-000002015232', refusing anything outside runs/."""
    run, name = brain_id.split('/', 1)
    for suffix in NETWORK_FILES:
        path = (runs / run / 'checkpoints' / f'{name}{suffix}').resolve()
        if runs.resolve() in path.parents and path.is_file():
            return path
    raise ValueError(f'No checkpoint named {brain_id!r}')


@lru_cache(maxsize=8)
def load_policy(path: Path):
    return load_network(path)


def spec_value(name, slider):
    """A car spec from a 0-1 slider: the training range, from SPEC_SPREAD below the default to as far above."""
    return SPEC_DEFAULTS[name] * (1 + SPEC_SPREAD * (2 * slider - 1))


class RaceSession:
    def __init__(self, brain_id=SCRIPTED, seed=None, cars=6, laps=3, runs=RUNS, yours=None, edited=None, names=None, pitwall_id=DRIVER_DECIDES,
                 forecast=None, location=None):
        """`yours` is the key of the driver you race with (the cast's first when not given; they always make the
        field). `edited` maps driver keys to the stats you gave them, which they keep exactly; everyone else turns
        up as they usually are, give or take the race day. `names` maps driver keys to the names you gave them. `forecast` picks the weather (see weather.FORECASTS)
        and `location` the scenery and everything in it the cars can hit (see scenery.LOCATIONS); without them, the
        seed decides."""
        self.brain_id, self.pitwall_id = brain_id, pitwall_id
        self.policy = None if brain_id == SCRIPTED else load_policy(brain_path(brain_id, runs))
        self.pitwall = None if pitwall_id in (DRIVER_DECIDES, SCRIPTED_STRATEGIST) else load_policy(brain_path(pitwall_id, runs))
        self.seed = int(np.random.default_rng().integers(2 ** 31)) if seed is None else seed
        track = generate_track(self.seed)
        self.race = Race(track, CarSpecs.uniform(cars, **SPEC_DEFAULTS), laps, weather=Weather.for_race(self.seed, track.length, laps, forecast))
        self.location = location_for(self.seed) if location is None else location
        self.race.place_props(place(track, self.race.pit, self.seed, self.location))
        self.timing = Timing(self.race.track.length, cars)
        self.driver = NetworkDriver(self.policy, cars) if self.policy is not None else None
        # Drivers trained with human hands and feet (see human.py) race with them here too.
        if self.driver is not None and self.driver.feels:
            self.race.driving = Driving(cars, np.random.default_rng(self.seed + 1))
        your_key = cast.CAST[0].key if yours is None else yours
        if your_key not in cast.BY_KEY:
            raise ValueError(f'No driver {your_key!r}; the cast is {", ".join(cast.BY_KEY)}')
        self.names = {driver.key: driver.name for driver in cast.CAST} | dict(names or {})
        # Its own stream, so the cast's draw doesn't shift the circuit's weather or the drivers' hands.
        cast_rng = np.random.default_rng(self.seed + 2)
        self.lineup = cast.lineup(cars, cast_rng, must_race=your_key)
        self.yours = [driver.key for driver in self.lineup].index(your_key)
        self.personality = traits.neutral(cars)
        self.stats = [cast.race_day(driver, cast_rng) for driver in self.lineup]
        self.edited = {driver.key for driver in self.lineup if driver.key in (edited or {})}
        for car, driver in enumerate(self.lineup):
            self._apply(car, {**self.stats[car], **(edited or {}).get(driver.key, {})})
        self.controls = None
        # Your pit strategy: 'ai' leaves it to the driver, 'mine' uses your plan and stops only when you call.
        self.strategy = 'ai'
        self.my_plan = {'compound': 'medium', 'fuel': 1.0, 'wing': 0.5, 'engine': 0.5, 'repair': True, 'brakes': False}
        self.box_called = False

    def set_pit(self, strategy=None, plan=None, box=None):
        """Changes your pit strategy, your plan for the next stop, or calls (or cancels) a stop. Anything unknown
        or out of range is refused, naming what was wrong."""
        if strategy is not None:
            if strategy not in ('ai', 'mine'):
                raise ValueError(f"Strategy must be 'ai' or 'mine', got {strategy!r}")
            self.strategy = strategy
        for name, value in (plan or {}).items():
            if name not in self.my_plan:
                raise ValueError(f'Unknown plan item {name!r}; known ones are {", ".join(self.my_plan)}')
            if name == 'compound' and value not in COMPOUNDS:
                raise ValueError(f'Compound must be one of {", ".join(COMPOUNDS)}, got {value!r}')
            if name in ('fuel', 'wing', 'engine') and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f'Plan item {name!r} must be a number from 0 to 1, got {value!r}')
            if name in ('repair', 'brakes') and not isinstance(value, bool):
                raise ValueError(f'Plan item {name!r} must be true or false, got {value!r}')
            self.my_plan[name] = value
        if box is not None:
            self.box_called = bool(box)

    def set_stats(self, stats, car=None):
        """Changes a driver's traits and their car's specs (yours when no car is named), each a slider value from 0
        to 1, before the start: once the race is under way drivers are who they are and cars what they are (you're
        the pit wall then). Unknown names, cars and values outside 0-1, and changes after the start, are refused,
        naming what was wrong."""
        car = self._on_the_grid(self.yours if car is None else car)
        for name, value in stats.items():
            if name not in STATS:
                raise ValueError(f'Unknown stat {name!r}; known ones are {", ".join(STATS)}')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f'Stat {name!r} must be a number from 0 to 1, got {value!r}')
        self._apply(car, {**self.stats[car], **stats})
        self.edited.add(self.lineup[car].key)

    def reset_stats(self, car):
        """Puts a driver back to their usual self, in their usual car."""
        car = self._on_the_grid(car)
        self._apply(car, self.lineup[car].character)
        self.edited.discard(self.lineup[car].key)

    def rename(self, car, name):
        """Gives a driver a new name, before the start. Names that aren't one, or that someone already has, are
        refused (see cast.tidy_name)."""
        key = self.lineup[self._on_the_grid(car)].key
        self.names[key] = cast.tidy_name(name, [other for other_key, other in self.names.items() if other_key != key])

    def set_yours(self, car):
        """Makes another car yours before the start. Every driver keeps their own traits and car."""
        self.yours = self._on_the_grid(car)

    def _on_the_grid(self, car):
        if self.race.tick > 0:
            raise ValueError(f'Drivers and cars are set before the start; this race is {self.race.time:.0f} s in')
        if isinstance(car, bool) or not isinstance(car, int) or not 0 <= car < self.race.count:
            raise ValueError(f'Car must be a number from 0 to {self.race.count - 1}, got {car!r}')
        return car

    def _apply(self, car, stats):
        self.stats[car] = {name: float(stats[name]) for name in STATS}
        for index, name in enumerate(traits.TRAITS):
            self.personality[car, index] = self.stats[car][name]
        for name in SPEC_DEFAULTS:
            getattr(self.race.specs, name)[car] = spec_value(name, self.stats[car][name])
        self.race.specs.balance[car] = BALANCE_RANGE * (2 * self.stats[car]['balance'] - 1)
        if self.race.driving is not None:
            self.race.driving.consistency = traits.trait(self.personality, 'consistency').copy()
            self.race.driving.stamina = traits.trait(self.personality, 'stamina').copy()

    def intro(self):
        track = self.race.track
        return {
            'type': 'race', 'seed': self.seed, 'location': self.location,
            'props': [prop.describe(number) for number, prop in enumerate(self.race.props.props)],
            'brain': self.brain_id, 'pitwall': self.pitwall_id, 'laps': self.race.laps, 'dt': DT,
            'track': {
                'points': np.round(track.points, 2).tolist(), 'normals': np.round(track.normals, 4).tolist(),
                'curvature': np.round(track.curvature, 5).tolist(), 'width': track.width, 'length': track.length,
            },
            # A driver's number is theirs, wherever they start.
            'cars': [{'key': driver.key, 'name': self.names[driver.key], 'color': driver.color, 'number': cast.CAST.index(driver) + 1, 'usual': driver.character,
                      'edited': driver.key in self.edited} for driver in self.lineup],
            'yours': self.yours, 'stats': self.stats,
            # Everyone's names, racing or not, since a new name can't be anyone's.
            'castNames': dict(self.names),
            'strategy': self.strategy, 'myPlan': self.my_plan, 'compounds': list(COMPOUNDS), 'fuelCapacity': FUEL_CAPACITY,
            'pitLane': self.race.pit.describe(),
            'weather': {'forecast': self.race.weather.forecast, 'forecasts': list(FORECASTS)},
            'makesPitCalls': makes_pit_calls(self.policy),
            'specRanges': {name: [round(spec_value(name, 0), 2), round(spec_value(name, 1), 2)] for name in SPEC_DEFAULTS},
            # Whether this brain reads personality at all: networks trained before it existed don't.
            'readsPersonality': self.driver is not None and (self.driver.feels or self.policy.observation_space.shape[0] >= len(OBSERVATION_NAMES)),
            # Called before the first tick, so this is where every car lines up.
            'grid': [{'x': round(float(x), 2), 'y': round(float(y), 2), 'heading': round(float(heading), 4)}
                     for (x, y), heading in zip(self.race.cars.position, self.race.cars.heading)],
        }

    def advance(self, ticks):
        """Plays up to `ticks` physics ticks and returns a frame describing where everything is now."""
        race = self.race
        happened = {'contacts': [], 'laps': [], 'finished': [], 'pitEntered': [], 'pitStopped': [], 'pitReleased': [], 'pitExited': [],
                    'punctures': [], 'engineFailures': [], 'mistakes': [], 'debris': [], 'timing': [], 'fastestLap': None,
                    'propHits': [], 'crashed': []}
        for _ in range(ticks):
            if race.done:
                break
            # Drivers decide every DECISION_TICKS ticks and hold their controls in between, as in training.
            if self.controls is None or race.tick % DECISION_TICKS == 0:
                self.controls = self._decide()
            time_before, progress_before = race.time, race.progress.copy()
            events = race.step(*self.controls)
            happened['timing'] += self.timing.record(time_before, race.time, progress_before, race.progress)
            self.timing.sample(race.time, race.cars.position, race.cars.heading)
            happened['contacts'] += [[round(float(contact.point[0]), 2), round(float(contact.point[1]), 2), round(contact.impulse, 2)] for contact in events.contacts]
            happened['debris'] += np.round(race.new_debris, 2).tolist()
            for key, values in (('laps', events.laps), ('finished', events.finished), ('pitEntered', events.pit_entered),
                                ('pitReleased', events.pit_released), ('pitExited', events.pit_exited), ('punctures', events.punctures),
                                ('engineFailures', events.engine_failures), ('mistakes', events.mistakes), ('crashed', events.crashed)):
                happened[key] += [int(car) for car in values]
            happened['propHits'] += [{'car': hit.car, 'prop': hit.prop, 'impulse': round(hit.impulse, 2), 'broke': hit.broke,
                                      'x': round(float(hit.point[0]), 2), 'y': round(float(hit.point[1]), 2)} for hit in events.prop_hits]
            for car, jobs, standing in events.pit_stopped:
                plan = race.service_jobs[car]['plan']
                happened['pitStopped'].append({
                    'car': car, 'jobs': jobs, 'standing': standing, 'compound': COMPOUNDS[int(plan.compound[car])],
                    'wing': round(float(plan.wing[car]), 3), 'engine': round(float(plan.engine[car]), 3),
                })
            if self.yours in events.pit_released:
                self.box_called = False
        happened['fastestLap'] = self.timing.take_fastest_trace()
        return self._frame(happened)

    def _decide(self):
        steer, throttle, brake, call, plan = field_controls(self.race, self.driver, np.ones(self.race.count, dtype=bool), self.personality, pitwall=self.pitwall)
        if self.pitwall_id == SCRIPTED_STRATEGIST:
            call, plan = scripted_strategy(self.race)
        if self.strategy == 'mine':
            car, mine = self.yours, self.my_plan
            call[car] = self.box_called
            plan.compound[car] = COMPOUNDS.index(mine['compound'])
            plan.fuel[car] = mine['fuel'] * FUEL_CAPACITY
            plan.wing[car], plan.engine[car] = mine['wing'], mine['engine']
            plan.repair[car], plan.brakes[car] = mine['repair'], mine['brakes']
        return steer, throttle, brake, call, plan

    frames = 0

    @staticmethod
    def _times(seconds):
        return [None if np.isnan(time) else round(float(time), 3) for time in seconds]

    def _frame(self, happened):
        race, cars = self.race, self.race.cars
        steer, throttle, brake = self.controls[:3] if self.controls else (np.zeros(race.count),) * 3
        rounded = lambda values, digits=2: np.round(np.asarray(values, dtype=float), digits).tolist()
        conditions = race.conditions()
        self.frames += 1
        return {
            'type': 'frame', 'time': round(race.time, 2), 'done': race.done,
            'cars': {
                'x': rounded(cars.position[:, 0]), 'y': rounded(cars.position[:, 1]), 'heading': rounded(cars.heading, 4),
                'speed': rounded(cars.speed), 'steer': rounded(steer), 'throttle': rounded(throttle), 'brake': rounded(brake),
                'sliding': cars.sliding.astype(int).tolist(), 'offTrack': (~race.on_track).astype(int).tolist(),
                'tyreWear': rounded(cars.tyre_wear, 3), 'fuel': rounded(cars.fuel / FUEL_CAPACITY, 3),
                'lap': race.lap_of().tolist(), 'progress': rounded(race.progress, 1),
                'finishTime': [None if np.isnan(time) else round(float(time), 2) for time in race.finish_time],
                'compound': [COMPOUNDS[index] for index in cars.compound], 'wing': rounded(cars.wing, 3), 'engine': rounded(cars.engine, 3),
                'damage': rounded(cars.damage, 3), 'brakeWear': rounded(cars.brake_wear, 3),
                'lapStart': self._times(self.timing.lap_start), 'lastLap': self._times(self.timing.last_lap), 'bestLap': self._times(self.timing.best_lap),
                'pit': race.pit_state.tolist(), 'stops': race.stops.tolist(), 'serviceLeft': rounded(race.service_left, 2),
                'tyreTemp': rounded(cars.tyre_temp, 3), 'brakeTemp': rounded(cars.brake_temp, 3), 'engineTemp': rounded(cars.engine_temp, 3),
                'punctured': cars.punctured.astype(int).tolist(), 'retired': race.retired.astype(int).tolist(),
                'wingDamage': rounded(cars.wing_damage, 3), 'suspensionDamage': rounded(cars.suspension_damage, 3),
                'draft': rounded(conditions.draft), 'dirtyAir': rounded(conditions.dirty_air), 'balance': rounded(race.specs.balance),
            },
            'debris': np.round(race.debris, 2).tolist(),
            # The track's rubber, marbles and dry line change slowly, so they go out about once a second.
            'surface': surface_map(race.surface) if self.frames % SURFACE_EVERY == 1 else None,
            'order': race.standings().tolist(),
            'weather': {'wetness': round(race.weather.wetness, 3), 'rain': round(race.weather.rain, 3),
                        'temperature': round(race.weather.temperature, 3), 'wind': np.round(race.weather.wind, 2).tolist()},
            'yourPit': {'strategy': self.strategy, 'boxCalled': self.box_called},
            'events': happened,
        }


SURFACE_EVERY = 20   # frames between surface maps (about a second at normal speed)


def surface_map(surface):
    """The track surface for the viewer: rubber, marbles and dry line, each a byte (0-255) per cell, row by row
    along the track and lane by lane across it, base64-encoded."""
    layer = lambda values: base64.b64encode(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8).tobytes()).decode('ascii')
    rows, lanes = surface.rubber.shape
    return {'rows': rows, 'lanes': lanes, 'cellLength': CELL_LENGTH,
            'rubber': layer(surface.rubber), 'marbles': layer(surface.marbles), 'dryness': layer(surface.dryness)}
