"""Training the pit wall (see pitwall.py for what it sees and decides).

How it trains (`PitWallEnv`): cars race alone (ghost races, see Race), with the driver network at the wheel.
Each training step is one pit decision followed by a lap of driving. Every car pauses at its decision point
until all the cars in its group have reached theirs, so a whole batch of decisions is made at once. The reward
is minus the time the lap took (in minutes), so the pit wall learns whatever finishes the race soonest; a car
that runs out of time or fuel pays for the race it didn't finish.
"""
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from . import pit
from .brains import NetworkDriver
from .car import DT, SPEC_SPREAD
from .drivers import DECISION_TICKS
from .env import LAPS, random_specs
from .human import Driving
from .personality import random_personalities
from .pitwall import PITWALL_ACTION_NAMES, PITWALL_OBSERVATION_NAMES, pitwall_decisions, pitwall_observe
from .race import SECONDS_PER_LAP_LIMIT, Race
from .strategy import RACING_SPEED
from .track import generate_track
from .weather import Weather

DNF_PENALTY = 1.0     # minutes added on top of the time an unfinished race would still have needed
STUCK_SECONDS = 60.0  # a car that gains no ground for this long (its own driving time) is out of the race
# No lap takes anywhere near this many ticks: a group that drives this long between decisions has a bug, and
# training stops with what every unsettled car was doing rather than run on forever.
MAX_TICKS_PER_DECISION = 100_000

# Everything a car carries from tick to tick, saved and put back while it waits at its decision point.
CAR_FIELDS = ('position', 'heading', 'velocity', 'tyre_wear', 'fuel', 'compound', 'wing', 'engine', 'damage', 'brake_wear', 'sliding', 'lateral_load',
              'tyre_temp', 'brake_temp', 'engine_temp', 'engine_failed', 'punctured', 'wing_damage', 'suspension_damage',
              'yaw_rate', 'longitudinal_g', 'lateral_g', 'wheelspin')
RACE_FIELDS = ('progress', 'track_index', 'lateral', 'finish_time', 'retired', 'pit_state', 'lane_along', 'lane_speed', 'lane_start',
               'lane_shift', 'service_left', 'pit_decided', 'stops', 'fuel_used', 'tyre_used', 'off_track_ticks', 'contact_ticks')


def random_race(rng, cars, laps, ghosts=False):
    """A race on a new circuit with random cars, in the weather its seed brings."""
    seed = int(rng.integers(2 ** 31))
    track = generate_track(seed)
    return Race(track, random_specs(cars, rng, SPEC_SPREAD), laps, ghosts=ghosts, weather=Weather.for_race(seed, track.length, laps),
                rng=np.random.default_rng(seed), driving=Driving(cars, np.random.default_rng(seed + 1)))


def network_driver(model):
    """Driving by a driver network: steer, throttle and brake for every car, from what each one senses. Each race
    gets its own NetworkDriver, so a recurrent driver's memory follows its race."""
    at_the_wheel = {}

    def drive(race, personality):
        if id(race) not in at_the_wheel:
            # Races come and go; only the ones still running need their drivers' memories.
            if len(at_the_wheel) > 64:
                at_the_wheel.clear()
            at_the_wheel[id(race)] = (race, NetworkDriver(model, race.count))
        actions = at_the_wheel[id(race)][1].act(race, personality)
        return actions[:, 0], np.clip(actions[:, 1], 0, 1), np.clip(-actions[:, 1], 0, 1)
    return drive


def awaiting_decision(race):
    """Cars at their decision point that haven't made this lap's decision yet."""
    distance = race.progress % race.track.length
    return race.pit.in_window(distance) & ~race.pit_decided & (race.pit_state == pit.RACING) & ~race.finished


class PitWallEnv(VecEnv):
    def __init__(self, drive, groups=4, cars=32, laps=LAPS, seed=0):
        """`drive(race, personality)` gives every car's steer, throttle and brake (see `network_driver`)."""
        self.drive, self.group_count, self.cars, self.laps = drive, groups, cars, laps
        self.rng = np.random.default_rng(seed)
        self.render_mode = None
        observation_space = spaces.Box(-np.inf, np.inf, (len(PITWALL_OBSERVATION_NAMES),), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (len(PITWALL_ACTION_NAMES),), np.float32)
        super().__init__(groups * cars, observation_space, action_space)
        self.groups: list[dict[str, Any]] = []
        self.returns = np.zeros(self.num_envs)
        self.lengths = np.zeros(self.num_envs, dtype=int)

    # ---- Groups of ghost races ---------------------------------------------------------------------------

    def _new_group(self):
        laps = self.laps if isinstance(self.laps, int) else int(self.rng.integers(self.laps[0], self.laps[1] + 1))
        race = random_race(self.rng, self.cars, laps, ghosts=True)
        group = {
            'race': race, 'personality': random_personalities(self.cars, self.rng),
            'own_ticks': np.zeros(self.cars, dtype=int), 'out': np.zeros(self.cars, dtype=bool),   # out: ran dry or out of time
            'reported': np.zeros(self.cars, dtype=bool), 'call': np.zeros(self.cars, dtype=bool),
            'plan': pit.PitPlan.standard(self.cars), 'penalty': np.zeros(self.cars),
            'best_progress': race.progress.copy(), 'gained_at': np.zeros(self.cars, dtype=int),
        }
        self._drive(group, deciding=np.zeros(self.cars, dtype=bool))
        return group

    def _over(self, group):
        return group['race'].finished | group['out']

    def _drive(self, group, deciding):
        """Drives every car that isn't over until it reaches its next decision point. Cars in `deciding` are at
        their decision point now and make it on the first tick; cars that get to theirs wait there."""
        race = group['race']
        waiting = self._over(group) & ~deciding
        controls, tick = None, 0
        while not waiting.all():
            if tick >= MAX_TICKS_PER_DECISION:
                raise RuntimeError(f'Cars never reached a decision point or the end of their race after {tick} ticks: {_describe(race, group, ~waiting)}')
            if controls is None or tick % DECISION_TICKS == 0:
                controls = self.drive(race, group['personality'])
            saved = _save(race, waiting)
            race.step(*controls, group['call'], group['plan'])
            _restore(race, saved, waiting)
            group['own_ticks'][~waiting] += 1
            tick += 1
            self._retire(group, waiting)
            waiting |= self._over(group) | awaiting_decision(race)

    def _retire(self, group, waiting):
        """Cars that ran dry or ran out of time stop here and pay for the race they didn't finish."""
        race = group['race']
        racing = ~waiting & ~race.finished & ~group['out']
        gaining = race.progress > group['best_progress'] + 1.0
        group['best_progress'] = np.where(gaining, race.progress, group['best_progress'])
        group['gained_at'] = np.where(gaining, group['own_ticks'], group['gained_at'])
        dry = (race.cars.fuel <= 0) & (np.abs(race.cars.speed) < 0.5) & ~race.in_lane
        stuck = (group['own_ticks'] - group['gained_at']) * DT > STUCK_SECONDS
        late = group['own_ticks'] * DT > race.laps * SECONDS_PER_LAP_LIMIT
        retiring = racing & (dry | stuck | late)
        if retiring.any():
            remaining = np.maximum(race.laps * race.track.length - race.progress, 0.0)
            group['penalty'][retiring] = remaining[retiring] / RACING_SPEED / 60 + DNF_PENALTY
            group['out'] |= retiring

    # ---- VecEnv -----------------------------------------------------------------------------------------

    def reset(self):
        self.groups = [self._new_group() for _ in range(self.group_count)]
        self.returns[:] = 0
        self.lengths[:] = 0
        return self._observations()

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=float).reshape(self.num_envs, -1)

    def step_wait(self):
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for number, group in enumerate(self.groups):
            slots = slice(number * self.cars, (number + 1) * self.cars)
            race = group['race']
            deciding = awaiting_decision(race)
            call, plan = pitwall_decisions(self.actions[slots])
            group['call'] = call
            group['plan'] = plan
            ticks_before = group['own_ticks'].copy()
            over_before = self._over(group)
            self._drive(group, deciding)
            ended = self._over(group) & ~over_before
            rewards[slots] = -(group['own_ticks'] - ticks_before) * DT / 60 - group['penalty'] * ended
            # Cars whose race is over: the episode ends now (and again, as an empty step, every step after, until
            # the whole group is over and a new one starts).
            dones[slots] = self._over(group)
            if self._over(group).all():
                observations_before = pitwall_observe(race, group['personality'])
                self.groups[number] = self._new_group()
                for car, slot in enumerate(range(slots.start, slots.stop)):
                    infos[slot]['terminal_observation'] = observations_before[car]
            else:
                for car, slot in enumerate(range(slots.start, slots.stop)):
                    if dones[slot]:
                        infos[slot]['terminal_observation'] = pitwall_observe(race, group['personality'])[car]
            for car, slot in enumerate(range(slots.start, slots.stop)):
                if ended[car]:
                    infos[slot]['race'] = {'finished': float(race.finished[car]), 'stops': int(race.stops[car]), 'laps': race.laps,
                                           'minutes': group['own_ticks'][car] * DT / 60}
        self.returns += rewards
        self.lengths += 1
        for slot in np.flatnonzero(dones):
            if 'race' in infos[slot]:
                infos[slot]['episode'] = {'r': float(self.returns[slot]), 'l': int(self.lengths[slot])}
        self.returns[dones] = 0
        self.lengths[dones] = 0
        return self._observations(), rewards.astype(np.float32), dones, infos

    def _observations(self):
        return np.concatenate([pitwall_observe(group['race'], group['personality']) for group in self.groups])

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else list(indices)


def _describe(race, group, cars):
    return {int(car): {'progress': round(float(race.progress[car]), 1), 'laps': race.laps, 'speed': round(float(race.cars.speed[car]), 1),
                       'fuel': round(float(race.cars.fuel[car]), 1), 'pit_state': int(race.pit_state[car]), 'decided': bool(race.pit_decided[car]),
                       'service_left': float(race.service_left[car]), 'own_seconds': round(group['own_ticks'][car] * DT, 1),
                       'lateral': round(float(race.lateral[car]), 1)} for car in np.flatnonzero(cars)}


def _save(race, cars):
    """Copies of everything about the given cars that a tick could change."""
    if not cars.any():
        return None
    return {
        'car': {name: getattr(race.cars, name)[cars].copy() for name in CAR_FIELDS},
        'race': {name: getattr(race, name)[cars].copy() for name in RACE_FIELDS},
        'jobs': [race.service_jobs[car] for car in np.flatnonzero(cars)],
    }


def _restore(race, saved, cars):
    """Puts the given cars back exactly as they were, so a tick passes them by."""
    if saved is None:
        return
    for name, values in saved['car'].items():
        getattr(race.cars, name)[cars] = values
    for name, values in saved['race'].items():
        getattr(race, name)[cars] = values
    for car, jobs in zip(np.flatnonzero(cars), saved['jobs']):
        race.service_jobs[car] = jobs


# ---- Benchmark ----------------------------------------------------------------------------------------------

# (seed, laps, forecast): the dry races the pit wall has always been measured on, then the same circuits in
# changing weather, where the tyres have to suit the track.
BENCHMARK_RACES = ((1002, 8, 'dry'), (1005, 8, 'dry'), (1007, 8, 'dry'), (1002, 10, 'dry'), (1005, 10, 'dry'), (1007, 10, 'dry'),
                   (1002, 8, 'rain_coming'), (1005, 8, 'drying'), (1007, 10, 'shower'), (1003, 8, 'wet'))
TRAFFIC_SEEDS = (1003, 1008)
TRAFFIC_LAPS = 8


def relative_pace(reference, candidate):
    """The reference's race time over the candidate's, so above 1 means the candidate was faster. A candidate
    that didn't finish scores 0. A race the reference didn't finish says nothing about the candidate's pace, so
    it's left out (None): scored, it made the whole benchmark's average infinite."""
    if not np.isfinite(reference):
        return None
    if not np.isfinite(candidate):
        return 0.0
    return reference / candidate


def benchmark(pitwall, driver):
    """The driver alone over long races on fixed circuits, three ways: the pit wall making the calls, the scripted
    strategist making them, and never stopping. `pace` is the strategist's race time over the pit wall's (above 1,
    the pit wall is faster; 0 if it didn't finish), `beats_no_stop` the same against never stopping over the races
    never stopping finishes (`no_stop_failed` is the share it doesn't, like staying on slicks in the rain), `stops` the
    pit wall's average stops and `finished` its share of races finished. Then races in traffic: `traffic_place`
    is the average place of the cars the pit wall calls for, racing cars the scripted strategist calls for."""
    # Imported here: field.py uses this module, so it can't be imported at the top.
    from .car import CarSpecs
    from .field import field_controls
    from .strategy import scripted_strategy
    paces, versus_none, scripted_failed, no_stop_failed, stops, finished = [], [], [], [], [], []
    for seed, laps, forecast in BENCHMARK_RACES:
        track = generate_track(seed)
        times = {}
        for label in ('pitwall', 'scripted', 'none'):
            race = Race(track, CarSpecs.uniform(1), laps, weather=Weather.for_race(seed, track.length, laps, forecast),
                        rng=np.random.default_rng(seed), driving=Driving(1, np.random.default_rng(seed + 1)))
            at_the_wheel = NetworkDriver(driver, 1)
            while not race.done:
                steer, throttle, brake, call, plan = field_controls(race, at_the_wheel, np.array([True]), pitwall=pitwall if label == 'pitwall' else None)
                if label == 'scripted':
                    call, plan = scripted_strategy(race)
                elif label == 'none':
                    call = np.array([False])
                race.step(steer, throttle, brake, call, plan)
            times[label] = race.finish_time[0] if race.finished[0] else np.inf
            if label == 'pitwall':
                stops.append(race.stops[0])
                finished.append(float(race.finished[0]))
        scripted_failed.append(float(not np.isfinite(times['scripted'])))
        no_stop_failed.append(float(not np.isfinite(times['none'])))
        for scores, reference in ((paces, times['scripted']), (versus_none, times['none'])):
            pace = relative_pace(reference, times['pitwall'])
            if pace is not None:
                scores.append(pace)
    places = []
    for seed in TRAFFIC_SEEDS:
        # Six cars, all driven by the driver network: the pit wall calls the stops for every other one, the
        # scripted strategist for the rest. 0 means the pit wall's cars took the top places, 1 the bottom ones.
        track = generate_track(seed)
        race = Race(track, CarSpecs.uniform(6), TRAFFIC_LAPS, weather=Weather.for_race(seed, track.length, TRAFFIC_LAPS),
                    rng=np.random.default_rng(seed), driving=Driving(6, np.random.default_rng(seed + 1)))
        walled = np.arange(6) % 2 == 0
        at_the_wheel = NetworkDriver(driver, 6)
        while not race.done:
            # One call, so the driver's memory moves on once a tick; the pit wall's calls go to every other car.
            steer, throttle, brake, wall_call, wall_plan = field_controls(race, at_the_wheel, np.ones(6, dtype=bool), pitwall=pitwall)
            call = np.where(walled, wall_call, scripted_strategy(race)[0])
            plan = scripted_strategy(race)[1]
            for name in pit.PitPlan.__dataclass_fields__:
                getattr(plan, name)[:] = np.where(walled, getattr(wall_plan, name), getattr(plan, name))
            race.step(steer, throttle, brake, call, plan)
        place = np.empty(6)
        place[race.standings()] = np.arange(6) / 5
        places.append(place[walled].mean())
    return {'pace': float(np.mean(paces)), 'beats_no_stop': float(np.mean(versus_none)), 'no_stop_failed': float(np.mean(no_stop_failed)),
            'scripted_failed': float(np.mean(scripted_failed)),
            'stops': float(np.mean(stops)),
            'finished': float(np.mean(finished)), 'traffic_place': float(np.mean(places))}


# ---- Learning with traffic ----------------------------------------------------------------------------------

PLACE_VALUE = 0.5     # minutes' worth of reward for winning, sliding to nothing for last


class TrafficRaces:
    """Full races, cars in traffic, for the pit wall to learn where a stop puts it among the others (see
    train_pitwall.TrafficPPO). Nobody waits: each car asks the pit wall the moment it reaches its decision point,
    and each decision's reward is minus the time to the car's next decision (in minutes). At the end of the race
    a car also gets PLACE_VALUE scaled by its finishing place, or pays for the race it didn't finish.

    Every car is a slot. `tick(decide, record)` plays one tick of every race: `decide(observations)` gives
    (actions, values, log_probs) for the cars deciding now, and `record(slot, decision, reward, done, following)`
    hears about each decision once its reward is known (`following` is the slot's next observation, None at the end).
    """

    def __init__(self, drive, races=8, cars=6, laps=LAPS, seed=0):
        self.drive, self.race_count, self.cars, self.laps = drive, races, cars, laps
        self.rng = np.random.default_rng(seed)
        self.slots = races * cars
        self.races = [self._new_race() for _ in range(races)]
        self.controls = [None] * races
        self.ticks = 0

    def _new_race(self):
        laps = self.laps if isinstance(self.laps, int) else int(self.rng.integers(self.laps[0], self.laps[1] + 1))
        race = random_race(self.rng, self.cars, laps)
        return {'race': race, 'personality': random_personalities(self.cars, self.rng), 'call': np.zeros(self.cars, dtype=bool),
                'plan': pit.PitPlan.standard(self.cars), 'pending': [None] * self.cars, 'decided_at': np.zeros(self.cars)}

    def pending_observations(self):
        """Each slot's latest decision still waiting for its reward (zeros where there's none)."""
        observations = np.zeros((self.slots, len(PITWALL_OBSERVATION_NAMES)), dtype=np.float32)
        for number, entry in enumerate(self.races):
            for car, pending in enumerate(entry['pending']):
                if pending is not None:
                    observations[number * self.cars + car] = pending['observation']
        return observations

    def tick(self, decide, record):
        """One tick of every race. Returns summaries of races that ended."""
        deciders = []
        for number, entry in enumerate(self.races):
            for car in np.flatnonzero(awaiting_decision(entry['race'])):
                deciders.append((number, car))
        if deciders:
            observations = np.stack([pitwall_observe(self.races[number]['race'], self.races[number]['personality'])[car] for number, car in deciders])
            actions, values, log_probs = decide(observations)
            for index, (number, car) in enumerate(deciders):
                entry = self.races[number]
                race, slot = entry['race'], number * self.cars + car
                decision = {'observation': observations[index], 'action': actions[index], 'value': values[index],
                            'log_prob': log_probs[index], 'start': entry['pending'][car] is None}
                if entry['pending'][car] is not None:
                    record(slot, entry['pending'][car], -(race.time - entry['decided_at'][car]) / 60, False, decision['observation'])
                entry['pending'][car] = decision
                entry['decided_at'][car] = race.time
                call, plan = pitwall_decisions(np.clip(actions[index], -1, 1)[None])
                entry['call'][car] = call[0]
                for name in pit.PitPlan.__dataclass_fields__:
                    getattr(entry['plan'], name)[car] = getattr(plan, name)[0]
        ended = []
        for number, entry in enumerate(self.races):
            race = entry['race']
            if self.controls[number] is None or self.ticks % DECISION_TICKS == 0:
                self.controls[number] = self.drive(race, entry['personality'])
            race.step(*self.controls[number], entry['call'], entry['plan'])
            if race.done:
                ended.append(self._finish(number, record))
        self.ticks += 1
        return ended

    def _finish(self, number, record):
        entry = self.races[number]
        race = entry['race']
        places = np.empty(race.count)
        places[race.standings()] = np.arange(race.count)
        remaining = np.maximum(race.laps * race.track.length - race.progress, 0.0)
        for car, pending in enumerate(entry['pending']):
            if pending is None:
                continue
            if race.finished[car]:
                reward = -(race.finish_time[car] - entry['decided_at'][car]) / 60 + PLACE_VALUE * (race.count - 1 - places[car]) / (race.count - 1)
            else:
                reward = -(race.time - entry['decided_at'][car]) / 60 - remaining[car] / RACING_SPEED / 60 - DNF_PENALTY
            record(number * self.cars + car, pending, reward, True, None)
        summary = {'finished': float(race.finished.mean()), 'stops': float(race.stops.mean()), 'laps': race.laps}
        self.races[number] = self._new_race()
        self.controls[number] = None
        return summary
