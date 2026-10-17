"""The driver exam: fixed scenarios a snapshot either passes or doesn't, and scores it's compared on.

Watching races and reading numbers misses things (a pit test that drove the lane on rails hid for weeks that the
racing driver couldn't drive into the pit lane at all). So every behaviour that matters is a test here, run the
same way on the same circuits with the same seeds for every snapshot, through the game's own race session
(session.RaceSession), so the exam sees what a player sees.

- A `must` test passes or fails on a rule (finishes all 12 races, makes every stop it's called for...).
- A `scored` test gives a number, higher or lower being better, which gate.py compares between snapshots with a
  margin for noise.

Groups: A driving alone, B starts, C racing and traffic, D the pit lane, E strategy and long races, F weather,
G personality (each trait does what it says), H the car (each spec and setup does what it says, and every extreme
still races), I regressions (every bug found becomes a test). The quick tier is a subset for every snapshot; the
full exam runs before a snapshot can be published.

    .venv/bin/python -m slipstream.exam racer-2/step-000080018720            # the full exam, into runs/<run>/exam/
    .venv/bin/python -m slipstream.exam racer-2/step-000080018720 --quick
    .venv/bin/python -m slipstream.exam scripted                             # the scripted driver, as a reference
"""
import argparse
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import personality as traits
from .car import COMPOUNDS, FUEL_CAPACITY
from .race import HARD_HIT
from .scenery import LOCATIONS
from .session import RUNS, SCRIPTED, SCRIPTED_STRATEGIST, STATS, RaceSession

# Circuits, by seed, kept apart from the ones training and the checker use.
SOLO_SEEDS = tuple(range(9101, 9113))
FEW_SEEDS = SOLO_SEEDS[:4]
FIELD_SEEDS = (9201, 9202, 9203)
PIT_SEEDS = tuple(range(9301, 9309))
NEUTRAL = 0.5
CHUNK = 10                 # ticks between the exam's looks at the race (half a second)
LIGHTS_SECONDS = 9.0       # the longest the lights can take, so a race is under way after it


@dataclass
class Result:
    group: str
    name: str
    kind: str              # 'must' or 'scored'
    value: float
    passed: bool | None = None       # for must tests
    better: str | None = None        # 'higher' or 'lower', for scored tests
    detail: str = ''


@dataclass
class Played:
    """One race as the exam saw it: every tick's events, and samples every CHUNK ticks."""
    session: RaceSession
    contacts: list = field(default_factory=list)     # (first, second, impulse)
    penalties: list = field(default_factory=list)    # (car, seconds, reason)
    crashed: list = field(default_factory=list)
    mistakes: list = field(default_factory=list)
    pit_entered: list = field(default_factory=list)
    stopped: list = field(default_factory=list)      # (car, standing seconds)
    lap_times: dict = field(default_factory=dict)    # car -> [lap time, ...]
    passes: list = field(default_factory=list)       # (passer, passed, hard hit within 2 s)
    top_speed: np.ndarray = None
    sliding: np.ndarray = None
    racing_samples: np.ndarray = None
    worn_samples: np.ndarray = None
    fuel_outs: set = field(default_factory=set)
    hardest_braking: np.ndarray = None               # m/s² of the strongest slowing seen between looks
    lane_contacts: int = 0                           # contacts with either car in the pit lane at the time

    @property
    def race(self):
        return self.session.race


def neutral(session, cars=None):
    """Every driver and car at the middle of every slider, so scenarios compare driving, not the cast's characters."""
    for car in range(session.race.count) if cars is None else cars:
        session.set_stats({name: NEUTRAL for name in STATS}, car)


def lights_out_now(session):
    session.race.marshals.lights_out_at = session.race.time


def place_car(session, car, distance, lateral=0.0, turn=0.0, speed=0.0):
    """Puts `car` `distance` metres round the lap, `lateral` metres to the left of the centre line, turned `turn`
    radians from the track's direction, moving at `speed`."""
    race = session.race
    track = race.track
    race.cars.position[car] = track.point_at(distance, lateral)
    race.cars.heading[car] = track.heading_at(distance) + turn
    race.cars.velocity[car] = race.cars.forward[car] * speed
    race.progress[car] = distance
    race.track_index, _, race.lateral = track.locate(race.cars.position)
    race.furthest[car] = distance
    race.gained_at[car] = race.time


def play(session, seconds=None, until=None):
    """Plays `session` to the end, for `seconds` of race time, or until `until(played)` is true."""
    played = Played(session)
    race = session.race
    count = race.count
    played.top_speed, played.sliding = np.zeros(count), np.zeros(count)
    played.racing_samples, played.worn_samples = np.zeros(count), np.zeros(count)
    played.hardest_braking = np.zeros(count)
    step = race.step

    def watched(*args, **kwargs):
        events = step(*args, **kwargs)
        order_before = watched.order
        played.contacts += [(contact.first, contact.second, contact.impulse) for contact in events.contacts]
        played.lane_contacts += sum(bool(race.in_lane[contact.first] or race.in_lane[contact.second]) for contact in events.contacts)
        played.penalties += [(penalty.car, penalty.seconds, penalty.reason) for penalty in events.penalties]
        played.crashed += list(events.crashed)
        played.mistakes += list(events.mistakes)
        played.pit_entered += list(events.pit_entered)
        played.stopped += [(car, standing) for car, _, standing in events.pit_stopped]
        order = race.standings()
        running = ~race.retired
        for place, car in enumerate(order):
            before = int(np.flatnonzero(order_before == car)[0])
            if place < before and running[car] and not race.in_lane[car]:
                for passed in order[place + 1:before + 1]:
                    if running[passed] and not race.in_lane[passed]:
                        rough = any(car in (first, second) and impulse >= HARD_HIT for first, second, impulse in played.contacts[-40:])
                        played.passes.append((int(car), int(passed), rough))
        watched.order = order
        return events
    watched.order = race.standings()
    race.step = watched
    end = None if seconds is None else race.time + seconds
    while not race.done and (end is None or race.time < end) and not (until and until(played)):
        speed_before, time_before = race.cars.speed.copy(), race.time
        frame = session.advance(CHUNK)
        if race.time > time_before:
            played.hardest_braking = np.maximum(played.hardest_braking, (speed_before - race.cars.speed) / (race.time - time_before))
        for event in frame['events']['timing']:
            if event['kind'] == 'lap':
                played.lap_times.setdefault(event['car'], []).append(event['time'])
        racing = ~race.finished & ~race.retired
        played.top_speed = np.maximum(played.top_speed, race.cars.speed * racing)
        played.sliding += race.cars.sliding & racing
        played.racing_samples += racing
        played.worn_samples += (race.cars.tyre_wear > 0.9) & racing
        played.fuel_outs |= set(np.flatnonzero((race.cars.fuel <= 0.01) & racing).tolist())
    race.step = step
    return played


class Exam:
    """The exam of one brain ('scripted', a run's snapshot, or a published one), with the scenarios several tests
    share played once."""

    def __init__(self, brain, runs=RUNS):
        self.brain, self.runs = brain, runs

    def session(self, seed, cars=1, laps=3, brain=None, stats=None, **options):
        session = RaceSession(brain or self.brain, seed=seed, cars=cars, laps=laps, runs=self.runs, **options)
        neutral(session)
        for car, changes in (stats or {}).items():
            session.set_stats({**{name: NEUTRAL for name in STATS}, **changes}, car)
        return session

    # ---- Shared scenarios --------------------------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def solo(self, seed, brain=None, location=None, forecast=None, stats=None, laps=3, stops=False):
        """One car alone for `laps` laps; with `stops` the team calls the stops, otherwise there are none."""
        session = self.session(seed, laps=laps, brain=brain, location=location, forecast=forecast,
                               stats={0: dict(stats)} if stats else None)
        session.set_pit(strategy='ai' if stops else 'mine')
        if stops:
            session.pitwall_id = SCRIPTED_STRATEGIST
        return play(session)

    def solo_location(self, index):
        return LOCATIONS[index % len(LOCATIONS)]

    @lru_cache(maxsize=None)
    def field(self, seed, cars=12, laps=3, stats=None, pitwall=None, race_format='standard', minutes=None):
        """A field of the brain's own copies."""
        options = {'pitwall_id': pitwall} if pitwall else {}
        session = self.session(seed, cars=cars, laps=laps, race_format=race_format, minutes=minutes, **options)
        if stats:
            for car in range(cars):
                session.set_stats({**{name: NEUTRAL for name in STATS}, **dict(stats)}, car)
        return play(session)

    @lru_cache(maxsize=None)
    def called_in(self, seed):
        """One car called in from the first lap, its own plan; plays until it has stopped, or two laps."""
        session = self.session(seed, laps=4)
        session.set_pit(strategy='mine')
        played = play(session, until=lambda played: played.race.lap_of()[0] >= 1)
        session.set_pit(box=True)
        rest = play(session, until=lambda later: later.race.stops[0] > 0 and later.race.pit_state[0] == 0 or later.race.lap_of()[0] >= 3)
        for name in ('penalties', 'pit_entered', 'stopped', 'contacts'):
            getattr(played, name).extend(getattr(rest, name))
        return played

    # ---- The tests ---------------------------------------------------------------------------------------------

    def run(self, quick=False):
        results = []
        for test in TESTS:
            if quick and not test.quick:
                continue
            began = time.time()
            try:
                result = test.function(self)
            except Exception as error:   # a test that can't run fails, saying why, rather than ending the exam
                result = Result(test.group, test.name, test.kind, float('nan'), False if test.kind == 'must' else None,
                                test.better, f'could not run: {type(error).__name__}: {error}')
            result.group, result.name, result.kind = test.group, test.name, test.kind
            if test.kind == 'scored':
                result.better = test.better
            result.detail = (result.detail + f' ({time.time() - began:.0f} s)').strip()
            results.append(result)
            print(f"{'PASS' if result.passed else 'FAIL' if result.passed is False else '    '} {test.group} {test.name}: "
                  f'{result.value:.3f} {result.detail}', flush=True)
        return results


@dataclass
class Test:
    group: str
    name: str
    kind: str
    better: str | None
    quick: bool
    function: object


TESTS: list[Test] = []


def test(group, name, kind='must', better=None, quick=False):
    """Registers an exam test: a function of the Exam returning a Result (or a (value, passed) pair for a must test,
    or a value for a scored one)."""
    def register(function):
        def wrapped(exam):
            outcome = function(exam)
            if isinstance(outcome, Result):
                return outcome
            if kind == 'must':
                value, passed, *detail = outcome
                return Result(group, name, kind, float(value), bool(passed), detail=detail[0] if detail else '')
            value, *detail = outcome if isinstance(outcome, tuple) else (outcome,)
            return Result(group, name, kind, float(value), better=better, detail=detail[0] if detail else '')
        TESTS.append(Test(group, name, kind, better, quick, wrapped))
        return function
    return register


def lap_time(played, car=0):
    return float(played.race.finish_time[car] / played.race.laps) if played.race.finished[car] else float('nan')


def scripted_ratio(exam, seeds, **options):
    """The scripted driver's race time over this brain's, alone on each circuit (above 1 is faster; 0 where this brain
    doesn't finish). Circuits the scripted driver doesn't finish say nothing about pace and are left out."""
    ratios = []
    for seed in seeds:
        mine, script = exam.solo(seed, **options), exam.solo(seed, brain=SCRIPTED, **options)
        if script.race.finished[0]:
            ratios.append(script.race.finish_time[0] / mine.race.finish_time[0] if mine.race.finished[0] else 0.0)
    return float(np.mean(ratios)) if ratios else float('nan')


# ---- A. Driving alone ----------------------------------------------------------------------------------------------

@test('A', 'finishes alone on 12 circuits', quick=True)
def finishes_alone(exam):
    done = sum(bool(exam.solo(seed, location=exam.solo_location(index)).race.finished[0]) for index, seed in enumerate(SOLO_SEEDS))
    return done, done == len(SOLO_SEEDS), f'{done}/{len(SOLO_SEEDS)}'


@test('A', 'dry pace against the scripted driver', 'scored', 'higher', quick=True)
def dry_pace(exam):
    return scripted_ratio(exam, SOLO_SEEDS[:6])


@test('A', 'lap time spread (s)', 'scored', 'lower')
def lap_spread(exam):
    spreads = [np.std(times[1:]) for seed in SOLO_SEEDS[:6] if len(times := exam.solo(seed).lap_times.get(0, [])) >= 3]
    return float(np.mean(spreads)) if spreads else float('nan')


@test('A', 'share of time off track alone', 'scored', 'lower', quick=True)
def off_track_alone(exam):
    return float(np.mean([exam.solo(seed).race.off_track_ticks[0] / max(exam.solo(seed).race.tick, 1) for seed in SOLO_SEEDS[:6]]))


@test('A', 'track-limit strikes per lap alone', 'scored', 'lower')
def strikes_alone(exam):
    return float(np.mean([exam.solo(seed).race.stewards.strikes[0] / exam.solo(seed).race.laps for seed in SOLO_SEEDS[:6]]))


@test('A', 'damage taken alone', 'scored', 'lower')
def damage_alone(exam):
    return float(np.mean([exam.solo(seed).race.cars.damage[0] for seed in SOLO_SEEDS[:6]]))


def recovers(exam, set_up, seconds, check):
    """Starts each of FEW_SEEDS' cars in trouble (`set_up(session)`), lights out, and asks whether `check` holds within
    `seconds`."""
    recovered = 0
    for seed in FEW_SEEDS:
        session = exam.session(seed)
        lights_out_now(session)
        start = set_up(session)
        played = play(session, seconds=seconds, until=lambda played: check(played, start))
        recovered += bool(check(played, start))
    return recovered, recovered == len(FEW_SEEDS), f'{recovered}/{len(FEW_SEEDS)}'


def mid_corner(session):
    """The distance round the lap of the circuit's tightest bend."""
    return float(np.argmax(np.abs(session.race.track.curvature)) * session.race.track.length / len(session.race.track.curvature))


@test('A', 'recovers from a spin within 10 s', quick=True)
def spin_recovery(exam):
    def set_up(session):
        distance = mid_corner(session)
        place_car(session, 0, distance, turn=np.pi / 2)
        return distance
    return recovers(exam, set_up, 10, lambda played, start: played.race.progress[0] > start + 40 and played.race.cars.speed[0] > 12)


@test('A', 'turns round from facing the wrong way within 25 s')
def wrong_way(exam):
    def set_up(session):
        place_car(session, 0, 300.0, turn=np.pi)
        return 300.0
    return recovers(exam, set_up, 25, lambda played, start: played.race.progress[0] > start + 40)


@test('A', 'gets back from the grass within 8 s')
def grass_escape(exam):
    def set_up(session):
        place_car(session, 0, 300.0, lateral=session.race.track.width / 2 + 5.0)
        return 300.0
    return recovers(exam, set_up, 8, lambda played, start: bool(played.race.on_track[0]) and played.race.progress[0] > start + 20)


@test('A', 'frees itself from a wall within 20 s')
def wall_escape(exam):
    def set_up(session):
        race = session.race
        walls = [prop for prop in race.props.props if prop.kind in ('armco', 'wall', 'pitwall')]
        prop = walls[len(walls) // 3]
        index, _, _ = race.track.locate(np.array([[prop.x, prop.y]]))
        distance = float(index[0] * race.track.length / len(race.track.curvature))
        _, _, lateral = race.track.locate(np.array([[prop.x, prop.y]]))
        side = np.sign(lateral[0]) or 1.0
        # Nose against it: just inside the barrier, turned to face it.
        place_car(session, 0, distance, lateral=lateral[0] - side * 2.2, turn=side * np.pi / 2)
        return distance
    return recovers(exam, set_up, 20, lambda played, start: played.race.progress[0] > start + 30)


# ---- B. Starts -----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def full_grid(exam, seed):
    return play(exam.session(seed, cars=20, laps=3), seconds=LIGHTS_SECONDS + 30)


@test('B', 'no jump starts in any field race', quick=True)
def no_jump_starts(exam):
    jumps = sum(reason == 'jump start' for seed in FIELD_SEEDS for _, _, reason in exam.field(seed).penalties)
    jumps += sum(reason == 'jump start' for seed in FIELD_SEEDS[:1] for _, _, reason in full_grid(exam, seed).penalties)
    return jumps, jumps == 0


@test('B', 'reaction to the lights (s)', 'scored', 'lower')
def reaction(exam):
    times = []
    for seed in FEW_SEEDS:
        session = exam.session(seed)
        out = session.race.marshals.lights_out_at
        played = play(session, until=lambda played: played.race.cars.speed[0] > 2.0 or played.race.time > out + 5)
        # Moving before the lights go out is a jump start: it scores as the slowest reaction, not the quickest.
        reacted = played.race.time - out
        times.append(5.0 if reacted < 0 else reacted)
    return float(np.mean(times))


@test('B', 'all 20 cars get away from a full grid', quick=True)
def full_grid_gets_away(exam):
    away = [int((full_grid(exam, seed).race.progress > 60).sum()) for seed in FIELD_SEEDS]
    return min(away), min(away) == 20, f'{away} of 20'


@test('B', 'hard hits per car on the first lap of a full grid', 'scored', 'lower')
def first_lap_hits(exam):
    return float(np.mean([sum(impulse >= HARD_HIT for _, _, impulse in full_grid(exam, seed).contacts) / 20 for seed in FIELD_SEEDS]))


# ---- C. Racing and traffic -------------------------------------------------------------------------------------------

def stuck_share(played):
    return float(np.mean(played.race.retired_by == 'stuck'))


@test('C', 'cars stuck in a field of its copies', 'scored', 'lower', quick=True)
def field_stuck(exam):
    return float(np.mean([stuck_share(exam.field(seed)) for seed in FIELD_SEEDS]))


@test('C', 'no more than 15% stuck in a field of its copies', quick=True)
def field_stuck_ceiling(exam):
    share = max(stuck_share(exam.field(seed)) for seed in FIELD_SEEDS)
    return share, share <= 0.15


@test('C', 'hard hits per car in a field of its copies', 'scored', 'lower', quick=True)
def field_hits(exam):
    return float(np.mean([sum(impulse >= HARD_HIT for _, _, impulse in exam.field(seed).contacts) / 12 for seed in FIELD_SEEDS]))


@test('C', 'penalty seconds per car in a field of its copies', 'scored', 'lower', quick=True)
def field_penalties(exam):
    return float(np.mean([exam.field(seed).race.stewards.seconds.mean() for seed in FIELD_SEEDS]))


@test('C', 'place against scripted rivals (0 first, 1 last)', 'scored', 'lower')
def mixed_place(exam):
    places = []
    for seed in FIELD_SEEDS:
        networked = [car % 2 == 0 for car in range(8)]
        played = play(exam.session(seed, cars=8, laps=3, networked=networked))
        order = list(played.race.standings())
        places += [order.index(car) / 7 for car in range(8) if networked[car]]
    return float(np.mean(places))


def stopped_ahead(exam, followers, seconds):
    """A car stopped on the racing line 150 m ahead of `followers` cars of this brain; how many get past it."""
    passed = 0
    for seed in FEW_SEEDS:
        session = exam.session(seed, cars=followers + 1, networked=[False] + [True] * followers)
        lights_out_now(session)
        wreck = 400.0
        place_car(session, 0, wreck, lateral=0.0)
        session.race.retire(0, 'stuck')
        session.recovered[0] = True          # it stays where it is: nobody comes to clear it
        for follower in range(1, followers + 1):
            place_car(session, follower, wreck - 150 - 12 * follower, lateral=(-1) ** follower * 2.0, speed=25.0)
        played = play(session, seconds=seconds, until=lambda played: (played.race.progress[1:] > wreck + 10).all())
        passed += int((played.race.progress[1:] > wreck + 10).sum())
    return passed


@test('C', 'goes round a car stopped on the racing line within 15 s', quick=True)
def round_a_stopped_car(exam):
    passed = stopped_ahead(exam, 1, 15)
    return passed, passed == len(FEW_SEEDS), f'{passed}/{len(FEW_SEEDS)}'


@test('C', 'a queue of five gets round a stopped car within 40 s')
def queue_clears(exam):
    passed = stopped_ahead(exam, 5, 40)
    return passed, passed == 5 * len(FEW_SEEDS), f'{passed}/{5 * len(FEW_SEEDS)}'


def behind_a_slow_car(exam, seed, seconds=90):
    """This brain 40 m behind a scripted car with the slowest car on the grid."""
    session = exam.session(seed, cars=2, laps=5, networked=[False, True],
                           stats={0: {'top_speed': 0.0, 'acceleration': 0.0, 'grip': 0.2}})
    lights_out_now(session)
    place_car(session, 0, 300.0, lateral=1.5, speed=25.0)
    place_car(session, 1, 260.0, lateral=-1.5, speed=25.0)
    return play(session, seconds=seconds)


@test('C', 'clean passes of a slower car', 'scored', 'higher')
def clean_passes(exam):
    clean = 0
    for seed in FEW_SEEDS:
        played = behind_a_slow_car(exam, seed)
        clean += any(passer == 1 and not rough for passer, _, rough in played.passes)
    return clean / len(FEW_SEEDS)


@test('C', 'collision penalties defending against a faster car', 'scored', 'lower')
def defending(exam):
    blamed = 0
    for seed in FEW_SEEDS:
        session = exam.session(seed, cars=2, laps=5, networked=[True, False],
                               stats={1: {'top_speed': 1.0, 'acceleration': 1.0, 'grip': 1.0, 'aggression': 1.0}})
        lights_out_now(session)
        place_car(session, 0, 300.0, lateral=1.5, speed=25.0)
        place_car(session, 1, 270.0, lateral=-1.5, speed=25.0)
        played = play(session, seconds=90)
        blamed += sum(car == 0 and reason == 'collision' for car, _, reason in played.penalties)
    return blamed / len(FEW_SEEDS)


@test('C', 'no blue-flag penalties when lapped')
def blue_flags(exam):
    penalties = 0
    for seed in FEW_SEEDS:
        # The lapping car is this brain too, in the fastest car: a car that can't get past would make a fair driver
        # look guilty of holding it up.
        session = exam.session(seed, cars=2, laps=6, stats={1: {'top_speed': 1.0, 'grip': 1.0, 'acceleration': 1.0}})
        lights_out_now(session)
        length = session.race.track.length
        place_car(session, 0, 300.0, lateral=1.5, speed=20.0)
        place_car(session, 1, 260.0, lateral=-1.5, speed=30.0)
        session.race.progress[1] += length          # a lap ahead: the car behind is lapping it
        played = play(session, seconds=60)
        penalties += sum(car == 0 and reason == 'blue flags' for car, _, reason in played.penalties)
    return penalties, penalties == 0


@test('C', 'no passing under yellow flags')
def yellow_flags(exam):
    passes = sum(reason == 'yellow flag' for seed in FIELD_SEEDS for _, _, reason in exam.field(seed).penalties)
    return passes, passes == 0


@test('C', 'contact per lap following a car closely', 'scored', 'lower')
def close_following(exam):
    touches = []
    for seed in FEW_SEEDS:
        session = exam.session(seed, cars=2, laps=4)
        lights_out_now(session)
        place_car(session, 0, 300.0, speed=25.0)
        place_car(session, 1, 280.0, speed=25.0)
        played = play(session, seconds=60)
        touches.append(len(played.contacts) / max(played.race.lap_of()[1] + 1, 1))
    return float(np.mean(touches))


# ---- D. The pit lane -----------------------------------------------------------------------------------------------

@test('D', 'takes the pit call on 8 circuits', quick=True)
def takes_the_call(exam):
    stops = sum(exam.called_in(seed).race.stops[0] > 0 for seed in PIT_SEEDS)
    return stops, stops == len(PIT_SEEDS), f'{stops}/{len(PIT_SEEDS)}'


@test('D', 'no pit-lane speeding', quick=True)
def pit_speeding(exam):
    speeding = sum(reason == 'pit speeding' for seed in PIT_SEEDS for _, _, reason in exam.called_in(seed).penalties)
    return speeding, speeding == 0


@test('D', 'no crossing the pit-lane lines', quick=True)
def pit_lines(exam):
    crossed = sum(reason == 'pit line' for seed in PIT_SEEDS for _, _, reason in exam.called_in(seed).penalties)
    return crossed, crossed == 0


@test('D', 'standing time in the box (s)', 'scored', 'lower')
def box_time(exam):
    standing = [time for seed in PIT_SEEDS for _, time in exam.called_in(seed).stopped]
    return float(np.mean(standing)) if standing else float('nan'), f'{len(standing)} stops'


@test('D', 'time a stop costs against the scripted driver', 'scored', 'lower')
def pit_loss(exam):
    def loss(brain):
        costs = []
        for seed in PIT_SEEDS[:4]:
            plain = exam.solo(seed, brain=brain, laps=3)
            session = exam.session(seed, laps=3, brain=brain)
            session.set_pit(strategy='mine')
            play(session, until=lambda played: played.race.lap_of()[0] >= 1)
            session.set_pit(box=True)
            stopping = play(session)
            if plain.race.finished[0] and stopping.race.finished[0] and stopping.race.stops[0]:
                costs.append(stopping.race.finish_time[0] - plain.race.finish_time[0])
        return float(np.mean(costs)) if costs else float('nan')
    return loss(None) - loss(SCRIPTED)


@test('D', 'never goes into the pit lane uncalled', quick=True)
def only_when_called(exam):
    entries = sum(len(exam.solo(seed).pit_entered) for seed in SOLO_SEEDS[:6])
    return entries, entries == 0


@test('D', 'three cars called together all stop without contact')
def shared_lane(exam):
    good = 0
    for seed in FEW_SEEDS:
        session = exam.session(seed, cars=3, laps=4)
        play(session, until=lambda played: played.race.lap_of().min() >= 1)
        session.set_pit(strategy='mine')
        original = session._decide

        def everyone_in():
            controls = list(original())
            controls[3] = controls[3] | (session.race.stops == 0)
            return tuple(controls)
        session._decide = everyone_in
        played = play(session, until=lambda later: (later.race.stops > 0).all() or later.race.lap_of().min() >= 3)
        good += bool((played.race.stops > 0).all() and played.lane_contacts == 0)
    return good, good == len(FEW_SEEDS), f'{good}/{len(FEW_SEEDS)}'


# ---- E. Strategy and long races ------------------------------------------------------------------------------------

def long_race(exam, seed):
    return exam.field(seed, cars=6, laps=20, pitwall=SCRIPTED_STRATEGIST)


@test('E', 'a 20-lap race finishes, every car stopping', quick=True)
def twenty_laps(exam):
    good = 0
    for seed in FIELD_SEEDS[:2]:
        race = long_race(exam, seed).race
        good += int(race.finished.sum() >= 5 and (race.stops[race.finished] >= 1).all())
    return good, good == 2, f'{good}/2 races'


@test('E', 'never runs out of fuel in a 20-lap race', quick=True)
def never_dry(exam):
    dry = sum(len(long_race(exam, seed).fuel_outs) for seed in FIELD_SEEDS[:2])
    return dry, dry == 0


@test('E', 'share of racing on worn-out tyres', 'scored', 'lower')
def worn_tyres(exam):
    races = [long_race(exam, seed) for seed in FIELD_SEEDS[:2]]
    return float(sum(played.worn_samples.sum() for played in races) / max(sum(played.racing_samples.sum() for played in races), 1))


@test('E', 'endurance: both mandatory stops made')
def endurance_stops(exam):
    played = exam.field(FIELD_SEEDS[0], cars=6, laps=5, race_format='endurance')
    made = int((played.race.stops >= 2).sum())
    return made, made == 6, f'{made}/6 cars'


def radioed(exam, seed, pace):
    session = exam.session(seed, laps=4)
    session.set_pit(strategy='mine')
    session.set_radio(pace=pace)
    return play(session)


@test('E', 'the radio: push is faster than save')
def radio_push(exam):
    faster = sum(lap_time(radioed(exam, seed, 'push')) < lap_time(radioed(exam, seed, 'save')) for seed in FEW_SEEDS)
    return faster, faster >= len(FEW_SEEDS) - 1, f'{faster}/{len(FEW_SEEDS)}'


@test('E', 'the radio: save uses less fuel than push')
def radio_save(exam):
    used = lambda played: (FUEL_CAPACITY - played.race.cars.fuel[0]) / max(played.race.laps, 1)
    less = sum(used(radioed(exam, seed, 'save')) < used(radioed(exam, seed, 'push')) for seed in FEW_SEEDS)
    return less, less >= len(FEW_SEEDS) - 1, f'{less}/{len(FEW_SEEDS)}'


@test('E', 'orders: attack passes more than bring it home', 'scored', 'higher')
def attack_order(exam):
    def passes(approach):
        made = 0
        for seed in FEW_SEEDS:
            session = exam.session(seed, cars=2, laps=5, networked=[False, True], stats={0: {'top_speed': 0.2, 'grip': 0.3}})
            session.set_yours(1)
            session.set_radio(approach=approach)
            lights_out_now(session)
            place_car(session, 0, 300.0, lateral=1.5, speed=25.0)
            place_car(session, 1, 260.0, lateral=-1.5, speed=25.0)
            made += sum(passer == 1 for passer, _, _ in play(session, seconds=90).passes)
        return made
    return passes('attack') - passes('bring_it_home')


# ---- F. Weather ----------------------------------------------------------------------------------------------------

@test('F', 'wet pace against the scripted driver', 'scored', 'higher', quick=True)
def wet_pace(exam):
    return scripted_ratio(exam, FEW_SEEDS, forecast='wet')


@test('F', 'share of time off track in the wet', 'scored', 'lower')
def wet_off_track(exam):
    return float(np.mean([exam.solo(seed, forecast='wet').race.off_track_ticks[0] / max(exam.solo(seed, forecast='wet').race.tick, 1) for seed in FEW_SEEDS]))


@test('F', 'finishes when rain arrives mid-race')
def rain_arrives(exam):
    done = sum(bool(exam.solo(seed, forecast='rain_coming', laps=6, stops=True).race.finished[0]) for seed in FEW_SEEDS)
    return done, done == len(FEW_SEEDS), f'{done}/{len(FEW_SEEDS)}'


@test('F', 'pace on a drying track against the scripted driver', 'scored', 'higher')
def drying_pace(exam):
    return scripted_ratio(exam, FEW_SEEDS, forecast='drying')


# ---- G. Personality: each trait does what it says ------------------------------------------------------------------

def solo_with(exam, trait, value, seeds=FEW_SEEDS, laps=3):
    return [exam.solo(seed, stats=((trait, value),), laps=laps) for seed in seeds]


def mean(values):
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float('nan')


@test('G', 'more risk: faster')
def risk_pace(exam):
    low, high = (mean([lap_time(played) for played in solo_with(exam, 'risk', value)]) for value in (0.0, 1.0))
    return low - high, high < low, f'{low:.2f} s a lap at 0, {high:.2f} s at 1'


@test('G', 'more risk: more time off track')
def risk_off_track(exam):
    share = lambda value: mean([played.race.off_track_ticks[0] / max(played.race.tick, 1) for played in solo_with(exam, 'risk', value)])
    low, high = share(0.0), share(1.0)
    return high - low, high >= low, f'{low:.3f} at 0, {high:.3f} at 1'


@test('G', 'more conservation: less tyre wear')
def conservation_wear(exam):
    wear = lambda value: mean([played.race.cars.tyre_wear[0] for played in solo_with(exam, 'conservation', value)])
    low, high = wear(0.0), wear(1.0)
    return low - high, high < low, f'{low:.3f} at 0, {high:.3f} at 1'


@test('G', 'more consistency: steadier laps')
def consistency_spread(exam):
    spread = lambda value: mean([np.mean(np.abs(np.diff(played.lap_times.get(0, [0, 0, 0])[1:]))) for played in solo_with(exam, 'consistency', value, laps=5)])
    low, high = spread(0.0), spread(1.0)
    return low - high, high <= low, f'{low:.3f} s at 0, {high:.3f} s at 1'


@test('G', 'more stamina: less pace lost late in a long race')
def stamina_fade(exam):
    def fade(value):
        losses = []
        for played in solo_with(exam, 'stamina', value, seeds=FEW_SEEDS, laps=12):
            times = played.lap_times.get(0, [])
            if len(times) >= 10:
                losses.append(np.mean(times[-3:]) - np.mean(times[1:4]))
        return mean(losses)
    low, high = fade(0.0), fade(1.0)
    return low - high, high <= low, f'{low:.2f} s lost at 0, {high:.2f} s at 1'


def field_with(exam, trait, value):
    return [exam.field(seed, cars=8, stats=((trait, value),)) for seed in FIELD_SEEDS[:2]]


@test('G', 'more aggression: more contact')
def aggression_contact(exam):
    contact = lambda value: mean([len(played.contacts) / 8 for played in field_with(exam, 'aggression', value)])
    low, high = contact(0.0), contact(1.0)
    return high - low, high >= low, f'{low:.2f} a car at 0, {high:.2f} at 1'


@test('G', 'more overtaking: more passes')
def overtaking_passes(exam):
    passes = lambda value: mean([len(played.passes) / 8 for played in field_with(exam, 'overtaking', value)])
    low, high = passes(0.0), passes(1.0)
    return high - low, high >= low, f'{low:.2f} a car at 0, {high:.2f} at 1'


# ---- H. The car: each setting does what it says, and every extreme still races -------------------------------------

def car_with(exam, changes, seeds=FEW_SEEDS[:3], **options):
    return [exam.solo(seed, stats=tuple(sorted(changes.items())), **options) for seed in seeds]


def spec_test(spec, measure, more_is, description):
    """A spec at its lowest and highest: `measure(played)` must move `more_is` ('higher' or 'lower') with it, and the
    car must finish at both ends."""
    @test('H', f'{spec}: {description}')
    def direction(exam):
        low, high = (mean([measure(played) for played in car_with(exam, {spec: value})]) for value in (0.0, 1.0))
        moved = high > low if more_is == 'higher' else high < low
        return high - low, moved, f'{low:.2f} at 0, {high:.2f} at 1'

    @test('H', f'{spec}: finishes at both ends')
    def finishes(exam):
        done = sum(bool(played.race.finished[0]) for value in (0.0, 1.0) for played in car_with(exam, {spec: value}))
        return done, done == 6, f'{done}/6'


spec_test('top_speed', lambda played: played.top_speed[0], 'higher', 'faster on the straight')
spec_test('acceleration', lambda played: lap_time(played), 'lower', 'quicker laps')
spec_test('braking', lambda played: played.hardest_braking[0], 'higher', 'brakes harder')
spec_test('grip', lambda played: lap_time(played), 'lower', 'quicker laps')


@test('H', 'balance: finishes at both ends')
def balance_finishes(exam):
    done = sum(bool(played.race.finished[0]) for value in (0.0, 1.0) for played in car_with(exam, {'balance': value}))
    return done, done == 6, f'{done}/6'


@test('H', 'balance: sliding with the most oversteer', 'scored', 'lower')
def oversteer_sliding(exam):
    return mean([played.sliding[0] / max(played.racing_samples[0], 1) for played in car_with(exam, {'balance': 1.0})])


def setup_race(exam, seed, wing=None, compound=None, forecast=None):
    """Alone with the wing or the starting tyres set (the engine mode is the driver's own button)."""
    session = exam.session(seed, laps=3, forecast=forecast)
    session.set_pit(strategy='mine')
    cars = session.race.cars
    if wing is not None:
        cars.wing[0] = wing
    if compound is not None:
        cars.compound[0] = COMPOUNDS.index(compound)
    return play(session)


@test('H', 'wing: finishes with none and with full')
def wing_finishes(exam):
    done = sum(bool(setup_race(exam, seed, wing=wing).race.finished[0]) for seed in FEW_SEEDS[:3] for wing in (0.0, 1.0))
    return done, done == 6, f'{done}/6'


@test('H', 'every tyre in the dry finishes; softs wear fastest')
def dry_compounds(exam):
    races = {compound: [setup_race(exam, seed, compound=compound) for seed in FEW_SEEDS[:2]] for compound in ('soft', 'medium', 'hard')}
    done = all(played.race.finished[0] for played_list in races.values() for played in played_list)
    wear = {compound: mean([played.race.cars.tyre_wear[0] for played in played_list]) for compound, played_list in races.items()}
    return wear['soft'] - wear['hard'], done and wear['soft'] > wear['hard'], ', '.join(f'{name} {value:.2f}' for name, value in wear.items())


@test('H', 'weather tyres beat slicks in the wet')
def wets_in_the_wet(exam):
    lap = lambda compound: mean([lap_time(setup_race(exam, seed, compound=compound, forecast='wet')) for seed in FEW_SEEDS[:2]])
    weather = min(lap('intermediate'), lap('wet'))
    slick = lap('medium')
    return slick - weather, weather < slick or not np.isfinite(slick), f'{weather:.2f} s a lap on the better weather tyre, {slick:.2f} on mediums'


EXTREMES = {
    'every setting lowest': {name: 0.0 for name in ('top_speed', 'acceleration', 'braking', 'grip', 'balance')},
    'every setting highest': {name: 1.0 for name in ('top_speed', 'acceleration', 'braking', 'grip', 'balance')},
    'worst brakes, highest top speed': {'braking': 0.0, 'top_speed': 1.0},
    'most oversteer, least grip': {'balance': 1.0, 'grip': 0.0},
}


@test('H', 'every extreme car finishes', quick=True)
def extreme_cars(exam):
    done = sum(bool(played.race.finished[0]) for changes in EXTREMES.values() for played in car_with(exam, changes, seeds=FEW_SEEDS[:2]))
    total = 2 * len(EXTREMES)
    return done, done == total, f'{done}/{total}'


@test('H', 'random cars from the grid finish')
def random_cars(exam):
    rng = np.random.default_rng(7)
    done = 0
    for seed in FEW_SEEDS:
        changes = {name: float(rng.uniform()) for name in ('top_speed', 'acceleration', 'braking', 'grip', 'balance')}
        done += bool(car_with(exam, changes, seeds=(seed,))[0].race.finished[0])
    return done, done == len(FEW_SEEDS), f'{done}/{len(FEW_SEEDS)}'


@test('H', 'every format runs to the flag')
def every_format(exam):
    from .formats import FORMATS
    done = sum(bool(exam.field(FIELD_SEEDS[0], cars=5, laps=3, race_format=kind, minutes=10).race.done) for kind in FORMATS)
    return done, done == len(FORMATS), f'{done}/{len(FORMATS)}'


@test('H', 'share finishing a full 20-car race', 'scored', 'higher')
def full_field_finishes(exam):
    return float(np.mean([exam.field(seed, cars=20, laps=3).race.finished.mean() for seed in FIELD_SEEDS[:2]]))


# ---- I. Regressions: every bug found is a test ---------------------------------------------------------------------

@test('I', 'circuit 21 with 20 cars: half the field finishes (the whole-field freeze)')
def circuit_21(exam):
    share = float(play(exam.session(21, cars=20, laps=3)).race.finished.mean())
    return share, share >= 0.5


# ---- Running it ----------------------------------------------------------------------------------------------------

def exam_path(brain, runs=RUNS, quick=False):
    run, _, step = brain.partition('/')
    folder = runs / run / 'exam' if step else Path('runs') / 'exam'
    return folder / f"{step or run}{'.quick' if quick else ''}.json"


def write(results, path, brain, quick):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'brain': brain, 'quick': quick, 'when': time.strftime('%Y-%m-%d %H:%M'),
                                'results': [result.__dict__ for result in results]}, indent=1, default=float) + '\n')


def examined(brain, runs, quick):
    """The brain's saved exam of this tier, running it first if there's none."""
    path = exam_path(brain, runs, quick)
    if not path.exists():
        write(Exam(brain, runs).run(quick=quick), path, brain, quick)
    return json.loads(path.read_text())


def watch(run_name, runs=RUNS, wait=120):
    """Examines a run's newest snapshot whenever there's a new one (skipping any it fell behind on) and puts it to
    the gate against the published driver: the quick exam first, the full exam only for one the quick exam passes.
    Each decision goes to runs/<run>/gate.jsonl with its reasons; a full pass is printed as PROMOTE."""
    from .gate import DEFAULT_DRIVER, decide
    run = runs / run_name
    while True:
        published = DEFAULT_DRIVER.read_text().strip()
        snapshots = sorted((run / 'checkpoints').glob('step-*.zip'))
        newest = f'{run_name}/{snapshots[-1].stem}' if snapshots else None
        if newest and newest != published and not exam_path(newest, runs, quick=True).exists():
            for quick in (True, False):
                challenger, champion = examined(newest, runs, quick), examined(published, runs, quick)
                passed, reasons, improvements, known = decide(challenger['results'], champion['results'])
                entry = {'brain': newest, 'against': published, 'tier': 'quick' if quick else 'full', 'passed': passed,
                         'reasons': reasons, 'improvements': improvements, 'known': known, 'when': time.strftime('%Y-%m-%d %H:%M')}
                with open(run / 'gate.jsonl', 'a') as log:
                    log.write(json.dumps(entry) + '\n')
                print(f"{'PROMOTE' if passed and not quick else 'pass' if passed else 'FAIL'} {entry['tier']} {newest}: "
                      f"{'; '.join(improvements if passed else reasons)}", flush=True)
                if not passed:
                    break
        time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description='Run the driver exam on a brain and save its results')
    parser.add_argument('brain', nargs='?', help="'scripted', or a snapshot like racer-2/step-000080018720")
    parser.add_argument('--quick', action='store_true', help='only the quick tier')
    parser.add_argument('--runs', default=str(RUNS), help='where the snapshot is (runs, or brains for a published one)')
    parser.add_argument('--watch', metavar='RUN', help="keep examining a run's newest snapshot and gating it")
    arguments = parser.parse_args()
    runs = Path(arguments.runs)
    if arguments.watch:
        return watch(arguments.watch, runs)
    if not arguments.brain:
        parser.error('name a brain, or --watch a run')
    results = Exam(arguments.brain, runs).run(quick=arguments.quick)
    path = exam_path(arguments.brain, runs, arguments.quick)
    write(results, path, arguments.brain, arguments.quick)
    failed = [result for result in results if result.passed is False]
    print(f'{len(results)} tests, {len(failed)} must-pass failed; saved to {path}')


if __name__ == '__main__':
    main()
