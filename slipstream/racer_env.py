"""The training world for racing drivers (senses.RACER_SENSE_NAMES), trained from scratch.

Everything a race in the game has, a training race has: the stewards, the marshals' lights and flags, a pit lane
the driver drives itself, props to crash into, any weather. And races look like the game's: anywhere from
FIELD[0] to FIELD[1] cars, over LAPS[0] to LAPS[1] laps, cars set up with any wing. LEARNERS cars in every race
are the network learning; of the rest, up to LEAGUE_SHARE are driven by its own earlier snapshots (the league, so
it learns to race drivers like itself, not just the scripted driver who gives way to nobody and never attacks)
and the others by the scripted driver. The team's scripted strategist calls the stops and talks on the radio.

What a racing driver is paid for, on top of a driver by feel's (env.REWARDS, train.TRAFFIC_REWARDS):
- `penalty`, per second of penalty the stewards hand out: the same as that much race time is worth.
- `damage`, per unit of damage taken (all parts) per lap still to race, up to DAMAGE_LAPS: damage costs pace for
  the rest of the race, far beyond the few seconds a driver looks ahead, so it's charged when it happens.
- clean passes: a place gained within CLEAN_SECONDS of a hard hit it was in pays only `rough_pass` of a clean one.
- `stop` for stopping when the pit wall called (a missed call costs as much): the stop's pay-off is the pit
  wall's business, the driver's is to bring the car in.
- `radio`, per decision, for doing what the radio asks (the engine mode down when asked to save, up to push).

Like personality, these only count once the network can drive (train.CAN_DRIVE), fading in with the same ramp: a
first racing driver run paid them from the start, and a driver that could barely get round found that every start
in traffic cost it penalties and damage, so by 2M decisions it no longer left the grid.

- `still`, per decision a car sits below STILL_SPEED once the lights are out (not in the pit lane), on top of the
  time it loses: a field of a network's own copies once sat on the grid after the lights, each car waiting for the
  one ahead, and later queued for good behind a car that had stopped, rather than go round it.

A share of races are drills (PIT_DRILL): the learners are called in from the first lap, so the lane, the limiter
and the box come up far more often than real strategy would bring them. In START_DRILL of races the learners start
from the back of the grid, where getting away means finding a way round cars still slow off the line.
"""
from pathlib import Path

import numpy as np

from . import personality as traits
from . import pit
from .brains import NetworkDriver
from .drivers import DECISION_TICKS, scripted_controls
from .driver_env import DriverEnv, random_cars
from .env import _places, contact_shares
from .human import Driving
from .numpy_network import NumpyNetwork
from .personality import trait
from .race import HARD_HIT, Race
from .scenery import location_for, place
from .senses import RACER_SENSE_NAMES, racer_senses
from .strategy import RACER_ACTIONS, RADIO_CALLS, decode_racer, laps_left, scripted_radio, scripted_strategy
from .track import generate_track
from .weather import Weather

LEARNERS = 5
FIELD = (6, 20)
LAPS = (3, 25)
WING = (0.0, 1.0)
LEAGUE_SHARE = 0.5
LEAGUE_SIZE = 5
DAMAGE_LAPS = 10.0
CLEAN_SECONDS = 2.0
PIT_DRILL = 0.25
START_DRILL = 0.25
STILL_SPEED = 2.0
STALLED_SECONDS = 15.0
# On top of env.REWARDS and train.TRAFFIC_REWARDS. A second of race time is worth 0.12 (train.TRAFFIC_REWARDS
# charges 0.012 a decision), and a penalty second is charged the same. A hard hit worth 0.1 damage with five laps
# to go costs 0.75, about a place.
RACER_REWARDS = {'podium': 2.0, 'places': 2.0, 'time': 0.012, 'retire': 3.5, 'off_track': 0.01, 'hard_hit': 0.25,
                 'penalty': 0.12, 'damage': 1.5, 'rough_pass': 0.25, 'stop': 0.5, 'radio': 0.002, 'still': 0.024}


class RacerEnv(DriverEnv):
    has_league = True
    # Even the scripted driver finishes only about a fifth of these races (big fields, props, nobody avoiding
    # anybody, no help off a wall), so how many training cars finish never shows whether the network can drive.
    judged_alone = True

    def __init__(self, races=32, seed=0, rewards=None):
        self.league: list[NumpyNetwork] = []
        self.league_cars, self.league_drivers, self.drills, self.hit_at = {}, {}, {}, {}
        self.start_drills = {}
        super().__init__(races=races, cars=FIELD[1], laps=LAPS, seed=seed, rewards={**RACER_REWARDS, **(rewards or {})},
                         rivals=FIELD[1] - LEARNERS)

    def observation_size(self):
        return len(RACER_SENSE_NAMES)

    def action_size(self):
        return len(RACER_ACTIONS)

    def observe(self, race, personality):
        return racer_senses(race, personality)

    def field_size(self):
        return int(self.rng.integers(FIELD[0], FIELD[1] + 1))

    def add_to_league(self, path: Path):
        """Lets an exported snapshot (an .npz, see numpy_network.py) drive rivals from the next races on, keeping the
        LEAGUE_SIZE latest."""
        self.league = (self.league + [NumpyNetwork.load(path)])[-LEAGUE_SIZE:]

    def _new_race(self, personality):
        seed = int(self.rng.integers(2 ** 31))
        track = generate_track(seed)
        count = len(personality)
        # Race lengths drawn evenly on a log scale, so short races (most of the game's) come up as often as long ones
        # without the long ones (where stops and saving matter) being rare.
        laps = int(round(np.exp(self.rng.uniform(np.log(LAPS[0]), np.log(LAPS[1])))))
        driving = Driving(count, np.random.default_rng(seed + 1), consistency=trait(personality, 'consistency'),
                          stamina=trait(personality, 'stamina'))
        race = Race(track, random_cars(count, self.rng), laps, weather=Weather.for_race(seed, track.length, laps),
                    rng=np.random.default_rng(seed), driving=driving, stewards=True, assisted=False, flags=True)
        race.cars.wing[:] = self.rng.uniform(*WING, count)
        race.place_props(place(track, race.pit, seed, location_for(seed)))
        self.drills[id(race)] = bool(self.rng.random() < PIT_DRILL)
        self.start_drills[id(race)] = bool(self.rng.random() < START_DRILL)
        self.hit_at[id(race)] = np.full(count, -np.inf)
        return race

    def _seat_learners(self, race):
        """The grid slots go front to back, so a start drill's learners take the last ones."""
        if self.start_drills[id(race)]:
            return np.arange(race.count - self.learners, race.count)
        return super()._seat_learners(race)

    def _seat_league(self, race, mine):
        """Which of the cars the network isn't learning with a snapshot from the league drives, decided at a race's
        first decision (once the learners have their seats)."""
        others = np.setdiff1d(np.arange(race.count), mine)
        most = int(LEAGUE_SHARE * len(others)) if self.league else 0
        chosen = np.sort(self.rng.permutation(others)[:int(self.rng.integers(0, most + 1))]) if most else np.zeros(0, dtype=int)
        self.league_cars[id(race)] = chosen
        if len(chosen):
            snapshot = self.league[int(self.rng.integers(len(self.league)))]
            self.league_drivers[id(race)] = NetworkDriver(snapshot, race.count)

    def after_drive(self, race):
        """Retires cars that have stopped making progress, and charges them for it: but not a car on the grid
        waiting for the lights, or in the pit lane (its box can take longer than STALLED_SECONDS)."""
        best, gained_at = self.furthest[id(race)]
        waiting = race.in_lane | (not race.marshals.started(race.time))
        gaining = (race.progress > best + 1.0) | waiting
        best = np.where(gaining, np.maximum(race.progress, best), best)
        gained_at = np.where(gaining, race.time, gained_at)
        self.furthest[id(race)] = (best, gained_at)
        beached = ~race.finished & ~race.retired & (race.time - gained_at > STALLED_SECONDS)
        for car in np.flatnonzero(beached):
            race.retire(car, 'stuck')
        if race.done:
            for store in (self.furthest, self.league_cars, self.league_drivers, self.drills, self.hit_at,
                          self.start_drills):
                store.pop(id(race), None)
        return -self.crash_cost * beached

    def _drive(self, race, personality, actions, mine):
        count = race.count
        steer, throttle, brake, limiter, reverse, engine = decode_racer(actions, count)
        limiter, reverse, engine = limiter.astype(float), reverse.astype(float), engine.copy()
        if id(race) not in self.league_cars:
            self._seat_league(race, mine)
        league = self.league_cars[id(race)]
        if len(league):
            league_steer, league_throttle, league_brake, *league_buttons = decode_racer(self.league_drivers[id(race)].act(race, personality), count)
            steer[league], throttle[league], brake[league] = league_steer[league], league_throttle[league], league_brake[league]
            for button, value in zip((limiter, reverse, engine), league_buttons):
                button[league] = value[league]
        scripted = ~np.isin(np.arange(count), np.concatenate([mine, league]))
        scripted_steer, scripted_throttle, scripted_brake = scripted_controls(race)
        steer, throttle, brake = (np.where(scripted, script, own) for script, own
                                  in ((scripted_steer, steer), (scripted_throttle, throttle), (scripted_brake, brake)))
        for button in (limiter, reverse, engine):
            button[scripted] = np.nan
        call, plan = scripted_strategy(race)
        if self.drills.get(id(race)):
            learner = np.isin(np.arange(count), mine)
            call |= learner & (race.stops == 0) & (race.lap_of() >= 1)
        race.radio = scripted_radio(race)
        return self._rewarded(race, personality, mine, (steer, throttle, brake, call, plan, limiter, reverse, engine))

    def _rewarded(self, race, personality, mine, controls):
        """Plays the decision out and pays every car for it (see the module notes)."""
        rewards, count = self.rewards, race.count
        # personality_strength is the shaping ramp (train.Progress): 0 until the network can drive, then up to 1.
        weights, strength = traits.WEIGHTS, self.personality_strength
        scaled = lambda multiplier: 1 + (multiplier - 1) * strength
        contact_cost = rewards['contact'] * scaled(traits.blend(weights['contact'], traits.trait(personality, 'aggression')))
        off_track_cost = rewards['off_track'] * scaled(traits.blend(weights['off_track'], traits.trait(personality, 'risk')))
        free_grip = traits.blend(weights['grip_margin'], traits.trait(personality, 'risk'))
        grip_excess = weights['grip_excess'] * strength
        place_value = traits.blend(weights['place'], traits.trait(personality, 'overtaking')) * strength * rewards['places']
        conservation = traits.trait(personality, 'conservation') * strength
        hit_at = self.hit_at[id(race)]
        racing = ~race.finished & ~race.retired
        order_before = race.standings()
        called_before = race.pit_state == pit.CALLED
        reward = np.zeros(count)
        for _ in range(DECISION_TICKS):
            progress, wear, fuel = race.progress.copy(), race.cars.tyre_wear.copy(), race.cars.fuel.copy()
            harm = race.cars.damage + race.cars.wing_damage + race.cars.suspension_damage
            distance_before = race.progress % race.track.length
            events = race.step(*controls)
            reward += rewards['progress'] * (race.progress - progress) * race.on_track * racing
            reward -= off_track_cost / DECISION_TICKS * ~race.on_track
            for contact in events.contacts:
                for car, share in contact_shares(race.progress, contact.first, contact.second, rewards['contact_ahead']).items():
                    reward[car] -= contact_cost[car] * contact.impulse * share
                    if contact.impulse >= HARD_HIT:
                        reward[car] -= rewards['hard_hit'] * share
                        hit_at[car] = race.time
            reward -= grip_excess / DECISION_TICKS * np.maximum(race.cars.lateral_load - free_grip, 0) * racing
            used_tyres = np.maximum(race.cars.tyre_wear - wear, 0.0)
            used_fuel = np.maximum(fuel - race.cars.fuel, 0.0)
            reward -= conservation * (weights['tyre_wear'] * used_tyres + weights['fuel'] * used_fuel)
            harmed = np.maximum(race.cars.damage + race.cars.wing_damage + race.cars.suspension_damage - harm, 0.0)
            reward -= strength * rewards['damage'] * harmed * np.minimum(laps_left(race), DAMAGE_LAPS)
            for penalty in events.penalties:
                reward[penalty.car] -= strength * rewards['penalty'] * penalty.seconds
            for car, _, _ in events.pit_stopped:
                reward[car] += strength * rewards['stop']
            missed = called_before & (race.pit_state == pit.CALLED) & race.pit.entered(distance_before, race.progress % race.track.length)
            reward -= strength * rewards['stop'] * missed
            reward[events.crashed] -= self.crash_cost
            for car in events.finished:
                place_now = int(np.flatnonzero(race.standings() == car)[0])
                reward[car] += rewards['finish'] + rewards['podium'] * (count - 1 - place_now) / max(count - 1, 1)
        running = ~race.retired
        moved = _places(order_before, running) - _places(race.standings(), running)
        rough = race.time - hit_at < CLEAN_SECONDS
        reward += place_value * np.where((moved > 0) & rough, rewards['rough_pass'] * moved, moved) * racing
        engine = controls[-1]
        heard = race.radio
        followed = ((heard == RADIO_CALLS.index('save')) & (engine < 0.4)) | ((heard == RADIO_CALLS.index('push')) & (engine > 0.6))
        reward += strength * rewards['radio'] * followed
        reward -= rewards['time'] * racing
        stalled = racing & (race.cars.speed < STILL_SPEED) & ~race.in_lane & race.marshals.started(race.time)
        reward -= rewards['still'] * stalled
        return reward
