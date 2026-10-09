"""Races as a Stable-Baselines3 vectorized environment.

Every car in every race is one slot, and one network drives them all (parameter sharing): with 16 races of 6
cars, each step is 96 decisions. A driver decides every DECISION_TICKS physics ticks (10 times a second) and
holds its controls in between. When a race ends, all its slots end together and a new race on a new circuit
takes their place.

Every driver gets a random personality and a random car each race. Both are inputs to the network, and the
personality reweighs that driver's rewards (see personality.py and `_drive`), so one network learns to drive
every combination.
"""
from dataclasses import fields
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .car import SPEC_DEFAULTS, SPEC_SPREAD, CarSpecs
from .drivers import DECISION_TICKS, scripted_controls
from . import personality as traits
from .observe import OBSERVATION_SIZE, observe
from .pit import PitPlan
from .strategy import ACTION_NAMES, condition_value, decode, scripted_strategy
from .race import Race
from .track import generate_track
from .weather import Weather

# Per metre of progress (a lap is about 12), per decision while still racing (so standing still always loses),
# per decision spent off the tarmac, per m/s of contact taken, and on taking the flag: `finish` for finishing at
# all plus `podium` scaled from last place (0) to the win (1). The off-track and contact costs are then scaled by
# the driver's personality, which also adds its own terms. An earlier off-track cost of 0.02 outweighed the progress a
# crawling car makes, and the first run learned to sit on the grid with the brake on. Progress made off the
# tarmac earns nothing (see _drive): the second run learned to cut corners across the grass. `places` multiplies
# what the personality pays per place gained or lost along the way (see personality.WEIGHTS).
REWARDS = {'progress': 0.01, 'time': 0.005, 'off_track': 0.005, 'contact': 0.03, 'finish': 1.0, 'podium': 1.0, 'places': 1.0}
# Training races are anywhere from this short (where a stop never pays) to this long (where one usually does).
LAPS = (3, 10)


def random_specs(count, rng, spread):
    """Car specs within ±spread of the defaults, each car drawn on its own."""
    return CarSpecs(*(value * rng.uniform(1 - spread, 1 + spread, count) for value in SPEC_DEFAULTS.values()))


class RaceVecEnv(VecEnv):
    def __init__(self, races=16, cars=6, laps=LAPS, seed=0, spec_spread=SPEC_SPREAD, rewards=None, rivals=0):
        """`laps` is a number of laps, or a (fewest, most) range each race draws from. `rivals` of each race's
        `cars` are driven by the scripted driver: the network only drives (and learns from) the rest, but races
        against all of them. With every car its own copy, a network can learn to hold back, since a place one copy
        gains another loses and only the cost of contact is sure; a scripted rival gives way to nobody."""
        if not 0 <= rivals < cars:
            raise ValueError(f'A race of {cars} cars can have from 0 to {cars - 1} scripted rivals, got {rivals}')
        self.race_count, self.cars, self.laps, self.spec_spread = races, cars, laps, spec_spread
        self.rivals = rivals
        self.learners = cars - rivals
        self.rewards = {**REWARDS, **(rewards or {})}
        self.rng = np.random.default_rng(seed)
        observation_space = spaces.Box(-np.inf, np.inf, (self.observation_size(),), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (self.action_size(),), np.float32)
        self.render_mode = None
        super().__init__(races * self.learners, observation_space, action_space)
        self.races: list[Race] = []
        self.personalities: list[np.ndarray] = []
        # Per race, which cars the network drives: a different set of grid slots each race.
        self.learner_cars: list[np.ndarray] = []
        # How much personality reshapes rewards, from 0 (everyone rewarded like a neutral driver) to 1. Training
        # raises it once the network can drive: personality costs that only a moving car pays (tyres, fuel, grip)
        # otherwise make sitting still look safest to a network that can't drive yet.
        self.personality_strength = 1.0
        # The same for condition (see strategy.py): its costs (damage from crashing, fuel burned while driving
        # erratically) also fall only on cars that move, so it too waits until the network can drive.
        self.condition_strength = 1.0
        self.returns = np.zeros(self.num_envs)
        self.lengths = np.zeros(self.num_envs, dtype=int)
        self.actions = np.zeros((self.num_envs, self.action_size()))
        self.condition: list[np.ndarray] = []

    # ---- What subclasses can change: what a driver observes, what its outputs do, and what races look like ----

    def observation_size(self):
        return OBSERVATION_SIZE

    def action_size(self):
        # Steering (left positive), one pedal (forward is throttle, back is brake), then the pit call and the plan
        # for the next stop (see strategy.decode).
        return len(ACTION_NAMES)

    def observe(self, race, personality):
        return observe(race, personality)

    def after_drive(self, race):
        """Anything else each car earns or pays once its decision has played out (nothing, here)."""
        return np.zeros(race.count)

    def controls(self, race, actions):
        """Steer, throttle, brake, pit call and stop plan for every car in a race."""
        return decode(actions, race.count)

    def new_race(self, personality=None):
        """A race on a new circuit. `personality` is the drivers' traits, for subclasses that need them."""
        seed = int(self.rng.integers(2 ** 31))
        track = generate_track(seed)
        laps = self.laps if isinstance(self.laps, int) else int(self.rng.integers(self.laps[0], self.laps[1] + 1))
        return Race(track, random_specs(self.cars, self.rng, self.spec_spread), laps, weather=Weather.for_race(seed, track.length, laps))

    def reset(self):
        self.personalities = [traits.random_personalities(self.cars, self.rng) for _ in range(self.race_count)]
        self.races = [self.new_race(personality) for personality in self.personalities]
        self.learner_cars = [self._seat_learners() for _ in self.races]
        self.condition = [self.condition_strength * condition_value(race) for race in self.races]
        self.returns[:] = 0
        self.lengths[:] = 0
        return self._observations()

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=float).reshape(self.num_envs, self.action_size())

    def step_wait(self):
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        observations = np.empty((self.num_envs, self.observation_size()), dtype=np.float32)
        for number, race in enumerate(self.races):
            slots = slice(number * self.learners, (number + 1) * self.learners)
            mine = self.learner_cars[number]
            actions = np.zeros((self.cars, self.action_size()))
            actions[mine] = self.actions[slots]
            earned = self._drive(race, self.personalities[number], actions, mine) + self.after_drive(race)
            observations[slots] = self.observe(race, self.personalities[number])[mine]
            # Condition shaping: every change in what the car's condition is worth, as it happens, and nothing
            # else. Over a race it adds up to (value at the end - value at the start): a car that reaches the flag
            # ends at 0, one that runs out of time ends below it, so never finishing can only cost. (A first try
            # paid 0.99 x new - old, the textbook form: with a value that stays below zero, that paid 1% of it every
            # decision for doing nothing, and that run learned to wander about and never finish.)
            condition = self.condition_strength * condition_value(race)
            rewards[slots] = (earned + condition - self.condition[number])[mine]
            self.condition[number] = condition
            if race.done:
                dones[slots] = True
                for slot in range(slots.start, slots.stop):
                    infos[slot]['terminal_observation'] = observations[slot].copy()
                infos[slots.start]['race'] = summarize(race)
                self.personalities[number] = traits.random_personalities(self.cars, self.rng)
                self.races[number] = race = self.new_race(self.personalities[number])
                self.learner_cars[number] = self._seat_learners()
                self.condition[number] = self.condition_strength * condition_value(race)
                observations[slots] = self.observe(race, self.personalities[number])[self.learner_cars[number]]
        self.returns += rewards
        self.lengths += 1
        for slot in np.flatnonzero(dones):
            infos[slot]['episode'] = {'r': float(self.returns[slot]), 'l': int(self.lengths[slot])}
        self.returns[dones] = 0
        self.lengths[dones] = 0
        return observations, rewards.astype(np.float32), dones, infos

    def _seat_learners(self):
        return np.sort(self.rng.permutation(self.cars)[:self.learners])

    def _drive(self, race, personality, actions, mine):
        steer, throttle, brake, pit_call, plan = self.controls(race, actions)
        if self.rivals:
            # Scripted rivals drive and run their stops as the scripted driver and strategist would.
            rival = ~np.isin(np.arange(race.count), mine)
            scripted_steer, scripted_throttle, scripted_brake = scripted_controls(race)
            steer, throttle, brake = (np.where(rival, scripted, own) for scripted, own
                                      in ((scripted_steer, steer), (scripted_throttle, throttle), (scripted_brake, brake)))
            scripted_call, scripted_plan = scripted_strategy(race)
            pit_call = np.where(rival, scripted_call, pit_call)
            plan = PitPlan(**{field.name: np.where(rival, getattr(scripted_plan, field.name), getattr(plan, field.name))
                              for field in fields(PitPlan)})
        racing = ~race.finished & ~race.retired
        weights, strength = traits.WEIGHTS, self.personality_strength
        # At strength 0 the multipliers are 1 and the extra terms vanish, which is how the basic run was rewarded.
        scaled = lambda multiplier: 1 + (multiplier - 1) * strength
        contact_cost = self.rewards['contact'] * scaled(traits.blend(weights['contact'], traits.trait(personality, 'aggression')))
        off_track_cost = self.rewards['off_track'] * scaled(traits.blend(weights['off_track'], traits.trait(personality, 'risk')))
        free_grip = traits.blend(weights['grip_margin'], traits.trait(personality, 'risk'))
        grip_excess = weights['grip_excess'] * strength
        place_value = traits.blend(weights['place'], traits.trait(personality, 'overtaking')) * strength * self.rewards['places']
        conservation = traits.trait(personality, 'conservation') * strength
        places_before = _places(race)
        reward = np.zeros(race.count)
        for _ in range(DECISION_TICKS):
            progress, wear, fuel = race.progress.copy(), race.cars.tyre_wear.copy(), race.cars.fuel.copy()
            events = race.step(steer, throttle, brake, pit_call, plan)
            # Track limits: ground gained off the tarmac doesn't count, so cutting across the infield never pays.
            reward += self.rewards['progress'] * (race.progress - progress) * race.on_track * racing
            reward -= off_track_cost / DECISION_TICKS * ~race.on_track
            for contact in events.contacts:
                reward[[contact.first, contact.second]] -= contact_cost[[contact.first, contact.second]] * contact.impulse
            # A cautious driver pays for using grip beyond its comfort margin; a risky one uses all of it freely.
            reward -= grip_excess / DECISION_TICKS * np.maximum(race.cars.lateral_load - free_grip, 0) * racing
            # Only what's used up counts: a stop refills the tank and resets the tyres, and if those counted as
            # negative use, a careful driver would be paid for every stop (a run learned to stop every lap).
            used_tyres = np.maximum(race.cars.tyre_wear - wear, 0.0)
            used_fuel = np.maximum(fuel - race.cars.fuel, 0.0)
            reward -= conservation * (weights['tyre_wear'] * used_tyres + weights['fuel'] * used_fuel)
            for car in events.finished:
                place = int(np.flatnonzero(race.standings() == car)[0])
                reward[car] += self.rewards['finish'] + self.rewards['podium'] * (race.count - 1 - place) / max(race.count - 1, 1)
        reward += place_value * (places_before - _places(race)) * racing
        reward -= self.rewards['time'] * racing
        return reward

    def _observations(self):
        return np.concatenate([self.observe(race, personality)[mine]
                               for race, personality, mine in zip(self.races, self.personalities, self.learner_cars)])

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


def _places(race):
    """Each car's position in the race, 0 for the leader."""
    places = np.empty(race.count, dtype=int)
    places[race.standings()] = np.arange(race.count)
    return places


def summarize(race):
    """How a race went, for the training charts."""
    finished = race.finished
    return {
        'finished': float(finished.mean()),
        # Average speed of the winner over the race distance, comparable across circuits of different lengths.
        'winner_speed': race.laps * race.track.length / float(np.nanmin(race.finish_time)) if finished.any() else 0.0,
        'off_track': float(race.off_track_ticks.sum() / (race.tick * race.count)),
        'contact': float(race.contact_ticks.sum() / (race.tick * race.count)),
        'laps_done': float(np.mean(np.minimum(race.lap_of(), race.laps))),
        'stops': float(race.stops.mean()),
    }
