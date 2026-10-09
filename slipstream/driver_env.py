"""The training world for drivers that drive by feel (see senses.py).

The network here is only the driver. It steers and works one pedal; the team's scripted strategist calls the stops.
It senses its car rather than reading its numbers, and every race hands it a different car: wider spreads of
power, grip and braking than before, a random handling balance (some cars understeer, some oversteer), a random
setup, and whatever the weather brings. Its own hands and feet sit between what it decides and what the car does
(see human.py), with its consistency and stamina among its traits. To drive well across all that, it has to feel
out each car and the conditions as it goes, which is why it trains with a memory (an LSTM, sb3-contrib's
RecurrentPPO).
"""
import numpy as np

from .car import BALANCE_RANGE, SPEC_DEFAULTS, CarSpecs
from .env import LAPS, RaceVecEnv
from .human import Driving
from .personality import TRAITS, trait
from .race import Race
from .senses import SENSE_NAMES, senses
from .strategy import scripted_strategy
from .track import generate_track
from .weather import Weather

SPEC_SPREAD = 0.35          # each spec anywhere this far either side of the default
WING_RANGE = (0.2, 0.8)     # setups the team might send a car out with
# A car that gains no ground for this long is beached and out of the race, and pays RETIRE_COST: otherwise a car
# stuck on the grass sits there for the rest of the race, and getting stuck would be a way out of paying for time.
STALLED_SECONDS = 15.0
RETIRE_COST = 2.0


def random_cars(count, rng):
    """Cars drawn wide: every spec ±SPEC_SPREAD and a handling balance anywhere in ±BALANCE_RANGE."""
    specs = [value * rng.uniform(1 - SPEC_SPREAD, 1 + SPEC_SPREAD, count) for value in SPEC_DEFAULTS.values()]
    return CarSpecs(*specs, balance=rng.uniform(-BALANCE_RANGE, BALANCE_RANGE, count))


class DriverEnv(RaceVecEnv):
    def __init__(self, races=16, cars=6, laps=LAPS, seed=0, rewards=None, rivals=0):
        self.furthest = {}
        super().__init__(races=races, cars=cars, laps=laps, seed=seed, rewards=rewards, rivals=rivals)
        # The driver makes no stops, so the value of the car's condition (which pays for stops) has no part here.
        self.condition_strength = 0.0

    def new_race(self, personality=None):
        race = self._new_race(personality)
        # The furthest each car has got, and when it last got further, for spotting beached cars.
        self.furthest[id(race)] = (race.progress.copy(), np.zeros(race.count))
        return race

    def after_drive(self, race):
        """Retires cars that have stopped making progress, and charges them for it."""
        best, gained_at = self.furthest[id(race)]
        gaining = race.progress > best + 1.0
        best = np.where(gaining, race.progress, best)
        gained_at = np.where(gaining, race.time, gained_at)
        self.furthest[id(race)] = (best, gained_at)
        beached = ~race.finished & ~race.retired & (race.time - gained_at > STALLED_SECONDS)
        race.retired |= beached
        if race.done:
            del self.furthest[id(race)]
        return -RETIRE_COST * beached

    def observation_size(self):
        return len(SENSE_NAMES)

    def action_size(self):
        return 2

    def observe(self, race, personality):
        return senses(race, personality)

    def controls(self, race, actions):
        actions = np.asarray(actions, dtype=float).reshape(race.count, 2)
        call, plan = scripted_strategy(race)
        return actions[:, 0], np.clip(actions[:, 1], 0, 1), np.clip(-actions[:, 1], 0, 1), call, plan

    def _new_race(self, personality):
        seed = int(self.rng.integers(2 ** 31))
        track = generate_track(seed)
        laps = self.laps if isinstance(self.laps, int) else int(self.rng.integers(self.laps[0], self.laps[1] + 1))
        personality = personality if personality is not None else np.full((self.cars, len(TRAITS)), 0.5)
        driving = Driving(self.cars, np.random.default_rng(seed + 1), consistency=trait(personality, 'consistency'),
                          stamina=trait(personality, 'stamina'))
        race = Race(track, random_cars(self.cars, self.rng), laps, weather=Weather.for_race(seed, track.length, laps),
                    rng=np.random.default_rng(seed), driving=driving)
        race.cars.wing[:] = self.rng.uniform(*WING_RANGE, self.cars)
        return race
