"""The pit wall: a second, small network that makes every pit decision, once a lap, for a car the driver network
drives.

Why a separate network: the driver decides ten times a second, and only one decision in about four hundred
falls at the pit entry. Learning when to stop from that one, among all the steering, drowned it out: a driver
network went thirteen million decisions without learning to stop. The pit wall sees one decision per lap and
nothing else, so every sample it learns from is a pit decision.

What it decides: whether to come in at the end of this lap and, if so, the plan (compound, fuel, wing, engine,
repairs, brakes). What it sees: the car's condition and setup, how much race is left, the fuel it needs, the
driver's personality and the car's specs (`pitwall_observe`).

How the pit wall trains is in pitwall_env.py; this is what a race needs to let it make its calls.
"""
import numpy as np

from .car import COMPOUNDS, FUEL_CAPACITY, SLICK_COUNT, centred
from .observe import SPEC_SCALE, WEATHER_NAMES, weather_inputs
from .personality import PERSONALITY_TRAITS, neutral
from .strategy import ACTION_NAMES, decode, fuel_to_finish, laps_left, tyre_outlook

PITWALL_ACTION_NAMES = ACTION_NAMES[2:]
TELEMETRY_NAMES = ['punctured', 'wing_damage', 'suspension_damage', 'tyre_temp', 'brake_temp', 'engine_temp']
PITWALL_OBSERVATION_NAMES = (
    ['tyre_wear', 'fuel', 'fuel_margin', 'laps_left', 'race_laps', 'damage', 'brake_wear', 'wing', 'engine', 'tyre_outlook', 'stops', 'track_length']
    + [f'compound.{name}' for name in COMPOUNDS[:SLICK_COUNT]]
    + [f'personality.{name}' for name in PERSONALITY_TRAITS]
    + [f'spec.{name}' for name in SPEC_SCALE]
    + ['place', 'gap_ahead', 'gap_behind']
    + WEATHER_NAMES
    # The team's telemetry, which the driver doesn't get: added with the race's wear and tear.
    + TELEMETRY_NAMES
)
LAPS_SCALE = 10.0
GAP_SCALE = 30.0      # seconds



def pitwall_observe(race, personality=None):
    # Laps and stops are capped where training's races (up to 10 laps) left them: a 50-lap race read as 5 times the
    # longest race the network had seen, and it called every car in nearly every lap.
    cars = race.cars
    personality = (neutral(race.count) if personality is None else np.asarray(personality, dtype=float))[:, :len(PERSONALITY_TRAITS)]
    laps = laps_left(race)
    numbers = np.stack([
        cars.tyre_wear, cars.fuel / FUEL_CAPACITY, np.clip((cars.fuel - fuel_to_finish(race)) / FUEL_CAPACITY, -1, 1),
        np.minimum(laps / LAPS_SCALE, 1.0), np.full(race.count, min(race.laps / LAPS_SCALE, 1.0)), cars.damage, cars.brake_wear,
        centred(cars.wing), centred(cars.engine), tyre_outlook(cars.tyre_wear, cars.compound, laps) / 5, np.minimum(race.stops / 3, 1.0),
        np.full(race.count, race.track.length / 1500),
    ], axis=1)
    specs = np.stack([getattr(race.specs, name) / scale for name, scale in SPEC_SCALE.items()], axis=1)
    slicks = np.eye(len(COMPOUNDS))[cars.compound][:, :SLICK_COUNT]
    telemetry = np.stack([cars.punctured.astype(float), cars.wing_damage, cars.suspension_damage, cars.tyre_temp, cars.brake_temp,
                          cars.engine_temp], axis=1)
    return np.concatenate([numbers, slicks, personality * 2 - 1, specs, traffic(race), weather_inputs(race), telemetry], axis=1).astype(np.float32)


def traffic(race):
    """Each car's place (0 leading, 1 last) and the time to the cars ahead and behind, in units of GAP_SCALE
    seconds and capped at 1. Ghost cars race alone: they lead, with nobody near."""
    count = race.count
    if race.ghosts or count == 1:
        return np.tile([0.0, 1.0, 1.0], (count, 1))
    standings = race.standings()
    place = np.empty(count)
    place[standings] = np.arange(count) / (count - 1)
    ordered = race.progress[standings]
    pace = max(float(np.median(np.maximum(race.cars.speed, 0))), 15.0)
    gaps = np.clip((ordered[:-1] - ordered[1:]) / pace / GAP_SCALE, 0, 1)
    ahead, behind = np.ones(count), np.ones(count)
    ahead[standings[1:]] = gaps
    behind[standings[:-1]] = gaps
    return np.stack([place, ahead, behind], axis=1)


def pitwall_decisions(actions):
    """The pit call and stop plan from the pit wall's outputs (the same meaning as the driver network's own
    strategy outputs, see strategy.decode)."""
    actions = np.atleast_2d(np.asarray(actions, dtype=float))
    _, _, _, call, plan = decode(np.concatenate([np.zeros((len(actions), 2)), actions], axis=1), len(actions))
    return call, plan
