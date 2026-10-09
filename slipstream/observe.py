"""What each driver perceives, as one flat vector of numbers roughly in [-1, 1].

Everything is in the car's own frame (x forward, y to its left), so a corner looks the same wherever it is on
the map. The names in OBSERVATION_NAMES line up with the vector, for anything that wants to show what a
driver is looking at.

New inputs only ever go at the end. A network trained before they existed reads the vector cut down to its own
input size (see `for_network`), so older runs can still be watched.
"""
import numpy as np

from . import pit
from .car import COMPOUNDS, FUEL_CAPACITY, SLICK_COUNT, centred
from .personality import PERSONALITY_TRAITS, neutral
from .track import MIN_RADIUS, SPACING

CURVATURE_AHEAD = (6, 12, 20, 30, 42, 56, 72, 90, 110, 135, 160)  # metres along the centre line
LINE_AHEAD = (12, 30, 56, 90, 135)        # centre-line points, for where the road actually goes
NEARBY = 4                                # closest other cars it tracks
NEARBY_RANGE = 60.0                       # metres
SCALE = {'speed': 80.0, 'sideways': 10.0, 'line': 100.0, 'relative_speed': 20.0, 'gap': 100.0}
SPEC_SCALE = {'top_speed': 80.0, 'acceleration': 12.0, 'braking': 30.0, 'grip': 2.0}

# The weather came last: which wet-weather tyres a car is on, how wet the track is and how hard it's raining.
WEATHER_NAMES = [f'compound.{name}' for name in COMPOUNDS[SLICK_COUNT:]] + ['wetness', 'rain']

OBSERVATION_NAMES = (
    ['speed', 'sideways', 'lateral', 'heading_sin', 'heading_cos', 'tyre_wear', 'fuel', 'on_track', 'sliding']
    + [f'curvature.{distance}' for distance in CURVATURE_AHEAD]
    + [f'line.{distance}.{axis}' for distance in LINE_AHEAD for axis in ('x', 'y')]
    + [f'car.{slot}.{feature}' for slot in range(NEARBY) for feature in ('present', 'x', 'y', 'vx', 'vy')]
    + ['position', 'remaining', 'gap_ahead', 'gap_behind']
    + [f'spec.{name}' for name in SPEC_SCALE]
    + [f'personality.{name}' for name in PERSONALITY_TRAITS]
    + [f'compound.{name}' for name in COMPOUNDS[:SLICK_COUNT]]
    + ['wing', 'engine', 'damage', 'brake_wear', 'pit_called', 'in_pit_lane', 'fuel_margin', 'laps_left', 'pit_entry_ahead']
    + ['pit_decision_now']
    + WEATHER_NAMES
)
LAPS_SCALE = 10.0
OBSERVATION_SIZE = len(OBSERVATION_NAMES)


def to_car_frame(vectors, heading):
    """Rotates world vectors (..., 2) into frames whose x axis points along `heading` (broadcast over cars)."""
    cos, sin = np.cos(heading), np.sin(heading)
    return np.stack([vectors[..., 0] * cos + vectors[..., 1] * sin, -vectors[..., 0] * sin + vectors[..., 1] * cos], axis=-1)


def observe(race, personality=None):
    """`personality` is one row of traits per car (see personality.py); neutral when not given."""
    track, cars, specs = race.track, race.cars, race.specs
    count = race.count
    heading = cars.heading
    distance = race.progress % track.length
    index = race.track_index

    own_velocity = to_car_frame(cars.velocity, heading)
    error = heading - np.arctan2(track.tangents[index, 1], track.tangents[index, 0])
    own = np.stack([
        own_velocity[:, 0] / SCALE['speed'], own_velocity[:, 1] / SCALE['sideways'], race.lateral / (track.width / 2),
        np.sin(error), np.cos(error), cars.tyre_wear, cars.fuel / FUEL_CAPACITY,
        race.on_track.astype(float), cars.sliding.astype(float),
    ], axis=1)

    samples = (np.floor((distance[:, None] + np.array(CURVATURE_AHEAD)[None, :]) / SPACING).astype(int)) % track.size
    curvature = np.clip(track.curvature[samples] * MIN_RADIUS, -1, 1)
    line_points = track.point_at(distance[:, None] + np.array(LINE_AHEAD)[None, :])
    line = to_car_frame(line_points - cars.position[:, None, :], heading[:, None]) / SCALE['line']

    # Ghost cars (see Race) race alone: nobody else is out there, and every one of them is leading.
    nearby = np.zeros((count, NEARBY * 5)) if race.ghosts else _nearby_cars(cars, heading, count)

    standings = race.standings()
    place = np.empty(count, dtype=int)
    place[standings] = np.arange(count)
    ordered = race.progress[standings]
    gap_ahead = np.ones(count)
    gap_behind = np.ones(count)
    gap_ahead[standings[1:]] = np.clip((ordered[:-1] - ordered[1:]) / SCALE['gap'], 0, 1)
    gap_behind[standings[:-1]] = np.clip((ordered[:-1] - ordered[1:]) / SCALE['gap'], 0, 1)
    total = race.laps * track.length
    if race.ghosts:
        place, gap_ahead, gap_behind = np.zeros(count), np.ones(count), np.ones(count)
    situation = np.stack([place / max(count - 1, 1), np.clip((total - race.progress) / total, 0, 1), gap_ahead, gap_behind], axis=1)

    car_specs = np.stack([getattr(specs, name) / scale for name, scale in SPEC_SCALE.items()], axis=1)
    traits = (neutral(count) if personality is None else np.asarray(personality, dtype=float))[:, :len(PERSONALITY_TRAITS)] * 2 - 1
    return np.concatenate([own, curvature, line.reshape(count, -1), nearby, situation, car_specs, traits, _strategy(race), weather_inputs(race)],
                          axis=1).astype(np.float32)


def weather_inputs(race):
    """The inputs named in WEATHER_NAMES, one row per car."""
    wet_tyres = np.eye(len(COMPOUNDS))[race.cars.compound][:, SLICK_COUNT:]
    conditions = np.tile([race.weather.wetness, race.weather.rain], (race.count, 1))
    return np.concatenate([wet_tyres, conditions], axis=1)


def _strategy(race):
    """The car's setup and condition, and what it needs to know to decide on a stop."""
    from .strategy import fuel_to_finish, laps_left
    cars = race.cars
    compound = np.eye(len(COMPOUNDS))[cars.compound][:, :SLICK_COUNT]
    margin = np.clip((cars.fuel - fuel_to_finish(race)) / FUEL_CAPACITY, -1, 1)
    entry_ahead = ((race.pit.entry - race.progress % race.track.length) % race.track.length) / race.track.length
    return np.concatenate([compound, np.stack([
        centred(cars.wing), centred(cars.engine), cars.damage, cars.brake_wear,
        (race.pit_state == pit.CALLED).astype(float), race.in_lane.astype(float),
        margin, np.minimum(laps_left(race) / LAPS_SCALE, 1.0), entry_ahead,
        # 1 when this decision is the one that commits the car to a stop or not this lap.
        (race.pit.in_window(race.progress % race.track.length) & ~race.pit_decided & (race.pit_state == pit.RACING)).astype(float),
    ], axis=1)], axis=1)


def for_network(observations, model):
    """The observations cut down to the inputs `model` was trained with (older runs know fewer)."""
    return observations[:, :model.observation_space.shape[0]]


def _nearby_cars(cars, heading, count):
    """The NEARBY closest other cars within range: present flag, position and velocity relative to this car,
    nearest first. Empty slots are all zeros."""
    offsets = cars.position[None, :, :] - cars.position[:, None, :]
    distance = np.linalg.norm(offsets, axis=2)
    np.fill_diagonal(distance, np.inf)
    distance[distance > NEARBY_RANGE] = np.inf
    order = np.argsort(distance, axis=1)[:, :NEARBY]
    rows = np.arange(count)[:, None]
    present = np.isfinite(distance[rows, order])
    relative_position = to_car_frame(offsets[rows, order], heading[:, None]) / NEARBY_RANGE
    relative_velocity = to_car_frame(cars.velocity[order] - cars.velocity[:, None, :], heading[:, None]) / SCALE['relative_speed']
    features = np.concatenate([present[..., None], relative_position, relative_velocity], axis=2) * present[..., None]
    if features.shape[1] < NEARBY:
        features = np.concatenate([features, np.zeros((count, NEARBY - features.shape[1], 5))], axis=1)
    return features.reshape(count, -1)
