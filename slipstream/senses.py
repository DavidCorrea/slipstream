"""What a driver perceives: the inputs of the drivers trained to *drive* rather than to read numbers.

A real driver isn't told the car's top speed, how much grip the tyres have left, or how wet the track is. They feel
the car (how it slides, how hard it turns, brakes and accelerates, whether the wheels spin), see the road and the
cars they can see, and get told a few things over the radio and on the dash: which tyres are on, how much fuel is
left, the engine mode, the laps to go, and whether the team wants them in. Everything else they have to work out
as they go, which is why these drivers have a memory (see driver_env.py).

What a driver can see is limited, too. Mirrors cover the car behind but leave a blind spot to each side, where a
car alongside can only be sensed once it's right there. Spray off a wet track, and the smoke of a damaged car, hide
the cars further up the road.
"""
import numpy as np

from . import pit
from .car import COMPOUNDS, FUEL_CAPACITY, centred
from .observe import CURVATURE_AHEAD, LINE_AHEAD, SCALE, to_car_frame
from .personality import TRAITS, neutral
from .strategy import laps_left
from .track import MIN_RADIUS, SPACING

NEARBY = 4
LOOKING_AHEAD = 70.0          # metres a driver can see up the road in clear air
RAIN_VISIBILITY = 0.5         # share of that lost in the heaviest rain
MIRRORS = {'range': 50.0, 'half_angle': np.radians(35)}   # mirrors see straight back, this wide
ALONGSIDE = {'ahead': 3.5, 'across': 4.0}                 # a car this close is felt, wherever it is
SPRAY = {'clear': 8.0, 'thick': 25.0, 'hides': 0.5}       # spray and smoke start to hide a car beyond `clear` m, fully by `clear + thick`
SMOKE_FROM_DAMAGE = 0.35
FEEL_SCALE = {'yaw_rate': 2.0, 'g': 3.0}
LAPS_SCALE = 10.0

SENSE_NAMES = (
    ['speed', 'sideways', 'yaw_rate', 'longitudinal_g', 'lateral_g', 'wheelspin', 'sliding', 'on_track',
     'lateral', 'heading_sin', 'heading_cos']
    + [f'curvature.{distance}' for distance in CURVATURE_AHEAD]
    + [f'line.{distance}.{axis}' for distance in LINE_AHEAD for axis in ('x', 'y')]
    + [f'car.{slot}.{feature}' for slot in range(NEARBY) for feature in ('present', 'x', 'y', 'vx', 'vy')]
    + ['position', 'remaining', 'gap_ahead', 'gap_behind']
    + [f'personality.{name}' for name in TRAITS]
    + [f'compound.{name}' for name in COMPOUNDS]
    + ['fuel', 'wing', 'engine', 'pit_called', 'in_pit_lane', 'laps_left']
)


def visible(race):
    """[looking, seen]: whether each driver can see (or sense) each other car right now."""
    cars = race.cars
    count = race.count
    offset = to_car_frame(cars.position[None, :, :] - cars.position[:, None, :], cars.heading[:, None])
    ahead, across = offset[..., 0], offset[..., 1]
    distance = np.hypot(ahead, across)
    alongside = (np.abs(ahead) < ALONGSIDE['ahead']) & (np.abs(across) < ALONGSIDE['across'])
    in_view = (ahead > 0) & (distance < LOOKING_AHEAD * (1 - RAIN_VISIBILITY * race.weather.rain))
    angle_from_behind = np.abs(np.arctan2(across, -ahead))
    in_mirrors = (ahead < 0) & (angle_from_behind < MIRRORS['half_angle']) & (distance < MIRRORS['range'])
    # A car ahead disappears into its own spray (wet track, at speed) or smoke (badly damaged) at a distance.
    speed = np.maximum(cars.speed, 0.0)
    spray = race.weather.wetness * np.minimum(speed / 50.0, 1.0)
    smoke = np.where(np.maximum(cars.damage, cars.engine_failed) > SMOKE_FROM_DAMAGE, 1.0, 0.0)
    cloud = np.maximum(spray, smoke)[None, :]
    lost = (ahead > 0) & (cloud * np.clip((distance - SPRAY['clear']) / SPRAY['thick'], 0, 1) > SPRAY['hides'])
    seen = alongside | (in_view & ~lost) | in_mirrors
    return seen & ~np.eye(count, dtype=bool)


def senses(race, personality=None):
    """Every driver's senses, one row per car, in the order of SENSE_NAMES."""
    track, cars = race.track, race.cars
    count = race.count
    heading = cars.heading
    distance = race.progress % track.length
    own_velocity = to_car_frame(cars.velocity, heading)
    error = heading - np.arctan2(track.tangents[race.track_index, 1], track.tangents[race.track_index, 0])
    feel = np.stack([
        own_velocity[:, 0] / SCALE['speed'], own_velocity[:, 1] / SCALE['sideways'],
        np.clip(cars.yaw_rate / FEEL_SCALE['yaw_rate'], -1, 1), np.clip(cars.longitudinal_g / FEEL_SCALE['g'], -1, 1),
        np.clip(cars.lateral_g / FEEL_SCALE['g'], -1, 1), cars.wheelspin, cars.sliding.astype(float),
        race.on_track.astype(float), np.clip(race.lateral / (track.width / 2), -3, 3), np.sin(error), np.cos(error),
    ], axis=1)

    samples = (np.floor((distance[:, None] + np.array(CURVATURE_AHEAD)[None, :]) / SPACING).astype(int)) % track.size
    curvature = np.clip(track.curvature[samples] * MIN_RADIUS, -1, 1)
    line_points = track.point_at(distance[:, None] + np.array(LINE_AHEAD)[None, :])
    line = to_car_frame(line_points - cars.position[:, None, :], heading[:, None]) / SCALE['line']

    nearby = np.zeros((count, NEARBY, 5)) if race.ghosts else _seen_cars(race, visible(race))
    situation = _situation(race)
    traits = (neutral(count) if personality is None else np.asarray(personality, dtype=float)) * 2 - 1
    told = np.concatenate([np.eye(len(COMPOUNDS))[cars.compound], np.stack([
        cars.fuel / FUEL_CAPACITY, centred(cars.wing), centred(cars.engine),
        (race.pit_state == pit.CALLED).astype(float), race.in_lane.astype(float), np.minimum(laps_left(race) / LAPS_SCALE, 1.0),
    ], axis=1)], axis=1)
    return np.concatenate([feel, curvature, line.reshape(count, -1), nearby.reshape(count, -1), situation, traits, told],
                          axis=1).astype(np.float32)


def _seen_cars(race, seen):
    """The NEARBY closest cars each driver can see: present, where (in its own frame) and how fast they're closing."""
    cars = race.cars
    count = race.count
    offset = to_car_frame(cars.position[None, :, :] - cars.position[:, None, :], cars.heading[:, None])
    relative = to_car_frame(cars.velocity[None, :, :] - cars.velocity[:, None, :], cars.heading[:, None])
    distance = np.where(seen, np.hypot(offset[..., 0], offset[..., 1]), np.inf)
    closest = np.argsort(distance, axis=1)[:, :NEARBY]
    result = np.zeros((count, NEARBY, 5))
    for slot in range(min(NEARBY, count - 1)):
        other = closest[:, slot]
        present = np.isfinite(distance[np.arange(count), other])
        result[:, slot, 0] = present
        result[:, slot, 1:3] = np.where(present[:, None], offset[np.arange(count), other] / SCALE['gap'], 0.0)
        result[:, slot, 3:5] = np.where(present[:, None], relative[np.arange(count), other] / SCALE['relative_speed'], 0.0)
    return result


def _situation(race):
    """What the pit board says: place, how much race is left, and the gaps to the cars ahead and behind."""
    count = race.count
    if race.ghosts:
        return np.stack([np.zeros(count), _remaining(race), np.ones(count), np.ones(count)], axis=1)
    standings = race.standings()
    place = np.empty(count)
    place[standings] = np.arange(count)
    ordered = race.progress[standings]
    gap_ahead, gap_behind = np.ones(count), np.ones(count)
    gap_ahead[standings[1:]] = np.clip((ordered[:-1] - ordered[1:]) / SCALE['gap'], 0, 1)
    gap_behind[standings[:-1]] = np.clip((ordered[:-1] - ordered[1:]) / SCALE['gap'], 0, 1)
    return np.stack([place / max(count - 1, 1), _remaining(race), gap_ahead, gap_behind], axis=1)


def _remaining(race):
    total = race.laps * race.track.length
    return np.clip((total - race.progress) / total, 0, 1)
