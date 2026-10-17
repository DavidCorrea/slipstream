"""What a driver perceives: the inputs of the drivers trained to *drive* rather than to read numbers.

A real driver isn't told the car's top speed, how much grip the tyres have left, or how wet the track is. They feel
the car (how it slides, how hard it turns, brakes and accelerates, whether the wheels spin), see the road and the
cars they can see, and get told a few things over the radio and on the dash: which tyres are on, how much fuel is
left, the engine mode, the laps to go, and whether the team wants them in. Everything else they have to work out
as they go, which is why these drivers have a memory (see driver_env.py).

What a driver can see is limited, too. Mirrors cover the car behind but leave a blind spot to each side, where a
car alongside can only be sensed once it's right there. Spray off a wet track, and the smoke of a damaged car, hide
the cars further up the road.

Drivers that race (RACER_SENSE_NAMES, from the second generation of drivers by feel) know everything a racing
driver knows on top: what the dash and the engineer say about the car (tyre wear and temperature, brakes,
engine, damage, a puncture, the gear, the limiter), the rain now and on the radar, where the track ahead is
wetter or drier, the debris they can see, where the pit lane and their box are, the lights and the marshals'
flags, their penalties and warnings, and what the pit wall says on the radio.
"""
import numpy as np

from . import pit
from .car import COMPOUNDS, FUEL_CAPACITY, centred
from .flags import LIGHTS
from .observe import CURVATURE_AHEAD, LINE_AHEAD, SCALE, to_car_frame
from .personality import TRAITS, neutral
from .strategy import RADIO_CALLS, laps_left
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
DEBRIS_SEEN = 2               # pieces of debris a driver keeps an eye on
LOOKING_OUT = 300.0           # metres ahead the radar, the pit board and the marshals are read up to
RADAR = 300.0                 # seconds ahead the engineer can see rain coming
WET_AHEAD = 40.0              # metres ahead a driver reads the track's wet and dry patches
RACER_SENSE_NAMES = SENSE_NAMES + (
    ['tyre_wear', 'tyre_temp', 'brake_wear', 'brake_temp', 'engine_temp', 'damage', 'wing_damage', 'suspension_damage',
     'punctured', 'reversing', 'limiter', 'wetness_here', 'rain', 'rain_in', 'rain_coming']
    + [f'wetness_ahead.{side}' for side in ('left', 'middle', 'right')]
    + [f'debris.{slot}.{feature}' for slot in range(DEBRIS_SEEN) for feature in ('present', 'x', 'y')]
    + ['pit_entry_in', 'in_box', 'speed_zone', 'box_along', 'box_across', 'lights', 'started', 'yellow_ahead',
       'in_yellow', 'blue_flag', 'penalties', 'strikes']
    + [f'radio.{call}' for call in RADIO_CALLS[1:]]
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


def racer_senses(race, personality=None):
    """Every racing driver's senses, one row per car, in the order of RACER_SENSE_NAMES: everything a driver by feel
    senses, then everything a racing driver knows on top (see the module notes)."""
    cars, track, count = race.cars, race.track, race.count
    distance = race.progress % track.length
    half = track.width / 2
    surface, weather = race.surface, race.weather
    rain_in, rain_coming = _rain_on_the_radar(weather)
    across = np.array([-half / 2, 0.0, half / 2])
    ahead = np.repeat((distance + WET_AHEAD)[:, None], 3, axis=1)
    car = np.stack([
        cars.tyre_wear, cars.tyre_temp, cars.brake_wear, cars.brake_temp, cars.engine_temp, cars.damage,
        cars.wing_damage, cars.suspension_damage, cars.punctured.astype(float), cars.reversing.astype(float),
        race.limiter.astype(float), surface.wetness(distance, race.lateral, weather.wetness), np.full(count, weather.rain),
        np.full(count, rain_in), np.full(count, rain_coming),
    ], axis=1)
    wet_ahead = surface.wetness(ahead.ravel(), np.tile(across, count), weather.wetness).reshape(count, 3)
    return np.concatenate([senses(race, personality), car, wet_ahead, _debris_ahead(race), _pit_and_flags(race, distance),
                           np.eye(len(RADIO_CALLS))[race.radio][:, 1:]], axis=1).astype(np.float32)


def _rain_on_the_radar(weather):
    """How soon the next rain arrives (0 raining now, 1 none within RADAR seconds) and how hard it will fall."""
    if weather.rain > 0:
        return 0.0, float(weather.rain)
    coming = [(start - weather.time, intensity) for start, _, intensity in weather.showers if 0 <= start - weather.time < RADAR]
    if not coming:
        return 1.0, 0.0
    seconds, intensity = min(coming)
    return seconds / RADAR, float(intensity)


def _debris_ahead(race):
    """The DEBRIS_SEEN nearest pieces of debris ahead of each driver within sight: present, and where in its own
    frame."""
    cars, count = race.cars, race.count
    result = np.zeros((count, DEBRIS_SEEN, 3))
    if not len(race.debris):
        return result.reshape(count, -1)
    offset = to_car_frame(race.debris[None, :, :] - cars.position[:, None, :], cars.heading[:, None])
    reach = np.where((offset[..., 0] > 0), np.hypot(offset[..., 0], offset[..., 1]), np.inf)
    reach = np.where(reach < LOOKING_AHEAD, reach, np.inf)
    nearest = np.argsort(reach, axis=1)[:, :DEBRIS_SEEN]
    for slot in range(min(DEBRIS_SEEN, reach.shape[1])):
        piece = nearest[:, slot]
        present = np.isfinite(reach[np.arange(count), piece])
        result[:, slot, 0] = present
        result[:, slot, 1:] = np.where(present[:, None], offset[np.arange(count), piece] / SCALE['gap'], 0.0)
    return result.reshape(count, -1)


def _pit_and_flags(race, distance):
    """Where the pit lane and each car's box are, the start lights, the flags, and the stewards' tally."""
    lane, count = race.pit, race.count
    along = lane.along(distance)
    entry_in = np.clip((-along % race.track.length) / LOOKING_OUT, 0, 1)
    speed_zone = race.in_lane & (along >= pit.RAMP) & (along <= lane.length - pit.RAMP)
    box_along = np.clip((lane.boxes[:count] - along) / LOOKING_OUT, -1, 1)
    # How far across to the box: in the lane, and for a called car on its way to the entry too. The lane is on
    # either side of the track, and a car told this only once in it never learned which way to turn in: called, it
    # slowed for the entry and drove past it on the far side.
    approaching = (race.pit_state == pit.CALLED) & ~race.in_lane & ((-along % race.track.length) <= LOOKING_OUT)
    box_across = np.where(race.in_lane | approaching,
                          np.clip(lane.side * (lane.working_lane - race.lateral) / (2 * pit.LANES_APART), -1, 1), 0.0)
    lights = np.zeros(count)
    started = np.ones(count)
    yellow_ahead, in_yellow, blue = np.ones(count), np.zeros(count), np.zeros(count)
    marshals = race.marshals
    if marshals is not None:
        lights[:] = marshals.lights(race.time) / LIGHTS['count']
        started[:] = float(marshals.started(race.time))
        in_yellow = marshals.under_yellow(distance, race.track.length).astype(float)
        for start, _, _ in marshals.zones:
            yellow_ahead = np.minimum(yellow_ahead, ((start - distance) % race.track.length) / LOOKING_OUT)
        blue = marshals.blue.astype(float)
    penalties, strikes = np.zeros(count), np.zeros(count)
    if race.stewards is not None:
        penalties = np.clip(race.stewards.seconds / 20.0, 0, 1)
        strikes = np.clip(race.stewards.strikes / 4.0, 0, 1)
    return np.stack([entry_in, (race.pit_state == pit.STOPPED).astype(float), speed_zone.astype(float), box_along, box_across,
                     lights, started, np.clip(yellow_ahead, 0, 1), in_yellow, blue, penalties, strikes], axis=1)
