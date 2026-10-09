"""Car physics for every car at once.

A car is a point with a heading and a velocity. Each tick the tyres share one grip budget between speeding up
or slowing down and turning (the friction circle): brake hard and there's less left for the corner. Asking for
more turn than the grip allows makes the car run wide (understeer) and scrubs its tyres.

What a car can do also depends on its setup and condition, all of which a pit stop can change:
- tyre compound: softs grip more and wear faster, hards grip less and last; every compound loses grip as it
  wears, slowly at first and then sharply. On a wet track slicks lose grip fast; intermediates are best when it's
  damp and full wets when it's soaked, but both overheat and wear quickly on a dry track.
- wing angle: more wing, more cornering grip and a lower top speed.
- engine mode: more power and a little more top speed, for more fuel.
- damage from contact: less grip, top speed and acceleration until repaired.
- brake wear: worn pads stop the car less hard.
- fuel: a full tank is heavy, slowing both speeding up and stopping; an empty one stops the car.

And what a race does to it:
- temperatures: tyres grip best inside their compound's window, cold out of the pits and overheating when slid
  or worked too hard (rain cools them); brakes fade when hot; engines run hotter pushing and in another car's dirty
  air, and a hot engine can fail.
- handling: a car's balance makes it understeer (run wide) or oversteer (rotate past its grip and slide, maybe spin)
  at the limit. A broken front wing pushes it toward understeer.
- local damage: a broken front wing costs front grip; bent suspension pulls the car to one side; a puncture
  leaves a car slow and sliding until new tyres go on.
- the world (`Conditions`): how wet the track is under each car, the slipstream of a car ahead (less drag, a higher
  top speed), its dirty air (less grip in corners, less cooling), and the surface (rubber on the racing line grips
  more, marbles off it less).
"""
from dataclasses import dataclass, field

import numpy as np

DT = 0.05               # seconds per physics tick
GRAVITY = 9.81
WHEELBASE = 3.0
MAX_STEER = 0.32        # radians at walking pace; less at speed, like a real steering ratio
STEER_FADE_SPEED = 35.0 # m/s at which the available steering lock has halved
EMPTY_MASS = 750.0      # kg, car and driver
FUEL_CAPACITY = 60.0    # kg: about five laps at racing pace, so long races need a stop or a lot of saving
ROLLING_DRAG = 0.4      # m/s^2
OFF_TRACK = {'grip': 0.55, 'drag': 5.0}
# Tyre wear per second at full cornering load, and extra while sliding: mediums go off after four or five laps,
# so one stop clearly pays from about seven laps on (at a gentler rate it only broke even around nine).
WEAR = {'load': 0.006, 'slide': 0.016}
FUEL_BURN = 0.42        # kg per second at full throttle in the standard engine mode
SLIDE_THRESHOLD = 1.0   # m/s of sideways speed that counts as a slide (tyre smoke, skid marks)

# Wet-weather tyres come after the slicks, so the slicks keep the indices older networks and runs know them by.
COMPOUNDS = ('soft', 'medium', 'hard', 'intermediate', 'wet')
SLICK_COUNT = 3
# Grip of each compound on a dry, damp and soaked track. Intermediates overtake slicks at about a third wet, and
# full wets overtake intermediates at about 70%.
WETNESS_POINTS = np.array([0.0, 0.5, 1.0])
COMPOUND_GRIP = np.array([
    [1.06, 0.78, 0.55],
    [1.00, 0.75, 0.53],
    [0.95, 0.72, 0.50],
    [0.90, 0.88, 0.74],
    [0.84, 0.82, 0.83],
])
# Where the grips above cross: wetter than this, intermediates beat slicks, and full wets beat intermediates.
INTERMEDIATE_FROM, WET_FROM = 0.32, 0.7
COMPOUND_WEAR = np.array([1.8, 1.0, 0.55, 1.1, 0.9])
# A tread made to clear water overheats once the track is drier than it needs: below `from` wetness, wear grows
# to `extra` times more on a bone-dry track. Water cools every tyre, so wear also falls as the track gets wetter.
OVERHEAT_FROM = np.array([0.0, 0.0, 0.0, 0.25, 0.55])
OVERHEAT_EXTRA = np.array([0.0, 0.0, 0.0, 3.0, 6.0])
WATER_COOLING = 0.5
MEDIUM = 1
WING = {'grip': 0.08, 'top_speed': 0.07}              # ± at full and no wing, around the middle setting
ENGINE = {'power': 0.08, 'top_speed': 0.03, 'fuel': (0.75, 1.35)}   # ± around standard; fuel use from lean to push
# Damage costs grip, top speed and acceleration. Only the part of an impact above `gentle` m/s does damage, so
# cars rubbing wheels in a pack don't wreck each other.
DAMAGE = {'grip': 0.15, 'top_speed': 0.10, 'acceleration': 0.10, 'per_impulse': 0.01, 'gentle': 1.0}
BRAKE_WEAR = {'per_second': 0.006, 'loss': 0.4}       # pad wear per second of full braking; braking lost when gone

# Temperatures run from 0 to 1 (the viewer shows them in degrees). Each compound works best in a window around its
# own temperature: wet-weather tyres run much cooler than slicks.
TYRE_WINDOW = np.array([0.55, 0.6, 0.65, 0.42, 0.32])
TYRE_WINDOW_HALF = 0.06        # either side of the centre, full grip
TYRE_TEMPERATURE = {
    'grip_loss': 0.2,          # grip lost 0.25 outside the window (and no more beyond)
    'load': 0.08, 'slide': 0.04, 'brake': 0.02,    # heating per second: cornering, sliding, braking (each × speed/60)
    'cooling': 0.05, 'ambient': 0.25,              # cooling per second toward the air's temperature
    'wet_cooling': 2.0, 'wet_ambient': 0.13,       # a soaked track cools three times as fast, toward a colder air
    'overheat_wear': 3.0,      # extra wear per 0.2 above the window
}
TYRE_BLANKETS = 0.45           # new tyres come out of their blankets warm, but not yet at their best
BRAKE_TEMPERATURE = {'heating': 0.75, 'cooling': 0.15, 'ambient': 0.1, 'fade_from': 0.85, 'fade': 0.5}
ENGINE_TEMPERATURE = {'heating': 0.010, 'push': 0.8, 'cooling': 0.03, 'ambient': 0.3, 'dirty_air': 0.5, 'start': 0.4}
SLIPSTREAM_TOP_SPEED = 0.07    # extra top speed right in another car's slipstream
DIRTY_AIR_GRIP = 0.1           # cornering grip lost right behind another car
# Handling: at balance +1 a car rotates up to OVERSTEER beyond what its grip allows (and slides); at -1 it gives up
# turning UNDERSTEER sooner. A broken front wing shifts the balance toward understeer by WING_DAMAGE.
HANDLING = {'oversteer': 0.6, 'understeer': 0.3, 'wing_damage': 0.8}
PART_DAMAGE = {'wing_grip': 0.15, 'suspension_grip': 0.1, 'suspension_pull': 0.12}
PUNCTURE = {'grip': 0.75, 'top_speed': 0.85, 'drag': 1.5}   # slow, but a car can limp back to the pits
# The day: how far the air's temperature (0 cold to 1 hot, 0.5 mild) moves the tyres' and engine's resting
# temperature, and the top speed lost per m/s of headwind (or gained per m/s of tailwind).
AIR_TEMPERATURE = {'tyre': 0.12, 'engine': 0.1}
WIND_TOP_SPEED = 0.006


@dataclass
class CarSpecs:
    """What each car can do, one entry per car. Defaults are a mid-field car."""
    top_speed: np.ndarray      # m/s
    acceleration: np.ndarray   # m/s^2 from rest with an empty tank
    braking: np.ndarray        # m/s^2 with an empty tank
    grip: np.ndarray           # cornering grip in g on fresh tyres
    balance: np.ndarray = None # handling, -1 understeer to +1 oversteer; neutral when not given

    def __post_init__(self):
        if self.balance is None:
            self.balance = np.zeros(len(self.top_speed))

    @classmethod
    def uniform(cls, count, top_speed=72.0, acceleration=9.0, braking=22.0, grip=1.7, balance=0.0):
        return cls(*(np.full(count, value, dtype=float) for value in (top_speed, acceleration, braking, grip, balance)))


@dataclass
class Conditions:
    """What the world is doing to each car this tick (see the module docstring)."""
    wetness: np.ndarray        # 0 dry to 1 soaked, under this car
    draft: np.ndarray          # 0 clear air to 1 right in a slipstream
    dirty_air: np.ndarray      # 0 clear air to 1 right behind another car
    surface_grip: np.ndarray   # 1 for plain tarmac; more on rubber, less on marbles
    air_temperature: np.ndarray = None   # 0 cold to 1 hot; mild when not given
    headwind: np.ndarray = None          # m/s of wind against the car (negative is a tailwind); still when not given

    def __post_init__(self):
        count = len(self.wetness)
        if self.air_temperature is None:
            self.air_temperature = np.full(count, 0.5)
        if self.headwind is None:
            self.headwind = np.zeros(count)

    @classmethod
    def uniform(cls, count, wetness=0.0, draft=0.0, dirty_air=0.0, surface_grip=1.0, air_temperature=0.5, headwind=0.0):
        return cls(*(np.full(count, value, dtype=float) for value in (wetness, draft, dirty_air, surface_grip, air_temperature, headwind)))


@dataclass
class CarState:
    position: np.ndarray   # (K, 2)
    heading: np.ndarray    # (K,) radians
    velocity: np.ndarray   # (K, 2)
    tyre_wear: np.ndarray  # (K,) 0 fresh, 1 gone
    fuel: np.ndarray       # (K,) kg
    compound: np.ndarray = field(default=None)     # (K,) index into COMPOUNDS
    wing: np.ndarray = field(default=None)         # (K,) 0 no wing to 1 full wing
    engine: np.ndarray = field(default=None)       # (K,) 0 lean to 1 push
    damage: np.ndarray = field(default=None)       # (K,) 0 none to 1 wrecked
    brake_wear: np.ndarray = field(default=None)   # (K,) 0 new pads to 1 gone
    sliding: np.ndarray = field(default=None)      # (K,) bool, set each tick
    lateral_load: np.ndarray = field(default=None) # (K,) share of grip used for cornering this tick
    tyre_temp: np.ndarray = field(default=None)    # (K,) 0-1, see TYRE_WINDOW
    brake_temp: np.ndarray = field(default=None)   # (K,) 0-1
    engine_temp: np.ndarray = field(default=None)  # (K,) 0-1
    engine_failed: np.ndarray = field(default=None)       # (K,) bool
    punctured: np.ndarray = field(default=None)           # (K,) bool
    wing_damage: np.ndarray = field(default=None)         # (K,) 0-1, the front wing
    suspension_damage: np.ndarray = field(default=None)   # (K,) 0-1
    pull_side: np.ndarray = field(default=None)           # (K,) which way bent suspension pulls, +1 left or -1 right
    # What the car did this tick, for a driver to feel: turning rate (rad/s), acceleration along and across it
    # (m/s^2), and how much drive the tyres couldn't put down (0-1).
    yaw_rate: np.ndarray = field(default=None)
    longitudinal_g: np.ndarray = field(default=None)
    lateral_g: np.ndarray = field(default=None)
    wheelspin: np.ndarray = field(default=None)

    @classmethod
    def at(cls, positions, headings, fuel=FUEL_CAPACITY):
        count = len(positions)
        return cls(position=np.array(positions, dtype=float), heading=np.array(headings, dtype=float),
                   velocity=np.zeros((count, 2)), tyre_wear=np.zeros(count), fuel=np.full(count, fuel, dtype=float),
                   compound=np.full(count, MEDIUM), wing=np.full(count, 0.5), engine=np.full(count, 0.5),
                   damage=np.zeros(count), brake_wear=np.zeros(count),
                   tyre_temp=np.full(count, TYRE_BLANKETS), brake_temp=np.full(count, BRAKE_TEMPERATURE['ambient']),
                   engine_temp=np.full(count, ENGINE_TEMPERATURE['start']), engine_failed=np.zeros(count, dtype=bool),
                   punctured=np.zeros(count, dtype=bool), wing_damage=np.zeros(count), suspension_damage=np.zeros(count),
                   pull_side=np.where(np.arange(count) % 2 == 0, 1.0, -1.0), yaw_rate=np.zeros(count),
                   longitudinal_g=np.zeros(count), lateral_g=np.zeros(count), wheelspin=np.zeros(count),
                   sliding=np.zeros(count, dtype=bool), lateral_load=np.zeros(count))

    @property
    def forward(self):
        return np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)

    @property
    def left(self):
        return np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)

    @property
    def speed(self):
        return np.einsum('ck,ck->c', self.velocity, self.forward)


def tyre_grip_factor(wear):
    """Worn tyres keep most of their grip until about half worn, then fall away."""
    return 1.0 - 0.45 * np.clip(wear, 0, 1) ** 2


def centred(value):
    """A 0-1 setting as -1 to 1 around its middle."""
    return 2 * np.asarray(value, dtype=float) - 1


def compound_grip(compound, wetness):
    """How much of a car's grip each compound gives on a track this wet (0 dry to 1 soaked); `wetness` is one value
    for every car or one per car."""
    compound = np.asarray(compound)
    wetness = np.broadcast_to(np.asarray(wetness, dtype=float), compound.shape)
    dry, damp, soaked = (COMPOUND_GRIP[compound, column] for column in range(3))
    halfway = WETNESS_POINTS[1]
    return np.where(wetness <= halfway, dry + (damp - dry) * wetness / halfway,
                    damp + (soaked - damp) * (wetness - halfway) / (1 - halfway))


def effective_grip(state: CarState, specs: CarSpecs, wetness, surface_grip=1.0, dirty_air=0.0):
    """Cornering grip in g right now on tarmac: the car's grip shaped by compound and how wet the track is, tyre
    wear and temperature, wing, damage, a puncture, the surface and any dirty air."""
    parts = (1 - PART_DAMAGE['wing_grip'] * state.wing_damage) * (1 - PART_DAMAGE['suspension_grip'] * state.suspension_damage)
    return (specs.grip * compound_grip(state.compound, wetness) * tyre_grip_factor(state.tyre_wear)
            * tyre_temperature_grip(state.tyre_temp, state.compound)
            * (1 + WING['grip'] * centred(state.wing)) * (1 - DAMAGE['grip'] * state.damage) * parts
            * np.where(state.punctured, PUNCTURE['grip'], 1.0) * surface_grip * (1 - DIRTY_AIR_GRIP * dirty_air))


def tyre_temperature_grip(temperature, compound):
    """Share of their grip tyres this hot give: all of it inside their window, less the further out they are."""
    outside = np.maximum(np.abs(temperature - TYRE_WINDOW[compound]) - TYRE_WINDOW_HALF, 0.0)
    return 1 - TYRE_TEMPERATURE['grip_loss'] * np.minimum(outside / 0.25, 1.0)


def brake_fade(temperature):
    """Share of their stopping power brakes this hot still have."""
    over = np.clip((temperature - BRAKE_TEMPERATURE['fade_from']) / (1 - BRAKE_TEMPERATURE['fade_from']), 0.0, 1.0)
    return 1 - BRAKE_TEMPERATURE['fade'] * over


def tyre_temperature_wear(compound, wetness):
    """How much faster than usual each tyre wears on a track this wet: wet-weather tyres overheating as it dries,
    and every tyre cooled by water."""
    start = OVERHEAT_FROM[compound]
    dryness_past_start = np.clip((start - wetness) / np.maximum(start, 1e-6), 0.0, 1.0)
    return (1 + OVERHEAT_EXTRA[compound] * dryness_past_start) * (1 - WATER_COOLING * wetness)


def suited_compound(wetness, slick=MEDIUM):
    """The kind of tyre that grips best on a track this wet, with `slick` when it's dry enough for slicks."""
    if wetness >= WET_FROM:
        return COMPOUNDS.index('wet')
    if wetness >= INTERMEDIATE_FROM:
        return COMPOUNDS.index('intermediate')
    return slick


def effective_top_speed(state: CarState, specs: CarSpecs, draft=0.0, headwind=0.0):
    return specs.top_speed * (1 - WING['top_speed'] * centred(state.wing)) * (1 + ENGINE['top_speed'] * centred(state.engine)) \
        * (1 - DAMAGE['top_speed'] * state.damage) * np.where(state.punctured, PUNCTURE['top_speed'], 1.0) \
        * (1 + SLIPSTREAM_TOP_SPEED * draft) * (1 - WIND_TOP_SPEED * headwind)


def fuel_burn_rate(engine):
    """kg per second at full throttle for an engine mode."""
    low, high = ENGINE['fuel']
    return FUEL_BURN * (low + (high - low) * np.asarray(engine, dtype=float))


def step_cars(state: CarState, specs: CarSpecs, steer, throttle, brake, on_track, conditions: Conditions | None = None):
    """Advances every car one tick. steer in [-1, 1] (positive turns left), throttle and brake in [0, 1]. Without
    `conditions`, the track is dry, clean and the air clear."""
    conditions = conditions if conditions is not None else Conditions.uniform(len(state.heading))
    wetness = conditions.wetness
    # Bent suspension pulls the car to one side whatever the driver does.
    steer = np.clip(steer + state.pull_side * PART_DAMAGE['suspension_pull'] * state.suspension_damage, -1, 1)
    running = (state.fuel > 0) & ~state.engine_failed
    throttle = np.where(running, np.clip(throttle, 0, 1), 0.0)
    brake = np.clip(brake, 0, 1)
    forward = state.forward
    along = np.einsum('ck,ck->c', state.velocity, forward)
    speed = np.maximum(along, 0.0)

    surface_grip = np.where(on_track, 1.0, OFF_TRACK['grip'])
    grip = effective_grip(state, specs, wetness, conditions.surface_grip, conditions.dirty_air) * GRAVITY * surface_grip
    heaviness = EMPTY_MASS / (EMPTY_MASS + state.fuel)
    top_speed = effective_top_speed(state, specs, conditions.draft, conditions.headwind)
    power = specs.acceleration * (1 + ENGINE['power'] * centred(state.engine)) * (1 - DAMAGE['acceleration'] * state.damage)
    braking = specs.braking * (1 - BRAKE_WEAR['loss'] * state.brake_wear) * brake_fade(state.brake_temp)

    # Engine force fades toward top speed; brakes, the grass and a flat tyre slow the car whatever the engine does.
    drive = throttle * power * heaviness * np.clip(1 - (speed / top_speed) ** 2, 0, 1)
    stopping = brake * braking * heaviness + ROLLING_DRAG + np.where(on_track, 0.0, OFF_TRACK['drag']) \
        + np.where(state.punctured, PUNCTURE['drag'], 0.0)
    # Slowing down can't take more than stopping the car outright this tick needs: the grip left over goes to
    # stopping a slide. (Otherwise a car barely rolling forward spends it all and slides sideways for ever.)
    asked = drive - np.minimum(np.where(speed > 0, stopping, 0.0), speed / DT)
    longitudinal = np.clip(asked, -grip, grip)

    # Whatever grip the pedals leave over is all there is for turning.
    cornering_grip = np.sqrt(np.maximum(grip ** 2 - longitudinal ** 2, 0.0))
    lock = MAX_STEER / (1 + speed / STEER_FADE_SPEED) * steer
    wanted_turn = speed * np.tan(lock) / WHEELBASE
    # At the limit the balance decides: an oversteering car rotates past what its grip can follow, an
    # understeering one gives up turning sooner.
    balance = specs.balance - HANDLING['wing_damage'] * state.wing_damage
    reach = 1 + np.where(balance > 0, HANDLING['oversteer'] * balance, HANDLING['understeer'] * balance)
    turn_limit = cornering_grip / np.maximum(speed, 1.0)
    turn = np.clip(wanted_turn, -turn_limit * reach, turn_limit * reach)
    understeer = np.abs(wanted_turn) > turn_limit * reach + 1e-9

    heading = state.heading + turn * DT
    new_forward = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    new_left = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    # The car turned under its velocity: what was straight ahead is now partly sideways. Tyres pull that back
    # into line up to the grip left over; the rest is a slide.
    along = np.einsum('ck,ck->c', state.velocity, new_forward)
    sideways = np.einsum('ck,ck->c', state.velocity, new_left)
    correction = np.clip(-sideways, -cornering_grip * DT, cornering_grip * DT)
    sideways = sideways + correction
    along = np.maximum(along + longitudinal * DT, 0.0)

    previous_velocity = state.velocity
    state.heading = (heading + np.pi) % (2 * np.pi) - np.pi
    state.velocity = new_forward * along[:, None] + new_left * sideways[:, None]
    state.position = state.position + state.velocity * DT
    state.sliding = (np.abs(sideways) > SLIDE_THRESHOLD) | (understeer & (speed > 10))
    state.lateral_load = np.abs(speed * turn) / np.maximum(grip, 1e-6)
    acceleration = (state.velocity - previous_velocity) / DT
    state.yaw_rate = turn
    state.longitudinal_g = np.einsum('ck,ck->c', acceleration, new_forward) / GRAVITY
    state.lateral_g = np.einsum('ck,ck->c', acceleration, new_left) / GRAVITY + speed * turn / GRAVITY
    state.wheelspin = np.where(asked > grip, np.clip((asked - grip) / np.maximum(asked, 1e-6), 0, 1), 0.0)

    overheating = 1 + TYRE_TEMPERATURE['overheat_wear'] * np.maximum(state.tyre_temp - TYRE_WINDOW[state.compound] - TYRE_WINDOW_HALF, 0.0) / 0.2
    wear_rate = COMPOUND_WEAR[state.compound] * tyre_temperature_wear(state.compound, wetness) * overheating \
        * (WEAR['load'] * state.lateral_load + WEAR['slide'] * state.sliding)
    state.tyre_wear = np.minimum(1.0, state.tyre_wear + DT * wear_rate)
    state.brake_wear = np.minimum(1.0, state.brake_wear + DT * BRAKE_WEAR['per_second'] * brake * (speed > 1))
    state.fuel = np.maximum(0.0, state.fuel - DT * fuel_burn_rate(state.engine) * throttle)
    _heat(state, speed, brake, throttle, conditions)
    return state


def _heat(state, speed, brake, throttle, conditions):
    """Tyres, brakes and engine warm with work and cool toward the air, faster the faster the car goes."""
    airflow = speed / 60
    tyre = TYRE_TEMPERATURE
    tyre_heating = (tyre['load'] * np.minimum(state.lateral_load, 1.5) + tyre['slide'] * state.sliding + tyre['brake'] * brake) * airflow
    ambient = tyre['ambient'] - tyre['wet_ambient'] * conditions.wetness + AIR_TEMPERATURE['tyre'] * (conditions.air_temperature - 0.5)
    tyre_cooling = tyre['cooling'] * (1 + tyre['wet_cooling'] * conditions.wetness) * (state.tyre_temp - ambient)
    state.tyre_temp = np.clip(state.tyre_temp + DT * (tyre_heating - tyre_cooling), 0.0, 1.0)

    brakes = BRAKE_TEMPERATURE
    brake_heating = brakes['heating'] * brake * airflow
    brake_cooling = brakes['cooling'] * (state.brake_temp - brakes['ambient']) * (0.3 + airflow)
    state.brake_temp = np.clip(state.brake_temp + DT * (brake_heating - brake_cooling), 0.0, 1.0)

    engine = ENGINE_TEMPERATURE
    engine_heating = engine['heating'] * throttle * (1 - engine['push'] / 2 + engine['push'] * state.engine)
    engine_ambient = engine['ambient'] + AIR_TEMPERATURE['engine'] * (conditions.air_temperature - 0.5)
    engine_cooling = engine['cooling'] * (state.engine_temp - engine_ambient) * (airflow + 0.2) * (1 - engine['dirty_air'] * conditions.dirty_air)
    state.engine_temp = np.clip(state.engine_temp + DT * (engine_heating - engine_cooling), 0.0, 1.0)
