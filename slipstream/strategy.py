"""Race strategy: turning a network's outputs into pit calls and stop plans, judging what a car's condition is
worth, and a simple scripted strategist.

Why condition has a value: a stop costs time now (the lane and the standing time) and pays off over every lap
that follows. A driver looks about ten seconds ahead, far too short to see that pay-off, so on its own it would
never stop. Instead each car carries a value for its condition (`condition_value`): worn tyres, damage and worn
brakes cost more the more race there is left to drive on them, and so does fuel short of what's needed to
finish. Training rewards every change in that value as it happens (a form of potential-based shaping, which tells the
network sooner without changing which strategy is best). A stop that fixes the car earns its worth
straight away; letting tyres go off costs a little every lap; near the flag, condition stops mattering at all.
"""
import numpy as np

from .car import COMPOUND_WEAR, COMPOUNDS, FUEL_CAPACITY, INTERMEDIATE_FROM, WET_FROM, fuel_burn_rate, suited_compound, tyre_grip_factor
from .pit import CALLED, PitPlan

# The network's outputs, all in [-1, 1]: driving, then the pit call and the plan for the next stop. `compound`
# picks a slick; `weather_tyres`, added with the weather, overrides it with intermediates or full wets.
ACTION_NAMES = ('steer', 'pedal', 'pit', 'compound', 'fuel', 'wing', 'engine', 'repair', 'brakes', 'weather_tyres')
WEATHER_TYRES = ('slick', 'intermediate', 'wet')
DRIVING_ONLY = 2          # networks trained before pit stops existed only steer and press the pedal
CALL_THRESHOLD = 0.0      # above this, the lap's pit decision (see pit.py) is "come in"

RACING_SPEED = 30.0       # m/s, a rough average lap speed, used to estimate fuel use
WET_SLOWDOWN = 0.25       # share of that speed lost on a soaked track: longer laps burn more fuel
THROTTLE_SHARE = 0.6      # rough share of the time spent on full throttle
FUEL_MARGIN = 3.0         # kg a careful strategist adds to what it expects to need

# What condition is worth, in reward: per lap per unit of grip lost to tyre wear (a lap is worth about 12), per
# lap left per unit of damage and of brake wear, and per full tank short of the finish.
CONDITION_VALUE = {'tyres': 4.2, 'damage': 2.4, 'brakes': 1.0, 'fuel_shortfall': 20.0}
# Tyre wear per lap at racing pace on mediums (measured), for looking ahead at the wear still to come.
WEAR_PER_LAP = 0.13
LOOKAHEAD_STEP = 0.25     # laps between the points the tyre outlook is added up at
MAX_LAPS = 12


def decode(actions, count):
    """Steer, throttle, brake, pit call and stop plan from the network's outputs. Networks that only drive never
    call for a stop, and their cars get the standard stop if someone else calls them in."""
    actions = np.asarray(actions, dtype=float).reshape(count, -1)
    steer = actions[:, 0]
    throttle, brake = np.clip(actions[:, 1], 0, 1), np.clip(-actions[:, 1], 0, 1)
    if actions.shape[1] <= DRIVING_ONLY:
        return steer, throttle, brake, np.zeros(count, dtype=bool), PitPlan.standard(count)
    to_share = lambda column: (np.clip(actions[:, column], -1, 1) + 1) / 2
    plan = PitPlan(
        compound=_tyre_choice(actions),
        fuel=to_share(4) * FUEL_CAPACITY,
        wing=to_share(5), engine=to_share(6),
        repair=actions[:, 7] > 0, brakes=actions[:, 8] > 0,
    )
    return steer, throttle, brake, actions[:, 2] > CALL_THRESHOLD, plan


def _tyre_choice(actions):
    """The compound for the next stop. Networks from before the weather have no weather-tyre output and only
    ever choose slicks."""
    three_ways = lambda column: np.digitize(actions[:, column], [-1 / 3, 1 / 3])
    slick = three_ways(ACTION_NAMES.index('compound'))
    weather_column = ACTION_NAMES.index('weather_tyres')
    if actions.shape[1] <= weather_column:
        return slick
    weather = three_ways(weather_column)
    return np.select([weather == 1, weather == 2], [COMPOUNDS.index('intermediate'), COMPOUNDS.index('wet')], slick)


def distance_left(race):
    return np.maximum(race.laps * race.track.length - race.progress, 0.0) * ~race.finished


def laps_left(race):
    return distance_left(race) / race.track.length


def fuel_to_finish(race):
    """Rough kg each car needs to reach the flag at racing pace in its current engine mode."""
    seconds = distance_left(race) / racing_speed(race)
    return seconds * fuel_burn_rate(race.cars.engine) * THROTTLE_SHARE


def tyre_outlook(wear, compound, laps):
    """Grip each car's tyres will lose over the rest of its race, summed lap by lap (in grip x laps), if they go on
    wearing at their compound's usual rate. Worn tyres cost more than their wear today: they keep getting worse."""
    ahead = np.arange(0.0, MAX_LAPS, LOOKAHEAD_STEP)
    future = np.minimum(wear[:, None] + WEAR_PER_LAP * COMPOUND_WEAR[compound][:, None] * ahead[None, :], 1.0)
    # Each step counts for the share of a quarter lap that is still to be driven.
    share = np.clip(laps[:, None] - ahead[None, :], 0.0, LOOKAHEAD_STEP)
    return ((1 - tyre_grip_factor(future)) * share).sum(axis=1)


def condition_value(race):
    """What each car's condition is worth (zero or less): see the module notes."""
    cars, laps = race.cars, laps_left(race)
    shortfall = np.maximum(fuel_to_finish(race) - cars.fuel, 0.0) / FUEL_CAPACITY
    return -(CONDITION_VALUE['tyres'] * tyre_outlook(cars.tyre_wear, cars.compound, laps)
             + laps * (CONDITION_VALUE['damage'] * cars.damage + CONDITION_VALUE['brakes'] * cars.brake_wear)
             + CONDITION_VALUE['fuel_shortfall'] * shortfall)


def fuel_per_lap(race):
    return race.track.length / racing_speed(race) * fuel_burn_rate(race.cars.engine) * THROTTLE_SHARE


def racing_speed(race):
    return RACING_SPEED * (1 - WET_SLOWDOWN * race.weather.wetness)


# How far past a crossover (see car.py) the scripted strategist waits before changing tyres, so a track drifting
# around one doesn't bring it in every lap.
CHANGE_MARGIN = 0.06


def tyres_for(wetness, laps):
    """The compound the scripted strategist fits for a track this wet: wets, intermediates, or softs for a short
    last stint and mediums otherwise."""
    slick = np.where(laps < 4, COMPOUNDS.index('soft'), COMPOUNDS.index('medium'))
    return np.where(wetness >= INTERMEDIATE_FROM, suited_compound(wetness), slick)


def wrong_tyres(compound, wetness):
    """Whether each car is clearly on the wrong kind of tyre for the conditions: each kind suits the wetness
    between two crossovers, and only past the margin either side does it have to come off."""
    kind = np.select([compound == COMPOUNDS.index('wet'), compound == COMPOUNDS.index('intermediate')], [2, 1], 0)
    lowest = np.array([0.0, INTERMEDIATE_FROM, WET_FROM])[kind] - CHANGE_MARGIN
    highest = np.array([INTERMEDIATE_FROM, WET_FROM, 1.0])[kind] + CHANGE_MARGIN
    return (wetness < lowest) | (wetness > highest)


def scripted_strategy(race):
    """A sensible fixed strategy: come in when the fuel won't reach the flag and is about to run dry, for worn
    tyres or real damage with enough race left to make it pay, or for the right tyres when the track gets wetter or
    drier; take enough fuel to finish and the tyres that suit the track."""
    cars, laps, needed = race.cars, laps_left(race), fuel_to_finish(race)
    wetness = race.weather.wetness
    short_of_fuel = (cars.fuel < needed) & (cars.fuel < fuel_per_lap(race) * 1.4 + FUEL_MARGIN)
    worn_out = (cars.tyre_wear > 0.7) & (laps > 2)
    broken = (np.maximum.reduce([cars.damage, cars.wing_damage, cars.suspension_damage]) > 0.4) & (laps > 2)
    changing_conditions = wrong_tyres(cars.compound, wetness) & (laps > 0.7)
    call = (short_of_fuel | worn_out | broken | changing_conditions | cars.punctured | (race.pit_state == CALLED)) & ~race.finished & (laps > 0.3)
    plan = PitPlan(
        compound=tyres_for(wetness, laps),
        fuel=np.minimum(needed + FUEL_MARGIN, FUEL_CAPACITY),
        wing=cars.wing.copy(), engine=cars.engine.copy(),
        repair=np.ones(race.count, dtype=bool), brakes=cars.brake_wear > 0.5,
    )
    return call, plan
