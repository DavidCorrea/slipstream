"""The drivers of the viewer's championship: the same twenty people every race, each with a character of their own.

A driver's traits are who they are and don't change from race to race. Their car is roughly the same car each
time, but no two race days are alike: set-up, engine and tyre batch move its specs a little either side of what
it usually is (RACE_DAY). The grid is drawn afresh each race, and when the field is smaller than the cast some
drivers sit the race out. All of it comes from the race's seed, so the same circuit brings back the same day.

Drivers can be renamed, so everything that follows a driver from race to race goes by their `key`, which stays.

Every value is a slider from 0 to 1, as the grid menu shows it (see session.spec_value for what the car ones
mean in m/s and g, and personality.py for the traits).
"""
from dataclasses import dataclass, field

import numpy as np

RACE_DAY = 0.03      # how far each car slider moves either side of the driver's usual car, race to race
NAME_LENGTH = 16
NAME_MARKS = " .'-"   # what a name may hold besides letters, as in "Da Costa", "O'Neil" or "Abu-Bakr"


@dataclass(frozen=True)
class Driver:
    key: str
    name: str
    color: str
    character: dict = field(default_factory=dict)


def _driver(name, color, aggression, risk, overtaking, conservation, consistency, stamina,
            top_speed, acceleration, braking, grip, balance):
    return Driver(name.lower(), name, color, {
        'aggression': aggression, 'risk': risk, 'overtaking': overtaking, 'conservation': conservation,
        'consistency': consistency, 'stamina': stamina,
        'top_speed': top_speed, 'acceleration': acceleration, 'braking': braking, 'grip': grip, 'balance': balance,
    })


CAST = (
    # Fast and fearless, and it shows on the tyres and in the barriers.
    _driver('Vega', '#e63946', 0.85, 0.8, 0.8, 0.2, 0.35, 0.5, 0.58, 0.55, 0.5, 0.5, 0.65),
    # The tyre whisperer: never the quickest lap, often the last stop.
    _driver('Okafor', '#f4a261', 0.3, 0.4, 0.45, 0.85, 0.75, 0.7, 0.48, 0.5, 0.52, 0.55, 0.45),
    # Ice cold: the same lap again and again, to the flag.
    _driver('Lindqvist', '#2a9d8f', 0.4, 0.55, 0.5, 0.5, 0.9, 0.8, 0.5, 0.52, 0.55, 0.52, 0.5),
    # A street fighter who lives for the move up the inside.
    _driver('Moreau', '#457b9d', 0.75, 0.65, 0.9, 0.35, 0.5, 0.45, 0.52, 0.58, 0.58, 0.45, 0.55),
    # Smooth and tidy, in the car with the most grip.
    _driver('Sato', '#e9c46a', 0.2, 0.5, 0.35, 0.6, 0.7, 0.6, 0.45, 0.48, 0.5, 0.62, 0.4),
    # Brilliant for half a race, then the mistakes come.
    _driver('Ferreira', '#9b5de5', 0.6, 0.75, 0.65, 0.3, 0.55, 0.25, 0.55, 0.6, 0.48, 0.5, 0.6),
    # A rookie: quick in a straight line, still learning the rest.
    _driver('Kowalski', '#00bbf9', 0.55, 0.85, 0.7, 0.25, 0.25, 0.65, 0.62, 0.52, 0.45, 0.45, 0.55),
    # The veteran: cautious, wily, and never tired.
    _driver('Haddad', '#f15bb5', 0.35, 0.3, 0.4, 0.7, 0.8, 0.9, 0.47, 0.47, 0.53, 0.55, 0.45),
    # The rest of a full grid. The colours are spread round the wheel, so twenty cars stay twenty colours.
    # Late-braking specialist: gains everything into the corners.
    _driver('Novak', '#06d6a0', 0.6, 0.7, 0.75, 0.4, 0.6, 0.6, 0.5, 0.5, 0.65, 0.5, 0.55),
    # Qualifying hero, never quite as quick on old tyres.
    _driver('Reyes', '#ff7f11', 0.5, 0.8, 0.6, 0.15, 0.55, 0.5, 0.56, 0.58, 0.5, 0.56, 0.6),
    # The thinker: plans the race three stops ahead.
    _driver('Adeyemi', '#8338ec', 0.3, 0.45, 0.5, 0.75, 0.8, 0.75, 0.5, 0.5, 0.52, 0.52, 0.5),
    # Fearless in the wet, wild in the dry.
    _driver('Larsen', '#3a86ff', 0.7, 0.75, 0.7, 0.35, 0.4, 0.6, 0.53, 0.55, 0.5, 0.48, 0.65),
    # The wall: impossible to pass, and slow to pass anyone.
    _driver('Castillo', '#d62828', 0.8, 0.45, 0.3, 0.5, 0.65, 0.7, 0.5, 0.48, 0.55, 0.53, 0.45),
    # Calm and precise, a future champion.
    _driver('Ivanova', '#90be6d', 0.4, 0.6, 0.6, 0.55, 0.85, 0.8, 0.52, 0.53, 0.53, 0.55, 0.5),
    # Smooth on tyres, ruthless on the last lap.
    _driver('Tanaka', '#ffbe0b', 0.5, 0.55, 0.85, 0.65, 0.7, 0.65, 0.49, 0.5, 0.51, 0.54, 0.5),
    # A slipstream artist, fastest down the straights.
    _driver('Mbeki', '#fb5607', 0.55, 0.6, 0.7, 0.4, 0.6, 0.55, 0.64, 0.55, 0.47, 0.46, 0.55),
    # Erratic and exciting: a podium or the gravel.
    _driver('Brennan', '#ff006e', 0.75, 0.9, 0.85, 0.2, 0.2, 0.5, 0.55, 0.57, 0.5, 0.5, 0.7),
    # Old school: brave, kind to the car, never gives up.
    _driver('Rossi', '#2ec4b6', 0.5, 0.5, 0.6, 0.6, 0.65, 0.85, 0.48, 0.5, 0.55, 0.53, 0.45),
    # Grip merchant: the best car through the fast corners.
    _driver('Kaur', '#7209b7', 0.35, 0.65, 0.55, 0.5, 0.7, 0.6, 0.46, 0.5, 0.5, 0.64, 0.4),
    # Hard as nails, and harder still on brakes.
    _driver('Petrov', '#a7c957', 0.7, 0.6, 0.65, 0.3, 0.5, 0.7, 0.52, 0.53, 0.62, 0.5, 0.55),
)
CAR_SLIDERS = ('top_speed', 'acceleration', 'braking', 'grip', 'balance')
BY_KEY = {driver.key: driver for driver in CAST}


def lineup(count, rng, must_race=None):
    """Who races, in grid order: `count` drivers drawn at random, always including `must_race` (a key)."""
    drawn = [CAST[index] for index in rng.permutation(len(CAST))[:count]]
    if must_race is not None and must_race not in (driver.key for driver in drawn):
        drawn[rng.integers(count)] = BY_KEY[must_race]
    return drawn


def tidy_name(name, taken):
    """A name a driver can race under: trimmed, spaces collapsed, made of letters (any alphabet) and NAME_MARKS,
    starting with a letter, at most NAME_LENGTH long, and not `taken` (other drivers' names, any case)."""
    if not isinstance(name, str):
        raise ValueError(f'A name must be text, got {name!r}')
    tidied = ' '.join(name.split())
    if not tidied:
        raise ValueError('A name cannot be empty')
    if len(tidied) > NAME_LENGTH:
        raise ValueError(f'A name can be at most {NAME_LENGTH} characters, got {len(tidied)} in {tidied!r}')
    if not tidied[0].isalpha() or not all(character.isalpha() or character in NAME_MARKS for character in tidied):
        raise ValueError(f'A name is letters, spaces and {NAME_MARKS.strip()} starting with a letter, got {tidied!r}')
    if tidied.casefold() in {other.casefold() for other in taken}:
        raise ValueError(f'{tidied!r} is already racing under that name')
    return tidied


def race_day(driver, rng):
    """The driver's stats for one race: their own traits, and their usual car give or take RACE_DAY."""
    stats = dict(driver.character)
    for name in CAR_SLIDERS:
        stats[name] = float(np.clip(stats[name] + rng.uniform(-RACE_DAY, RACE_DAY), 0, 1))
    return stats
