"""The weather over a race: when it rains and how wet the track is.

Each race gets a forecast: dry, wet throughout, drying from a wet start, rain on the way, or a passing shower.
Rain soaks the track quickly and it dries slowly once the rain stops, so a shower leaves a damp track behind it
for a few laps. How wet the track is decides which tyres are fastest (see car.compound_grip); nothing else is a
rule, so when to change tyres is for the strategists to work out.
"""
from dataclasses import dataclass, field

import numpy as np

# How likely each forecast is, from a race's seed. Half the races are dry, so dry racing stays the main thing.
FORECASTS = {'dry': 0.5, 'rain_coming': 0.15, 'shower': 0.13, 'drying': 0.12, 'wet': 0.10}
SOAK_RATE = 0.03       # per second in full rain, slowing as the track gets wetter (about a minute to soaked)
DRY_RATE = 0.0025      # per second once it stops: a soaked track takes about seven minutes to dry
RAIN_RANGE = (0.4, 1.0)
RACE_PACE = 30.0       # m/s, a rough average lap speed, for how long a race will last
WIND_MOST = 9.0        # m/s
RAINY_COOLING = 0.4    # share of the warmth a rainy day loses


@dataclass
class Weather:
    forecast: str
    showers: list = field(default_factory=list)   # (start, end, intensity) in race seconds
    wetness: float = 0.0                          # 0 dry to 1 soaked
    time: float = 0.0
    temperature: float = 0.5                      # the air: 0 a cold day to 1 a hot one
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))   # m/s, the way it blows across the circuit

    @classmethod
    def dry(cls):
        return cls('dry')

    @classmethod
    def for_race(cls, seed, track_length, laps, forecast=None):
        """The weather for a race on a circuit: from the seed, or with the given forecast."""
        race_seconds = laps * track_length / RACE_PACE
        if forecast is None:
            return cls.from_seed(seed, race_seconds)
        return cls.from_forecast(forecast, race_seconds, np.random.default_rng([seed, 1]))

    @classmethod
    def from_seed(cls, seed, race_seconds):
        # Its own stream from the seed, so the weather doesn't change the circuit the same seed builds.
        rng = np.random.default_rng([seed, 1])
        forecast = rng.choice(list(FORECASTS), p=list(FORECASTS.values()))
        return cls.from_forecast(str(forecast), race_seconds, rng)

    @classmethod
    def from_forecast(cls, forecast, race_seconds, rng):
        if forecast not in FORECASTS:
            raise ValueError(f'Unknown forecast {forecast!r}; known ones are {", ".join(FORECASTS)}')
        intensity = float(rng.uniform(*RAIN_RANGE))
        endless = race_seconds * 10
        # The day: rain comes with cooler air, and the wind blows from anywhere at up to WIND_MOST.
        rainy = forecast != 'dry'
        temperature = float(rng.uniform(0, 1)) * (1 - RAINY_COOLING * rainy)
        direction = rng.uniform(0, 2 * np.pi)
        wind = np.array([np.cos(direction), np.sin(direction)]) * rng.uniform(0, WIND_MOST)
        day = {'temperature': temperature, 'wind': wind}
        if forecast == 'dry':
            return cls(forecast, **day)
        if forecast == 'wet':
            return cls(forecast, [(0.0, endless, intensity)], wetness=float(rng.uniform(0.6, 0.9)), **day)
        if forecast == 'drying':
            return cls(forecast, [], wetness=float(rng.uniform(0.7, 1.0)), **day)
        start = float(rng.uniform(0.2, 0.5)) * race_seconds
        if forecast == 'rain_coming':
            return cls(forecast, [(start, endless, intensity)], **day)
        return cls(forecast, [(start, start + float(rng.uniform(0.15, 0.3)) * race_seconds, intensity)], **day)

    @property
    def rain(self):
        """How hard it's raining right now, 0 to 1."""
        return next((intensity for start, end, intensity in self.showers if start <= self.time < end), 0.0)

    def step(self, seconds):
        rain = self.rain
        soaking = rain * SOAK_RATE * (1 - self.wetness)
        drying = (1 - rain) * DRY_RATE
        self.wetness = float(np.clip(self.wetness + (soaking - drying) * seconds, 0.0, 1.0))
        self.time += seconds
