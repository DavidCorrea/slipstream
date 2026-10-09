"""The driver's hands and feet: what a driver asks of the car arrives a moment late and never quite as asked.

A trained network decides what to do; a person then has to do it. Reactions take a few hundredths of a second, so
every input arrives LATENCY_TICKS late. Hands and feet are never perfectly steady, and they get less steady under
pressure (a car right behind or right ahead), with fatigue as the race wears on, and the less consistent the
driver is. Once in a while, more so under pressure, a driver makes a real mistake: a moment's too much steering,
a stab of brake, a lift where it should have been flat. Two of the driver's traits shape this: `consistency`
(how small the errors are and how rare the mistakes) and `stamina` (how slowly fatigue sets in).
"""
import numpy as np

LATENCY_TICKS = 3                        # 0.15 s
NOISE = {'steer': 0.025, 'pedal': 0.03}  # everyday unsteadiness, before pressure and fatigue
PRESSURE_NOISE = 2.0                     # how much more unsteady at full pressure
FATIGUE_SECONDS = 1200.0                 # a race this long tires a driver of average stamina out
FATIGUE_NOISE = 1.0
# Chance per tick of a real mistake (about one every four minutes calmly, every minute under full pressure), how
# long one lasts and how big it is.
MISTAKE = {'chance': 0.0002, 'pressure': 3.0, 'ticks': 8, 'steer': 0.25, 'pedal': 0.5}


class Driving:
    def __init__(self, count, rng, consistency=None, stamina=None):
        """`consistency` and `stamina` are 0-1 per driver (0.5 when not given)."""
        self.rng = rng
        self.consistency = np.full(count, 0.5) if consistency is None else np.asarray(consistency, dtype=float)
        self.stamina = np.full(count, 0.5) if stamina is None else np.asarray(stamina, dtype=float)
        self.queue = [(np.zeros(count), np.zeros(count), np.zeros(count)) for _ in range(LATENCY_TICKS)]
        self.mistake_left = np.zeros(count, dtype=int)
        self.mistake = np.zeros((count, 2))
        self.just_erred = np.zeros(count, dtype=bool)   # drivers whose mistake began this tick

    def apply(self, steer, throttle, brake, pressure, time):
        """What the car gets this tick, given what the driver asks for now. `pressure` is 0-1 per driver and
        `time` the seconds raced so far."""
        self.queue.append((np.asarray(steer, dtype=float), np.asarray(throttle, dtype=float), np.asarray(brake, dtype=float)))
        steer, throttle, brake = self.queue.pop(0)
        count = len(steer)
        fatigue = np.minimum(time / FATIGUE_SECONDS, 1.5) * (1.2 - self.stamina)
        unsteadiness = (1.5 - self.consistency) * (1 + PRESSURE_NOISE * pressure + FATIGUE_NOISE * fatigue)
        steer = steer + self.rng.normal(0, NOISE['steer'], count) * unsteadiness
        pedal = throttle - brake + self.rng.normal(0, NOISE['pedal'], count) * unsteadiness

        starting = (self.mistake_left == 0) & (self.rng.random(count) < MISTAKE['chance'] * (1 + MISTAKE['pressure'] * pressure)
                                               * (1 + fatigue) * (1.5 - self.consistency))
        self.just_erred = starting
        self.mistake_left[starting] = MISTAKE['ticks']
        self.mistake[starting] = self.rng.normal(0, 1, (int(starting.sum()), 2)) * [MISTAKE['steer'], MISTAKE['pedal']]
        erring = self.mistake_left > 0
        steer = steer + np.where(erring, self.mistake[:, 0], 0.0)
        pedal = pedal + np.where(erring, self.mistake[:, 1], 0.0)
        self.mistake_left[erring] -= 1
        return np.clip(steer, -1, 1), np.clip(pedal, 0, 1), np.clip(-pedal, 0, 1)
