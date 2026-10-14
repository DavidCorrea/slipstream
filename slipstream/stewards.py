"""The stewards: time penalties for breaking the rules, added to a car's race time when it finishes.

Drivers learned to race from rewards alone, and rewards only reach a few seconds ahead: a hit that wrecks a rival's
race, or a short cut across the grass, looked nearly free. A real driver weighs the stewards too, so here they
judge what the race does:
- track limits: every run off the tarmac that lasts TRACK_LIMITS['lasting'] is a strike; the first few are
  warnings, every one after that costs time.
- causing a collision: a crash (not the knocks of a close fight: with 20 drivers trained before the stewards, a
  penalty for every hard hit came to four a car a race) is the fault of the car that did most of the moving into
  the other; when both did, it's a racing incident. It costs time, more if the other car is out of the race
  because of it. One crash is judged once, however many ticks the cars stay in touch.
- reversing on the track: backing up out of trouble off the road is fine, backing down the racing line isn't.
- the pit lane (pit.py): speeding in it, and crossing the white lines that part its entry and exit from the track.
"""
from dataclasses import dataclass

import numpy as np

from .car import DT

TRACK_LIMITS = {'lasting': 0.3, 'warnings': 3, 'seconds': 5.0}   # s off to count; free strikes; s a strike after
# m/s of impulse that makes a crash; how many times more the car to blame must have moved into the other; s; s if the
# other car crashed out; s before the same pair is judged again.
COLLISION = {'impulse': 10.0, 'blame': 2.0, 'seconds': 5.0, 'out': 10.0, 'apart': 3.0}
REVERSING = {'lasting': 2.0, 'seconds': 5.0}                      # s backing up on the tarmac before it's dangerous
PIT_SPEEDING = 5.0
PIT_LINE = 5.0


@dataclass
class Penalty:
    car: int
    seconds: float
    reason: str      # 'track limits', 'collision', 'reversing', 'pit speeding', 'pit line'


class Stewards:
    def __init__(self, count):
        self.seconds = np.zeros(count)          # time added to each car's race
        self.strikes = np.zeros(count, dtype=int)
        self.off_for = np.zeros(count)          # seconds of the current run off the tarmac
        self.reversing_for = np.zeros(count)    # seconds of the current spell backing up on the tarmac
        self.judged = {}                        # (at fault, other) -> race time the pair was last judged

    def penalise(self, car, seconds, reason):
        self.seconds[car] += seconds
        return Penalty(int(car), float(seconds), reason)

    def judge(self, race, contacts, crashed):
        """Every penalty this tick: track limits, collisions (from this tick's `contacts`) and reversing on the
        track."""
        racing = ~race.finished & ~race.retired
        return self._track_limits(race, racing) + self._collisions(race, contacts, crashed) + self._reversing(race, racing)

    def _track_limits(self, race, racing):
        off = ~race.on_track & racing
        before = self.off_for.copy()
        self.off_for = np.where(off, self.off_for + DT, 0.0)
        striking = (before < TRACK_LIMITS['lasting']) & (self.off_for >= TRACK_LIMITS['lasting'])
        penalties = []
        for car in np.flatnonzero(striking):
            self.strikes[car] += 1
            if self.strikes[car] > TRACK_LIMITS['warnings']:
                penalties.append(self.penalise(car, TRACK_LIMITS['seconds'], 'track limits'))
        return penalties

    def _collisions(self, race, contacts, crashed):
        penalties = []
        for contact in contacts:
            at_fault = _to_blame(contact)
            if contact.impulse < COLLISION['impulse'] or at_fault is None:
                continue
            other = contact.second if at_fault == contact.first else contact.first
            pair = (at_fault, other)
            if race.time - self.judged.get(pair, -np.inf) < COLLISION['apart']:
                continue
            self.judged[pair] = race.time
            seconds = COLLISION['out'] if other in crashed else COLLISION['seconds']
            penalties.append(self.penalise(at_fault, seconds, 'collision'))
        return penalties

    def _reversing(self, race, racing):
        backing = race.cars.reversing & race.on_track & ~race.in_lane & racing
        before = self.reversing_for.copy()
        self.reversing_for = np.where(backing, self.reversing_for + DT, 0.0)
        dangerous = (before < REVERSING['lasting']) & (self.reversing_for >= REVERSING['lasting'])
        return [self.penalise(car, REVERSING['seconds'], 'reversing') for car in np.flatnonzero(dangerous)]


def _to_blame(contact):
    """The car that did most of the moving into the other, or None for a racing incident."""
    first, second = max(contact.first_into, 0.0), max(contact.second_into, 0.0)
    if first > COLLISION['blame'] * second:
        return contact.first
    if second > COLLISION['blame'] * first:
        return contact.second
    return None
