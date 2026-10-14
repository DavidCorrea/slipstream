"""The marshals: the start lights, and the flags they show the drivers during a race.

- Start lights: five red lights come on a second apart, then after a hold nobody can predict they all go out, and
  that's the start. A car that moves off its grid slot before then has jumped the start.
- Yellow flags: around an incident (a crash, a hard hit, a car stopped by the track) the marshals wave yellow from
  YELLOW['before'] metres before it to YELLOW['after'] beyond, until YELLOW['lasting'] seconds after the last
  thing that happened there. Under yellow nobody may overtake, except a car that is crawling or stopped.
- Blue flags: a car about to be lapped is shown blue while the car lapping it is close behind, and has to let it
  by.

The stewards (stewards.py) judge what drivers do under them.
"""
import numpy as np

from .car import DT

LIGHTS = {'count': 5, 'apart': 1.0, 'hold': (0.5, 3.0)}   # lights; s between them; s held at five, at random
JUMP_START = 0.5                                           # metres off its grid slot before lights out
YELLOW = {'before': 200.0, 'after': 30.0, 'lasting': 10.0, 'hit': 10.0, 'stopped': 2.0, 'reach': 4.0}
# m before an incident, m after, s after it, m/s impulse that counts, s standing still that does, m off the
# tarmac a stopped car still counts from
CRAWLING = 15.0                                            # m/s: slower than this, a car may be passed under yellow
BLUE_REACH = 60.0                                          # metres behind a lapping car shows the blue flag from


class Marshals:
    def __init__(self, race, start_lights):
        self.lights_out_at = 0.0
        if start_lights:
            rng = np.random.default_rng([race.track.seed, 7])
            self.lights_out_at = LIGHTS['count'] * LIGHTS['apart'] + float(rng.uniform(*LIGHTS['hold']))
        self.grid = race.cars.position.copy()
        self.zones = []                      # [start, end, until]: circuit distances and race time
        self.stopped_for = np.zeros(race.count)
        self.blue = np.zeros(race.count, dtype=bool)
        self.lapping = np.full(race.count, -1)   # the car each blue-flagged car must let by

    def lights(self, time):
        """How many red lights are lit (0 once they've gone out, and before the first)."""
        if time >= self.lights_out_at:
            return 0
        return int(min(LIGHTS['count'], np.floor(time / LIGHTS['apart']) + 1))

    def started(self, time):
        return time >= self.lights_out_at

    def jumped(self, race):
        """Cars that have moved off their grid slot before the lights went out."""
        if self.started(race.time):
            return np.zeros(race.count, dtype=bool)
        return np.linalg.norm(race.cars.position - self.grid, axis=1) > JUMP_START

    def update(self, race, events):
        """Puts out yellow flags around this tick's incidents, takes in the old ones, and shows blue to cars about
        to be lapped. Returns the new zones' circuit distances, for the race feed."""
        length = race.track.length
        distance = race.progress % length
        racing = ~race.finished & ~race.retired & ~race.in_lane
        incidents = set(events.crashed)
        incidents |= {contact.first for contact in events.contacts if contact.impulse >= YELLOW['hit']}
        incidents |= {hit.car for hit in events.prop_hits if hit.impulse >= YELLOW['hit']}
        near_track = np.abs(race.lateral) <= race.track.width / 2 + YELLOW['reach']
        still = (race.cars.speed < 1.0) & near_track & ~race.finished & ~race.in_lane & self.started(race.time)
        self.stopped_for = np.where(still, self.stopped_for + DT, 0.0)
        incidents |= set(np.flatnonzero(self.stopped_for >= YELLOW['stopped']).tolist())
        new = []
        for car in sorted(incidents):
            start, end = distance[car] - YELLOW['before'], distance[car] + YELLOW['after']
            zone = next((zone for zone in self.zones if _covers(zone, distance[car], length)), None)
            if zone is None:
                self.zones.append([start, end, race.time + YELLOW['lasting']])
                new.append(float(distance[car]))
            else:
                zone[2] = race.time + YELLOW['lasting']
        self.zones = [zone for zone in self.zones if zone[2] > race.time]
        self._blue(race, racing)
        return new

    def under_yellow(self, distance, length):
        """Whether each circuit distance is inside a yellow zone."""
        distance = np.asarray(distance)
        covered = np.zeros(distance.shape, dtype=bool)
        for zone in self.zones:
            covered |= _covers(zone, distance, length)
        return covered

    def _blue(self, race, racing):
        """Blue for every running car with a car a lap or more ahead of it closing within BLUE_REACH behind."""
        length = race.track.length
        ahead_by = race.progress[None, :] - race.progress[:, None]        # [car, other]: how far other is ahead
        behind_on_track = (race.progress[:, None] - race.progress[None, :]) % length
        lapping = (ahead_by >= length - BLUE_REACH) & (behind_on_track < BLUE_REACH) & racing[:, None] & racing[None, :]
        np.fill_diagonal(lapping, False)
        self.blue = lapping.any(axis=1)
        self.lapping = np.where(self.blue, np.argmax(lapping, axis=1), -1)

    def describe(self, race):
        """What the viewer shows: the lights, the yellow zones (circuit distances) and who is shown blue."""
        return {'lights': self.lights(race.time), 'started': self.started(race.time),
                'yellow': [[round(start, 1), round(end, 1)] for start, end, _ in self.zones],
                'blue': self.blue.astype(int).tolist()}


def _covers(zone, distance, length):
    """Whether a yellow zone covers circuit distance `distance` (zones can wrap past the line)."""
    start, end, _ = zone
    return ((np.asarray(distance) - start) % length) <= (end - start)
