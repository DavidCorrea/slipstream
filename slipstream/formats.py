"""Race formats: the same race, run to other rules. None of them asks the drivers for anything new: the rules act on
the race around them (its length, the grid, who's still in it), and the team calls the stops as ever.

- standard     the laps asked for, from the grid as drawn (or as qualified).
- reverse      the grid turned round: the front row starts at the back.
- sprint       a third of the laps (never under SPRINT_LAPS), flat out.
- elimination  from ELIMINATE_FROM laps on, each time the leader completes a lap the last car still running is
               out, until LAST_STANDING are left. An eliminated car leaves the circuit (it's moved off it), so
               nobody runs into it where it would have pulled up.
- endurance    twice the laps with MANDATORY_STOPS stops: the team calls a car in when a stop it owes falls due, and
               one that takes the flag short of its stops gets UNSERVED_STOPS_PENALTY seconds.
- timed        a race against the clock, like Le Mans: when the time is up, the leader's next crossing of the line
               is the last lap. Until then the race's laps are what the time left should take at the leader's pace,
               so the drivers (who sense the laps left) and the strategy plan for a race of about that length.
"""
import math

import numpy as np

STANDARD, REVERSE, SPRINT, ELIMINATION, ENDURANCE, TIMED = 'standard', 'reverse', 'sprint', 'elimination', 'endurance', 'timed'
FORMATS = (STANDARD, REVERSE, SPRINT, ELIMINATION, ENDURANCE, TIMED)
SPRINT_LAPS = 2
ELIMINATE_FROM = 2
LAST_STANDING = 3
MANDATORY_STOPS = 2
UNSERVED_STOPS_PENALTY = 30.0
TIMED_MINUTES = (10, 20, 30, 60)
# Lap times before anyone has set one: a lap at about the average racing speed.
EXPECTED_SPEED = 30.0
OFF_THE_CIRCUIT = 100_000.0     # metres out where eliminated cars are parked, far from anything they could touch


def race_laps(kind, laps, cars, track_length, minutes=None):
    """How many laps a race of this format runs (a timed race's first estimate), from the laps asked for."""
    if kind == SPRINT:
        return max(SPRINT_LAPS, round(laps / 3))
    if kind == ENDURANCE:
        return laps * 2
    if kind == ELIMINATION:
        return max(laps, ELIMINATE_FROM + cars - LAST_STANDING)
    if kind == TIMED:
        return max(1, math.ceil(_minutes(minutes) * 60 / (track_length / EXPECTED_SPEED)))
    return laps


def _minutes(minutes):
    if minutes not in TIMED_MINUTES:
        raise ValueError(f'A timed race lasts one of {", ".join(map(str, TIMED_MINUTES))} minutes, got {minutes!r}')
    return minutes


class FormatRules:
    """Applies a format's rules to a race as it runs."""

    def __init__(self, kind, race, minutes=None):
        if kind not in FORMATS:
            raise ValueError(f'Format must be one of {", ".join(FORMATS)}, got {kind!r}')
        self.kind = kind
        self.seconds = _minutes(minutes) * 60.0 if kind == TIMED else None
        if self.seconds is not None:
            # The race's own cut-off must outlast the clock and the lap after it.
            race.time_limit = max(race.time_limit, self.seconds * 2)

    def calls(self, race, call, forced):
        """The team's pit calls with the stops an endurance race demands: a car owing one is called in once it's due,
        for the cars in `forced` (not the one whose stops you call yourself)."""
        if self.kind != ENDURANCE:
            return call
        owed = race.stops < MANDATORY_STOPS
        # Stop n falls due n/(stops + 1) of the way through the race.
        due_lap = np.floor(race.laps * (race.stops + 1) / (MANDATORY_STOPS + 1))
        due = owed & (race.lap_of() >= due_lap) & ~race.in_lane & ~race.finished & ~race.retired
        return call | (due & forced)

    def after_tick(self, race, events):
        """What the format did this tick: cars eliminated and penalties handed out, each a list."""
        happened = {'eliminated': []}
        if self.kind == ELIMINATION:
            happened['eliminated'] = self._eliminate(race, events)
        if self.kind == ENDURANCE:
            happened['penalties'] = self._unserved_stops(race, events)
        if self.kind == TIMED:
            self._set_laps_by_the_clock(race, events)
        return happened

    def _leader_crossed(self, race, events):
        leader = int(race.standings()[0])
        return leader in events.laps, leader

    def _eliminate(self, race, events):
        crossed, leader = self._leader_crossed(race, events)
        running = np.flatnonzero(~race.retired & ~race.finished)
        if not crossed or race.lap_of()[leader] < ELIMINATE_FROM or len(running) <= LAST_STANDING:
            return []
        last = int([car for car in race.standings() if car in running][-1])
        race.retire(last, 'eliminated')
        race.cars.position[last] = [OFF_THE_CIRCUIT + 20.0 * last, OFF_THE_CIRCUIT]
        race.cars.velocity[last] = 0.0
        return [last]

    def _unserved_stops(self, race, events):
        return [race.stewards.penalise(car, UNSERVED_STOPS_PENALTY, 'mandatory stops')
                for car in events.finished if race.stops[car] < MANDATORY_STOPS]

    def _set_laps_by_the_clock(self, race, events):
        crossed, leader = self._leader_crossed(race, events)
        if not crossed:
            return
        done = race.lap_of()[leader]
        left = self.seconds - race.time
        if left <= 0:
            race.laps = done + 1
            return
        per_lap = race.time / max(done, 1)
        race.laps = done + max(1, math.ceil(left / per_lap))

    def describe(self, race):
        """The format for the viewer: its kind, and the time left in a timed race."""
        described = {'kind': self.kind}
        if self.seconds is not None:
            described['timeLeft'] = round(max(0.0, self.seconds - race.time), 1)
        if self.kind == ENDURANCE:
            described['mandatoryStops'] = MANDATORY_STOPS
        return described
