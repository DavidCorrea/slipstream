"""The pit lane and what happens in it.

The lane runs beside the start straight, on the outside of the circuit, from ENTRY_BEFORE metres before the line
to EXIT_AFTER metres after it. Each car has a box along it. A car that has been called in leaves the track when
it reaches the entry and drives itself through the lane: it slows to the speed limit, stops in its box while
the crew works, then drives out at the limit and rejoins. Drivers only decide whether to come in and what the
crew should do; nobody has to learn to steer down a pit lane.

Whether to come in is decided once a lap, like a pit wall's "box this lap": the first decision a driver makes
after passing a point DECIDE_BEFORE metres before the entry commits it, in or not, until it has passed the entry.
(When any decision could call a stop, only the one that happened to fall at the entry mattered, and a run learned
nothing about stopping in 7 million decisions.)

The lane has two lanes, as a real one does: a fast lane nearest the track for driving through, and a working lane
along the garages where the boxes are. A car moves over to the working lane just before its box and back out after
its stop, so cars being worked on never block the cars driving past. Nobody overtakes under the speed limit: cars
queue nose to tail, and a car leaving its box waits for a gap in the fast lane before pulling out.

What the crew does decides how long the car stands still. Tyres, fuel, wing and engine are done side by side,
so the longest of them counts; repairs and new brake pads come on top.
"""
from dataclasses import dataclass

import numpy as np

from .car import COMPOUNDS, FUEL_CAPACITY, TYRE_BLANKETS
from .track import SPACING

ENTRY_BEFORE = 160.0    # metres before the line where the lane leaves the track
EXIT_AFTER = 120.0      # metres after the line where it rejoins, at least: a bigger field's boxes take it further
RAMP = 45.0             # metres the lane takes to move off the track and back on
LANE_OFFSET = 9.0       # lane centre, metres beyond the track edge
ENTRY_REACH = 15.0      # metres beyond the track edge a car can still turn into the lane from
SPEED_LIMIT = 22.0      # m/s (about 80 km/h)
DECIDE_BEFORE = 150.0   # metres before the entry where the lap's pit decision is made
SLOW_DOWN = 14.0        # m/s^2 a car brakes at to reach the limit and to stop in its box
FIRST_BOX = -60.0       # metres from the line to the first box (negative is before the line)
BOX_SPACING = 12.0
LANES_APART = 3.6       # metres between the fast lane's centre and the working lane's
# Moving between lanes happens within the gap between two boxes (BOX_SPACING less a car length), so a car turning
# in or pulling out never swings across the next box.
TURN_IN = 6.0           # metres before its box a car starts moving over to the working lane
PULL_OUT = 6.0          # metres after its box it takes to get back to the fast lane
QUEUE_GAP = 9.0         # metres, centre to centre, a car keeps to the one ahead in its lane
BACK_OFF = 3.0          # m/s a car eases off for every metre it's closer than that, until the gap opens
HARD_BRAKING = 30.0     # m/s^2 a car brakes at when it's closing on the car ahead inside that gap
NOSE_TO_TAIL = 5.5      # metres, centre to centre, two cars in the same lane can never get closer than
MERGE_CLEAR = (15.0, 9.0)   # a car leaving its box waits while a fast-lane car is this far behind or ahead of it
SERVICE = {'base': 2.0, 'tyres': 2.4, 'fuel_per_second': 12.0, 'wing': 1.5, 'engine': 1.0, 'repair_per_damage': 10.0, 'brakes': 4.0}
CHANGE_THRESHOLD = 0.05  # a wing or engine setting has to move this much for the crew to bother

RACING, CALLED, IN_LANE, STOPPED = 0, 1, 2, 3


@dataclass
class PitPlan:
    """What the crew should do at each car's next stop, one entry per car."""
    compound: np.ndarray    # index into COMPOUNDS of the new tyres
    fuel: np.ndarray        # kg the tank should hold when the car leaves
    wing: np.ndarray        # 0-1
    engine: np.ndarray      # 0-1
    repair: np.ndarray      # bool: fix the damage
    brakes: np.ndarray      # bool: new brake pads

    @classmethod
    def standard(cls, count):
        """New mediums, a full tank, the car otherwise as it is: what a car gets when nobody says otherwise."""
        return cls(compound=np.full(count, COMPOUNDS.index('medium')), fuel=np.full(count, FUEL_CAPACITY),
                   wing=np.full(count, 0.5), engine=np.full(count, 0.5), repair=np.ones(count, dtype=bool), brakes=np.zeros(count, dtype=bool))


def service_times(plan: PitPlan, cars, car):
    """How long each job takes for one car, and how long it stands still in all."""
    worst_damage = max(float(cars.damage[car]), float(cars.wing_damage[car]), float(cars.suspension_damage[car]))
    jobs = {
        'tyres': SERVICE['tyres'],
        'fuel': max(0.0, min(float(plan.fuel[car]), FUEL_CAPACITY) - float(cars.fuel[car])) / SERVICE['fuel_per_second'],
        'wing': SERVICE['wing'] if abs(plan.wing[car] - cars.wing[car]) > CHANGE_THRESHOLD else 0.0,
        'engine': SERVICE['engine'] if abs(plan.engine[car] - cars.engine[car]) > CHANGE_THRESHOLD else 0.0,
        'repair': SERVICE['repair_per_damage'] * worst_damage if plan.repair[car] and worst_damage > 0.02 else 0.0,
        'brakes': SERVICE['brakes'] if plan.brakes[car] else 0.0,
    }
    total = SERVICE['base'] + max(jobs['tyres'], jobs['fuel'], jobs['wing'], jobs['engine']) + jobs['repair'] + jobs['brakes']
    return {key: round(value, 2) for key, value in jobs.items()}, total


def apply_service(plan: PitPlan, cars, car, jobs):
    cars.compound[car] = int(plan.compound[car])
    cars.tyre_wear[car] = 0.0
    cars.tyre_temp[car] = TYRE_BLANKETS
    cars.punctured[car] = False
    cars.fuel[car] = max(float(cars.fuel[car]), min(float(plan.fuel[car]), FUEL_CAPACITY))
    if jobs['wing']:
        cars.wing[car] = float(np.clip(plan.wing[car], 0, 1))
    if jobs['engine']:
        cars.engine[car] = float(np.clip(plan.engine[car], 0, 1))
    if jobs['repair']:
        cars.damage[car] = cars.wing_damage[car] = cars.suspension_damage[car] = 0.0
    if jobs['brakes']:
        cars.brake_wear[car] = 0.0


class PitLane:
    """Where the lane is on this circuit and where every car's box is, in metres along the lane from its entry."""

    def __init__(self, track, count):
        self.track = track
        self.boxes = ENTRY_BEFORE + FIRST_BOX + np.arange(count) * BOX_SPACING
        # The lane runs on past the last box far enough to pull out of it and ramp back onto the track. Twenty cars'
        # boxes reach well beyond EXIT_AFTER, and a box past the lane's end had its car leave the moment it stopped.
        self.length = max(ENTRY_BEFORE + EXIT_AFTER, self.boxes[-1] + PULL_OUT + RAMP)
        self.entry = track.length - ENTRY_BEFORE
        # The lane goes on the outside of the start straight: the side facing away from the middle of the circuit.
        middle = track.points.mean(axis=0)
        start, left = track.points[0], track.normals[0]
        self.side = 1.0 if np.linalg.norm(start + left * 20 - middle) > np.linalg.norm(start - left * 20 - middle) else -1.0
        self.offset = self.side * (track.width / 2 + LANE_OFFSET)
        self.fast_lane = self.offset - self.side * LANES_APART / 2
        self.working_lane = self.offset + self.side * LANES_APART / 2

    def lateral(self, along, start, shift=0.0):
        """Offset from the centre line `along` metres into the lane, for a car that came in at lateral `start` and is
        `shift` of the way (0 to 1) from the fast lane over to the working lane."""
        into = np.clip(along / RAMP, 0, 1)
        out = np.clip((self.length - along) / RAMP, 0, 1)
        ease = lambda value: value * value * (3 - 2 * value)
        lateral = start + (self.fast_lane - start) * ease(into)
        rejoin = self.side * self.track.width / 4
        lateral = np.where(along > self.length - RAMP, rejoin + (self.fast_lane - rejoin) * ease(out), lateral)
        return lateral + shift * (self.working_lane - self.fast_lane)

    def stretch(self, along, start, shift=0.0):
        """How many metres of a car's actual path one metre along the centre line is, `along` metres into the lane:
        less round the inside of a bend (where the lane runs) and more round the outside."""
        index = (np.floor(self.distance(along) / SPACING).astype(int)) % self.track.size
        return np.maximum(1 - self.track.curvature[index] * self.lateral(along, start, shift), 0.1)

    def distance(self, along):
        """Distance along the circuit for a point `along` metres into the lane."""
        return (self.entry + along) % self.track.length

    def in_window(self, distance):
        """Whether each car is between the decision point and the lane entry."""
        return ((self.entry - distance) % self.track.length) < DECIDE_BEFORE

    def entered(self, before, after):
        """Whether a car moved from `before` to `after` (distances along the circuit) across the lane entry."""
        length = self.track.length
        # The signed move along the loop, so sliding backwards is a small step back, not nearly a lap forward.
        moved = (after - before + length / 2) % length - length / 2
        return (moved > 0) & (((self.entry - before) % length) <= moved)

    def describe(self):
        return {'entry': self.entry, 'length': self.length, 'side': self.side, 'offset': self.offset, 'ramp': RAMP,
                'fastLane': self.fast_lane, 'workingLane': self.working_lane,
                'boxes': self.boxes.tolist(), 'speedLimit': SPEED_LIMIT}
