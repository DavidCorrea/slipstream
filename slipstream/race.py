"""A race: cars on a track, contact between them, laps and standings.

For contact each car is its footprint as drawn: a rectangle from its front wing to its rear wing, as wide as its
wheels, turned with the car. Every bump does damage. Progress is measured along the centre line from the start line; the
grid starts behind it, so progress begins negative and a car finishes when it has covered `laps` full lengths.
Cars in the pit lane (see pit.py) drive themselves and can't touch cars on track.
"""
from dataclasses import dataclass, field

import numpy as np

from . import pit
from .aero import air_between
from .car import DAMAGE, DT, CarSpecs, CarState, Conditions, step_cars, suited_compound
from .human import Driving
from .obstacles import Obstacles
from .surface import Surface
from .pit import PitLane, PitPlan
from .scenery import SOFT
from .track import Track
from .weather import Weather

# The footprint: 2.2 m behind the car's centre to 2.8 m ahead (the front wing reaches further than the rear), and
# 1.06 m either side to the outside of the wheels. (Two circles stood in for it before; they left a pinch at the
# sidepods and stopped short of the front wing, so cars drove visibly into each other without touching.)
FOOTPRINT_HALF_LENGTH = 2.5
FOOTPRINT_OFFSET = 0.3     # how far ahead of the car's centre the footprint's centre is
FOOTPRINT_HALF_WIDTH = 1.06
FOOTPRINT_REACH = float(np.hypot(FOOTPRINT_HALF_LENGTH, FOOTPRINT_HALF_WIDTH))
RESTITUTION = 0.25
GRID_GAP = 9.0             # metres between grid rows
SECONDS_PER_LAP_LIMIT = 150.0
AFTER_WINNER = 45.0        # seconds the rest of the field gets to finish once the winner has
# Where a hit lands decides what breaks: a hit to the front takes the front wing, to the side the suspension.
# Part damage builds faster than the general damage every hit also does.
PART_HARM = 1.5
# A hard enough hit leaves bits on the track; running over one can puncture a tyre.
DEBRIS = {'impulse': 3.5, 'per_impulse': 0.5, 'most': 6, 'spread': 3.0, 'reach': 1.1, 'puncture': 0.03}
# A hot engine can fail: the chance per second grows from nothing at `from` to `rate` at the hottest.
ENGINE_FAILURE = {'from': 0.85, 'rate': 0.002}
# Hitting props (see scenery.py, obstacles.py). A solid one bounces a car off with RESTITUTION and takes some of its
# speed along it too, more the harder the hit (`scrape` of it per m/s of impulse, at most `scrape_most`): rubbing
# along a wall costs a little, a real impact a lot. A soft one (a tyre wall) gives more and harms less; a breakable
# one breaks, taking a share of the car's speed and a light knock. At first the speed along went at a flat 10% a
# tick of contact, and a 20-car field rubbing along the armco and pit wall lost whole laps. A hit harder than CRASH_OUT (about 100 km/h head on) ends the race,
# and so does getting less than STUCK_METRES further in STUCK_SECONDS, there being no reverse gear: a car creeping
# along a wall on the grass at walking pace got a metre further every few seconds and stayed in the race for ever.
PROPS = {'scrape': 0.015, 'scrape_most': 0.5, 'soft_restitution': 0.05, 'soft_absorbs': 0.03, 'soft_harm': 0.35,
         'breaks_speed': 0.15, 'breaks_knock': 0.2}
CRASH_OUT = 35.0
STUCK_SECONDS = 20.0
STUCK_METRES = 25.0
# A car up against something at walking pace backs off it and swings its nose round to the way the track goes, as a
# driver would in reverse: cars have no reverse gear and can't steer standing still, and one nosed into the pit
# wall otherwise pushed against it until it retired, with the cars behind piling up on it.
RECOVERY = {'below': 3.0, 'turn': 0.9, 'back': 1.2}   # m/s; radians a second; m/s off the prop
PRESSURE_GAP = 20.0        # metres to the car ahead or behind at which a driver starts to feel the pressure


@dataclass
class Contact:
    first: int
    second: int
    impulse: float          # m/s of closing speed taken out
    point: np.ndarray       # where they touched


@dataclass
class PropHit:
    car: int
    prop: int
    impulse: float          # m/s of speed taken out, as for a contact between cars
    point: np.ndarray       # where it hit
    broke: bool             # a breakable prop the car went through


@dataclass
class TickEvents:
    contacts: list = field(default_factory=list)
    prop_hits: list = field(default_factory=list)  # PropHits this tick
    crashed: list = field(default_factory=list)    # cars out of the race from a big hit, or stuck where they hit
    laps: list = field(default_factory=list)       # car indices that crossed the line this tick
    finished: list = field(default_factory=list)   # car indices that took the flag this tick
    pit_entered: list = field(default_factory=list)
    pit_stopped: list = field(default_factory=list)    # (car, jobs, standing time) for cars that just reached their box
    pit_released: list = field(default_factory=list)
    pit_exited: list = field(default_factory=list)
    punctures: list = field(default_factory=list)          # cars that just got a puncture
    engine_failures: list = field(default_factory=list)    # cars whose engine just failed
    mistakes: list = field(default_factory=list)           # drivers who just made a mistake


class Race:
    def __init__(self, track: Track, specs: CarSpecs, laps: int, ghosts=False, weather: Weather | None = None,
                 rng=None, driving: Driving | None = None):
        """`ghosts` makes every car race on its own: all start from pole, pass through each other, and the race
        only ends when all of them have finished. That's many solo races run side by side, for training the pit
        wall (see pitwall.py). Without a `weather`, the race is dry. `rng` drives chance (punctures, failures,
        where debris lands); `driving`, when given, puts the drivers' hands and feet between what they ask for and
        what the cars get (see human.py)."""
        self.track, self.specs, self.laps, self.ghosts = track, specs, laps, ghosts
        self.rng = rng if rng is not None else np.random.default_rng(track.seed)
        self.driving = driving
        self.surface = Surface(track)
        self.debris = np.zeros((0, 2))         # where each piece of debris lies
        self.new_debris = np.zeros((0, 2))     # the pieces that landed this tick, for the viewer
        self.weather = weather if weather is not None else Weather.dry()
        count = len(specs.top_speed)
        self.count = count
        # Two columns, staggered: odd slots half a row back, like a real grid. Ghosts all take pole.
        slot = np.zeros(count, dtype=int) if ghosts else np.arange(count)
        rows = slot // 2
        behind = (rows + 1) * GRID_GAP + (slot % 2) * GRID_GAP / 2
        lateral = np.where(slot % 2 == 0, 1, -1) * track.width / 4
        distance = track.length - behind
        self.cars = CarState.at(track.point_at(distance, lateral), track.heading_at(distance))
        # Teams start on whatever suits the track on the grid.
        self.cars.compound[:] = suited_compound(self.weather.wetness)
        self.track_index, _, self.lateral = track.locate(self.cars.position)
        self.progress = -behind.astype(float)
        self.tick = 0
        self.finish_time = np.full(count, np.nan)
        self.retired = np.zeros(count, dtype=bool)   # out of the race without finishing (an engine failure, a crash)
        self.props = None                             # what stands around the track, once placed (place_props)
        # Running totals per car, for summaries: ticks spent off the tarmac and ticks spent touching another car.
        self.off_track_ticks = np.zeros(count, dtype=int)
        self.contact_ticks = np.zeros(count, dtype=int)
        # And what each car has used up, counting only consumption (stops refill and reset, they don't un-use).
        self.fuel_used = np.zeros(count)
        self.tyre_used = np.zeros(count)
        self.time_limit = laps * SECONDS_PER_LAP_LIMIT
        self.pit = PitLane(track, count)
        self.pit_state = np.full(count, pit.RACING)
        self.lane_along = np.zeros(count)       # metres into the lane
        self.lane_speed = np.zeros(count)
        self.lane_start = np.zeros(count)       # lateral offset where the car turned in
        self.lane_shift = np.zeros(count)       # 0 in the fast lane to 1 in the working lane (see pit.py)
        self.service_left = np.zeros(count)     # seconds the crew still needs
        self.service_jobs = [None] * count
        self.stops = np.zeros(count, dtype=int)
        # Whether each car has made this lap's pit decision (see pit.py); cleared once it passes the entry.
        self.pit_decided = np.zeros(count, dtype=bool)

    @property
    def time(self):
        return self.tick * DT

    def conditions(self):
        """What the world does to each car this tick (see car.Conditions): the wetness under it (less on a dry
        line), the air of the cars ahead and the surface."""
        distance = self.progress % self.track.length
        racing = ~self.finished & ~self.retired & ~self.in_lane
        if self.ghosts:
            draft = dirty = np.zeros(self.count)
        else:
            draft, dirty = air_between(self.cars.position, self.cars.heading, self.cars.velocity, racing)
        on_tarmac = racing
        return Conditions(
            wetness=np.where(on_tarmac, self.surface.wetness(distance, self.lateral, self.weather.wetness), self.weather.wetness),
            draft=draft, dirty_air=dirty,
            surface_grip=np.where(on_tarmac, self.surface.grip(distance, self.lateral), 1.0),
            air_temperature=np.full(self.count, self.weather.temperature),
            headwind=-self.cars.forward @ self.weather.wind,
        )

    def pressure(self):
        """How hard each driver is being pushed or is pushing, 0-1: how close the nearest car ahead or behind is."""
        if self.ghosts or self.count < 2:
            return np.zeros(self.count)
        order = self.standings()
        ordered = self.progress[order]
        gaps = np.full(self.count, np.inf)
        between = ordered[:-1] - ordered[1:]
        gaps[order[1:]] = np.minimum(gaps[order[1:]], between)
        gaps[order[:-1]] = np.minimum(gaps[order[:-1]], between)
        return np.clip(1 - gaps / PRESSURE_GAP, 0, 1) * ~self.finished

    @property
    def in_lane(self):
        return self.pit_state >= pit.IN_LANE

    @property
    def on_track(self):
        """On the tarmac, which includes the pit lane."""
        return (np.abs(self.lateral) <= self.track.width / 2) | self.in_lane

    @property
    def finished(self):
        return ~np.isnan(self.finish_time)

    @property
    def done(self):
        out = self.finished | self.retired
        if self.ghosts:
            return bool(out.all())
        if out.all() or self.time >= self.time_limit:
            return True
        return bool(self.finished.any() and self.time - np.nanmin(self.finish_time) >= AFTER_WINNER)

    def lap_of(self):
        """Laps completed by each car (0 on the way to the first line crossing)."""
        return np.maximum(np.floor(self.progress / self.track.length), 0).astype(int)

    def standings(self):
        """Car indices from first to last: finishers by finishing time, then everyone else by distance covered, and
        cars that retired last."""
        key = np.where(self.finished, -1e9 + np.nan_to_num(self.finish_time), np.where(self.retired, 1e9, 0) - self.progress)
        return np.argsort(key, kind='stable')

    def step(self, steer, throttle, brake, pit_call=None, plan: PitPlan | None = None) -> TickEvents:
        """One tick. `pit_call` says which cars want to come in; it only counts once a lap, when a car first
        reaches the decision point before the pit entry (see pit.py). `plan` is what the crew does at each stop."""
        events = TickEvents()
        if self.done:
            return events
        laps_before = self.lap_of()
        distance_before = self.progress % self.track.length
        if pit_call is not None:
            deciding = self.pit.in_window(distance_before) & ~self.pit_decided & (self.pit_state == pit.RACING) & ~self.finished
            self.pit_state = np.where(deciding & np.asarray(pit_call, dtype=bool), pit.CALLED, self.pit_state)
            self.pit_decided |= deciding
        if self.driving is not None:
            steer, throttle, brake = self.driving.apply(steer, throttle, brake, self.pressure(), self.time)
            events.mistakes = [int(car) for car in np.flatnonzero(self.driving.just_erred & ~self.finished & ~self.retired)]
        # A car that has taken the flag coasts to a stop, a retired one pulls up; cars in the lane are driven by it,
        # not by their drivers.
        driven = ~self.in_lane
        stopping = self.finished | self.retired
        throttle = np.where(stopping | ~driven, 0.0, throttle)
        brake = np.where(stopping, 0.3, np.where(driven, brake, 0.0))
        steer = np.where(driven & ~self.retired, steer, 0.0)
        fuel, wear = self.cars.fuel.copy(), self.cars.tyre_wear.copy()
        step_cars(self.cars, self.specs, steer, throttle, brake, self.on_track, self.conditions())
        self.weather.step(DT)
        self.fuel_used += fuel - self.cars.fuel
        shed = np.maximum(self.cars.tyre_wear - wear, 0.0)
        self.tyre_used += self.cars.tyre_wear - wear
        events.contacts = self._resolve_contacts()
        self.new_debris = np.zeros((0, 2))
        for contact in events.contacts:
            self._harm(contact)
        if self.props is not None:
            self._hit_props(events)
        self._run_over_debris(events)
        self._fail_engines(events)
        self._drive_pit_lane(plan or PitPlan.standard(self.count), events)

        self.track_index, distance, self.lateral = self.track.locate(self.cars.position, self.track_index)
        # Progress follows the change in distance along the loop, wrapped so crossing the line counts as a small
        # step forward, not a whole lap back.
        before = (self.progress % self.track.length)
        change = (distance - before + self.track.length / 2) % self.track.length - self.track.length / 2
        self.progress = self.progress + change
        self._cross_pit_entry(distance_before, events)
        self.surface.update(self.progress % self.track.length, self.lateral, self.on_track & ~self.in_lane, shed, self.weather.rain, DT)
        self.tick += 1
        self.off_track_ticks += ~self.on_track
        if self.props is not None:
            self._retire_stuck(events)

        crossed = (self.lap_of() > laps_before) & ~self.finished
        events.laps = list(np.flatnonzero(crossed))
        finishing = crossed & (self.progress >= self.laps * self.track.length)
        self.finish_time[finishing] = self.time
        events.finished = list(np.flatnonzero(finishing))
        return events

    def place_props(self, props):
        """Puts the circuit's props (scenery.place) where the cars can hit them. Races without them, like training's,
        have nothing to hit beyond the grass."""
        self.props = Obstacles(props)
        self.furthest = self.progress.copy()
        self.gained_at = np.zeros(self.count)

    def _hit_props(self, events):
        props, cars = self.props, self.cars
        # Props all stand off the tarmac, so a car well inside its edges can't be touching one.
        close = np.flatnonzero(~self.in_lane & (np.abs(self.lateral) > self.track.width / 2 - FOOTPRINT_REACH - 1.0))
        for car in close:
            centre = cars.position[car] + cars.forward[car] * FOOTPRINT_OFFSET
            for prop, away, depth, point in props.touching(centre, cars.forward[car], cars.left[car], FOOTPRINT_HALF_LENGTH, FOOTPRINT_HALF_WIDTH):
                closing = float(np.dot(cars.velocity[car], away))
                if props.breaks(prop):
                    props.broken[prop] = True
                    impulse = abs(min(closing, 0.0)) * PROPS['breaks_knock']
                    cars.velocity[car] *= 1 - PROPS['breaks_speed']
                    events.prop_hits.append(PropHit(int(car), int(prop), impulse, point, True))
                    self._harm_by_prop(car, impulse, point)
                    continue
                soft = props.hardness[prop] == SOFT
                cars.position[car] += away * depth
                impulse = 0.0
                if closing < 0:
                    impulse = -(1 + (PROPS['soft_restitution'] if soft else RESTITUTION)) * closing
                    sliding = cars.velocity[car] - away * closing
                    kept = 1 - min(PROPS['scrape_most'], impulse * (PROPS['soft_absorbs'] if soft else PROPS['scrape']))
                    cars.velocity[car] = sliding * kept + away * (impulse + closing)
                events.prop_hits.append(PropHit(int(car), int(prop), impulse, point, False))
                self._harm_by_prop(car, impulse * (PROPS['soft_harm'] if soft else 1.0), point)
                if np.linalg.norm(cars.velocity[car]) < RECOVERY['below']:
                    self._back_off(car, away)
                if impulse >= CRASH_OUT and not self.retired[car]:
                    self.retired[car] = True
                    events.crashed.append(int(car))

    def _back_off(self, car, away):
        cars = self.cars
        tangent = self.track.tangents[self.track_index[car]]
        turn = np.arctan2(tangent[1], tangent[0]) - cars.heading[car]
        turn = np.arctan2(np.sin(turn), np.cos(turn))
        cars.heading[car] += np.clip(turn, -RECOVERY['turn'] * DT, RECOVERY['turn'] * DT)
        cars.position[car] += away * RECOVERY['back'] * DT

    def _harm_by_prop(self, car, impulse, point):
        """Damage from a prop, as from another car: general damage, and to the part of the car that hit."""
        cars = self.cars
        self.contact_ticks[car] += 1
        harm = DAMAGE['per_impulse'] * max(impulse - DAMAGE['gentle'], 0.0)
        cars.damage[car] = min(1.0, cars.damage[car] + harm)
        towards = np.asarray(point) - cars.position[car]
        towards /= max(np.linalg.norm(towards), 1e-6)
        if np.dot(towards, cars.forward[car]) > 0.7:
            cars.wing_damage[car] = min(1.0, cars.wing_damage[car] + PART_HARM * harm)
        elif abs(np.dot(towards, cars.left[car])) > 0.7:
            cars.suspension_damage[car] = min(1.0, cars.suspension_damage[car] + PART_HARM * harm)
        if impulse >= DEBRIS['impulse']:
            pieces = int(min(DEBRIS['most'], np.ceil(impulse * DEBRIS['per_impulse'])))
            landed = np.asarray(point) + self.rng.normal(0, DEBRIS['spread'], (pieces, 2))
            self.debris = np.concatenate([self.debris, landed])
            self.new_debris = np.concatenate([self.new_debris, landed])

    def _retire_stuck(self, events):
        """A car that hasn't got STUCK_METRES further in STUCK_SECONDS is out: stuck against, or beached by,
        something it hit."""
        gaining = self.progress > self.furthest + STUCK_METRES
        self.furthest = np.where(gaining, self.progress, self.furthest)
        self.gained_at = np.where(gaining | self.in_lane, self.time, self.gained_at)
        stuck = ~self.finished & ~self.retired & (self.time - self.gained_at > STUCK_SECONDS)
        self.retired |= stuck
        events.crashed += [int(car) for car in np.flatnonzero(stuck)]

    def _harm(self, contact):
        """Damage from one contact: general damage to both cars, and to whichever part of each car was hit.
        A hard enough hit leaves debris."""
        cars = self.cars
        pair = [contact.first, contact.second]
        self.contact_ticks[pair] += 1
        harm = DAMAGE['per_impulse'] * max(contact.impulse - DAMAGE['gentle'], 0.0)
        cars.damage[pair] = np.minimum(1.0, cars.damage[pair] + harm)
        for car, other in ((contact.first, contact.second), (contact.second, contact.first)):
            towards = cars.position[other] - cars.position[car]
            towards /= max(np.linalg.norm(towards), 1e-6)
            if np.dot(towards, cars.forward[car]) > 0.7:
                cars.wing_damage[car] = min(1.0, cars.wing_damage[car] + PART_HARM * harm)
            elif abs(np.dot(towards, cars.left[car])) > 0.7:
                cars.suspension_damage[car] = min(1.0, cars.suspension_damage[car] + PART_HARM * harm)
        if contact.impulse >= DEBRIS['impulse']:
            pieces = int(min(DEBRIS['most'], np.ceil(contact.impulse * DEBRIS['per_impulse'])))
            landed = np.asarray(contact.point) + self.rng.normal(0, DEBRIS['spread'], (pieces, 2))
            self.debris = np.concatenate([self.debris, landed])
            self.new_debris = np.concatenate([self.new_debris, landed])

    def _run_over_debris(self, events):
        """Each tick a car's wheels are over a piece of debris, there's a chance it cuts a tyre; the piece is gone
        either way once a car has run over it."""
        if not len(self.debris):
            return
        racing = ~self.finished & ~self.retired & ~self.in_lane
        reach = np.linalg.norm(self.debris[None, :, :] - self.cars.position[:, None, :], axis=2) < DEBRIS['reach']
        reach &= racing[:, None]
        for car in np.flatnonzero(reach.any(axis=1)):
            if not self.cars.punctured[car] and self.rng.random() < DEBRIS['puncture'] * reach[car].sum():
                self.cars.punctured[car] = True
                events.punctures.append(int(car))
        self.debris = self.debris[~reach.any(axis=0)]

    def _fail_engines(self, events):
        """A hot engine can fail; its car is out of the race."""
        heat = np.clip((self.cars.engine_temp - ENGINE_FAILURE['from']) / (1 - ENGINE_FAILURE['from']), 0, 1)
        failing = ~self.cars.engine_failed & ~self.finished & (self.rng.random(self.count) < ENGINE_FAILURE['rate'] * heat ** 2 * DT)
        self.cars.engine_failed |= failing
        self.retired |= failing
        events.engine_failures += [int(car) for car in np.flatnonzero(failing)]

    def _cross_pit_entry(self, distance_before, events):
        """Cars that passed the pit entry this tick, judged from where they really were before and after it (on
        the inside of a bend a car gains more centre-line distance than its speed suggests, so a crossing guessed
        from speed alone can slip through): their lap's decision is done with, and called cars turn in."""
        lane = self.pit
        distance_after = self.progress % self.track.length
        crossed = lane.entered(distance_before, distance_after)
        self.pit_decided &= ~crossed
        # A car off in the grass can't turn in; it stays called and comes in next time round.
        within_reach = np.abs(self.lateral) <= self.track.width / 2 + pit.ENTRY_REACH
        for car in np.flatnonzero(crossed & within_reach & (self.pit_state == pit.CALLED)):
            self.pit_state[car] = pit.IN_LANE
            self.lane_along[car] = (distance_after[car] - lane.entry) % self.track.length
            self.lane_speed[car] = max(float(self.cars.speed[car]), 0.0)
            self.lane_start[car] = self.lateral[car]
            events.pit_entered.append(int(car))

    def _drive_pit_lane(self, plan, events):
        cars, lane = self.cars, self.pit
        for car in np.flatnonzero(self.pit_state == pit.STOPPED):
            self.service_left[car] -= DT
            if self.service_left[car] <= 0:
                pit.apply_service(self.service_jobs[car]['plan'], cars, car, self.service_jobs[car]['jobs'])
                self.pit_state[car] = pit.IN_LANE
                events.pit_released.append(int(car))
        # Front of the queue first, so each car knows how fast the one ahead of it is going this tick.
        for car in sorted(np.flatnonzero(self.pit_state == pit.IN_LANE), key=lambda car: -self.lane_along[car]):
            box, along = lane.boxes[car], self.lane_along[car]
            serviced = self.service_jobs[car] is not None
            stopping_here = along < box and not serviced
            target = pit.SPEED_LIMIT
            if stopping_here:
                # Brake for the box, moving over into the working lane on the way in.
                target = min(target, np.sqrt(2 * pit.SLOW_DOWN * max(box - along, 0.0)))
                self.lane_shift[car] = np.clip(1 - (box - along) / pit.TURN_IN, 0, 1)
            elif serviced and along < box + pit.PULL_OUT:
                # Released: wait in the box for a gap in the fast lane, then pull back out into it.
                waiting = self.lane_speed[car] == 0 and along <= box + 1e-6 and self._fast_lane_blocked(car)
                target = 0.0 if waiting else target
                self.lane_shift[car] = np.clip(1 - (along - box) / pit.PULL_OUT, 0, 1)
            else:
                self.lane_shift[car] = 0.0
            room = self._room_ahead(car)
            braking = pit.HARD_BRAKING if room < target and room < self.lane_speed[car] else pit.SLOW_DOWN
            target = min(target, room)
            speed = self.lane_speed[car]
            speed = max(target, speed - braking * DT) if speed > target else min(target, speed + 8.0 * DT)
            self.lane_speed[car] = speed
            self.lane_along[car] += speed * DT
            self._keep_behind(car)
            if stopping_here and self.lane_along[car] >= box - 0.15:
                self.lane_along[car] = box
                self.lane_speed[car] = 0.0
                self.lane_shift[car] = 1.0
                jobs, standing = pit.service_times(plan, cars, car)
                frozen = PitPlan(*(np.array(getattr(plan, name)) for name in PitPlan.__dataclass_fields__))
                self.service_jobs[car] = {'jobs': jobs, 'plan': frozen}
                self.service_left[car] = standing
                self.pit_state[car] = pit.STOPPED
                self.stops[car] += 1
                events.pit_stopped.append((int(car), jobs, round(standing, 2)))
            if self.lane_along[car] >= lane.length:
                self.pit_state[car] = pit.RACING
                self.service_jobs[car] = None
                self.lane_shift[car] = 0.0
                events.pit_exited.append(int(car))
        for car in np.flatnonzero(self.in_lane | np.isin(np.arange(self.count), events.pit_exited)):
            along = np.array([min(self.lane_along[car], lane.length)])
            distance = lane.distance(along)
            ahead = lane.distance(along + 1.0)
            shift = self.lane_shift[car]
            point = self.track.point_at(distance, lane.lateral(along, self.lane_start[car], shift))[0]
            next_point = self.track.point_at(ahead, lane.lateral(along + 1.0, self.lane_start[car], shift))[0]
            direction = (next_point - point) / max(np.linalg.norm(next_point - point), 1e-6)
            cars.position[car] = point
            cars.heading[car] = np.arctan2(direction[1], direction[0])
            cars.velocity[car] = direction * self.lane_speed[car]

    def _lane_neighbours(self, car):
        """The other cars in the pit lane (driving or in their boxes): who, how far along, and whether each is in the
        same lane as `car`. Ghosts race alone, so they never share the lane."""
        if self.ghosts:
            return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=bool)
        others = np.flatnonzero(self.in_lane & (np.arange(self.count) != car))
        same_lane = np.abs(self.lane_shift[others] - self.lane_shift[car]) < 0.5
        return others, self.lane_along[others], same_lane

    def _room_ahead(self, car):
        """The fastest `car` may go without closing to within QUEUE_GAP of the car ahead in its lane; closer than that
        (two cars that came in together), slower than the car ahead until the gap opens."""
        others, along, same_lane = self._lane_neighbours(car)
        ahead = same_lane & (along > self.lane_along[car])
        if not ahead.any():
            return np.inf
        nearest = np.argmin(np.where(ahead, along, np.inf))
        gap = (along[nearest] - self.lane_along[car]) * self._lane_stretch(car)
        speed_ahead = self.lane_speed[others[nearest]]
        if gap < pit.QUEUE_GAP:
            return max(0.0, speed_ahead - pit.BACK_OFF * (pit.QUEUE_GAP - gap))
        return speed_ahead + np.sqrt(2 * pit.SLOW_DOWN * (gap - pit.QUEUE_GAP))

    def _keep_behind(self, car):
        """However the speeds worked out, a car never ends up inside the one ahead of it in its lane: it stays a car
        length back and no faster."""
        others, along, same_lane = self._lane_neighbours(car)
        ahead = same_lane & (along >= self.lane_along[car] - 1e-9)
        if not ahead.any():
            return
        nearest = np.argmin(np.where(ahead, along, np.inf))
        limit = along[nearest] - pit.NOSE_TO_TAIL / self._lane_stretch(car)
        if self.lane_along[car] > limit:
            self.lane_along[car] = max(limit, 0.0)
            self.lane_speed[car] = min(self.lane_speed[car], self.lane_speed[others[nearest]])

    def _lane_stretch(self, car):
        """Metres of `car`'s actual path per metre along the centre line, where it is in the lane."""
        return float(self.pit.stretch(np.array([self.lane_along[car]]), self.lane_start[car], self.lane_shift[car])[0])

    def _fast_lane_blocked(self, car):
        """Whether a car in the fast lane is too close behind or ahead for `car` to pull out of its box."""
        others, along, _ = self._lane_neighbours(car)
        in_fast_lane = self.lane_shift[others] < 0.5
        behind, ahead = pit.MERGE_CLEAR
        offset = along - self.lane_along[car]
        return bool((in_fast_lane & (offset > -behind) & (offset < ahead)).any())

    def _resolve_contacts(self):
        if self.ghosts:
            return []
        cars = self.cars
        forward, left = cars.forward, cars.left
        centres = cars.position + forward * FOOTPRINT_OFFSET
        gaps = centres[:, None, :] - centres[None, :, :]
        # Only footprints whose surrounding circles meet can touch; the exact test is for those few pairs.
        near = (np.linalg.norm(gaps, axis=2) < 2 * FOOTPRINT_REACH) & ~self.in_lane[:, None] & ~self.in_lane[None, :]
        contacts = []
        for first, second in zip(*np.nonzero(np.triu(near, 1))):
            touching = _footprint_overlap(centres[first], forward[first], left[first], centres[second], forward[second], left[second])
            if touching is None:
                continue
            normal, overlap = touching
            cars.position[first] += normal * overlap / 2
            cars.position[second] -= normal * overlap / 2
            closing = np.dot(cars.velocity[first] - cars.velocity[second], normal)
            impulse = 0.0
            if closing < 0:
                impulse = -(1 + RESTITUTION) * closing / 2
                cars.velocity[first] += normal * impulse
                cars.velocity[second] -= normal * impulse
            # Where they meet: between the two centres, on the line where the footprints overlap.
            middle = (centres[first] + centres[second]) / 2
            surface = np.dot(centres[first], normal) - _reach_along(forward[first], left[first], normal) + overlap / 2
            point = middle + normal * (surface - np.dot(middle, normal))
            contacts.append(Contact(int(first), int(second), float(abs(impulse) * 2), point))
        return contacts


def _reach_along(forward, left, axis):
    """How far a footprint reaches from its centre along `axis`."""
    return FOOTPRINT_HALF_LENGTH * abs(np.dot(forward, axis)) + FOOTPRINT_HALF_WIDTH * abs(np.dot(left, axis))


def _footprint_overlap(first_centre, first_forward, first_left, second_centre, second_forward, second_left):
    """Whether two footprints overlap, by the separating axis test (two rectangles are apart exactly when some
    edge direction of one of them separates them). Returns the direction to push the first car out (a unit vector
    from the second towards the first) and how deep they overlap, or None when they're apart."""
    between = first_centre - second_centre
    best = None
    for axis in (first_forward, first_left, second_forward, second_left):
        overlap = (_reach_along(first_forward, first_left, axis) + _reach_along(second_forward, second_left, axis)
                   - abs(np.dot(between, axis)))
        if overlap <= 0:
            return None
        if best is None or overlap < best[1]:
            best = (axis if np.dot(between, axis) >= 0 else -axis, overlap)
    return best
