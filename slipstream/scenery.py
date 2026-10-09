"""Everything around the track, placed where the race can hit it.

A circuit's location (countryside, desert, alpine, coast, city, or the city at night) decides what stands around
it: trees, rocks, cacti, hay bales, buildings, lamps, palms, a backdrop of mountains or mesas or towers. The track
itself brings sponsor boards, braking-marker boards, marshal posts, camera towers, a sponsor bridge, barriers (armco
on the open circuits, concrete walls on the street ones) and tyre walls on the outside of tight corners. All of
it is placed here, from the circuit's seed, and every piece has a footprint the cars can hit (race.py): the viewer
draws exactly this list (web/scenery.js).

Each prop is `solid` (walls, buildings, trees, rocks: a car stops or bounces off), `soft` (tyre walls: they give,
and spare the car), `breakable` (boards, signs, lamps, cacti, bales: they break, slowing the car a little), or
`scenic` (the lake, the sea: only to look at).

The grandstands, the start gantry, the pit wall and the garages come from the track and the pit lane alone, and
the viewer builds them from those (web/circuit.js); `track_furniture` places their footprints the same way.

Footprints are circles (`radius`) or boxes (`half_length` along `angle`, `half_width` across it). `angle` is the
viewer's turn about the vertical: a box's length runs along (cos angle, sin angle) in the race's coordinates.
"""
from dataclasses import dataclass, field

import numpy as np

from .pit import RAMP
from .track import SPACING

LOCATIONS = ('countryside', 'desert', 'alpine', 'coast', 'city', 'night')
WALLED = ('city', 'night')
# How a prop takes a hit; SCENIC ones (the lake, the sea) are only to look at and have no footprint.
SOLID, SOFT, BREAKABLE, SCENIC = 'solid', 'soft', 'breakable', 'scenic'
PIT_WALL_CURVATURE = 1 / 150   # the pit wall stands only where the track beside it bends less than this (also web/circuit.js)


@dataclass
class Prop:
    kind: str
    x: float
    y: float
    hardness: str
    radius: float = 0.0          # a circle's, or 0 for a box
    half_length: float = 0.0
    half_width: float = 0.0
    angle: float = 0.0
    look: dict = field(default_factory=dict)   # what only the viewer needs: sizes, shades, which sponsor

    def describe(self, number):
        shape = {'radius': round(self.radius, 2)} if self.radius else {'halfLength': round(self.half_length, 2), 'halfWidth': round(self.half_width, 2)}
        return {'id': number, 'kind': self.kind, 'x': round(self.x, 2), 'y': round(self.y, 2), 'angle': round(self.angle, 3),
                'hardness': self.hardness, **shape, **self.look}


def location_for(seed):
    """The location a circuit gets when none is chosen."""
    return LOCATIONS[abs(int(seed)) % len(LOCATIONS)]


def place(track, lane, seed, location):
    """Every prop around the circuit for this location, scenery and trackside alike."""
    if location not in LOCATIONS:
        raise ValueError(f'Unknown location {location!r}; the locations are {", ".join(LOCATIONS)}')
    surroundings = _Surroundings(track, lane, np.random.default_rng(int(seed) + 3))
    builders = {'countryside': _countryside, 'desert': _desert, 'alpine': _alpine, 'coast': _coast,
                'city': lambda where: _city(where, night=False), 'night': lambda where: _city(where, night=True)}
    return builders[location](surroundings) + _trackside(surroundings, location in WALLED) + track_furniture(track, lane)


# ---- Where things can go -------------------------------------------------------------------------------------

class _Surroundings:
    def __init__(self, track, lane, rng):
        self.track, self.lane, self.rng = track, lane, rng
        self.half = track.width / 2
        self.centre = track.points.mean(axis=0)
        self.radius = float(np.linalg.norm(track.points - self.centre, axis=1).max())
        self.coarse = track.points[::4]
        # Scenery keeps clear of the track, and well clear of the pits and grandstands behind them.
        steps = lane.entry + lane.length * np.arange(12) / 11
        self.reserved = track.point_at(steps, np.full(12, lane.offset + lane.side * 25))
        self.clear_of_lane = _lane_clearance(track, lane)

    def away_from_track(self, spots):
        spots = np.atleast_2d(spots)
        from_track = np.linalg.norm(spots[:, None] - self.coarse[None], axis=2).min(axis=1)
        from_pits = np.linalg.norm(spots[:, None] - self.reserved[None], axis=2).min(axis=1) - 25
        return np.minimum(from_track, from_pits)

    def scatter(self, count, clear, reach=None):
        """Up to `count` spots at least `clear` metres from the track, within `reach` of its middle."""
        reach = self.radius + 320 if reach is None else reach
        tries = self.centre + self.rng.uniform(-reach, reach, (count * 6, 2))
        return tries[self.away_from_track(tries) > clear][:count]

    def around(self, count, clearance, footprint):
        """A ring of big shapes on the horizon, each `clearance` metres beyond the circuit's reach from its foot.
        `footprint(rng)` gives a piece's half size across and the rest of the prop; returns (spot, piece) pairs."""
        placed = []
        for index in range(count):
            angle = index / count * 2 * np.pi + self.rng.random() * 0.2
            reach, piece = footprint(self.rng)
            spot = self.centre + np.array([np.cos(angle), -np.sin(angle)]) * (self.radius + clearance + reach)
            placed.append((spot, piece))
        return placed

    def along_track(self, spacing, lateral):
        """Spots on both sides of the track every `spacing` samples, `lateral` metres out, facing across it."""
        spots = []
        for index in range(0, self.track.size, spacing):
            if not (self.clear_of_lane(index, 1) and self.clear_of_lane(index, -1)):
                continue
            (x, y), (nx, ny) = self.track.points[index], self.track.normals[index]
            for side in (1, -1):
                spots.append((x + nx * side * lateral, y + ny * side * lateral, float(np.arctan2(-ny * side, -nx * side))))
        return spots


def _lane_clearance(track, lane):
    """Whether the track sample at `index` is clear of the pit lane on `side` (1 left, -1 right): the lane runs where
    trackside things would otherwise go when the start straight sits next to a corner."""
    count, margin = track.size, 4
    first = int(np.floor(lane.entry / SPACING)) - margin
    span = int(np.ceil(lane.length / SPACING)) + 2 * margin
    return lambda index, side: side != lane.side or ((index - first) % count) > span


def _trees(where, count, pine_share):
    props = []
    for x, y in where.scatter(count, where.half + 30):
        scale = 0.7 + where.rng.random() * 0.9
        props.append(Prop('tree', x, y, SOLID, radius=0.35 * scale, angle=where.rng.random() * np.pi,
                          look={'scale': round(scale, 2), 'pine': bool(where.rng.random() < pine_share)}))
    return props


# ---- Locations -------------------------------------------------------------------------------------------------

def _countryside(where):
    props = _trees(where, 900, 0.35)
    for x, y in where.scatter(60, where.half + 40):
        # A bale lies on its side: its length runs along the angle.
        props.append(Prop('bale', x, y, BREAKABLE, half_length=0.65, half_width=0.75, angle=where.rng.random() * np.pi))
    return props


def _desert(where):
    props = []
    for x, y in where.scatter(260, where.half + 25):
        size = [1 + where.rng.random() * 3, 0.6 + where.rng.random() * 1.6, 1 + where.rng.random() * 3]
        props.append(Prop('rock', x, y, SOLID, radius=(size[0] + size[2]) / 2, angle=where.rng.random() * np.pi,
                          look={'size': [round(value, 2) for value in size]}))
    for x, y in where.scatter(200, where.half + 25):
        scale = 0.8 + where.rng.random() * 0.8
        props.append(Prop('cactus', x, y, BREAKABLE, radius=0.4 * scale, angle=where.rng.random() * np.pi, look={'scale': round(scale, 2)}))

    def mesa(rng):
        height, width = 50 + rng.random() * 110, 140 + rng.random() * 240
        return width * 0.55, {'height': round(height, 1), 'width': round(width, 1)}
    for (x, y), look in where.around(16, 150, mesa):
        props.append(Prop('mesa', x, y, SOLID, radius=look['width'] * 0.55, look=look))
    return props


def _alpine(where):
    props = _trees(where, 1400, 0.85)

    def mountain(rng):
        height, width = 260 + rng.random() * 300, 280 + rng.random() * 220
        return width, {'height': round(height, 1), 'width': round(width, 1), 'turn': round(rng.random() * np.pi, 3)}
    for (x, y), look in where.around(18, 150, mountain):
        props.append(Prop('mountain', x, y, SOLID, radius=look['width'], look=look))
    lake = where.scatter(1, where.half + 120, where.radius + 260)
    if len(lake):
        props.append(Prop('lake', lake[0][0], lake[0][1], SCENIC, look={'size': round(70 + where.rng.random() * 40, 1)}))
    return props


def _coast(where):
    # The sea fills one side of the world beyond a beach; palms stand on the land side, boats are moored off it.
    direction = where.rng.random() * 2 * np.pi
    outward = np.array([np.cos(direction), -np.sin(direction)])
    shore = where.centre + outward * (where.radius + 90)
    props = [Prop('sea', shore[0], shore[1], SCENIC, look={'direction': round(direction, 3)})]
    for x, y in where.scatter(500, where.half + 20):
        if np.dot(np.array([x, y]) - shore, outward) >= -10:
            continue
        scale = 0.8 + where.rng.random() * 0.6
        props.append(Prop('palm', x, y, SOLID, radius=0.3 * scale, angle=where.rng.random() * 2 * np.pi, look={'scale': round(scale, 2)}))
    along = np.array([-outward[1], outward[0]])
    for index in range(24):
        x, y = shore + outward * (50 + (index % 3) * 14) + along * (index - 12) * 9
        props.append(Prop('boat', x, y, SOLID, half_length=3, half_width=1.1, angle=-direction))
    return props


def _city(where, night):
    props = []
    spots = where.scatter(320, where.half + 16, where.radius + 260)
    for (x, y), away in zip(spots, where.away_from_track(spots) if len(spots) else []):
        # Low blocks along the track, so the racing stays in view from above; towers further back.
        back = min(away / 120, 1)
        height = 6 + where.rng.random() * 10 + back * where.rng.random() ** 2 * 80
        width, depth = 10 + where.rng.random() * 16, 10 + where.rng.random() * 16
        props.append(Prop('block', x, y, SOLID, half_length=width / 2, half_width=depth / 2, angle=where.rng.random() * np.pi,
                          look={'height': round(height, 1), 'width': round(width, 1), 'depth': round(depth, 1), 'shade': round(where.rng.random(), 3)}))

    def tower(rng):
        height, width, depth = 60 + rng.random() * 180, 30 + rng.random() * 40, 30 + rng.random() * 40
        return max(width, depth) / 2, {'height': round(height, 1), 'width': round(width, 1), 'depth': round(depth, 1)}
    for (x, y), look in where.around(40, 500, tower):
        props.append(Prop('skyline', x, y, SOLID, half_length=look['width'] / 2, half_width=look['depth'] / 2, look=look))
    # Street lamps along the track, closer together at night when they light it.
    for x, y, angle in where.along_track(16 if night else 28, where.half + 4):
        props.append(Prop('lamp', x, y, BREAKABLE, radius=0.2, angle=angle))
    return props


# ---- Trackside -------------------------------------------------------------------------------------------------

def _trackside(where, walled):
    track, rng, half, count = where.track, where.rng, where.half, where.track.size
    curvature = track.curvature
    clear = where.clear_of_lane
    straight = lambda index: abs(curvature[index % count]) < 1 / 300

    def at(index, lateral):
        return track.points[index % count] + track.normals[index % count] * lateral

    def heading(index):
        (x0, y0), (x1, y1) = track.points[index % count], track.points[(index + 1) % count]
        return float(np.arctan2(y1 - y0, x1 - x0))

    props = []
    # Sponsor boards every so often along straights, on both sides.
    for index in range(0, count, 14):
        if not straight(index):
            continue
        for side in (1, -1):
            if not clear(index, side) or rng.random() < 0.4:
                continue
            x, y = at(index, side * (half + (1.5 if walled else 9)))
            props.append(Prop('board', x, y, BREAKABLE, half_length=4.5, half_width=0.15, angle=heading(index) + (np.pi if side > 0 else 0),
                              look={'sponsor': int(rng.integers(1 << 16))}))
    # Braking boards (3, 2, 1) before tight corners, on the outside of the approach.
    index = 0
    while index < count:
        ahead = (index + 25) % count
        entering = abs(curvature[ahead]) > 1 / 40 and abs(curvature[index]) < 1 / 200
        already = abs(curvature[(index - 1) % count]) < 1 / 200 and abs(curvature[(ahead - 1) % count]) > 1 / 40
        side = -1 if curvature[ahead] > 0 else 1
        if entering and not already and clear(index, side):
            for step, number in enumerate((3, 2, 1)):
                x, y = at(index + step * 5, side * (half + 3))
                props.append(Prop('marker', x, y, BREAKABLE, half_length=0.55, half_width=0.1, angle=heading(index) - np.pi / 2, look={'number': number}))
            index += 30
        index += 1
    # Marshal posts at corners, and TV camera towers on the outside of a few.
    for index in range(0, count, 20):
        side = -1 if curvature[index] > 0 else 1
        if abs(curvature[index]) >= 1 / 70 and clear(index, side):
            x, y = at(index, side * (half + 14))
            props.append(Prop('marshal', x, y, SOLID, half_length=1.0, half_width=1.0, angle=heading(index) + (np.pi if side > 0 else 0),
                              look={'phase': round(rng.random() * 2 * np.pi, 3)}))
    for index in range(7, count, 55):
        side = -1 if curvature[index] > 0 else 1
        if abs(curvature[index]) >= 1 / 90 and clear(index, side):
            x, y = at(index, side * (half + 22))
            props.append(Prop('camera', x, y, SOLID, half_length=0.95, half_width=0.95, angle=heading(index)))
    # A sponsor bridge over the longest straight away from the start: its two legs stand either side.
    best, run, best_run = -1, 0, 0
    for index in range(int(count * 0.25), int(count * 0.8)):
        run = run + 1 if straight(index) else 0
        if run > best_run:
            best_run, best = run, index - run // 2
    if best_run > 20:
        span = half * 2 + 8
        for end in (-1, 1):
            x, y = at(best, end * span / 2)
            props.append(Prop('bridge', x, y, SOLID, half_length=0.6, half_width=0.6, angle=heading(best), look={'span': round(span, 2), 'end': end}))
    # Barriers: armco along the outside of straights, or concrete walls along both sides of a street circuit, as a
    # chain of short pieces from one sample to the next.
    offset = half + 3.2 if walled else half + 12
    for side in (1, -1):
        for index in range(count):
            keep = lambda sample: (walled or straight(sample)) and clear(sample % count, side)
            if not (keep(index) and keep(index + 1)):
                continue
            start, end = at(index, side * offset), at(index + 1, side * offset)
            middle, run = (start + end) / 2, end - start
            props.append(Prop('wall' if walled else 'armco', middle[0], middle[1], SOLID, half_length=float(np.linalg.norm(run)) / 2 + 0.05,
                              half_width=0.3 if walled else 0.15, angle=float(np.arctan2(run[1], run[0])), look={'side': side}))
    # Tyre walls on the outside of the tightest corners (a street circuit has its walls instead).
    if not walled:
        for index in range(0, count, 2):
            side = -1 if curvature[index] > 0 else 1
            if abs(curvature[index]) >= 1 / 38 and clear(index, side):
                x, y = at(index, side * (half + 19))
                props.append(Prop('tyres', x, y, SOFT, radius=0.62, look={'shade': index // 2 % 4}))
    return props


def track_furniture(track, lane):
    """The footprints of what the viewer builds from the track and the pit lane (web/circuit.js): the start gantry's
    two posts, the three grandstands behind the garages, the pit wall and the garages themselves."""
    half, props = track.width / 2, []
    (x, y), (nx, ny) = track.points[0], track.normals[0]
    for side in (-1, 1):
        reach = half + 3
        props.append(Prop('gantry', x + nx * side * reach, y + ny * side * reach, SOLID, half_length=0.4, half_width=0.4))
    outside, distance = lane.side, abs(lane.offset) + 22
    for offset in (-34, 0, 34):
        index = int(offset / 2 + track.size) % track.size
        (x, y), (nx, ny) = track.points[index], track.normals[index]
        # A stand rises away from the track from its front edge, about 14 m deep and 31 m wide.
        front = np.array([x + nx * outside * distance, y + ny * outside * distance])
        middle = front + np.array([nx, ny]) * outside * 6.9
        props.append(Prop('grandstand', middle[0], middle[1], SOLID, half_length=15.5, half_width=7.0,
                          angle=float(np.arctan2(track.tangents[index][1], track.tangents[index][0]))))
    for along in np.arange(RAMP, lane.length - RAMP, 6.0):
        length = min(6.0, lane.length - RAMP - along)
        middle = along + length / 2
        index = int(np.floor(((lane.entry + middle) % track.length) / SPACING)) % track.size
        # Only beside the straight: a big field's lane runs on into the first corner, where cars running wide
        # piled into a wall on the outside of it. There, run-off parts the lane from the track instead.
        if abs(track.curvature[index]) >= PIT_WALL_CURVATURE:
            continue
        x, y = track.point_at(lane.entry + middle, lane.side * (half + 3.4))
        tangent = track.tangents[index]
        props.append(Prop('pitwall', x, y, SOLID, half_length=length / 2, half_width=0.25, angle=float(np.arctan2(tangent[1], tangent[0]))))
    for along in lane.boxes:
        x, y = track.point_at(lane.entry + along, lane.offset + lane.side * 10.5)
        tangent = track.tangents[int(np.floor(((lane.entry + along) % track.length) / SPACING)) % track.size]
        props.append(Prop('garage', x, y, SOLID, half_length=5.7, half_width=3.7, angle=float(np.arctan2(tangent[1], tangent[0]))))
    return props
