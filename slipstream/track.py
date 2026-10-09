"""Procedurally generated closed circuits.

A track is a smooth loop through randomly placed control points, sampled every `SPACING` metres along its
centre line. Index 0 is the start/finish line, placed at the end of the longest straight so the grid lines up
on it. Everything is in metres; the track's left is the direction of `normals`.

Every circuit has at least one corner with some character: a sharp turn (a straight into a corner that turns hard
in a few metres, made by pinning the line close either side of a control point) or a chicane (a quick left-right
jink laid into a straight other than the main one). `features` finds them on a finished track.
"""
from dataclasses import dataclass

import numpy as np

SPACING = 2.0          # metres between centre-line samples
WIDTH = 14.0           # tarmac width
MIN_RADIUS = 22.0      # tightest corner allowed, measured on the centre line
CLEARANCE = 30.0       # minimum gap between the centre lines of two separate parts of the track
LENGTH_RANGE = (900.0, 1700.0)
GRID_STRAIGHT = 160.0  # how much straight to look for before the start line
GRID_MIN_RADIUS = 60.0 # the grid behind the line must be at least this straight
SHARP = {'turn': np.radians(55), 'pins': (45.0, 12.0), 'exit': 25.0, 'room': 70.0, 'second': 0.3}   # see _with_sharp_turns
CHICANE = {'length': (70.0, 100.0), 'swerve': (8.0, 12.0), 'margin': 25.0, 'chance': 0.6, 'straight': 1 / 250}
# What counts (features): a sharp turn turns 90 degrees or more within 45 m, straight after 80 m of near-straight;
# a chicane is two bends the opposite way round, each turning 25 degrees within 25 m, their middles 45 m apart at
# most. Plain circuits meet these only now and then (27 of 60 did); every circuit here has at least one.
FEATURE = {'sharp_turn': np.radians(90), 'sharp_within': 45.0, 'approach': 80.0, 'bend': np.radians(25), 'bend_within': 25.0, 'bends_apart': 45.0}


@dataclass(frozen=True)
class Track:
    seed: int
    points: np.ndarray     # (M, 2) centre line
    tangents: np.ndarray   # (M, 2) unit direction of travel
    normals: np.ndarray    # (M, 2) unit vector to the left of travel
    curvature: np.ndarray  # (M,) signed, 1/m; positive turns left
    width: float

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        return self.size * SPACING

    def point_at(self, distance, lateral=0.0):
        """World position `distance` metres along the centre line (wrapping), `lateral` metres to its left."""
        distance = np.asarray(distance, dtype=float)
        index = np.floor(distance / SPACING).astype(int) % self.size
        along = distance - np.floor(distance / SPACING) * SPACING
        return self.points[index] + self.tangents[index] * along[..., None] + self.normals[index] * np.asarray(lateral, dtype=float)[..., None]

    def heading_at(self, distance):
        index = np.floor(np.asarray(distance, dtype=float) / SPACING).astype(int) % self.size
        return np.arctan2(self.tangents[index, 1], self.tangents[index, 0])

    def locate(self, positions, hints=None, behind=10, ahead=20):
        """Where each position is relative to the track: (nearest sample index, distance along the loop,
        lateral offset to the left of the centre line). With `hints` (each car's index last tick) only a window
        of samples around it is searched, which is what keeps a car from snapping to a nearby part of the track.
        """
        positions = np.asarray(positions, dtype=float)
        if hints is None:
            gaps = positions[:, None, :] - self.points[None, :, :]
            index = np.argmin(np.einsum('cmk,cmk->cm', gaps, gaps), axis=1)
        else:
            window = (np.asarray(hints)[:, None] + np.arange(-behind, ahead + 1)[None, :]) % self.size
            gaps = positions[:, None, :] - self.points[window]
            index = window[np.arange(len(positions)), np.argmin(np.einsum('cwk,cwk->cw', gaps, gaps), axis=1)]
        offset = positions - self.points[index]
        along = np.einsum('ck,ck->c', offset, self.tangents[index])
        lateral = np.einsum('ck,ck->c', offset, self.normals[index])
        return index, (index * SPACING + along) % self.length, lateral


def generate_track(seed: int, attempts: int = 500) -> Track:
    """A new circuit for `seed`. Layouts that come out too tight, too short or too long, or that pass too close
    to themselves, are thrown away and drawn again from the same seed's stream, so a seed always gives the same
    track."""
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        centre = _centre_line(rng)
        if centre is None:
            continue
        track = _finish(seed, centre)
        grid = track.curvature[-int(GRID_STRAIGHT * 0.5 / SPACING):]
        if np.abs(grid).max() < 1 / GRID_MIN_RADIUS:
            return track
    raise RuntimeError(f'No valid track found for seed {seed} in {attempts} attempts')


def _centre_line(rng):
    count = int(rng.integers(7, 13))
    angles = np.sort((np.arange(count) + rng.uniform(-0.35, 0.35, count)) * 2 * np.pi / count)
    scale = rng.uniform(170.0, 260.0)
    # Points pulled well in toward the middle make the hairpins and esses; the rest stay near the rim.
    radii = scale * np.where(rng.random(count) < 0.35, rng.uniform(0.3, 0.6, count), rng.uniform(0.75, 1.0, count))
    # Stretch one way so circuits aren't all round blobs.
    stretch = rng.uniform(0.7, 1.3)
    controls = np.stack([np.cos(angles) * radii * stretch, np.sin(angles) * radii / stretch], axis=1)
    controls = _with_straights(controls, rng)
    controls = _with_sharp_turns(controls, rng)
    dense = _catmull_rom_loop(controls, samples_per_segment=40)
    centre = _resample(dense, SPACING)
    if centre is None:
        return None
    centre = _with_chicanes(centre, rng)
    if centre is None:
        return None
    centre = _ease_tight_corners(centre)
    if centre is None:
        return None
    length = len(centre) * SPACING
    if not LENGTH_RANGE[0] <= length <= LENGTH_RANGE[1]:
        return None
    if _passes_too_close(centre):
        return None
    # Easing can round a sharp turn or a chicane off; a circuit that ends up with neither is drawn again.
    if not _features(_curvature(_tangents(centre))):
        return None
    return centre


def _with_straights(controls, rng):
    """Lines up two extra control points between some neighbours: a spline through four points in a row runs
    straight between them, which gives the circuit somewhere to overtake."""
    result = []
    spans = np.linalg.norm(np.roll(controls, -1, axis=0) - controls, axis=1)
    longest = int(np.argmax(spans))
    for index, point in enumerate(controls):
        result.append(point)
        following = controls[(index + 1) % len(controls)]
        # The longest gap always becomes a straight, so every circuit has somewhere to put the grid.
        if index == longest or (spans[index] > 120.0 and rng.random() < 0.5):
            result.extend([point + (following - point) * 0.3, point + (following - point) * 0.7])
    return np.array(result)


def _with_sharp_turns(controls, rng):
    """Makes one corner (sometimes two) sharp: where the line already turns by SHARP['turn'] or more at a control
    point with room either side, extra points pinned along the straight line in (SHARP['pins'] metres before it)
    and out (SHARP['exit'] after) make the spline run straight into the corner and turn hard in a few metres."""
    count = len(controls)
    incoming = controls - np.roll(controls, 1, axis=0)
    outgoing = np.roll(controls, -1, axis=0) - controls
    lengths_in, lengths_out = np.linalg.norm(incoming, axis=1), np.linalg.norm(outgoing, axis=1)
    cosine = np.einsum('mk,mk->m', incoming, outgoing) / np.maximum(lengths_in * lengths_out, 1e-9)
    turn = np.arccos(np.clip(cosine, -1, 1))
    candidates = np.flatnonzero((turn >= SHARP['turn']) & (lengths_in > SHARP['room']) & (lengths_out > SHARP['room']))
    if not len(candidates):
        return controls
    chosen = set(rng.choice(candidates, size=min(len(candidates), 2 if rng.random() < SHARP['second'] else 1), replace=False).tolist())
    result = []
    for index in range(count):
        point = controls[index]
        if index in chosen:
            result.extend(point - incoming[index] / lengths_in[index] * distance for distance in SHARP['pins'])
            result.append(point)
            result.append(point + outgoing[index] / lengths_out[index] * SHARP['exit'])
        else:
            result.append(point)
    return np.array(result)


def _with_chicanes(centre, rng):
    """Lays a chicane into some straights (always one, where any straight has room): a smooth jink left, then right,
    then back onto the line, CHICANE['length'] long and up to CHICANE['swerve'] off it. On the longest straight it
    goes in the first part only, leaving GRID_STRAIGHT clear for the grid."""
    curvature = _curvature(_tangents(centre))
    straight = np.abs(curvature) < CHICANE['straight']
    runs, start = [], None
    # Walk the loop from a bend, so a straight that wraps past index 0 is found whole.
    first = int(np.flatnonzero(~straight)[0]) if (~straight).any() else 0
    order = (np.arange(len(centre)) + first) % len(centre)
    for position, index in enumerate(order):
        if straight[index] and start is None:
            start = position
        if (not straight[index] or position == len(order) - 1) and start is not None:
            runs.append(order[start:position])
            start = None
    runs.sort(key=len, reverse=True)
    normals = np.stack([-_tangents(centre)[:, 1], _tangents(centre)[:, 0]], axis=1)
    shifted = centre.copy()
    grid = int(GRID_STRAIGHT / SPACING)
    margin = int(CHICANE['margin'] / SPACING)
    plans = []
    for order_index, run in enumerate(runs):
        samples = int(rng.uniform(*CHICANE['length']) / SPACING)
        # The longest straight keeps its last GRID_STRAIGHT clear, for the grid.
        room = len(run) - (grid if order_index == 0 else 0)
        if room >= samples + 2 * margin:
            plans.append((run[:room], samples, rng.random() < CHICANE['chance']))
    if plans and not any(wanted for _, _, wanted in plans):
        pick = int(rng.integers(len(plans)))
        plans[pick] = (*plans[pick][:2], True)
    for run, samples, wanted in plans:
        if not wanted:
            continue
        begin = int(rng.integers(margin, len(run) - samples - margin + 1))
        stretch = run[begin:begin + samples]
        progress = np.linspace(0, 1, samples)
        swerve = rng.uniform(*CHICANE['swerve']) * (1 if rng.random() < 0.5 else -1)
        # Left then right then back, starting and ending square to the straight.
        offset = swerve * np.sin(2 * np.pi * progress) * np.sin(np.pi * progress) ** 2
        shifted[stretch] = centre[stretch] + normals[stretch] * offset[:, None]
    return _resample(shifted, SPACING)


def features(track):
    """The track's sharp turns and chicanes, as (kind, index of where it is) pairs."""
    return _features(track.curvature)


def _features(curvature):
    """Sharp turns and chicanes as FEATURE defines them, one entry per corner."""
    size = len(curvature)

    def turned(metres):
        span = int(metres / SPACING)
        return np.convolve(np.concatenate([curvature, curvature[:span]]), np.ones(span), mode='valid')[:size] * SPACING

    found = []
    straight = np.abs(curvature) < CHICANE['straight']
    approach = int(FEATURE['approach'] / SPACING)
    for index in np.flatnonzero(np.abs(turned(FEATURE['sharp_within'])) >= FEATURE['sharp_turn']):
        if straight[(index - approach + np.arange(approach)) % size].mean() > 0.9:
            found.append(('sharp turn', int(index)))
    bend = turned(FEATURE['bend_within'])
    rights = np.flatnonzero(bend <= -FEATURE['bend'])
    reach = int(FEATURE['bends_apart'] / SPACING)
    for left in np.flatnonzero(bend >= FEATURE['bend']):
        apart = (rights - left + size // 2) % size - size // 2
        if np.any(np.abs(apart) <= reach):
            found.append(('chicane', int(left)))
    # One entry per corner: detections of a kind within 60 m of each other are the same one.
    merged = []
    for kind, index in sorted(found, key=lambda item: item[1]):
        if any(kind == other and (index - where) % size < int(60 / SPACING) for other, where in merged):
            continue
        merged.append((kind, index))
    return merged


def _catmull_rom_loop(controls, samples_per_segment):
    """Centripetal Catmull-Rom through every control point and back to the first, which avoids cusps and
    self-intersections inside a segment."""
    count = len(controls)
    curve = []
    for start in range(count):
        p0, p1, p2, p3 = (controls[(start + offset) % count] for offset in (-1, 0, 1, 2))
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = np.linspace(t1, t2, samples_per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        curve.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.concatenate(curve)


def _ease_tight_corners(points, rounds=400):
    """Smooths only where a corner is tighter than MIN_RADIUS, a little at a time, so hairpins open up to a
    drivable radius instead of the whole layout being thrown away. None if it won't settle."""
    for _ in range(rounds):
        tight = np.abs(_curvature(_tangents(points))) > 1 / MIN_RADIUS
        if not tight.any():
            return points
        # Widen the patch a few samples each side so the corner eases in rather than kinking at its edges.
        patch = np.convolve(np.concatenate([tight[-4:], tight, tight[:4]]).astype(float), np.ones(9), mode='valid') > 0
        neighbours = (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) / 2
        points = np.where(patch[:, None], points + 0.5 * (neighbours - points), points)
        points = _resample(points, SPACING)
        if points is None:
            return None
    return None


def _resample(loop, spacing):
    """Evenly spaced points along a closed polyline, the last one `spacing` short of the first."""
    closed = np.vstack([loop, loop[:1]])
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    count = int(cumulative[-1] // spacing)
    if count < 10:
        return None
    targets = np.arange(count) * cumulative[-1] / count
    return np.stack([np.interp(targets, cumulative, closed[:, axis]) for axis in (0, 1)], axis=1)


def _tangents(points):
    forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    return forward / np.linalg.norm(forward, axis=1, keepdims=True)


def _curvature(tangents):
    following = np.roll(tangents, -1, axis=0)
    turn = np.arctan2(tangents[:, 0] * following[:, 1] - tangents[:, 1] * following[:, 0], np.einsum('mk,mk->m', tangents, following))
    return turn / SPACING


def _passes_too_close(points):
    """True when two parts of the loop that are far apart along it come close in space (a crossing, or tarmac
    and run-off overlapping)."""
    coarse = points[::3]
    step = 3 * SPACING
    gaps = np.linalg.norm(coarse[:, None, :] - coarse[None, :, :], axis=2)
    index = np.arange(len(coarse))
    along = np.abs(index[:, None] - index[None, :]) * step
    along = np.minimum(along, len(coarse) * step - along)
    separate = along > CLEARANCE * 2.5
    return bool(np.any(gaps[separate] < CLEARANCE))


def _finish(seed, centre):
    tangents = _tangents(centre)
    curvature = _curvature(tangents)
    # The start line goes 60% of the way down the straightest stretch: the grid fits on the straight behind it,
    # and there's still a run to the first corner.
    window = int(GRID_STRAIGHT / SPACING)
    bend = np.convolve(np.concatenate([np.abs(curvature), np.abs(curvature[:window])]), np.ones(window), mode='valid')[:len(centre)]
    start = (int(np.argmin(bend)) + int(window * 0.6)) % len(centre)
    centre = np.roll(centre, -start, axis=0)
    tangents = np.roll(tangents, -start, axis=0)
    curvature = np.roll(curvature, -start)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return Track(seed=seed, points=centre, tangents=tangents, normals=normals, curvature=curvature, width=WIDTH)
