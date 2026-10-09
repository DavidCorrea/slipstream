"""The props a race's cars can hit (placed by scenery.py), and which of them a car's footprint touches.

A car's footprint is the same rectangle the cars collide with each other by (race.py). Props are circles or boxes.
Finding them is cheap: small props are filed in a grid of CELL-metre squares, and only the few big ones (the
backdrop's mountains and towers, the grandstands) are checked directly.
"""
import numpy as np

from .scenery import BREAKABLE, SCENIC

CELL = 10.0
BIG = 30.0   # props reaching further than this from their middle are checked directly, not through the grid


class Obstacles:
    def __init__(self, props):
        self.props = list(props)
        count = len(self.props)
        self.centre = np.array([[prop.x, prop.y] for prop in self.props], dtype=float).reshape(count, 2)
        self.radius = np.array([prop.radius for prop in self.props], dtype=float)
        self.half = np.array([[prop.half_length, prop.half_width] for prop in self.props], dtype=float).reshape(count, 2)
        angles = np.array([prop.angle for prop in self.props], dtype=float)
        self.along = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.across = np.stack([-self.along[:, 1], self.along[:, 0]], axis=1)
        self.hardness = [prop.hardness for prop in self.props]
        self.reach = np.where(self.radius > 0, self.radius, np.hypot(self.half[:, 0], self.half[:, 1]))
        self.broken = np.zeros(count, dtype=bool)
        hittable = np.array([hardness != SCENIC for hardness in self.hardness], dtype=bool)
        self.big = np.flatnonzero(hittable & (self.reach > BIG))
        self.grid = {}
        for index in np.flatnonzero(hittable & (self.reach <= BIG)):
            low = np.floor((self.centre[index] - self.reach[index]) / CELL).astype(int)
            high = np.floor((self.centre[index] + self.reach[index]) / CELL).astype(int)
            for cell_x in range(low[0], high[0] + 1):
                for cell_y in range(low[1], high[1] + 1):
                    self.grid.setdefault((cell_x, cell_y), []).append(index)

    def breaks(self, index):
        return self.hardness[index] == BREAKABLE

    def near(self, point, reach):
        """Props, not yet broken, that might be within `reach` of `point`."""
        low = np.floor((point - reach) / CELL).astype(int)
        high = np.floor((point + reach) / CELL).astype(int)
        found = set()
        for cell_x in range(low[0], high[0] + 1):
            for cell_y in range(low[1], high[1] + 1):
                found.update(self.grid.get((cell_x, cell_y), ()))
        if len(self.big):
            close = np.linalg.norm(self.centre[self.big] - point, axis=1) < self.reach[self.big] + reach
            found.update(self.big[close].tolist())
        return [index for index in found if not self.broken[index]]

    def touching(self, centre, forward, left, half_length, half_width):
        """Every prop the footprint (a rectangle at `centre`, its length along `forward`) overlaps: (prop, the unit
        direction pushing the car away from it, how deep they overlap, where they touch)."""
        reach = float(np.hypot(half_length, half_width))
        touches = []
        for index in self.near(centre, reach):
            if self.radius[index] > 0:
                touch = _circle_overlap(centre, forward, left, half_length, half_width, self.centre[index], self.radius[index])
            else:
                touch = _boxes_overlap(centre, forward, left, half_length, half_width,
                                       self.centre[index], self.along[index], self.across[index], *self.half[index])
            if touch is not None:
                touches.append((index, *touch))
        return touches


def _circle_overlap(centre, forward, left, half_length, half_width, middle, radius):
    """A rectangle against a circle: the nearest point of the rectangle to the circle's middle decides."""
    offset = middle - centre
    local = np.array([np.dot(offset, forward), np.dot(offset, left)])
    nearest = np.clip(local, [-half_length, -half_width], [half_length, half_width])
    gap = local - nearest
    distance = float(np.hypot(*gap))
    if distance >= radius:
        return None
    if distance > 1e-6:
        towards = (gap[0] * forward + gap[1] * left) / distance
        depth = radius - distance
    else:
        # The middle is inside the rectangle: out the nearest side.
        room = np.array([half_length - abs(local[0]), half_width - abs(local[1])])
        axis = int(np.argmin(room))
        towards = (forward if axis == 0 else left) * (1.0 if local[axis] >= 0 else -1.0)
        depth = radius + room[axis]
    point = centre + nearest[0] * forward + nearest[1] * left
    return -towards, float(depth), point


def _boxes_overlap(centre, forward, left, half_length, half_width, middle, along, across, half_along, half_across):
    """Two rectangles, by the separating axis test: apart exactly when an edge direction of one separates them."""
    between = centre - middle
    reach = lambda length, width, first, second, axis: length * abs(np.dot(first, axis)) + width * abs(np.dot(second, axis))
    best = None
    for axis in (forward, left, along, across):
        depth = (reach(half_length, half_width, forward, left, axis) + reach(half_along, half_across, along, across, axis)
                 - abs(np.dot(between, axis)))
        if depth <= 0:
            return None
        if best is None or depth < best[1]:
            best = (axis if np.dot(between, axis) >= 0 else -axis, float(depth))
    away, depth = best
    # Where they touch: the car's side facing the prop, at the middle of the overlap.
    point = centre - away * (reach(half_length, half_width, forward, left, away) - depth / 2)
    return away, depth, point
