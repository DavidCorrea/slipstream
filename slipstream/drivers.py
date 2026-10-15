"""A scripted driver that follows the centre line: no learning, and only the racecraft not to crash.

It aims at a point ahead on the centre line (pure pursuit) and picks a target speed from the corners coming
up: the fastest speed that grip allows through each one, plus what braking can shed before reaching it. It
proves the physics can be driven, and gives trained drivers a fixed opponent to beat.

It has traffic sense too: it keeps a braking distance from the car ahead in its path, and steers round a car
that is crawling or stopped in it. Without that a field of them drove into each other at the first corner and
piled up off the track, where each one stuck against another backed up and drove into it again: in 20-car races
with props only about a fifth of them finished.
"""
import numpy as np

from . import pit
from .car import BRAKE_WEAR, GRAVITY, MAX_STEER, OFF_TRACK, STEER_FADE_SPEED, WHEELBASE, effective_grip, effective_top_speed
from .observe import to_car_frame
from .track import SPACING

# Every driver, scripted or trained, decides every DECISION_TICKS physics ticks and holds its controls in between.
DECISION_TICKS = 2
LOOKAHEAD = (8.0, 0.55)   # metres, plus this many per m/s of speed
SPEED_HORIZON = 160.0     # metres of upcoming track considered for braking
CAUTION = 0.85            # share of the grip limit it's willing to use
REACTION = 0.3            # seconds ahead it aims its speed at, so the pedals act before the corner, not in it
# Off the tarmac it crawls back: on the grass, and above all on wet grass, full throttle would use up all the grip
# there is and leave none for turning back towards the track.
RECOVERY_SPEED = 9.0      # m/s
# Something in the way: a scripted driver always wants to go, so standing still this long means it's blocked (nosed
# into a wall, a car, a tree, or facing the wrong way after a spin). It backs up, swinging its nose round towards
# where it's going, for up to BACK_UP seconds or until it's pointing well enough, then drives on.
BLOCKED_SECONDS = 1.5
# Under the marshals (flags.py): it waits on the brakes for the lights to go out and reacts in this many seconds
# (each car its own, from its number), and shown blue it eases off to this share of its speed to let the car
# lapping it by.
START_REACTION = (0.15, 0.15)
BLUE_EASE = 0.85
# Traffic: a car in the way is one up to `looking` metres ahead, within `across` metres of its line; it keeps `gap`
# metres to it, braking with `braking` of what its brakes can do, and goes round one slower than `slow` of its own
# speed, aiming `beside` metres to the side of it.
TRAFFIC = {'looking': 60.0, 'across': 2.4, 'gap': 7.0, 'braking': 0.6, 'slow': 0.5, 'beside': 3.5}
BACK_UP = {'longest': 3.0, 'shortest': 0.8, 'aligned': 0.6}   # seconds, seconds, radians off the way ahead
# Driving the pit lane itself (in races that aren't assisted): from this far before the entry a called car moves
# over to the lane's side of the track, it aims this far under the limit and is down to it by the mouth of the lane
# (braking with this share of what grip and brakes allow), swings into its box over `turn_in` metres and back out
# over `pull_out`, and brakes gently enough for its box to stop on its marks.
PIT_APPROACH = {'from': 110.0, 'over': 70.0, 'edge': 2.5, 'under_limit': 1.0, 'turn_in': 13.0, 'pull_out': 14.0,
                'box_braking': 4.0, 'limit_braking': 0.6, 'lane_look': 4.0}


def scripted_controls(race):
    track, cars, specs = race.track, race.cars, race.specs
    speed = np.maximum(cars.speed, 0.0)
    distance = race.progress % track.length

    braking = specs.braking * (1 - BRAKE_WEAR['loss'] * cars.brake_wear)
    in_the_way, way_speed, going_round, around = _traffic(race, speed)
    target = track.point_at(distance + LOOKAHEAD[0] + LOOKAHEAD[1] * speed, around)
    to_target = target - cars.position
    angle = np.arctan2(to_target[:, 1], to_target[:, 0]) - cars.heading
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    reach = np.maximum(np.linalg.norm(to_target, axis=1), 1.0)
    wanted_lock = np.arctan(2 * WHEELBASE * np.sin(angle) / reach)
    steer = wanted_lock / (MAX_STEER / (1 + speed / STEER_FADE_SPEED))

    ahead = np.arange(0.0, SPEED_HORIZON, SPACING)
    indices = (np.floor((distance[:, None] + ahead[None, :]) / SPACING).astype(int)) % track.size
    curvature = np.abs(track.curvature[indices])
    grip = CAUTION * effective_grip(cars, specs, race.weather.wetness) * GRAVITY
    corner_speed = np.sqrt(grip[:, None] / np.maximum(curvature, 1e-4))
    profile = _speed_profile(corner_speed, curvature, grip, CAUTION * braking)
    lead = np.minimum((speed * REACTION / SPACING).astype(int), profile.shape[1] - 1)
    target_speed = np.minimum(profile[np.arange(len(speed)), lead], effective_top_speed(cars, specs))
    target_speed = np.where(race.on_track, target_speed, np.minimum(target_speed, RECOVERY_SPEED))
    if race.marshals is not None:
        target_speed = np.where(race.marshals.blue, target_speed * BLUE_EASE, target_speed)
    # Behind a car it isn't going round, no faster than lets it stop a gap short of it.
    following = way_speed + np.sqrt(2 * TRAFFIC['braking'] * braking * np.maximum(in_the_way - TRAFFIC['gap'], 0.0))
    target_speed = np.where(np.isfinite(in_the_way) & ~going_round, np.minimum(target_speed, following), target_speed)

    throttle = np.clip((target_speed - speed) / 1.0, 0, 1)
    # Brake only with the grip the corner it's in leaves over (the friction circle), or it can't turn.
    full_grip = grip / CAUTION
    cornering = speed ** 2 * curvature[:, 0]
    spare = np.sqrt(np.maximum(full_grip ** 2 - cornering ** 2, 0.0))
    brake = np.minimum(np.clip((speed - target_speed) / 0.5, 0, 1), spare / braking)
    steer = np.clip(steer, -1, 1)
    if not race.assisted:
        steer, throttle, brake = _pit_lane(race, speed, grip, np.minimum(braking, full_grip), target_speed, spare / braking, steer, throttle, brake)
    # Nobody reverses in the pit lane (a car standing in it is in its box, or waiting to be let out), nor on the
    # grid, where standing still is waiting for the lights.
    stopped_for = cars.stopped_for
    on_the_grid = np.zeros(race.count, dtype=bool)
    if race.marshals is not None:
        reaction = START_REACTION[0] + START_REACTION[1] * ((np.arange(race.count) * 0.618) % 1.0)
        going = race.marshals.lights_out_at + reaction
        on_the_grid = race.time < going
        throttle, brake = np.where(on_the_grid, 0.0, throttle), np.where(on_the_grid, 1.0, brake)
        stopped_for = np.minimum(stopped_for, race.time - going)
    return _backing_up(cars, angle, steer, throttle, brake, stopped_for, can_back_up=~race.in_lane & ~on_the_grid)


def _pit_lane(race, speed, grip, deceleration, target_speed, most_brake, steer, throttle, brake):
    """Drives the cars heading into or through the pit lane: along the lane's side of the track towards the
    entry, down the fast lane under the limit, into its box and back out if it was called, and on to the exit.
    `grip` is the cornering each car is willing to use, `deceleration` the hardest it can slow down, and
    `most_brake` the brake its cornering leaves grip for."""
    lane, cars, track = race.pit, race.cars, race.track
    along = lane.along(race.progress % track.length)
    approaching = (race.pit_state == pit.CALLED) & (along > -PIT_APPROACH['from']) & (along < lane.mouth)
    lane_cars = np.flatnonzero(approaching | race.in_lane)
    for car in lane_cars:
        # In the lane it looks closer, to follow its tighter turns into the box and out.
        look = PIT_APPROACH['lane_look'] + 0.3 * speed[car] if race.in_lane[car] else LOOKAHEAD[0] + LOOKAHEAD[1] * speed[car]
        ahead = along[car] + look
        target = track.point_at(lane.entry + ahead, _lane_path(race, car, ahead))
        to_target = target - cars.position[car]
        angle = np.arctan2(to_target[1], to_target[0]) - cars.heading[car]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        reach = max(np.linalg.norm(to_target), 1.0)
        lock = np.arctan(2 * WHEELBASE * np.sin(angle) / reach)
        steer[car] = np.clip(lock / (MAX_STEER / (1 + speed[car] / STEER_FADE_SPEED)), -1, 1)
        wanted = _lane_speed(race, car, along[car], deceleration[car])
        if wanted is None:
            continue
        wanted = min(wanted, target_speed[car]) if not race.in_lane[car] else wanted
        # The lane's own bends (into it, into the box and out) are tighter than the track's: no faster than grip
        # allows round the tightest of them just ahead.
        bend = max(_path_curvature(race, car, along[car] + step) for step in (0.0, 5.0, 10.0))
        wanted = min(wanted, np.sqrt(grip[car] / max(bend, 1e-4)))
        throttle[car] = np.clip((wanted - speed[car]) / 1.0, 0, 1)
        # On the grass there's only part of the grip there is on the tarmac to brake with, as for turning.
        grass = 1.0 if race.on_track[car] else OFF_TRACK['grip']
        brake[car] = min(np.clip((speed[car] - wanted) / 0.5, 0, 1), most_brake[car] * grass)
    return steer, throttle, brake


def _path_curvature(race, car, along, step=3.0):
    """How sharply car `car`'s path through the lane bends `along` metres in (1 / radius), from three points on it."""
    track = race.track
    points = [track.point_at(race.pit.entry + at, _lane_path(race, car, at)) for at in (along - step, along, along + step)]
    first, second = points[1] - points[0], points[2] - points[1]
    turn = np.arctan2(first[0] * second[1] - first[1] * second[0], first @ second)
    return abs(turn) / max(np.linalg.norm(first), 1e-6)


def _lane_path(race, car, along):
    """Where across the road car `car` wants to be `along` metres into the lane: at the lane's edge of the track
    before the entry, then in the fast lane, swinging over to the working lane for its box when it has a stop to
    make and back out after."""
    lane, half = race.pit, race.track.width / 2
    if along < 0:
        edge = lane.side * (half - PIT_APPROACH['edge'])
        return edge * np.clip((along + PIT_APPROACH['from']) / PIT_APPROACH['over'], 0, 1)
    box = lane.boxes[car]
    stopping = _stopping(race, car, along)
    serviced = race.service_jobs[car] is not None
    if stopping and along <= box:
        shift = np.clip(1 - (box - along) / PIT_APPROACH['turn_in'], 0, 1)
    elif stopping or (serviced and along >= box):
        shift = np.clip(1 - (along - box) / PIT_APPROACH['pull_out'], 0, 1) if serviced else 1.0
    else:
        shift = 0.0
    ease = lambda value: value * value * (3 - 2 * value)
    shift = ease(shift)
    into = np.clip(along / pit.RAMP, 0, 1)
    out = np.clip((lane.length - along) / pit.RAMP, 0, 1)
    spread = ease(min(into, out))
    return float(lane.centre(along)) + lane.side * pit.LANES_APART / 2 * (2 * shift - 1) * spread


def _stopping(race, car, along):
    """Whether car `car` still has its stop to make `along` metres into the lane: it was called in, the crew hasn't
    worked on it yet, and it hasn't overshot its box too far for them to reach (then it goes round again)."""
    return (race.lane_called[car] and race.service_jobs[car] is None and race.pit_state[car] >= pit.IN_LANE
            and along <= race.pit.boxes[car] + pit.BOX_MARKS['reach_along'])


def _lane_speed(race, car, along, deceleration):
    """The speed car `car` aims for `along` metres into the lane, or None where it races as usual: under the limit
    by the lane's mouth and through the speed zone, and down to a stop on its marks if it has a stop to make."""
    lane = race.pit
    cruise = pit.SPEED_LIMIT - PIT_APPROACH['under_limit']
    if along > lane.length - pit.RAMP or not (race.in_lane[car] or along > -PIT_APPROACH['from']):
        return None
    braking = PIT_APPROACH['limit_braking'] * deceleration
    wanted = cruise if along >= lane.mouth else np.sqrt(cruise ** 2 + 2 * braking * max(lane.mouth - along, 0.0))
    if _stopping(race, car, along):
        wanted = min(wanted, np.sqrt(2 * PIT_APPROACH['box_braking'] * max(lane.boxes[car] - along, 0.0)))
    if race.in_lane[car]:
        wanted = min(wanted, _room_in_lane(race, car, along))
    return wanted


def _room_in_lane(race, car, along):
    """The fastest car `car` can go and still stop behind the car ahead of it in the same lane (driving or in its
    box), keeping a queue's gap; nobody passes under the speed limit."""
    lane, half_width = race.pit, pit.LANES_APART * 0.6
    others = np.flatnonzero(race.in_lane & (np.arange(race.count) != car))
    ahead = race.lane_along[others] - along
    same_lane = np.abs(race.lateral[others] - race.lateral[car]) < half_width
    blocking = same_lane & (ahead > 0)
    if not blocking.any():
        return np.inf
    nearest = np.argmin(np.where(blocking, ahead, np.inf))
    gap = ahead[nearest] - pit.QUEUE_GAP
    return max(race.cars.speed[others[nearest]] + np.sqrt(2 * PIT_APPROACH['box_braking'] * max(gap, 0.0)) * (gap > 0), 0.0)


def _traffic(race, speed):
    """For each car: how far ahead the nearest car in its path is (inf for none), how fast that car is going its
    way, whether it's going round it, and where across the track to aim (0, the centre line, when it isn't). A car
    heading into the pit lane or in it queues instead of going round: a car ahead there is slowing for the limit, and
    swinging out round it took cars over the white lines and past the entry."""
    count = race.count
    nothing = np.full(count, np.inf), np.zeros(count), np.zeros(count, dtype=bool), np.zeros(count)
    if race.ghosts or count < 2:
        return nothing
    cars = race.cars
    offset = to_car_frame(cars.position[None, :, :] - cars.position[:, None, :], cars.heading[:, None])
    ahead, across = offset[..., 0], offset[..., 1]
    # Cars on rails in the lane, and the car itself, are never in the way.
    blocking = (ahead > 0) & (ahead < TRAFFIC['looking']) & (np.abs(across) < TRAFFIC['across'])
    blocking &= ~np.eye(count, dtype=bool) & ~race.on_rails[None, :] & ~race.on_rails[:, None]
    if not blocking.any():
        return nothing
    distance = np.where(blocking, ahead, np.inf)
    nearest = np.argmin(distance, axis=1)
    gap = distance[np.arange(count), nearest]
    their_speed = np.einsum('ck,ck->c', cars.velocity[nearest], cars.forward)
    along = race.pit.along(race.progress % race.track.length)
    for_the_lane = race.in_lane | ((race.pit_state == pit.CALLED) & (along > -PIT_APPROACH['from']) & (along < race.pit.length))
    slow = np.isfinite(gap) & (their_speed < TRAFFIC['slow'] * np.maximum(speed, 5.0)) & ~for_the_lane
    half = race.track.width / 2 - 1.5
    side = np.where(race.lateral[nearest] > 0, -1.0, 1.0)
    around = np.where(slow, np.clip(race.lateral[nearest] + side * TRAFFIC['beside'], -half, half), 0.0)
    return gap, np.maximum(their_speed, 0.0), slow, around


def _backing_up(cars, angle, steer, throttle, brake, stopped_for, can_back_up):
    """Overrides the controls of blocked cars (standing still `stopped_for` seconds while racing): both pedals to
    select reverse, then the throttle with the steering turned the opposite way (reverse turns the nose the other
    way), then the brake to stop and go back to first."""
    blocked = ~cars.reversing & (stopped_for > BLOCKED_SECONDS) & can_back_up
    backed_enough = (cars.reversed_for > BACK_UP['longest']) | \
        ((cars.reversed_for > BACK_UP['shortest']) & (np.abs(angle) < BACK_UP['aligned']))
    backing = cars.reversing & ~backed_enough
    finishing = cars.reversing & backed_enough
    steer = np.where(backing, -np.sign(angle), steer)
    throttle = np.where(blocked | backing, 1.0, np.where(finishing, 0.0, throttle))
    brake = np.where(blocked | finishing, 1.0, np.where(backing, 0.0, brake))
    return steer, throttle, brake


def _speed_profile(corner_speed, curvature, grip, braking):
    """The fastest speed at each point ahead from which every later corner can still be made. Works backward
    from the far end: each point may be faster than the next by what braking can shed over one sample, and
    braking only gets the grip that point's own cornering leaves over."""
    allowed = corner_speed.copy()
    for step in range(allowed.shape[1] - 2, -1, -1):
        following = allowed[:, step + 1]
        cornering = following ** 2 * curvature[:, step]
        deceleration = np.minimum(np.sqrt(np.maximum(grip ** 2 - cornering ** 2, 0.0)), braking)
        allowed[:, step] = np.minimum(allowed[:, step], np.sqrt(following ** 2 + 2 * deceleration * SPACING))
    return allowed
