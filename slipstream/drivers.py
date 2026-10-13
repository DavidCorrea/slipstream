"""A scripted driver that follows the centre line: no learning, no racecraft.

It aims at a point ahead on the centre line (pure pursuit) and picks a target speed from the corners coming
up: the fastest speed that grip allows through each one, plus what braking can shed before reaching it. It
proves the physics can be driven, and gives trained drivers a fixed opponent to beat.
"""
import numpy as np

from .car import BRAKE_WEAR, GRAVITY, MAX_STEER, STEER_FADE_SPEED, WHEELBASE, effective_grip, effective_top_speed
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
BACK_UP = {'longest': 2.0, 'shortest': 0.8, 'aligned': 0.6}   # seconds, seconds, radians off the way ahead


def scripted_controls(race):
    track, cars, specs = race.track, race.cars, race.specs
    speed = np.maximum(cars.speed, 0.0)
    distance = race.progress % track.length

    target = track.point_at(distance + LOOKAHEAD[0] + LOOKAHEAD[1] * speed)
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
    braking = specs.braking * (1 - BRAKE_WEAR['loss'] * cars.brake_wear)
    corner_speed = np.sqrt(grip[:, None] / np.maximum(curvature, 1e-4))
    profile = _speed_profile(corner_speed, curvature, grip, CAUTION * braking)
    lead = np.minimum((speed * REACTION / SPACING).astype(int), profile.shape[1] - 1)
    target_speed = np.minimum(profile[np.arange(len(speed)), lead], effective_top_speed(cars, specs))
    target_speed = np.where(race.on_track, target_speed, np.minimum(target_speed, RECOVERY_SPEED))

    throttle = np.clip((target_speed - speed) / 1.0, 0, 1)
    # Brake only with the grip the corner it's in leaves over (the friction circle), or it can't turn.
    full_grip = grip / CAUTION
    cornering = speed ** 2 * curvature[:, 0]
    spare = np.sqrt(np.maximum(full_grip ** 2 - cornering ** 2, 0.0))
    brake = np.minimum(np.clip((speed - target_speed) / 0.5, 0, 1), spare / braking)
    return _backing_up(cars, angle, np.clip(steer, -1, 1), throttle, brake)


def _backing_up(cars, angle, steer, throttle, brake):
    """Overrides the controls of blocked cars: both pedals to select reverse, then the throttle with the steering
    turned the opposite way (reverse turns the nose the other way), then the brake to stop and go back to first."""
    blocked = ~cars.reversing & (cars.stopped_for > BLOCKED_SECONDS)
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
