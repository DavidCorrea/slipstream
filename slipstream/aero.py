"""The air behind each car: its slipstream and its dirty air.

A car punches a hole in the air. Close behind it there's less drag, so a following car reaches a higher top speed
on the straights (the slipstream, or tow); but the same disturbed air robs the follower's wings of downforce, so
it has less grip in the corners and its engine and tyres get less cooling. The slipstream reaches further back than
the dirty air hurts, which is what makes a car close behind on a straight a threat and the same car in a corner
a sitting duck.
"""
import numpy as np

WAKE_LENGTH = 40.0         # metres behind a car its slipstream reaches
DIRTY_AIR_LENGTH = 20.0    # metres behind a car its dirty air costs grip
WAKE_HALF_WIDTH = 2.2      # metres either side of a car's line, widening a little further back
CLOSEST = 2.0              # nearer than this the cars are touching, not drafting
WAKE_SPEED = 15.0          # m/s a car needs to be doing for its wake to matter


def air_between(positions, headings, velocities, active):
    """For each car, how deep it sits in another car's slipstream and in its dirty air, both 0 to 1 (the strongest
    of any car ahead). `active` marks cars that are racing; others neither leave a wake nor feel one."""
    count = len(positions)
    forward = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    left = np.stack([-np.sin(headings), np.cos(headings)], axis=1)
    # [follower, leader]: where each leader is from each follower, in the follower's own frame.
    offset = positions[None, :, :] - positions[:, None, :]
    ahead = np.einsum('fck,fk->fc', offset, forward)
    across = np.abs(np.einsum('fck,fk->fc', offset, left))
    speeds = np.einsum('ck,ck->c', velocities, forward)
    leading = active[None, :] & active[:, None] & ~np.eye(count, dtype=bool) & (speeds[None, :] > WAKE_SPEED)
    half_width = WAKE_HALF_WIDTH * (1 + 0.4 * np.clip(ahead / WAKE_LENGTH, 0, 1))
    beside = np.clip(1 - across / half_width, 0, 1)
    behind = leading & (ahead > CLOSEST)
    draft = np.where(behind, np.clip(1 - ahead / WAKE_LENGTH, 0, 1) * beside, 0.0).max(axis=1, initial=0.0)
    dirty = np.where(behind, np.clip(1 - ahead / DIRTY_AIR_LENGTH, 0, 1) * beside, 0.0).max(axis=1, initial=0.0)
    return draft, dirty
