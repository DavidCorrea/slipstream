import numpy as np
import pytest

from slipstream.track import CLEARANCE, GRID_MIN_RADIUS, LENGTH_RANGE, MIN_RADIUS, SPACING, generate_track

SEEDS = range(12)


@pytest.fixture(scope='module')
def tracks():
    return [generate_track(seed) for seed in SEEDS]


class TestGeneratedTracks:
    def test_are_the_same_for_the_same_seed(self):
        assert np.array_equal(generate_track(7).points, generate_track(7).points)

    def test_differ_between_seeds(self):
        assert generate_track(1).size != generate_track(2).size or not np.allclose(generate_track(1).points, generate_track(2).points)

    def test_close_into_a_loop_with_even_spacing(self, tracks):
        for track in tracks:
            gaps = np.linalg.norm(np.diff(np.vstack([track.points, track.points[:1]]), axis=0), axis=1)
            assert np.allclose(gaps, SPACING, rtol=0.05), f'seed {track.seed}'

    def test_never_corner_tighter_than_the_minimum_radius(self, tracks):
        for track in tracks:
            assert np.abs(track.curvature).max() <= 1 / MIN_RADIUS + 1e-9, f'seed {track.seed}'

    def test_never_pass_close_to_themselves(self, tracks):
        for track in tracks:
            points = track.points[::3]
            index = np.arange(len(points))
            along = np.abs(index[:, None] - index[None, :]) * 3 * SPACING
            along = np.minimum(along, track.length - along)
            gaps = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            assert gaps[along > CLEARANCE * 2.5].min() >= CLEARANCE, f'seed {track.seed}'

    def test_stay_within_the_length_range(self, tracks):
        for track in tracks:
            assert LENGTH_RANGE[0] <= track.length <= LENGTH_RANGE[1]

    def test_put_the_grid_on_a_straight(self, tracks):
        for track in tracks:
            grid = track.curvature[-40:]
            assert np.abs(grid).max() < 1 / GRID_MIN_RADIUS, f'seed {track.seed}: tightest grid radius {1 / np.abs(grid).max():.0f} m'


class TestLocating:
    def test_finds_the_distance_and_offset_a_point_was_placed_at(self):
        track = generate_track(3)
        distances = np.array([5.0, 300.3, track.length - 3.1])
        laterals = np.array([2.0, -5.5, 0.0])
        _, distance, lateral = track.locate(track.point_at(distances, laterals))
        assert np.allclose(distance, distances, atol=0.3)
        assert np.allclose(lateral, laterals, atol=0.3)

    def test_with_hints_matches_a_full_search_for_a_car_moving_along(self):
        track = generate_track(4)
        positions = track.point_at(np.arange(0, track.length, 3.7), 3.0)
        hint = track.locate(positions[:1])[0]
        for position in positions:
            hinted, _, _ = track.locate(position[None], hint)
            full, _, _ = track.locate(position[None])
            assert hinted[0] == full[0]
            hint = hinted


class TestCornersWithCharacter:
    def test_every_circuit_has_a_chicane_or_a_sharp_turn(self, tracks):
        from slipstream.track import features
        for track in tracks:
            assert features(track), f'seed {track.seed} has neither'

    def test_circuits_get_both_kinds(self, tracks):
        from slipstream.track import features
        kinds = {kind for track in tracks for kind, _ in features(track)}
        assert kinds == {'chicane', 'sharp turn'}

    def test_a_chicane_turns_one_way_then_the_other_close_together(self):
        from slipstream.track import features
        track = next(track for track in (generate_track(seed) for seed in range(40)) if any(kind == 'chicane' for kind, _ in features(track)))
        index = next(index for kind, index in features(track) if kind == 'chicane')
        nearby = track.curvature[(index + np.arange(-40, 41)) % track.size]
        assert nearby.max() > 1 / 50 and nearby.min() < -1 / 50
