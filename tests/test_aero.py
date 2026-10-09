import numpy as np

from slipstream.aero import air_between


def cars(*placements, speed=50.0):
    """(x, y, heading) per car; all at the same speed along their heading."""
    placements = np.array(placements, dtype=float)
    velocity = np.stack([np.cos(placements[:, 2]), np.sin(placements[:, 2])], axis=1) * speed
    return placements[:, :2], placements[:, 2], velocity


class TestTheWakeOfACar:
    def test_a_car_close_behind_gets_both_the_slipstream_and_the_dirty_air(self):
        draft, dirty = air_between(*cars((0, 0, 0), (-10, 0, 0)), np.ones(2, dtype=bool))
        assert draft[1] > 0.5 and dirty[1] > 0.3
        assert draft[0] == 0 and dirty[0] == 0

    def test_the_slipstream_reaches_further_back_than_the_dirty_air(self):
        draft, dirty = air_between(*cars((0, 0, 0), (-30, 0, 0)), np.ones(2, dtype=bool))
        assert draft[1] > 0 and dirty[1] == 0

    def test_a_car_alongside_is_in_clean_air(self):
        draft, dirty = air_between(*cars((0, 0, 0), (-2, 6, 0)), np.ones(2, dtype=bool))
        assert draft[1] == 0 and dirty[1] == 0

    def test_a_slow_car_leaves_no_wake_worth_having(self):
        draft, _ = air_between(*cars((0, 0, 0), (-10, 0, 0), speed=5.0), np.ones(2, dtype=bool))
        assert draft[1] == 0

    def test_cars_out_of_the_race_leave_no_wake(self):
        draft, dirty = air_between(*cars((0, 0, 0), (-10, 0, 0)), np.array([False, True]))
        assert draft[1] == 0 and dirty[1] == 0
