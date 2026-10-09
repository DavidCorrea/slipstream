import numpy as np

from slipstream.surface import Surface
from slipstream.track import generate_track


def lap_on_line(surface, track, lateral=0.0, cars=6, laps=8, rain=0.0, wear=0.0005):
    """Drives `cars` cars `laps` times round the same line, a cell at a time."""
    step = 4.0
    for _ in range(laps * cars):
        for distance in np.arange(0, track.length, step):
            surface.update(np.array([distance]), np.array([lateral]), np.array([True]), np.array([wear]), rain, step / 40.0)


class TestTheRacingLine:
    def test_rubber_builds_up_where_cars_drive_and_grips_more(self):
        track = generate_track(3)
        surface = Surface(track)
        lap_on_line(surface, track)
        on_line = surface.grip(np.array([100.0]), np.array([0.0]))[0]
        off_line = surface.grip(np.array([100.0]), np.array([track.width / 2 - 0.5]))[0]
        assert on_line > 1.02 and on_line > off_line

    def test_worn_rubber_gathers_off_the_line_as_marbles_that_grip_less(self):
        track = generate_track(3)
        surface = Surface(track)
        lap_on_line(surface, track, wear=0.002)
        assert surface.grip(np.array([100.0]), np.array([track.width / 2 - 0.3]))[0] < 0.97

    def test_a_dry_line_forms_where_cars_drive_once_the_rain_stops(self):
        track = generate_track(3)
        surface = Surface(track)
        lap_on_line(surface, track, laps=3)
        on_line = surface.wetness(np.array([100.0]), np.array([0.0]), 0.6)[0]
        off_line = surface.wetness(np.array([100.0]), np.array([track.width / 2 - 0.5]), 0.6)[0]
        assert on_line < off_line == 0.6

    def test_rain_washes_the_dry_line_away(self):
        track = generate_track(3)
        surface = Surface(track)
        lap_on_line(surface, track, laps=3)
        drying = surface.wetness(np.array([100.0]), np.array([0.0]), 0.6)[0]
        for _ in range(200):
            surface.update(np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0), 1.0, 1.0)
        assert surface.wetness(np.array([100.0]), np.array([0.0]), 0.6)[0] > drying

    def test_off_the_tarmac_is_plain_ground(self):
        track = generate_track(3)
        surface = Surface(track)
        lap_on_line(surface, track)
        assert surface.grip(np.array([100.0]), np.array([track.width]))[0] == 1.0
