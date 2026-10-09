import numpy as np

from slipstream.car import CarSpecs
from slipstream.race import Race
from slipstream.senses import SENSE_NAMES, senses, visible
from slipstream.track import generate_track
from slipstream.weather import Weather


def pair(gap, sideways=0.0, weather=None):
    """Car 0 and car 1 on the start straight, car 1 `gap` metres ahead and `sideways` to the left, both at 50 m/s."""
    race = Race(generate_track(0), CarSpecs.uniform(2), laps=2, weather=weather)
    race.cars.heading[1] = race.cars.heading[0]
    race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * gap + race.cars.left[0] * sideways
    race.cars.velocity[:] = race.cars.forward[0] * 50.0
    return race


class TestWhatADriverKnows:
    def test_feels_the_car_but_is_never_told_its_numbers(self):
        for secret in ('spec.top_speed', 'spec.grip', 'wetness', 'rain', 'tyre_wear', 'tyre_temp', 'damage'):
            assert secret not in SENSE_NAMES
        for feel in ('yaw_rate', 'longitudinal_g', 'lateral_g', 'wheelspin', 'sliding'):
            assert feel in SENSE_NAMES

    def test_senses_line_up_with_their_names(self):
        race = pair(20.0)
        assert senses(race).shape == (2, len(SENSE_NAMES))


class TestWhatADriverSees:
    def test_sees_a_car_ahead_in_clear_air(self):
        assert visible(pair(30.0))[0, 1]

    def test_sees_a_car_straight_behind_in_the_mirrors(self):
        assert visible(pair(30.0))[1, 0]

    def test_misses_a_car_in_the_blind_spot(self):
        assert not visible(pair(-6.0, sideways=4.5))[0, 1]

    def test_feels_a_car_right_alongside(self):
        assert visible(pair(0.5, sideways=2.4))[0, 1]

    def test_loses_a_car_in_the_spray_of_a_wet_track(self):
        weather = Weather.from_forecast('wet', 600, np.random.default_rng(0))
        weather.wetness = 0.95
        assert not visible(pair(35.0, weather=weather))[0, 1]
        assert visible(pair(8.0, weather=weather))[0, 1]

    def test_loses_a_car_behind_the_smoke_of_a_damaged_one(self):
        race = pair(35.0)
        race.cars.damage[1] = 0.9
        assert not visible(race)[0, 1]
