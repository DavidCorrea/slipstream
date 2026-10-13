import numpy as np

from slipstream.benchmark import STALLED_SECONDS, field_race, run_race
from slipstream.car import CarSpecs
from slipstream.race import Race
from slipstream.track import generate_track


class TestABenchmarkRace:
    def test_ends_once_every_car_being_tested_is_stuck(self, monkeypatch):
        from slipstream import benchmark
        monkeypatch.setattr(benchmark, 'field_controls', lambda race, driver, trained, personality=None: (
            np.zeros(race.count), np.zeros(race.count), np.ones(race.count), np.zeros(race.count, dtype=bool), None))
        race = Race(generate_track(1), CarSpecs.uniform(1), laps=2)
        run_race(race, None, np.array([True]))
        assert race.time < STALLED_SECONDS + 5


class TestAFieldRace:
    def test_scores_how_a_whole_field_of_one_driver_races_among_the_scenery(self):
        scores = field_race(None, seeds=(2001,), cars=4, laps=1)
        assert set(scores) == {'field_out', 'field_off_track', 'field_hits'}
        assert 0 <= scores['field_out'] < 0.5 and 0 <= scores['field_off_track'] < 0.2 and scores['field_hits'] >= 0
