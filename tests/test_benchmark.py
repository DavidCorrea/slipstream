import numpy as np

from slipstream.benchmark import STALLED_SECONDS, run_race
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
