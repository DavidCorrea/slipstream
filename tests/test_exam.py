import numpy as np

from slipstream.exam import TESTS, Exam, exam_path, place_car, play
from slipstream.session import SCRIPTED, RaceSession


class TestTheExam:
    def test_every_test_has_a_group_a_rule_and_a_unique_name(self):
        names = [test.name for test in TESTS]
        assert len(names) == len(set(names))
        assert all(test.group in 'ABCDEFGHI' and test.kind in ('must', 'scored') for test in TESTS)
        assert all(test.better in ('higher', 'lower') for test in TESTS if test.kind == 'scored')

    def test_watches_a_race_tick_by_tick(self):
        played = play(RaceSession(SCRIPTED, seed=5, cars=3, laps=1))
        assert played.race.done and played.lap_times and played.top_speed.max() > 20

    def test_puts_a_car_where_a_scenario_needs_it(self):
        session = RaceSession(SCRIPTED, seed=5, cars=1, laps=1)
        place_car(session, 0, 300.0, lateral=2.0, turn=np.pi, speed=10.0)
        assert session.race.progress[0] == 300.0
        assert abs(session.race.lateral[0] - 2.0) < 0.5
        assert abs(session.race.cars.speed[0] - 10.0) < 1e-6

    def test_the_scripted_driver_takes_the_pit_call(self):
        test = next(test for test in TESTS if test.name == 'takes the pit call on 8 circuits')
        result = test.function(Exam(SCRIPTED))
        assert result.passed

    def test_saves_each_snapshots_exam_beside_its_run(self):
        from pathlib import Path
        assert exam_path('racer-2/step-000080018720', Path('runs')) == Path('runs/racer-2/exam/step-000080018720.json')
        assert exam_path('racer-2/step-000080018720', Path('runs'), quick=True).name == 'step-000080018720.quick.json'
