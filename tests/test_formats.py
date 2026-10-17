import numpy as np
import pytest

from slipstream.formats import (ELIMINATION, ENDURANCE, FORMATS, MANDATORY_STOPS, SPRINT, TIMED, UNSERVED_STOPS_PENALTY,
                                FormatRules, race_laps)
from slipstream.race import Race, TickEvents
from slipstream.car import DT, CarSpecs
from slipstream.session import SCRIPTED, RaceSession
from slipstream.track import generate_track


def race(cars=6, laps=8):
    return Race(generate_track(3), CarSpecs.uniform(cars), laps, stewards=True)


def lap_done(race, car, laps):
    """Puts `car` across the line after `laps` laps, with the others behind it in order."""
    length = race.track.length
    race.progress[:] = laps * length - 50.0 * np.arange(race.count)
    race.progress[car] = laps * length + 1.0
    events = TickEvents()
    events.laps = [car]
    return events


class TestHowLongRacesAre:
    def test_a_sprint_is_a_third_of_the_race_but_never_under_two_laps(self):
        assert race_laps(SPRINT, 9, cars=6, track_length=1200) == 3
        assert race_laps(SPRINT, 3, cars=6, track_length=1200) == 2

    def test_an_endurance_race_is_twice_as_long(self):
        assert race_laps(ENDURANCE, 8, cars=6, track_length=1200) == 16

    def test_an_elimination_race_is_long_enough_to_knock_out_all_but_three(self):
        assert race_laps(ELIMINATION, 3, cars=10, track_length=1200) == 9

    def test_a_timed_race_starts_from_the_laps_the_time_should_take(self):
        laps = race_laps(TIMED, 8, cars=6, track_length=1200, minutes=10)
        assert 10 <= laps <= 40

    def test_unknown_formats_are_refused_naming_the_choices(self):
        with pytest.raises(ValueError, match='sprint'):
            FormatRules('drag race', race())


class TestElimination:
    def test_the_last_car_is_knocked_out_each_time_the_leader_completes_a_lap_from_the_second(self):
        racing = race()
        rules = FormatRules(ELIMINATION, racing)
        assert rules.after_tick(racing, lap_done(racing, 0, 1)) == {'eliminated': []}
        out = rules.after_tick(racing, lap_done(racing, 0, 2))['eliminated']
        assert out == [5] and racing.retired_by[5] == 'eliminated'

    def test_an_eliminated_car_leaves_the_circuit_so_nobody_hits_it(self):
        racing = race()
        rules = FormatRules(ELIMINATION, racing)
        rules.after_tick(racing, lap_done(racing, 0, 2))
        assert np.linalg.norm(racing.cars.position[5]) > 10_000
        assert racing.cars.speed[5] == 0

    def test_stops_at_three_cars(self):
        racing = race(cars=5)
        rules = FormatRules(ELIMINATION, racing)
        for lap in range(2, 8):
            rules.after_tick(racing, lap_done(racing, 0, lap))
        assert (~racing.retired).sum() == 3


class TestEndurance:
    def test_calls_in_a_car_that_still_owes_a_stop_once_it_is_due(self):
        racing = race(laps=12)
        rules = FormatRules(ENDURANCE, racing)
        racing.progress[:] = 4.2 * racing.track.length
        call = rules.calls(racing, np.zeros(racing.count, dtype=bool), forced=np.ones(racing.count, dtype=bool))
        assert call.all()
        racing.stops[:] = MANDATORY_STOPS
        assert not rules.calls(racing, np.zeros(racing.count, dtype=bool), forced=np.ones(racing.count, dtype=bool)).any()

    def test_leaves_a_car_whose_stops_are_yours_to_call(self):
        racing = race(laps=12)
        rules = FormatRules(ENDURANCE, racing)
        racing.progress[:] = 4.2 * racing.track.length
        forced = np.ones(racing.count, dtype=bool)
        forced[2] = False
        assert not rules.calls(racing, np.zeros(racing.count, dtype=bool), forced)[2]

    def test_a_finisher_short_of_its_stops_gets_a_penalty(self):
        racing = race(laps=12)
        rules = FormatRules(ENDURANCE, racing)
        racing.stops[:] = [2, 1, 0, 2, 2, 2]
        events = TickEvents()
        events.finished = [0, 1, 2]
        penalties = rules.after_tick(racing, events)['penalties']
        assert [(penalty.car, penalty.seconds) for penalty in penalties] == [(1, UNSERVED_STOPS_PENALTY), (2, UNSERVED_STOPS_PENALTY)]
        assert racing.stewards.seconds[1] == UNSERVED_STOPS_PENALTY


class TestTimedRaces:
    def test_the_leaders_next_crossing_after_the_time_is_up_ends_it(self):
        racing = race(laps=30)
        rules = FormatRules(TIMED, racing, minutes=10)
        racing.tick = int(601.0 / DT)
        rules.after_tick(racing, lap_done(racing, 0, 7))
        assert racing.laps == 8

    def test_before_then_it_keeps_the_laps_the_time_left_should_take(self):
        racing = race(laps=30)
        rules = FormatRules(TIMED, racing, minutes=10)
        racing.tick = int(300.0 / DT)
        rules.after_tick(racing, lap_done(racing, 0, 5))
        assert racing.laps == 10
        assert rules.describe(racing)['timeLeft'] == pytest.approx(300.0, abs=0.1)


class TestInARace:
    def test_every_format_plays(self):
        for kind in FORMATS:
            session = RaceSession(SCRIPTED, seed=5, cars=4, laps=3, race_format=kind, minutes=10)
            frame = session.advance(40)
            assert frame['format']['kind'] == kind

    def test_a_reverse_grid_lines_the_field_up_back_to_front(self):
        usual = RaceSession(SCRIPTED, seed=5, cars=6, laps=3)
        reversed_grid = RaceSession(SCRIPTED, seed=5, cars=6, laps=3, race_format='reverse')
        assert [car['key'] for car in reversed_grid.intro()['cars']] == [car['key'] for car in usual.intro()['cars']][::-1]


class TestFramesOfEveryFormat:
    def test_frames_serialise_after_the_format_has_changed_the_race(self):
        import json
        for kind in FORMATS:
            session = RaceSession(SCRIPTED, seed=5, cars=4, laps=3, race_format=kind, minutes=10)
            leader = session.race
            events = lap_done(leader, 0, 2)
            leader.tick = int(601.0 / DT)
            session.format.after_tick(leader, events)
            json.dumps(session.advance(1))
