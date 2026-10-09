import numpy as np

from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.race import Race
from slipstream.track import generate_track


def run(race):
    events = []
    while not race.done:
        events.append(race.step(*scripted_controls(race)))
    return events


class TestTheGrid:
    def test_puts_every_car_behind_the_line_on_the_tarmac_without_touching(self):
        race = Race(generate_track(2), CarSpecs.uniform(10), laps=1)
        assert (race.progress < 0).all()
        assert race.on_track.all()
        assert race._resolve_contacts() == []


class TestScriptedDrivers:
    def test_finish_every_race_without_leaving_the_track_when_alone(self):
        for seed in range(6):
            race = Race(generate_track(seed), CarSpecs.uniform(1), laps=2)
            off_track = 0
            while not race.done:
                race.step(*scripted_controls(race))
                off_track += int(not race.on_track[0])
            assert race.finished[0], f'seed {seed}'
            assert off_track == 0, f'seed {seed}'


class TestLaps:
    def test_count_one_lap_per_line_crossing_and_none_for_leaving_the_grid(self):
        race = Race(generate_track(5), CarSpecs.uniform(1), laps=3)
        events = run(race)
        assert sum(len(tick.laps) for tick in events) == 3
        assert race.lap_of()[0] == 3

    def test_finish_cars_in_order_of_crossing(self):
        specs = CarSpecs.uniform(4)
        specs.top_speed[:] = [50.0, 60.0, 70.0, 80.0]
        race = Race(generate_track(6), specs, laps=2)
        run(race)
        assert race.finished.all()
        assert np.all(np.diff(race.finish_time[race.standings()]) >= 0)

    def test_rank_unfinished_cars_by_distance_covered(self):
        race = Race(generate_track(6), CarSpecs.uniform(3), laps=5)
        race.progress[:] = [100.0, 300.0, 200.0]
        assert list(race.standings()) == [1, 2, 0]

    def test_never_count_reversing_over_the_line_as_a_lap(self):
        race = Race(generate_track(1), CarSpecs.uniform(1), laps=2)
        track = race.track
        laps = 0
        for distance in [track.length - 4, 2.0, track.length - 4, 2.0]:
            race.cars.position[0] = track.point_at(np.array([distance]))[0]
            race.cars.velocity[0] = 0
            laps += len(race.step(np.zeros(1), np.zeros(1), np.zeros(1)).laps)
        assert laps <= 1


def two_cars():
    race = Race(generate_track(0), CarSpecs.uniform(2), laps=1)
    race.cars.heading[1] = race.cars.heading[0]
    race.cars.velocity[:] = 0.0
    return race


class TestContact:
    def test_pushes_overlapping_cars_apart_and_conserves_momentum(self):
        race = two_cars()
        race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * 4.7
        race.cars.velocity[0] = race.cars.forward[0] * 20
        race.cars.velocity[1] = race.cars.forward[0] * 10
        momentum = race.cars.velocity.sum(axis=0).copy()
        contacts = race._resolve_contacts()
        assert len(contacts) == 1
        assert np.allclose(race.cars.velocity.sum(axis=0), momentum)
        closing = np.dot(race.cars.velocity[0] - race.cars.velocity[1], race.cars.forward[0])
        assert closing <= 1e-9

    def test_feels_a_front_wing_in_another_cars_sidepod(self):
        race = two_cars()
        forward, left = race.cars.forward[0], race.cars.left[0]
        # Car 1 points straight at car 0's side, its front wing tip 0.3 m inside car 0's wheels.
        race.cars.position[1] = race.cars.position[0] + left * (1.06 + 2.8 - 0.3)
        race.cars.heading[1] = race.cars.heading[0] - np.pi / 2
        assert len(race._resolve_contacts()) == 1

    def test_feels_a_front_wing_tapping_the_car_ahead(self):
        race = two_cars()
        race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * (2.8 + 2.2 - 0.3)
        assert len(race._resolve_contacts()) == 1

    def test_leaves_cars_alone_with_a_hands_width_between_them(self):
        race = two_cars()
        race.cars.position[1] = race.cars.position[0] + race.cars.left[0] * (2 * 1.06 + 0.2)
        assert race._resolve_contacts() == []
        race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * (2.8 + 2.2 + 0.2)
        assert race._resolve_contacts() == []

    def test_leaves_cars_that_are_apart_alone(self):
        race = Race(generate_track(0), CarSpecs.uniform(4), laps=1)
        assert race._resolve_contacts() == []
