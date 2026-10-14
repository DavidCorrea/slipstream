import numpy as np

from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.flags import BLUE_REACH, CRAWLING, LIGHTS, YELLOW
from slipstream.race import Race
from slipstream.stewards import BLUE_FLAGS, JUMP_START, YELLOW_PASS
from slipstream.track import generate_track

STILL = lambda count: (np.zeros(count), np.zeros(count), np.ones(count))


def flagged_race(cars=2, laps=3, seed=4):
    return Race(generate_track(seed), CarSpecs.uniform(cars), laps, stewards=True, assisted=False, flags=True)


def place(race, car, progress, lateral=0.0, speed=0.0):
    track = race.track
    race.cars.position[car] = track.point_at(progress % track.length, lateral)
    race.cars.heading[car] = track.heading_at(progress % track.length)
    race.cars.velocity[car] = race.cars.forward[car] * speed
    race.progress[car] = progress
    race.track_index, _, race.lateral = track.locate(race.cars.position)


def start(race):
    while not race.marshals.started(race.time):
        race.step(*STILL(race.count))


class TestTheStart:
    def test_lights_come_on_a_second_apart_and_go_out_after_a_hold(self):
        race = flagged_race()
        seen = []
        while not race.marshals.started(race.time):
            seen.append(race.marshals.lights(race.time))
            race.step(*STILL(2))
        assert sorted(set(seen)) == [1, 2, 3, 4, 5]
        hold = race.marshals.lights_out_at - LIGHTS['count'] * LIGHTS['apart']
        assert LIGHTS['hold'][0] <= hold <= LIGHTS['hold'][1]
        assert race.marshals.lights(race.time) == 0

    def test_penalises_a_car_that_moves_before_the_lights_go_out(self):
        race = flagged_race()
        throttle = np.array([1.0, 0.0])
        penalties = []
        for _ in range(40):
            penalties += race.step(np.zeros(2), throttle, np.array([0.0, 1.0])).penalties
        assert [(penalty.car, penalty.reason, penalty.seconds) for penalty in penalties] == [(0, 'jump start', JUMP_START)]

    def test_scripted_drivers_wait_for_the_lights_then_go(self):
        race = flagged_race()
        penalties = []
        for _ in range(400):
            penalties += race.step(*scripted_controls(race)).penalties
        assert not [penalty for penalty in penalties if penalty.reason == 'jump start']
        assert (race.progress > 20).all()


class TestYellowFlags:
    def test_come_out_around_a_car_stopped_by_the_track_and_go_in_after(self):
        race = flagged_race()
        start(race)
        place(race, 0, 400.0, race.track.width / 2 + 1.0)
        place(race, 1, 900.0, speed=0.0)
        yellows = []
        for _ in range(int(3.0 / 0.05)):
            yellows += race.step(*STILL(2)).yellow_flags
        assert len(yellows) >= 1
        assert race.marshals.under_yellow(np.array([400.0 - YELLOW['before'] + 5, 400.0 + YELLOW['after'] + 5]), race.track.length).tolist() == [True, False]

    def test_penalise_passing_a_car_at_speed_under_yellow(self):
        race = flagged_race(cars=3)
        start(race)
        race.marshals.zones = [[300.0, 600.0, race.time + 60.0]]
        place(race, 2, 1200.0)
        place(race, 0, 450.0, 2.5, speed=CRAWLING + 10)
        place(race, 1, 440.0, -2.5, speed=CRAWLING + 20)
        race.stewards.progress = race.progress.copy()
        penalties = []
        for _ in range(30):
            penalties += race.step(np.zeros(3), np.array([0.5, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])).penalties
        assert [(penalty.car, penalty.reason, penalty.seconds) for penalty in penalties] == [(1, 'yellow flag', YELLOW_PASS)]

    def test_let_a_crawling_car_be_passed(self):
        race = flagged_race(cars=3)
        start(race)
        race.marshals.zones = [[300.0, 600.0, race.time + 60.0]]
        place(race, 2, 1200.0)
        place(race, 0, 450.0, 2.5, speed=5.0)
        place(race, 1, 440.0, -2.5, speed=30.0)
        race.stewards.progress = race.progress.copy()
        penalties = []
        for _ in range(30):
            penalties += race.step(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.0, 1.0])).penalties
        assert not [penalty for penalty in penalties if penalty.reason == 'yellow flag']


class TestBlueFlags:
    def test_show_blue_to_a_car_about_to_be_lapped_and_penalise_holding_up_the_leader(self):
        race = flagged_race()
        start(race)
        length = race.track.length
        place(race, 0, length + 500.0 - BLUE_REACH / 2)
        place(race, 1, 500.0)
        race.stewards.progress = race.progress.copy()
        penalties = []
        for _ in range(int((BLUE_FLAGS['lasting'] + 1) / 0.05)):
            events = race.step(*STILL(2))
            penalties += events.penalties
            assert race.marshals.blue.tolist() == [False, True]
        assert [(penalty.car, penalty.reason) for penalty in penalties] == [(1, 'blue flags')]
