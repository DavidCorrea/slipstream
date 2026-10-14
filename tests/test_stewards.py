import numpy as np

from slipstream.car import CarSpecs
from slipstream.race import Race
from slipstream.stewards import COLLISION, REVERSING, TRACK_LIMITS
from slipstream.track import generate_track

STILL = (np.zeros(2), np.zeros(2), np.zeros(2))


def race_on_a_straight(cars=2, stewards=True):
    return Race(generate_track(4), CarSpecs.uniform(cars), laps=3, stewards=stewards)


def place(race, car, distance, lateral=0.0, speed=0.0):
    track = race.track
    race.cars.position[car] = track.point_at(distance, lateral)
    race.cars.heading[car] = track.heading_at(distance)
    race.cars.velocity[car] = race.cars.forward[car] * speed
    race.progress[car] = distance
    race.track_index, _, race.lateral = track.locate(race.cars.position)


def hold(race, seconds, controls=STILL):
    penalties = []
    for _ in range(int(round(seconds / 0.05))):
        penalties += race.step(*controls).penalties
    return penalties


def run_off(race, car, seconds):
    distance = 200.0
    place(race, car, distance, race.track.width / 2 + 6)
    penalties = hold(race, seconds)
    place(race, car, distance)
    return penalties + hold(race, 0.2)


class TestTrackLimits:
    def test_warns_for_the_first_strikes_and_penalises_every_one_after(self):
        race = race_on_a_straight()
        place(race, 1, 400.0)
        penalties = []
        for _ in range(TRACK_LIMITS['warnings'] + 2):
            penalties += run_off(race, 0, TRACK_LIMITS['lasting'] + 0.2)
        assert race.stewards.strikes[0] == TRACK_LIMITS['warnings'] + 2
        assert [(penalty.car, penalty.reason) for penalty in penalties] == [(0, 'track limits')] * 2
        assert race.stewards.seconds[0] == 2 * TRACK_LIMITS['seconds']

    def test_lets_a_wheel_dropped_off_for_a_moment_go(self):
        race = race_on_a_straight()
        place(race, 1, 400.0)
        run_off(race, 0, TRACK_LIMITS['lasting'] / 2)
        assert race.stewards.strikes[0] == 0


class TestCollisions:
    def test_blames_the_car_that_ran_into_the_back_of_another(self):
        race = race_on_a_straight()
        place(race, 0, 220.0)
        place(race, 1, 212.0, speed=30.0)
        penalties = hold(race, 1.0)
        assert [(penalty.car, penalty.seconds, penalty.reason) for penalty in penalties] == [(1, COLLISION['seconds'], 'collision')]

    def test_judges_a_hit_once_however_long_the_cars_stay_in_touch(self):
        race = race_on_a_straight()
        place(race, 0, 220.0)
        place(race, 1, 212.0, speed=30.0)
        penalties = hold(race, 1.0, (np.zeros(2), np.array([0.0, 1.0]), np.zeros(2)))
        assert len(penalties) == 1

    def test_calls_two_cars_moving_into_each_other_a_racing_incident(self):
        race = race_on_a_straight()
        place(race, 0, 220.0, 1.3, speed=30.0)
        place(race, 1, 220.0, -1.3, speed=30.0)
        race.cars.velocity[0] -= race.cars.left[0] * 7.0
        race.cars.velocity[1] += race.cars.left[1] * 7.0
        events = race.step(*STILL)
        assert max(contact.impulse for contact in events.contacts) >= COLLISION['impulse']
        assert events.penalties == []

    def test_lets_rubbing_go(self):
        race = race_on_a_straight()
        place(race, 0, 220.0, speed=10.0)
        place(race, 1, 214.6, speed=11.0)
        assert hold(race, 1.0) == []
        assert race.contact_ticks[1] > 0


class TestReversing:
    def test_penalises_backing_down_the_track(self):
        race = race_on_a_straight()
        place(race, 1, 400.0)
        race.cars.reversing[0] = True
        penalties = hold(race, REVERSING['lasting'] + 0.5, (np.zeros(2), np.array([1.0, 0.0]), np.zeros(2)))
        assert [(penalty.car, penalty.reason) for penalty in penalties] == [(0, 'reversing')]

    def test_lets_a_car_back_out_of_trouble_off_the_road(self):
        race = race_on_a_straight()
        place(race, 1, 400.0)
        place(race, 0, 200.0, race.track.width / 2 + 8)
        race.cars.reversing[0] = True
        penalties = hold(race, REVERSING['lasting'] + 0.5, (np.zeros(2), np.array([1.0, 0.0]), np.zeros(2)))
        assert penalties == []


class TestTheResult:
    def test_ranks_finishers_by_their_time_with_penalties_added(self):
        race = race_on_a_straight()
        race.finish_time[:] = [100.0, 104.0]
        race.stewards.seconds[0] = 5.0
        assert list(race.standings()) == [1, 0]
        assert list(race.race_times()) == [105.0, 104.0]

    def test_hands_out_nothing_in_races_without_stewards(self):
        race = race_on_a_straight(stewards=False)
        place(race, 0, 220.0)
        place(race, 1, 212.0, speed=30.0)
        assert hold(race, 1.0) == []
        assert race.stewards is None
