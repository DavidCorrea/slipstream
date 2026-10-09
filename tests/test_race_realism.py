import numpy as np

from slipstream.car import COMPOUNDS, TYRE_BLANKETS, CarSpecs
from slipstream.human import LATENCY_TICKS, Driving
from slipstream.pit import PitPlan, apply_service, service_times
from slipstream.race import Race
from slipstream.track import generate_track


def two_cars(gap, sideways=0.0):
    """Two cars on the start straight: car 1 `gap` metres ahead of car 0 and `sideways` to its left."""
    race = Race(generate_track(0), CarSpecs.uniform(2), laps=2, rng=np.random.default_rng(0))
    race.cars.heading[1] = race.cars.heading[0]
    race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * gap + race.cars.left[0] * sideways
    race.cars.velocity[:] = race.cars.forward[0] * 50.0
    return race


class TestTheAirInARace:
    def test_a_car_close_behind_another_feels_its_wake(self):
        conditions = two_cars(12.0).conditions()
        assert conditions.draft[0] > 0.5 and conditions.dirty_air[0] > 0.2
        assert conditions.draft[1] == 0

    def test_ghosts_race_in_clean_air(self):
        race = two_cars(12.0)
        race.ghosts = True
        assert race.conditions().draft.max() == 0


class TestTheSurfaceInARace:
    def test_cars_lay_rubber_as_they_race(self):
        race = Race(generate_track(3), CarSpecs.uniform(1), laps=1)
        from slipstream.drivers import scripted_controls
        for _ in range(400):
            race.step(*scripted_controls(race))
        assert race.surface.rubber.max() > 0


class TestWhereAHitLands:
    def test_running_into_the_car_ahead_breaks_the_front_wing(self):
        race = two_cars(4.9)
        race.cars.velocity[0] = race.cars.forward[0] * 70.0
        race.step(np.zeros(2), np.zeros(2), np.zeros(2))
        assert race.cars.wing_damage[0] > 0 and race.cars.suspension_damage[0] == 0

    def test_being_hit_from_the_side_bends_the_suspension(self):
        race = two_cars(0.0, sideways=2.3)
        race.cars.velocity[1] = race.cars.velocity[0] - race.cars.left[0] * 15.0
        race.step(np.zeros(2), np.zeros(2), np.zeros(2))
        assert race.cars.suspension_damage[0] > 0


class TestDebris:
    def test_a_hard_hit_leaves_debris_on_the_track(self):
        race = two_cars(4.9)
        race.cars.velocity[0] = race.cars.forward[0] * 70.0
        race.step(np.zeros(2), np.zeros(2), np.zeros(2))
        assert len(race.debris) > 0 and len(race.new_debris) == len(race.debris)

    def test_driving_over_debris_can_puncture_a_tyre(self):
        race = Race(generate_track(0), CarSpecs.uniform(1), laps=1, rng=np.random.default_rng(1))
        race.cars.velocity[0] = race.cars.forward[0] * 30.0
        ahead = race.cars.position[0] + race.cars.forward[0] * np.arange(1, 40)[:, None]
        race.debris = ahead.copy()
        for _ in range(40):
            race.step(np.zeros(1), np.full(1, 0.5), np.zeros(1))
        assert race.cars.punctured[0]


class TestEngines:
    def test_an_engine_run_too_hot_fails_and_its_car_is_out(self):
        race = Race(generate_track(0), CarSpecs.uniform(2), laps=1, ghosts=True, rng=np.random.default_rng(2))
        for _ in range(4000):
            race.cars.engine_temp[0] = 1.0
            race.step(np.zeros(2), np.zeros(2), np.zeros(2))
            if race.cars.engine_failed[0]:
                break
        assert race.cars.engine_failed[0] and race.retired[0]
        assert race.standings()[-1] == 0


class TestTheCrew:
    def test_new_tyres_come_out_of_their_blankets_and_fix_a_puncture(self):
        race = Race(generate_track(0), CarSpecs.uniform(1), laps=1)
        race.cars.punctured[0], race.cars.tyre_temp[0] = True, 0.9
        plan = PitPlan.standard(1)
        jobs, _ = service_times(plan, race.cars, 0)
        apply_service(plan, race.cars, 0, jobs)
        assert not race.cars.punctured[0] and race.cars.tyre_temp[0] == TYRE_BLANKETS

    def test_repairs_fix_the_wing_and_suspension_and_take_longer_for_more_damage(self):
        race = Race(generate_track(0), CarSpecs.uniform(1), laps=1)
        plan = PitPlan.standard(1)
        race.cars.wing_damage[0] = 0.2
        light, _ = service_times(plan, race.cars, 0)
        race.cars.wing_damage[0], race.cars.suspension_damage[0] = 0.9, 0.6
        heavy, _ = service_times(plan, race.cars, 0)
        apply_service(plan, race.cars, 0, heavy)
        assert heavy['repair'] > light['repair'] > 0
        assert race.cars.wing_damage[0] == 0 and race.cars.suspension_damage[0] == 0


class TestTheDriversHandsAndFeet:
    def test_what_the_driver_asks_for_arrives_a_moment_later(self):
        driving = Driving(1, np.random.default_rng(0), consistency=np.ones(1))
        steered = [driving.apply(np.ones(1), np.zeros(1), np.zeros(1), np.zeros(1), 0.0)[0][0] for _ in range(LATENCY_TICKS + 1)]
        assert abs(steered[0]) < 0.2 and abs(steered[-1] - 1.0) < 0.2

    def test_pressure_and_fatigue_make_a_driver_less_precise(self):
        def spread(pressure, time):
            driving = Driving(400, np.random.default_rng(0))
            for _ in range(LATENCY_TICKS + 1):
                steer, _, _ = driving.apply(np.zeros(400), np.full(400, 0.5), np.zeros(400), np.full(400, pressure), time)
            return steer.std()
        assert spread(1.0, 0.0) > spread(0.0, 0.0)
        assert spread(0.0, 1500.0) > spread(0.0, 0.0)

    def test_a_consistent_driver_errs_less_than_an_erratic_one(self):
        def spread(consistency):
            driving = Driving(400, np.random.default_rng(0), consistency=np.full(400, consistency))
            for _ in range(LATENCY_TICKS + 1):
                steer, _, _ = driving.apply(np.zeros(400), np.full(400, 0.5), np.zeros(400), np.full(400, 0.5), 0.0)
            return steer.std()
        assert spread(1.0) < spread(0.0)
