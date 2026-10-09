import numpy as np

from slipstream.car import (COMPOUNDS, DT, CarSpecs, CarState, Conditions, brake_fade, step_cars, tyre_temperature_grip)


def one_car(speed=0.0, **specs):
    state = CarState.at([[0.0, 0.0]], [0.0])
    state.velocity[0] = [speed, 0.0]
    return state, CarSpecs.uniform(1, **specs)


def drive(state, specs, seconds, steer=0.0, throttle=0.0, brake=0.0, on_track=True, conditions=None):
    for _ in range(int(round(seconds / DT))):
        step_cars(state, specs, np.array([steer]), np.array([throttle]), np.array([brake]), np.array([on_track]), conditions)
    return state


class TestTyreTemperature:
    def test_tyres_grip_best_inside_their_window_and_less_cold_or_overheated(self):
        medium = np.array([COMPOUNDS.index('medium')])
        cold, warm, hot = (tyre_temperature_grip(np.array([temperature]), medium)[0] for temperature in (0.2, 0.6, 1.0))
        assert warm > cold and warm > hot and warm == 1.0

    def test_cornering_warms_tyres_and_a_straight_lets_them_cool(self):
        state, specs = one_car(speed=30.0)
        state.tyre_temp[0] = 0.3
        drive(state, specs, 20, steer=0.5, throttle=0.6)
        warmed = state.tyre_temp[0]
        drive(state, specs, 30, throttle=0.3)
        assert warmed > 0.4 and state.tyre_temp[0] < warmed

    def test_rain_keeps_tyres_cooler(self):
        dry, specs = one_car(speed=30.0)
        wet, _ = one_car(speed=30.0)
        drive(dry, specs, 30, steer=0.15, throttle=0.25)
        drive(wet, specs, 30, steer=0.15, throttle=0.25, conditions=Conditions.uniform(1, wetness=0.9))
        assert wet.tyre_temp[0] < dry.tyre_temp[0] - 0.05


class TestBrakes:
    def test_hot_brakes_stop_less_hard(self):
        assert brake_fade(np.array([0.95]))[0] < brake_fade(np.array([0.5]))[0] == 1.0

    def test_braking_heats_the_brakes_and_air_cools_them(self):
        state, specs = one_car(speed=60.0)
        drive(state, specs, 2, brake=1.0)
        hot = state.brake_temp[0]
        drive(state, specs, 10, throttle=1.0)
        assert hot > 0.4 and state.brake_temp[0] < hot


class TestEngine:
    def test_pushing_runs_the_engine_hotter_than_saving(self):
        push, specs = one_car(speed=40.0)
        lean, _ = one_car(speed=40.0)
        push.engine[0], lean.engine[0] = 1.0, 0.0
        drive(push, specs, 60, throttle=1.0)
        drive(lean, specs, 60, throttle=1.0)
        assert push.engine_temp[0] > lean.engine_temp[0] + 0.05

    def test_following_closely_starves_the_engine_of_air(self):
        clear, specs = one_car(speed=40.0)
        following, _ = one_car(speed=40.0)
        drive(clear, specs, 60, throttle=1.0)
        drive(following, specs, 60, throttle=1.0, conditions=Conditions.uniform(1, dirty_air=1.0))
        assert following.engine_temp[0] > clear.engine_temp[0]

    def test_a_blown_engine_gives_no_more_power(self):
        state, specs = one_car(speed=20.0)
        state.engine_failed[0] = True
        drive(state, specs, 3, throttle=1.0)
        assert state.speed[0] < 20.0


class TestTheAir:
    def test_the_slipstream_carries_a_car_past_its_usual_top_speed(self):
        alone, specs = one_car(speed=70.0)
        towed, _ = one_car(speed=70.0)
        drive(alone, specs, 30, throttle=1.0)
        drive(towed, specs, 30, throttle=1.0, conditions=Conditions.uniform(1, draft=1.0))
        assert towed.speed[0] > alone.speed[0] + 1.0

    def test_dirty_air_costs_grip_in_corners(self):
        clean, specs = one_car(speed=40.0)
        dirty, _ = one_car(speed=40.0)
        drive(clean, specs, 2, steer=1.0, throttle=0.5)
        drive(dirty, specs, 2, steer=1.0, throttle=0.5, conditions=Conditions.uniform(1, dirty_air=1.0))
        assert abs(dirty.heading[0]) < abs(clean.heading[0])


class TestHandling:
    def test_an_oversteering_car_slides_where_an_understeering_one_runs_wide(self):
        understeer, _ = one_car(speed=40.0)
        oversteer, _ = one_car(speed=40.0)
        drive(understeer, CarSpecs.uniform(1, balance=-0.8), 1.5, steer=1.0, throttle=0.5)
        drive(oversteer, CarSpecs.uniform(1, balance=0.8), 1.5, steer=1.0, throttle=0.5)
        # The oversteering car rotates more than its grip can follow, so it points further round and slides.
        assert abs(oversteer.heading[0]) > abs(understeer.heading[0])
        assert abs(np.dot(oversteer.velocity[0], oversteer.left[0])) > abs(np.dot(understeer.velocity[0], understeer.left[0]))


class TestLocalDamage:
    def test_bent_suspension_pulls_the_car_to_one_side(self):
        state, specs = one_car(speed=30.0)
        state.suspension_damage[0] = 1.0
        drive(state, specs, 3, throttle=0.4)
        assert abs(state.heading[0]) > 0.05

    def test_a_broken_front_wing_makes_the_car_turn_less(self):
        whole, specs = one_car(speed=40.0)
        broken, _ = one_car(speed=40.0)
        broken.wing_damage[0] = 1.0
        drive(whole, specs, 2, steer=1.0, throttle=0.5)
        drive(broken, specs, 2, steer=1.0, throttle=0.5)
        assert abs(broken.heading[0]) < abs(whole.heading[0])

    def test_a_puncture_slows_the_car_down(self):
        sound, specs = one_car(speed=60.0)
        flat, _ = one_car(speed=60.0)
        flat.punctured[0] = True
        drive(sound, specs, 10, throttle=1.0)
        drive(flat, specs, 10, throttle=1.0)
        assert flat.speed[0] < sound.speed[0] * 0.9


class TestTheSurface:
    def test_a_rubbered_line_grips_more_than_marbles(self):
        rubber, specs = one_car(speed=40.0)
        marbles, _ = one_car(speed=40.0)
        drive(rubber, specs, 2, steer=1.0, throttle=0.5, conditions=Conditions.uniform(1, surface_grip=1.05))
        drive(marbles, specs, 2, steer=1.0, throttle=0.5, conditions=Conditions.uniform(1, surface_grip=0.85))
        assert abs(rubber.heading[0]) > abs(marbles.heading[0])


class TestTheDay:
    def test_tyres_run_hotter_on_a_hot_day(self):
        cool, specs = one_car(speed=30.0)
        hot, _ = one_car(speed=30.0)
        # Along a straight, tyres settle toward the air's temperature.
        drive(cool, specs, 40, throttle=0.3, conditions=Conditions.uniform(1, air_temperature=0.0))
        drive(hot, specs, 40, throttle=0.3, conditions=Conditions.uniform(1, air_temperature=1.0))
        assert hot.tyre_temp[0] > cool.tyre_temp[0] + 0.03

    def test_a_headwind_costs_top_speed_and_a_tailwind_adds_it(self):
        still, specs = one_car(speed=70.0)
        into, _ = one_car(speed=70.0)
        with_it, _ = one_car(speed=70.0)
        drive(still, specs, 30, throttle=1.0)
        drive(into, specs, 30, throttle=1.0, conditions=Conditions.uniform(1, headwind=8.0))
        drive(with_it, specs, 30, throttle=1.0, conditions=Conditions.uniform(1, headwind=-8.0))
        assert into.speed[0] < still.speed[0] - 1 and with_it.speed[0] > still.speed[0] + 1
