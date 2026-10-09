import numpy as np

from slipstream.car import DT, FUEL_CAPACITY, GRAVITY, CarSpecs, CarState, step_cars


def one_car(speed=0.0, fuel=FUEL_CAPACITY, **specs):
    state = CarState.at([[0.0, 0.0]], [0.0], fuel=fuel)
    state.velocity[0] = [speed, 0.0]
    return state, CarSpecs.uniform(1, **specs)


def drive(state, specs, seconds, steer=0.0, throttle=0.0, brake=0.0, on_track=True):
    for _ in range(int(seconds / DT)):
        step_cars(state, specs, np.array([steer]), np.array([throttle]), np.array([brake]), np.array([on_track]))
    return state


class TestStraightLine:
    def test_accelerates_toward_but_never_past_top_speed(self):
        state, specs = one_car(top_speed=60.0)
        drive(state, specs, 60, throttle=1)
        assert 55.0 < state.speed[0] <= 60.0

    def test_stops_under_braking_and_never_reverses(self):
        state, specs = one_car(speed=40.0)
        drive(state, specs, 10, brake=1)
        assert state.speed[0] == 0.0

    def test_brakes_no_harder_than_the_tyres_allow(self):
        state, specs = one_car(speed=40.0, braking=60.0, grip=1.0)
        drive(state, specs, 1, brake=1)
        assert 40.0 - state.speed[0] <= GRAVITY * 1.0 + 0.5

    def test_is_slower_off_the_tarmac(self):
        tarmac, specs = one_car()
        grass, _ = one_car()
        drive(tarmac, specs, 5, throttle=1)
        drive(grass, specs, 5, throttle=1, on_track=False)
        assert grass.speed[0] < tarmac.speed[0] * 0.8

    def test_is_slower_to_pick_up_speed_with_a_full_tank(self):
        heavy, specs = one_car()
        light, _ = one_car(fuel=2.0)
        drive(heavy, specs, 3, throttle=1)
        drive(light, specs, 3, throttle=1)
        assert light.speed[0] > heavy.speed[0]


class TestCornering:
    def test_turns_without_sliding_when_within_grip(self):
        state, specs = one_car(speed=15.0)
        drive(state, specs, 0.5, steer=0.4, throttle=0.3)
        assert state.heading[0] > 0.05
        assert not state.sliding[0]

    def test_slides_wide_when_asking_for_more_turn_than_grip_allows(self):
        state, specs = one_car(speed=55.0)
        drive(state, specs, 0.3, steer=1.0)
        assert state.sliding[0]
        # The turn it managed is bounded by grip: centripetal acceleration never beats the tyres.
        assert state.lateral_load[0] <= 1.0 + 1e-6

    def test_has_less_turn_left_while_braking_hard(self):
        coasting, specs = one_car(speed=40.0)
        braking, _ = one_car(speed=40.0)
        drive(coasting, specs, 0.5, steer=1.0)
        drive(braking, specs, 0.5, steer=1.0, brake=1.0)
        assert braking.heading[0] < coasting.heading[0]


class TestWearAndFuel:
    def test_burns_fuel_only_on_the_throttle(self):
        state, specs = one_car()
        drive(state, specs, 5)
        assert state.fuel[0] == FUEL_CAPACITY
        drive(state, specs, 5, throttle=1)
        assert state.fuel[0] < FUEL_CAPACITY

    def test_has_no_drive_with_an_empty_tank(self):
        state, specs = one_car(fuel=0.0)
        drive(state, specs, 3, throttle=1)
        assert state.speed[0] == 0.0

    def test_wears_tyres_faster_when_sliding(self):
        gentle, specs = one_car(speed=15.0)
        sliding, _ = one_car(speed=55.0)
        drive(gentle, specs, 2, steer=0.2, throttle=0.3)
        drive(sliding, specs, 2, steer=1.0, throttle=1.0)
        assert sliding.tyre_wear[0] > gentle.tyre_wear[0] * 2


class TestSetupAndCondition:
    def test_soft_tyres_grip_more_and_wear_faster_than_hards(self):
        from slipstream.car import COMPOUNDS, effective_grip
        soft, specs = one_car(speed=30.0)
        hard, _ = one_car(speed=30.0)
        soft.compound[0], hard.compound[0] = COMPOUNDS.index('soft'), COMPOUNDS.index('hard')
        assert effective_grip(soft, specs, 0.0)[0] > effective_grip(hard, specs, 0.0)[0]
        drive(soft, specs, 3, steer=0.6, throttle=0.5)
        drive(hard, specs, 3, steer=0.6, throttle=0.5)
        assert soft.tyre_wear[0] > hard.tyre_wear[0] * 2

    def test_more_wing_grips_more_but_tops_out_lower(self):
        from slipstream.car import effective_grip, effective_top_speed
        low, specs = one_car()
        high, _ = one_car()
        low.wing[0], high.wing[0] = 0.0, 1.0
        assert effective_grip(high, specs, 0.0)[0] > effective_grip(low, specs, 0.0)[0]
        assert effective_top_speed(high, specs)[0] < effective_top_speed(low, specs)[0]

    def test_pushing_the_engine_is_quicker_and_thirstier(self):
        lean, specs = one_car()
        push, _ = one_car()
        lean.engine[0], push.engine[0] = 0.0, 1.0
        drive(lean, specs, 3, throttle=1)
        drive(push, specs, 3, throttle=1)
        assert push.speed[0] > lean.speed[0]
        assert push.fuel[0] < lean.fuel[0]

    def test_damage_costs_top_speed(self):
        clean, specs = one_car(top_speed=60.0)
        damaged, _ = one_car(top_speed=60.0)
        damaged.damage[0] = 1.0
        drive(clean, specs, 40, throttle=1)
        drive(damaged, specs, 40, throttle=1)
        assert damaged.speed[0] < clean.speed[0] * 0.95

    def test_worn_brakes_stop_less_hard_and_braking_wears_them(self):
        fresh, specs = one_car(speed=40.0, braking=12.0)
        worn, _ = one_car(speed=40.0, braking=12.0)
        worn.brake_wear[0] = 1.0
        drive(fresh, specs, 1, brake=1)
        drive(worn, specs, 1, brake=1)
        assert worn.speed[0] > fresh.speed[0]
        assert fresh.brake_wear[0] > 0


class TestSlidingToAStop:
    def test_a_car_sliding_sideways_on_the_grass_comes_to_rest(self):
        # Slowing a car that has (almost) stopped going forward takes almost no grip; the rest must go to stopping
        # the slide. (Spent on a forward speed of 1e-16 m/s, it left a car gliding sideways forever.)
        # Grip below what the grass drag asks for, like worn slicks on a wet track.
        state, specs = one_car(fuel=0.0, grip=0.5)
        # At most headings, rounding leaves a sideways slide with a forward speed of about 1e-16 every tick.
        state.heading[0] = 0.42
        state.velocity[0] = state.left[0] * 5.0
        drive(state, specs, 5, on_track=False)
        assert np.linalg.norm(state.velocity[0]) < 0.1
