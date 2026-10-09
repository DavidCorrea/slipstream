import numpy as np

from slipstream.car import CarSpecs
from slipstream.race import Race
from slipstream.strategy import condition_value
from slipstream.track import generate_track

STOP_COST = 2.7   # about nine seconds of progress at racing pace, in reward


def race_at(laps_left, wear, laps=10):
    race = Race(generate_track(5), CarSpecs.uniform(1), laps)
    race.progress[0] = (laps - laps_left) * race.track.length
    race.cars.tyre_wear[0] = wear
    return race


class TestConditionValue:
    def test_values_fresh_tyres_above_worn_ones(self):
        assert condition_value(race_at(5, 0.0))[0] > condition_value(race_at(5, 0.6))[0]

    def test_makes_a_mid_race_stop_on_worn_tyres_worth_more_than_it_costs(self):
        gain = condition_value(race_at(5, 0.0))[0] - condition_value(race_at(5, 0.6))[0]
        assert gain > STOP_COST

    def test_makes_new_tyres_worth_nothing_much_on_the_last_lap(self):
        gain = condition_value(race_at(0.5, 0.0))[0] - condition_value(race_at(0.5, 0.6))[0]
        assert gain < STOP_COST / 2

    def test_counts_the_wear_still_to_come(self):
        # Two laps on half-worn tyres cost more than twice one lap on them: they keep wearing.
        one = condition_value(race_at(1, 0.0))[0] - condition_value(race_at(1, 0.5))[0]
        two = condition_value(race_at(2, 0.0))[0] - condition_value(race_at(2, 0.5))[0]
        assert two > 2 * one
