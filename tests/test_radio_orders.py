import pytest

from slipstream import personality as traits
from slipstream.session import SCRIPTED, RaceSession
from slipstream.strategy import RADIO_CALLS


def session(cars=3, laps=3):
    race = RaceSession(SCRIPTED, seed=5, cars=cars, laps=laps)
    race.set_yours(1)
    return race


class TestPaceOrders:
    def test_push_goes_out_on_your_drivers_radio_whatever_the_pit_wall_would_say(self):
        race = session()
        race.set_radio(pace='save')
        race.advance(10)
        assert race.race.radio[1] == RADIO_CALLS.index('save')
        race.set_radio(pace='push')
        race.advance(10)
        assert race.race.radio[1] == RADIO_CALLS.index('push')

    def test_normal_says_nothing_and_auto_hands_the_radio_back_to_the_pit_wall(self):
        race = session()
        race.set_radio(pace='normal')
        race.advance(10)
        assert race.race.radio[1] == RADIO_CALLS.index('nothing')
        race.set_radio(pace='auto')
        assert race.radio_orders['pace'] == 'auto'

    def test_orders_reach_only_your_car(self):
        from slipstream.strategy import scripted_radio
        race = session()
        race.set_radio(pace='push')
        expected = scripted_radio(race.race)
        race._decide()
        assert [race.race.radio[car] for car in (0, 2)] == [expected[car] for car in (0, 2)]


class TestApproachOrders:
    def test_attack_makes_your_driver_more_aggressive_and_hungrier_for_places(self):
        race = session()
        base = race.driving_personality()[1].copy()
        race.set_radio(approach='attack')
        now = race.driving_personality()[1]
        for name in ('aggression', 'overtaking'):
            index = traits.TRAITS.index(name)
            assert now[index] > base[index]

    def test_bring_it_home_takes_fewer_risks_and_saves_the_car(self):
        race = session()
        base = race.driving_personality()[1].copy()
        race.set_radio(approach='bring_it_home')
        now = race.driving_personality()[1]
        assert now[traits.TRAITS.index('risk')] < base[traits.TRAITS.index('risk')]
        assert now[traits.TRAITS.index('conservation')] > base[traits.TRAITS.index('conservation')]

    def test_never_changes_anyone_else_or_the_drivers_usual_self(self):
        race = session()
        others = race.driving_personality()[[0, 2]].copy()
        race.set_radio(approach='attack')
        assert (race.driving_personality()[[0, 2]] == others).all()
        race.set_radio(approach='auto')
        assert (race.driving_personality()[1] == race.personality[1]).all()

    def test_traits_stay_in_range(self):
        race = session()
        race.set_stats({'aggression': 1.0, 'overtaking': 1.0, 'risk': 1.0}, 1)
        race.set_radio(approach='attack')
        assert race.driving_personality().max() <= 1.0


class TestPitOrders:
    def test_box_this_lap_brings_your_car_in_even_on_the_ai_strategy(self):
        race = session(laps=4)
        race.set_pit(box=True)
        stopped = False
        for _ in range(400):
            frame = race.advance(20)
            stopped |= any(stop['car'] == 1 for stop in frame['events']['pitStopped'])
            if stopped:
                break
        assert stopped

    def test_stay_out_keeps_your_car_out_for_the_lap_the_ai_would_call_it_in(self):
        race = session(laps=4)
        race.race.cars.tyre_wear[1] = 0.95
        race.set_pit(stay_out=True)
        stopped = False
        while race.race.lap_of()[1] < 1:
            frame = race.advance(20)
            stopped |= any(stop['car'] == 1 for stop in frame['events']['pitStopped'])
        assert not stopped


class TestRefusingOrders:
    def test_unknown_orders_are_refused_naming_the_choices(self):
        race = session()
        with pytest.raises(ValueError, match='push'):
            race.set_radio(pace='flat out')
        with pytest.raises(ValueError, match='attack'):
            race.set_radio(approach='kamikaze')

    def test_frames_say_what_your_driver_was_told(self):
        race = session()
        race.set_radio(pace='push', approach='defend')
        assert race.advance(5)['yourRadio'] == {'pace': 'push', 'approach': 'defend'}
