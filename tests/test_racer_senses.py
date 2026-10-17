import numpy as np

from slipstream import pit
from slipstream.car import CarSpecs
from slipstream.race import Race
from slipstream.senses import RACER_SENSE_NAMES, SENSE_NAMES, racer_senses, senses
from slipstream.strategy import RADIO_CALLS, decode_racer, scripted_radio
from slipstream.track import generate_track
from slipstream.weather import Weather


def racing(cars=3, forecast=None):
    track = generate_track(6)
    weather = Weather.for_race(6, track.length, 3, forecast) if forecast else None
    return Race(track, CarSpecs.uniform(cars), 3, stewards=True, assisted=False, flags=True, weather=weather)


def sense(race, name):
    return racer_senses(race)[:, RACER_SENSE_NAMES.index(name)]


class TestWhatARacingDriverKnows:
    def test_everything_a_driver_by_feel_senses_comes_first(self):
        race = racing()
        rows = racer_senses(race)
        assert rows.shape == (3, len(RACER_SENSE_NAMES))
        assert np.allclose(rows[:, :len(SENSE_NAMES)], senses(race))

    def test_hears_the_tyres_brakes_and_damage_from_the_dash(self):
        race = racing()
        race.cars.tyre_wear[1], race.cars.damage[2] = 0.6, 0.3
        assert sense(race, 'tyre_wear').tolist() == [0.0, np.float32(0.6), 0.0]
        assert sense(race, 'damage')[2] == np.float32(0.3)

    def test_counts_down_the_start_lights(self):
        race = racing()
        assert sense(race, 'lights')[0] == 0.2 and sense(race, 'started')[0] == 0.0
        while not race.marshals.started(race.time):
            race.step(np.zeros(3), np.zeros(3), np.ones(3))
        assert sense(race, 'lights')[0] == 0.0 and sense(race, 'started')[0] == 1.0

    def test_sees_rain_coming_on_the_radar(self):
        race = racing(forecast='rain_coming')
        assert 0 < sense(race, 'rain_in')[0] < 1
        assert sense(race, 'rain_coming')[0] > 0

    def test_knows_how_far_the_pit_entry_and_its_box_are(self):
        race = racing()
        lane = race.pit
        race.progress[:] = lane.entry - 150.0 + race.track.length
        assert np.allclose(sense(race, 'pit_entry_in'), 0.5, atol=0.01)
        assert (sense(race, 'box_along') > 0.5).all()

    def test_hears_the_pit_wall_on_the_radio(self):
        race = racing()
        race.radio[:] = [0, RADIO_CALLS.index('push'), RADIO_CALLS.index('save')]
        assert sense(race, 'radio.push').tolist() == [0.0, 1.0, 0.0]
        assert sense(race, 'radio.save').tolist() == [0.0, 0.0, 1.0]


class TestTheButtonsOnTheWheel:
    def test_limiter_reverse_and_engine_mode_come_from_the_outputs(self):
        steer, throttle, brake, limiter, reverse, engine = decode_racer([[0.1, -0.5, 1.0, -1.0, -1.0]], 1)
        assert (throttle[0], brake[0], limiter[0], reverse[0], engine[0]) == (0.0, 0.5, True, False, 0.0)

    def test_the_reverse_button_only_engages_with_the_car_stopped(self):
        race = racing(cars=1)
        race.cars.velocity[0] = race.cars.forward[0] * 10.0
        race.step(np.zeros(1), np.zeros(1), np.zeros(1), reverse=np.ones(1))
        assert not race.cars.reversing[0]
        race.cars.velocity[0] = 0.0
        race.step(np.zeros(1), np.zeros(1), np.zeros(1), reverse=np.ones(1))
        assert race.cars.reversing[0]
        race.step(np.zeros(1), np.zeros(1), np.ones(1), reverse=np.zeros(1))
        assert not race.cars.reversing[0]

    def test_the_driver_changes_the_engine_mode_as_it_goes(self):
        race = racing(cars=2)
        race.step(np.zeros(2), np.zeros(2), np.ones(2), engine=np.array([1.0, np.nan]))
        assert race.cars.engine.tolist() == [1.0, 0.5]


class TestThePitWallsRadio:
    def test_asks_for_care_when_the_fuel_wont_last_and_a_push_in_a_fight(self):
        race = racing()
        race.progress[:] = [500.0, 495.0, 100.0]
        race.cars.velocity[:] = race.cars.forward * 30.0
        race.cars.fuel[2] = 1.0
        assert [RADIO_CALLS[call] for call in scripted_radio(race)] == ['push', 'push', 'save']


class TestFindingThePitLane:
    def approaching(self, called):
        race = racing(cars=1)
        lane = race.pit
        race.place(0, race.track.length + lane.entry - 120.0, -lane.side * 3.0)
        if called:
            race.pit_state[0] = pit.CALLED
        return race

    def test_a_called_car_approaching_the_entry_knows_which_way_the_lane_is(self):
        # Without this a car only learned the lane's side once in it, and the lane is on either side.
        race = self.approaching(called=True)
        assert sense(race, 'box_across')[0] > 0.5

    def test_a_car_that_isnt_called_isnt_pointed_at_the_lane(self):
        race = self.approaching(called=False)
        assert sense(race, 'box_across')[0] == 0.0

    def test_far_from_the_entry_it_says_nothing(self):
        race = racing(cars=1)
        race.place(0, race.track.length + race.pit.entry - 600.0)
        race.pit_state[0] = pit.CALLED
        assert sense(race, 'box_across')[0] == 0.0
