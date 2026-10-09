import numpy as np

from slipstream.car import COMPOUNDS, DT, CarSpecs, CarState, Conditions, compound_grip, step_cars
from slipstream.drivers import scripted_controls
from slipstream.observe import OBSERVATION_NAMES, observe
from slipstream.pitwall import PITWALL_OBSERVATION_NAMES, pitwall_observe
from slipstream.race import Race
from slipstream.strategy import ACTION_NAMES, decode, scripted_strategy
from slipstream.track import generate_track
from slipstream.weather import FORECASTS, Weather

SLICKS = [COMPOUNDS.index(name) for name in ('soft', 'medium', 'hard')]
INTERMEDIATE, WET = COMPOUNDS.index('intermediate'), COMPOUNDS.index('wet')


def soak(weather, seconds):
    for _ in range(int(seconds / DT)):
        weather.step(DT)
    return weather


class TestTheSky:
    def test_a_dry_race_stays_dry(self):
        weather = soak(Weather.from_forecast('dry', race_seconds=600, rng=np.random.default_rng(1)), 600)
        assert weather.wetness == 0.0 and weather.rain == 0.0

    def test_rain_soaks_the_track_gradually(self):
        weather = Weather.from_forecast('wet', race_seconds=600, rng=np.random.default_rng(1))
        assert weather.rain > 0
        soak(weather, 10)
        partly = weather.wetness
        soak(weather, 300)
        assert 0 < partly < weather.wetness <= 1

    def test_a_track_dries_once_the_rain_stops(self):
        weather = Weather.from_forecast('drying', race_seconds=600, rng=np.random.default_rng(1))
        start = weather.wetness
        soak(weather, 600)
        assert start > 0.6 and weather.wetness < start / 2

    def test_rain_on_the_way_arrives_during_the_race(self):
        weather = Weather.from_forecast('rain_coming', race_seconds=600, rng=np.random.default_rng(1))
        assert weather.wetness == 0.0
        soak(weather, 600)
        assert weather.wetness > 0.3

    def test_every_forecast_comes_up_from_a_seed(self):
        kinds = {Weather.from_seed(seed, race_seconds=600).forecast for seed in range(200)}
        assert kinds == set(FORECASTS)


class TestTyresForTheConditions:
    def test_slicks_are_fastest_on_a_dry_track(self):
        assert compound_grip(np.array(SLICKS[:1]), 0.0)[0] > compound_grip(np.array([INTERMEDIATE, WET]), 0.0).max()

    def test_intermediates_are_fastest_on_a_damp_track(self):
        grip = compound_grip(np.arange(len(COMPOUNDS)), 0.45)
        assert grip.argmax() == INTERMEDIATE

    def test_full_wets_are_fastest_on_a_soaked_track(self):
        grip = compound_grip(np.arange(len(COMPOUNDS)), 1.0)
        assert grip.argmax() == WET

    def test_wet_tyres_wear_out_fast_on_a_dry_track(self):
        def wear_after_corner(compound, wetness):
            state = CarState.at([[0.0, 0.0]], [0.0])
            state.compound[0] = compound
            state.velocity[0] = [30.0, 0.0]
            specs = CarSpecs.uniform(1)
            for _ in range(int(10 / DT)):
                step_cars(state, specs, np.array([0.3]), np.array([0.6]), np.array([0.0]), np.array([True]), Conditions.uniform(1, wetness=wetness))
            return state.tyre_wear[0]
        assert wear_after_corner(WET, 0.0) > 2 * wear_after_corner(WET, 1.0)


    def test_intermediates_last_several_laps_on_a_damp_track(self):
        race = Race(generate_track(3), CarSpecs.uniform(1), laps=3, weather=Weather.from_forecast('drying', 600, np.random.default_rng(0)))
        race.weather.wetness = 0.4
        race.weather.showers = [(0.0, 1e9, 0.3)]
        race.cars.compound[:] = INTERMEDIATE
        while not race.done:
            race.step(*scripted_controls(race))
        assert race.finished.all() and race.cars.tyre_wear[0] < 0.6


class TestRacingInTheRain:
    def test_the_scripted_driver_still_finishes_a_wet_race_on_wets(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=1, weather=Weather.from_forecast('wet', 300, np.random.default_rng(0)))
        race.cars.compound[:] = WET
        while not race.done:
            race.step(*scripted_controls(race))
        assert race.finished.all()

    def test_the_scripted_driver_gets_back_on_track_from_a_slide_on_the_wrong_tyres(self):
        # Caught out on slicks in a downpour, it still finds its way round rather than ploughing across the grass.
        for seed in (3, 99):
            weather = Weather.from_forecast('wet', 300, np.random.default_rng(0))
            weather.wetness = 0.85
            race = Race(generate_track(seed), CarSpecs.uniform(1), laps=2, weather=weather)
            race.cars.compound[:] = COMPOUNDS.index('medium')
            furthest = 0.0
            while not race.done:
                race.step(*scripted_controls(race))
                furthest = max(furthest, abs(race.lateral[0]))
            # Sliding off at speed on wet grass carries a car some way; it's the hundreds of metres that mean lost.
            assert race.finished.all() and furthest < 80

    def test_the_scripted_strategist_fits_intermediates_when_it_rains(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=6, weather=Weather.from_forecast('wet', 600, np.random.default_rng(0)))
        race.weather.wetness = 0.5
        race.cars.compound[:] = COMPOUNDS.index('medium')
        call, plan = scripted_strategy(race)
        assert call.all() and (plan.compound == INTERMEDIATE).all()

    def test_the_scripted_strategist_leaves_slicks_on_in_the_dry(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=6)
        call, plan = scripted_strategy(race)
        assert not call.any() and np.isin(plan.compound, SLICKS).all()


    def test_cars_start_on_tyres_that_suit_the_track(self):
        wet = Race(generate_track(3), CarSpecs.uniform(2), laps=6, weather=Weather.from_forecast('wet', 600, np.random.default_rng(0)))
        dry = Race(generate_track(3), CarSpecs.uniform(2), laps=6)
        assert not np.isin(wet.cars.compound, SLICKS).any() and (dry.cars.compound == COMPOUNDS.index('medium')).all()

    def test_a_wet_track_means_more_fuel_to_finish(self):
        from slipstream.strategy import fuel_to_finish
        wet = Race(generate_track(3), CarSpecs.uniform(1), laps=6, weather=Weather.from_forecast('wet', 600, np.random.default_rng(0)))
        dry = Race(generate_track(3), CarSpecs.uniform(1), laps=6)
        assert fuel_to_finish(wet)[0] > fuel_to_finish(dry)[0] * 1.1


class TestNetworksSeeTheWeather:
    def test_weather_inputs_come_after_every_older_input(self):
        assert OBSERVATION_NAMES[-4:] == ['compound.intermediate', 'compound.wet', 'wetness', 'rain']
        weather = PITWALL_OBSERVATION_NAMES.index('compound.intermediate')
        assert PITWALL_OBSERVATION_NAMES[weather:weather + 4] == ['compound.intermediate', 'compound.wet', 'wetness', 'rain']
        assert weather == PITWALL_OBSERVATION_NAMES.index('gap_behind') + 1
        # Everything older networks read stays where they expect it: the slick flags, and up to their last input.
        assert OBSERVATION_NAMES.index('compound.hard') == 64 and OBSERVATION_NAMES[74] == 'pit_decision_now'
        assert PITWALL_OBSERVATION_NAMES.index('compound.hard') == 14 and PITWALL_OBSERVATION_NAMES[25] == 'gap_behind'

    def test_observations_report_the_wetness(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=2, weather=Weather.from_forecast('wet', 300, np.random.default_rng(0)))
        race.weather.wetness = 0.7
        assert np.allclose(observe(race)[:, OBSERVATION_NAMES.index('wetness')], 0.7)
        assert np.allclose(pitwall_observe(race)[:, PITWALL_OBSERVATION_NAMES.index('wetness')], 0.7)

    def test_the_weather_tyre_output_picks_inters_or_wets_over_the_slick_choice(self):
        actions = np.zeros((3, len(ACTION_NAMES)))
        actions[:, ACTION_NAMES.index('weather_tyres')] = [-1.0, 0.0, 1.0]
        assert list(decode(actions, 3)[4].compound) == [COMPOUNDS.index('medium'), INTERMEDIATE, WET]

    def test_networks_from_before_the_weather_only_ever_pick_slicks(self):
        actions = np.ones((2, len(ACTION_NAMES) - 1))
        assert np.isin(decode(actions, 2)[4].compound, SLICKS).all()


class TestWatchingTheWeather:
    def test_a_chosen_forecast_reaches_the_viewer_with_every_frame(self):
        from slipstream.session import SCRIPTED, RaceSession
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=2, forecast='wet')
        assert session.intro()['weather']['forecast'] == 'wet'
        frame = session.advance(20)
        assert frame['weather']['wetness'] > 0.5 and frame['weather']['rain'] > 0

    def test_an_unknown_forecast_is_refused_by_name(self):
        import pytest
        from slipstream.session import SCRIPTED, RaceSession
        with pytest.raises(ValueError, match='monsoon'):
            RaceSession(SCRIPTED, seed=5, cars=2, laps=2, forecast='monsoon')


class TestTheDay:
    def test_every_race_has_its_own_air_temperature_and_wind(self):
        days = [Weather.from_seed(seed, race_seconds=600) for seed in range(40)]
        temperatures = [day.temperature for day in days]
        winds = [np.hypot(*day.wind) for day in days]
        assert min(temperatures) < 0.25 and max(temperatures) > 0.75
        assert max(winds) > 5 and min(winds) < 2

    def test_a_car_driving_into_the_wind_feels_a_headwind(self):
        weather = Weather.dry()
        weather.wind = np.array([-6.0, 0.0])
        race = Race(generate_track(3), CarSpecs.uniform(1), laps=1, weather=weather)
        race.cars.heading[0] = 0.0
        assert abs(race.conditions().headwind[0] - 6.0) < 1e-6
