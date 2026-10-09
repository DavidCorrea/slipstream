import numpy as np

from slipstream import pit
from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.pitwall import PITWALL_ACTION_NAMES, PITWALL_OBSERVATION_NAMES, pitwall_decisions
from slipstream.pitwall_env import PitWallEnv, _restore, _save, awaiting_decision
from slipstream.race import Race
from slipstream.track import generate_track


def scripted(race, personality):
    return scripted_controls(race)


def no_stops(count):
    actions = np.zeros((count, len(PITWALL_ACTION_NAMES)))
    actions[:, 0] = -1.0
    actions[:, PITWALL_ACTION_NAMES.index('weather_tyres')] = -1.0
    return actions


class TestGhostRaces:
    def test_start_everyone_on_pole_and_let_them_pass_through_each_other(self):
        race = Race(generate_track(2), CarSpecs.uniform(4), laps=1, ghosts=True)
        assert np.allclose(race.cars.position, race.cars.position[0])
        assert race._resolve_contacts() == []

    def test_end_only_when_every_car_has_finished(self):
        race = Race(generate_track(2), CarSpecs.uniform(2), laps=1, ghosts=True)
        race.finish_time[0] = 1.0
        race.tick = 100000
        assert not race.done


class TestWaiting:
    def test_leaves_a_waiting_car_exactly_as_it_was(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=2, ghosts=True)
        for _ in range(50):
            race.step(*scripted_controls(race))
        waiting = np.array([True, False])
        position, progress = race.cars.position[0].copy(), race.progress[0]
        saved = _save(race, waiting)
        race.step(*scripted_controls(race))
        _restore(race, saved, waiting)
        assert np.array_equal(race.cars.position[0], position) and race.progress[0] == progress
        assert race.progress[1] > progress - 1


class TestPitWallEnvironment:
    def test_asks_for_one_decision_per_lap_with_each_car_at_its_decision_point(self):
        env = PitWallEnv(scripted, groups=1, cars=3, laps=3, seed=1)
        observations = env.reset()
        assert observations.shape == (3, len(PITWALL_OBSERVATION_NAMES))
        assert awaiting_decision(env.groups[0]['race']).all()
        laps_before = env.groups[0]['race'].lap_of().copy()
        env.step(no_stops(3))
        assert (env.groups[0]['race'].lap_of() == laps_before + 1).all()
        assert awaiting_decision(env.groups[0]['race']).all()

    def test_charges_every_lap_in_minutes_and_ends_the_episode_at_the_flag(self):
        env = PitWallEnv(scripted, groups=1, cars=2, laps=2, seed=2)
        env.reset()
        rewards = []
        for _ in range(4):
            _, reward, done, infos = env.step(no_stops(2))
            rewards.append(reward)
            if done.all():
                break
        assert done.all()
        assert all((reward < 0).all() for reward in rewards)
        assert all('episode' in info and info['race']['finished'] for info in infos)

    def test_stops_a_car_when_the_pit_wall_calls_it_in(self):
        env = PitWallEnv(scripted, groups=1, cars=2, laps=4, seed=3)
        env.reset()
        actions = no_stops(2)
        actions[0, 0] = 1.0
        env.step(actions)
        race = env.groups[0]['race']
        assert race.stops[0] == 1 and race.stops[1] == 0

    def test_reads_its_outputs_like_the_driver_networks_strategy_outputs(self):
        actions = no_stops(1)
        actions[0, 0], actions[0, 1] = 1.0, -1.0
        call, plan = pitwall_decisions(actions)
        assert call[0] and plan.compound[0] == 0


class TestTrafficRaces:
    def test_ask_each_car_once_per_lap_and_settle_every_decision_by_the_flag(self):
        from slipstream.pitwall_env import TrafficRaces
        traffic = TrafficRaces(scripted, races=1, cars=3, laps=2, seed=4)
        decisions, records = [], []

        def decide(observations):
            decisions.append(len(observations))
            count = len(observations)
            actions = no_stops(count)
            return actions, np.zeros(count), np.zeros(count)

        def record(slot, decision, reward, done, following):
            records.append((slot, reward, done))

        ended = []
        for _ in range(20000):
            ended += traffic.tick(decide, record)
            if ended:
                break
        assert ended and ended[0]['laps'] == 2
        assert sum(decisions) == 6          # two laps, three cars, one decision each per lap
        assert len(records) == 6 and sum(done for _, _, done in records) == 3
        assert all(reward < 0.5 for _, reward, _ in records)


class TestStuckCars:
    def test_retire_a_car_that_stops_gaining_ground_and_charge_it_for_the_race_left(self):
        env = PitWallEnv(lambda race, personality: (np.zeros(race.count), np.zeros(race.count), np.ones(race.count)), groups=1, cars=2, laps=3, seed=5)
        group = env._new_group()
        assert group['out'].all()
        assert (group['penalty'] > 1.0).all()

    def test_fail_loudly_rather_than_drive_forever(self, monkeypatch):
        from slipstream import pitwall_env
        monkeypatch.setattr(pitwall_env, 'MAX_TICKS_PER_DECISION', 50)
        monkeypatch.setattr(pitwall_env, 'STUCK_SECONDS', 1e9)
        env = PitWallEnv(lambda race, personality: (np.zeros(race.count), np.zeros(race.count), np.ones(race.count)), groups=1, cars=2, laps=3, seed=5)
        import pytest
        with pytest.raises(RuntimeError, match='never reached'):
            env._new_group()


class TestWithADriverByFeel:
    def test_the_pit_wall_trains_and_benchmarks_on_a_driver_with_a_memory(self):
        from sb3_contrib import RecurrentPPO
        from slipstream.driver_env import DriverEnv
        from slipstream.pitwall_env import network_driver
        driver = RecurrentPPO('MlpLstmPolicy', DriverEnv(races=1, cars=2, laps=1, seed=0), n_steps=8, batch_size=16, device='cpu', seed=0)
        env = PitWallEnv(network_driver(driver), groups=1, cars=2, laps=2, seed=1)
        env.reset()
        observations, rewards, dones, infos = env.step(no_stops(2))
        assert observations.shape == (2, len(PITWALL_OBSERVATION_NAMES))


class TestComparingRaceTimes:
    def test_compares_two_finished_races_by_time(self):
        from slipstream.pitwall_env import relative_pace
        assert relative_pace(110.0, 100.0) == 1.1

    def test_scores_nothing_for_a_pit_wall_that_didnt_finish(self):
        from slipstream.pitwall_env import relative_pace
        assert relative_pace(100.0, np.inf) == 0.0

    def test_leaves_out_a_race_the_reference_didnt_finish_instead_of_scoring_it_infinite(self):
        from slipstream.pitwall_env import relative_pace
        assert relative_pace(np.inf, 100.0) is None
