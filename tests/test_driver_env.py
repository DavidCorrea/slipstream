import numpy as np

from slipstream.car import BALANCE_RANGE
from slipstream.driver_env import RETIRE_COST, DriverEnv, random_cars
from slipstream.senses import SENSE_NAMES


class TestTheDriversWorld:
    def test_drivers_sense_rather_than_read_and_only_steer_and_press_a_pedal(self):
        env = DriverEnv(races=2, cars=3, seed=0)
        observations = env.reset()
        assert observations.shape == (6, len(SENSE_NAMES)) and env.action_space.shape == (2,)

    def test_every_race_brings_a_different_car_with_its_own_handling(self):
        env = DriverEnv(races=4, cars=6, seed=1)
        env.reset()
        balances = np.concatenate([race.specs.balance for race in env.races])
        assert balances.min() < -BALANCE_RANGE / 3 and balances.max() > BALANCE_RANGE / 3
        assert all(race.driving is not None for race in env.races)

    def test_cars_vary_only_as_far_as_the_games_own_cars_do(self):
        from slipstream.car import SPEC_DEFAULTS, SPEC_SPREAD
        cars = random_cars(500, np.random.default_rng(0))
        for name, default in SPEC_DEFAULTS.items():
            ratios = getattr(cars, name) / default
            assert ratios.min() >= 1 - SPEC_SPREAD and ratios.max() <= 1 + SPEC_SPREAD
            assert ratios.min() < 1 - SPEC_SPREAD * 0.9 and ratios.max() > 1 + SPEC_SPREAD * 0.9

    def test_drives_races_to_the_end_with_the_team_calling_the_stops(self):
        env = DriverEnv(races=1, cars=2, laps=1, seed=2)
        env.reset()
        for _ in range(3000):
            observations, rewards, dones, infos = env.step(np.tile([0.0, 0.3], (2, 1)))
            if dones.any():
                break
        assert dones.all() and 'race' in infos[0]


class TestLearningToDrive:
    def test_a_recurrent_network_trains_on_it(self):
        from sb3_contrib import RecurrentPPO
        env = DriverEnv(races=2, cars=2, laps=1, seed=3)
        model = RecurrentPPO('MlpLstmPolicy', env, n_steps=16, batch_size=32, n_epochs=1, device='cpu', seed=0)
        model.learn(64)
        assert model.num_timesteps >= 64


class TestBeachedCars:
    def test_a_car_that_stops_making_progress_is_retired_and_pays_for_it(self):
        from slipstream.driver_env import RETIRE_COST, STALLED_SECONDS
        env = DriverEnv(races=1, cars=2, laps=3, seed=4)
        env.reset()
        total = np.zeros(2)
        for _ in range(int(STALLED_SECONDS * 10) + 20):
            observations, rewards, dones, infos = env.step(np.tile([0.0, -1.0], (2, 1)))
            total += rewards
            if dones.any():
                break
        assert dones.all()
        assert total.max() < -RETIRE_COST


class TestRacingScriptedRivals:
    def test_only_its_own_cars_are_trained_and_every_rival_is_on_track(self):
        env = DriverEnv(races=2, cars=8, rivals=3, seed=0)
        observations = env.reset()
        assert env.num_envs == 2 * 5 and observations.shape[0] == 10
        assert all(race.count == 8 for race in env.races)
        assert all(len(learners) == 5 for learners in env.learner_cars)

    def test_rivals_drive_themselves_whatever_the_network_asks(self):
        env = DriverEnv(races=1, cars=4, rivals=2, seed=1)
        env.reset()
        race = env.races[0]
        rivals = np.setdiff1d(np.arange(4), env.learner_cars[0])
        # The network asks its cars to stand on the brake; the scripted rivals still get going.
        for _ in range(60):
            env.step(np.tile([0.0, -1.0], (env.num_envs, 1)).astype(np.float32))
        assert (race.progress[rivals] > 20).all()
        assert (race.progress[env.learner_cars[0]] < 5).all()

    def test_places_lost_to_a_rival_cost_the_network(self):
        env = DriverEnv(races=1, cars=2, rivals=1, seed=2, rewards={'places': 2.0})
        env.reset()
        race = env.races[0]
        mine, rival = env.learner_cars[0][0], 1 - env.learner_cars[0][0]
        race.progress[[mine, rival]] = [10.5, 10.0]
        race.cars.position[mine] = race.track.point_at(np.array([8.0]), np.array([4.0]))[0]
        race.cars.position[rival] = race.track.point_at(np.array([12.0]))[0]
        # Same move as the env test of a place gained, from the other side: the rival goes past.
        reward = env.step(np.zeros((1, 2), dtype=np.float32))[1][0]
        assert reward < -0.05

    def test_a_finished_race_starts_a_new_one_with_new_seats_for_its_cars(self):
        env = DriverEnv(races=1, cars=6, rivals=2, seed=3)
        env.reset()
        seen = set()
        for _ in range(5):
            env.races[0].tick = int(env.races[0].time_limit / 0.05)
            _, _, dones, infos = env.step(np.zeros((env.num_envs, 2), dtype=np.float32))
            assert dones.all() and 'race' in infos[0]
            seen.add(tuple(env.learner_cars[0]))
        assert len(seen) > 1


class TestRacingAmongProps:
    def test_training_races_stand_props_round_the_track_as_the_broadcast_does(self):
        env = DriverEnv(races=2, cars=2, seed=4)
        env.reset()
        assert all(race.props is not None and len(race.props.props) > 0 for race in env.races)

    def test_crashing_out_on_a_prop_costs_as_much_as_getting_beached(self):
        env = DriverEnv(races=1, cars=2, seed=5)
        env.reset()
        race = env.races[0]
        car = race.cars
        # Car 0 flung at 80 m/s across the track, into whatever stands beside it.
        car.velocity[0] = race.track.normals[int(race.progress[0] % race.track.length / 2.0) % race.track.size] * 80.0
        car.heading[0] = float(np.arctan2(car.velocity[0][1], car.velocity[0][0]))
        crashed_cost = None
        for _ in range(60):
            _, rewards, _, _ = env.step(np.tile([0.0, 1.0], (2, 1)))
            if race.retired[0]:
                crashed_cost = rewards[0]
                break
        assert crashed_cost is not None and crashed_cost <= -RETIRE_COST
