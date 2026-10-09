import numpy as np

from slipstream.car import COMPOUNDS, FUEL_CAPACITY
from slipstream.imitation import collect, strategist_actions
from slipstream.pitwall import pitwall_decisions, pitwall_observe
from slipstream.pitwall_env import PitWallEnv
from slipstream.race import Race
from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.strategy import scripted_strategy
from slipstream.track import generate_track
from slipstream.weather import Weather


def scripted(race, personality):
    return scripted_controls(race)


class TestTheStrategistsChoicesAsOutputs:
    def test_decode_back_to_exactly_what_the_strategist_decided(self):
        race = Race(generate_track(3), CarSpecs.uniform(4), laps=8, weather=Weather.from_forecast('wet', 600, np.random.default_rng(0)))
        race.cars.tyre_wear[:] = [0.1, 0.8, 0.3, 0.9]
        race.cars.fuel[:] = [50, 50, 5, 30]
        race.cars.compound[:] = [COMPOUNDS.index(name) for name in ('medium', 'soft', 'wet', 'hard')]
        for wetness in (0.0, 0.5, 0.9):
            race.weather.wetness = wetness
            call, plan = scripted_strategy(race)
            decoded_call, decoded_plan = pitwall_decisions(strategist_actions(race))
            assert (decoded_call == call).all()
            assert (decoded_plan.compound == plan.compound).all()
            assert np.allclose(decoded_plan.fuel, np.minimum(plan.fuel, FUEL_CAPACITY))
            assert (decoded_plan.repair == plan.repair).all() and (decoded_plan.brakes == plan.brakes).all()


class TestCollectingDecisions:
    def test_pair_every_real_decision_with_what_the_strategist_did(self):
        env = PitWallEnv(scripted, groups=1, cars=4, laps=3, seed=2)
        observations, actions = collect(env, decisions=20)
        assert len(observations) == len(actions) >= 20
        assert observations.shape[1] == env.observation_space.shape[0] and actions.shape[1] == env.action_space.shape[0]
        assert np.abs(actions).max() <= 1.0


class TestImitating:
    def test_a_fitted_pit_wall_makes_the_strategists_pit_calls(self):
        from stable_baselines3 import PPO
        from slipstream.imitation import imitate
        env = PitWallEnv(scripted, groups=1, cars=8, laps=(3, 6), seed=4)
        observations, actions = collect(env, decisions=400)
        model = PPO('MlpPolicy', env, device='cpu', seed=0, n_steps=8, batch_size=64, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
        imitate(model.policy, observations, actions, epochs=200)
        predicted, _ = model.predict(observations, deterministic=True)
        agree = (predicted[:, 0] > 0) == (actions[:, 0] > 0)
        assert agree.mean() > 0.9
