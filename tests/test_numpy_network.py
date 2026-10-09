import numpy as np
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from slipstream.brains import NetworkDriver, load_network
from slipstream.driver_env import DriverEnv
from slipstream.env import RaceVecEnv
from slipstream.numpy_network import export


def sb3_actions(model, observations, starts):
    actions, memory = [], None
    for step, observation in enumerate(observations):
        if model.policy.__class__.__name__.startswith('Recurrent'):
            action, memory = model.predict(observation, state=memory, episode_start=starts[step], deterministic=True)
        else:
            action, _ = model.predict(observation, deterministic=True)
        actions.append(action)
    return np.array(actions)


def numpy_actions(network, observations, starts):
    actions, memory = [], None
    for step, observation in enumerate(observations):
        action, memory = network.predict(observation, state=memory, episode_start=starts[step], deterministic=True)
        actions.append(action)
    return np.array(actions)


class TestNetworksWithoutPyTorch:
    def test_a_driver_with_a_memory_acts_exactly_as_trained_through_resets(self, tmp_path):
        model = RecurrentPPO('MlpLstmPolicy', DriverEnv(races=1, cars=3, seed=0), device='cpu', seed=1,
                             policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), lstm_hidden_size=16))
        rng = np.random.default_rng(2)
        observations = rng.normal(size=(12, 3, model.observation_space.shape[0])).astype(np.float32)
        starts = np.zeros((12, 3), dtype=bool)
        starts[0] = True
        starts[6, 1] = True    # one car's race ends and its memory starts again
        export(model, tmp_path / 'driver.npz')
        network = load_network(tmp_path / 'driver.npz')
        assert np.allclose(numpy_actions(network, observations, starts), sb3_actions(model, observations, starts), atol=1e-5)

    def test_a_network_without_memory_acts_exactly_as_trained(self, tmp_path):
        model = PPO('MlpPolicy', RaceVecEnv(races=1, cars=2, seed=0), device='cpu', seed=3)
        observations = np.random.default_rng(4).normal(size=(3, 2, model.observation_space.shape[0])).astype(np.float32)
        starts = np.ones((3, 2), dtype=bool)
        export(model, tmp_path / 'pitwall.npz')
        network = load_network(tmp_path / 'pitwall.npz')
        assert np.allclose(numpy_actions(network, observations, starts), sb3_actions(model, observations, starts), atol=1e-5)

    def test_keeps_actions_inside_what_the_car_accepts(self, tmp_path):
        model = PPO('MlpPolicy', RaceVecEnv(races=1, cars=2, seed=0), device='cpu', seed=3)
        with __import__('torch').no_grad():
            model.policy.action_net.bias.fill_(5.0)
        export(model, tmp_path / 'eager.npz')
        actions, _ = load_network(tmp_path / 'eager.npz').predict(np.zeros((2, model.observation_space.shape[0]), dtype=np.float32))
        assert (actions == 1.0).all()

    def test_drives_a_race_like_any_other_driver(self, tmp_path):
        model = RecurrentPPO('MlpLstmPolicy', DriverEnv(races=1, cars=2, seed=0), device='cpu',
                             policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16]), lstm_hidden_size=8))
        export(model, tmp_path / 'driver.npz')
        driver = NetworkDriver(load_network(tmp_path / 'driver.npz'), 2)
        env = DriverEnv(races=1, cars=2, seed=5)
        env.reset()
        assert driver.feels and driver.recurrent
        assert driver.act(env.races[0], None).shape == (2, 2)

    def test_refuses_a_network_it_cannot_reproduce(self, tmp_path):
        import torch
        model = PPO('MlpPolicy', RaceVecEnv(races=1, cars=2, seed=0), device='cpu', policy_kwargs=dict(activation_fn=torch.nn.ReLU))
        with pytest.raises(ValueError, match='Tanh'):
            export(model, tmp_path / 'relu.npz')
