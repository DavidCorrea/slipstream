"""Trained networks run with numpy alone, so a race can have them at the wheel where PyTorch isn't available (in a
browser, see browser.py).

A saved Stable-Baselines3 network is exported once (`export`, which runs where PyTorch is) to the weights its
actor uses: the layers that turn what a driver senses into its actions, and the memory of a driver by feel (one
LSTM layer). The critic, which only matters while training, is left out. `NumpyNetwork` then does what the
network's `predict` did, deterministic actions clipped to the action space, with the same handling of memory.

    .venv/bin/python -m slipstream.numpy_network runs/driver-rivals/checkpoints/step-000036016128.zip brains/driver-rivals/checkpoints/
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def export(model, path: Path):
    """Writes the actor of a Stable-Baselines3 model (PPO or RecurrentPPO, tanh layers) to an .npz file. Anything
    else is refused, naming what doesn't fit, rather than exported to act differently from how it was trained."""
    policy = model.policy
    layers = list(policy.mlp_extractor.policy_net)
    activations = {type(layer).__name__ for layer in layers[1::2]}
    if activations - {'Tanh'}:
        raise ValueError(f'Only networks with Tanh layers can be exported, this one has {", ".join(sorted(activations))}')
    if getattr(policy, 'squash_output', False):
        raise ValueError('Networks that squash their actions are not supported')
    weights = {
        'observation_size': np.array(model.observation_space.shape[0]),
        'action_low': model.action_space.low, 'action_high': model.action_space.high,
        'action_weight': _numpy(policy.action_net.weight), 'action_bias': _numpy(policy.action_net.bias),
    }
    for index, layer in enumerate(layers[0::2]):
        weights[f'layer_{index}_weight'] = _numpy(layer.weight)
        weights[f'layer_{index}_bias'] = _numpy(layer.bias)
    if hasattr(policy, 'lstm_actor'):
        lstm = policy.lstm_actor
        if lstm.num_layers != 1:
            raise ValueError(f'Only one LSTM layer can be exported, this network has {lstm.num_layers}')
        weights['lstm_input_weight'] = _numpy(lstm.weight_ih_l0)
        weights['lstm_hidden_weight'] = _numpy(lstm.weight_hh_l0)
        weights['lstm_bias'] = _numpy(lstm.bias_ih_l0) + _numpy(lstm.bias_hh_l0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file:
        np.savez_compressed(file, **weights)


def _numpy(parameter):
    return parameter.detach().cpu().numpy().astype(np.float32)


class NumpyNetwork:
    def __init__(self, weights):
        self.weights = weights
        self.layers = []
        while f'layer_{len(self.layers)}_weight' in weights:
            self.layers.append((weights[f'layer_{len(self.layers)}_weight'], weights[f'layer_{len(self.layers)}_bias']))
        self.has_memory = 'lstm_input_weight' in weights
        self.observation_space = SimpleNamespace(shape=(int(weights['observation_size']),))
        self.action_space = SimpleNamespace(shape=weights['action_low'].shape, low=weights['action_low'], high=weights['action_high'])

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as saved:
            return cls({name: saved[name] for name in saved.files})

    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        """Actions for a batch of observations, and the memory to hand back next time (None without one). As in
        Stable-Baselines3, a car whose `episode_start` is set forgets what it remembered."""
        features = np.asarray(observations, dtype=np.float32)
        if self.has_memory:
            features, state = self._remember(features, state, episode_start)
        for weight, bias in self.layers:
            features = np.tanh(features @ weight.T + bias)
        actions = features @ self.weights['action_weight'].T + self.weights['action_bias']
        return np.clip(actions, self.action_space.low, self.action_space.high), state

    def _remember(self, features, state, episode_start):
        hidden_size = self.weights['lstm_hidden_weight'].shape[1]
        if state is None:
            state = (np.zeros((len(features), hidden_size), np.float32), np.zeros((len(features), hidden_size), np.float32))
        hidden, cell = state
        if episode_start is not None:
            keep = 1.0 - np.asarray(episode_start, dtype=np.float32)[:, None]
            hidden, cell = hidden * keep, cell * keep
        gates = features @ self.weights['lstm_input_weight'].T + hidden @ self.weights['lstm_hidden_weight'].T + self.weights['lstm_bias']
        # PyTorch's gate order: input, forget, cell, output.
        remember, forget, candidate, reveal = np.split(gates, 4, axis=1)
        cell = _sigmoid(forget) * cell + _sigmoid(remember) * np.tanh(candidate)
        hidden = _sigmoid(reveal) * np.tanh(cell)
        return hidden, (hidden, cell)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    parser = argparse.ArgumentParser(description='Export trained networks for running with numpy alone')
    parser.add_argument('checkpoints', nargs='+', type=Path, help='saved .zip networks')
    parser.add_argument('destination', type=Path, help='folder to write the .npz files to')
    arguments = parser.parse_args()
    from .brains import load_network
    for checkpoint in arguments.checkpoints:
        target = arguments.destination / f'{checkpoint.stem}.npz'
        export(load_network(checkpoint), target)
        print(f'{checkpoint} -> {target} ({target.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
