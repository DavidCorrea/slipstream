"""Trained driver networks at the wheel: loading them, and keeping what a recurrent one remembers.

A driver that drives by feel (see senses.py) has a memory: what it felt a moment ago shapes what it does now. So
a network is handed to a race through a NetworkDriver, which carries that memory from one decision to the next for
every car in the race, starting fresh when the race does. Older drivers read the numbers instead (see observe.py)
and have no memory; a NetworkDriver drives them the same way.
"""
import json
import zipfile
from pathlib import Path

import numpy as np

from .observe import for_network, observe
from .senses import SENSE_NAMES, senses


def is_recurrent(path: Path):
    """Whether a saved network has a memory. Stable-Baselines3 records the policy's module in the file, and the
    recurrent ones come from sb3-contrib."""
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    return data['policy_class']['__module__'].startswith('sb3_contrib')


def load_network(path: Path):
    """Loads a saved driver or pit-wall network, recurrent or not."""
    if is_recurrent(path):
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device='cpu')
    from stable_baselines3 import PPO
    return PPO.load(path, device='cpu')


class NetworkDriver:
    def __init__(self, model, count):
        self.model = model
        self.recurrent = hasattr(model.policy, 'lstm_actor')
        # Drivers by feel sense (senses.py) and only steer and press a pedal.
        self.feels = model.observation_space.shape[0] == len(SENSE_NAMES) and model.action_space.shape[0] == 2
        self.memory = None
        self.starting = np.ones(count, dtype=bool)

    def act(self, race, personality):
        """The network's outputs for every car in the race, carrying its memory on to the next call."""
        observations = senses(race, personality) if self.feels else for_network(observe(race, personality), self.model)
        if not self.recurrent:
            actions, _ = self.model.predict(observations, deterministic=True)
            return actions
        actions, self.memory = self.model.predict(observations, state=self.memory, episode_start=self.starting, deterministic=True)
        self.starting[:] = False
        return actions
