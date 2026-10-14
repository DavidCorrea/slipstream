import functools

import numpy as np

from slipstream.parallel import ParallelRaces
from slipstream.racer_env import LEARNERS, RacerEnv
from slipstream.strategy import RACER_ACTIONS


class TestRacesOnSeveralCores:
    def test_play_every_workers_races_as_one_environment(self):
        races = ParallelRaces([functools.partial(RacerEnv, races=2, seed=seed) for seed in (1, 2)])
        try:
            observations = races.reset()
            assert races.num_envs == 4 * LEARNERS and observations.shape[0] == 4 * LEARNERS
            races.set_attr('personality_strength', 0.25)
            assert races.get_attr('personality_strength') == [0.25] * races.num_envs
            observations, rewards, dones, infos = races.step(np.zeros((races.num_envs, len(RACER_ACTIONS))))
            assert observations.shape[0] == rewards.shape[0] == dones.shape[0] == len(infos) == races.num_envs
        finally:
            races.close()
