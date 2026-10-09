import numpy as np

from slipstream.driver_env import DriverEnv
from slipstream.driver_imitation import collect_driving, imitate_driver
from slipstream.senses import SENSE_NAMES


class TestLearningFromTheScriptedDriver:
    def test_records_what_each_driver_sensed_and_did_in_sequences(self):
        env = DriverEnv(races=1, cars=3, laps=1, seed=0)
        observations, actions, starts = collect_driving(env, decisions=600, sequence_length=50)
        assert observations.shape[1:] == (50, len(SENSE_NAMES)) and actions.shape[1:] == (50, 2)
        assert starts.shape == observations.shape[:2] and starts[:, 0].all()
        assert np.abs(actions).max() <= 1.0

    def test_a_fitted_driver_steers_the_way_the_scripted_one_did(self):
        from sb3_contrib import RecurrentPPO
        env = DriverEnv(races=2, cars=3, laps=1, seed=1)
        observations, actions, starts = collect_driving(env, decisions=3000, sequence_length=50)
        model = RecurrentPPO('MlpLstmPolicy', env, n_steps=8, batch_size=16, device='cpu', seed=0)
        before = imitate_driver(model.policy, observations, actions, starts, epochs=0)
        after = imitate_driver(model.policy, observations, actions, starts, epochs=30)
        assert after < before / 2
