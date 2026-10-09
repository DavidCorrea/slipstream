import pytest
from sb3_contrib import RecurrentPPO

from slipstream.train import FEEL_CARS, FEEL_RIVALS, FEEL_SETTINGS, TRAFFIC_REWARDS, copy_driver, driver_env, resume_driver, source_path


class TestTrainingADriverByFeel:
    def test_races_full_fields_where_places_are_worth_more(self):
        env = driver_env(races=1, seed=0)
        assert env.cars == FEEL_CARS == 8
        assert env.rivals == FEEL_RIVALS == 3 and env.num_envs == 5
        assert env.rewards['places'] == TRAFFIC_REWARDS['places'] > 1
        assert env.rewards['podium'] == TRAFFIC_REWARDS['podium'] > 1

    def test_resumes_at_the_current_learning_rate_not_the_saved_one(self, tmp_path):
        env = driver_env(races=1, cars=2, seed=0, rivals=0)
        saved = RecurrentPPO('MlpLstmPolicy', env, device='cpu', learning_rate=3e-4, n_steps=8, batch_size=16)
        saved.save(tmp_path / 'model')
        resumed = resume_driver(tmp_path / 'model.zip', env)
        assert resumed.lr_schedule(1.0) == FEEL_SETTINGS['learning_rate'] < 3e-4
        assert resumed.policy.optimizer.param_groups[0]['lr'] == FEEL_SETTINGS['learning_rate']

    def test_starts_a_new_run_as_a_copy_of_a_snapshot_keeping_its_experience(self, tmp_path):
        env = driver_env(races=1, cars=2, seed=0, rivals=0)
        source = RecurrentPPO('MlpLstmPolicy', env, device='cpu', n_steps=8, batch_size=16)
        source.num_timesteps = 1234
        (tmp_path / 'old-run' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'old-run' / 'shaping.json').write_text('{"start": 99}')
        source.save(tmp_path / 'old-run' / 'checkpoints' / 'step-000000001234')
        (tmp_path / 'new-run').mkdir()
        copy = copy_driver(tmp_path / 'old-run' / 'checkpoints' / 'step-000000001234.zip', env, tmp_path / 'new-run')
        assert copy.num_timesteps == 1234
        assert (tmp_path / 'new-run' / 'shaping.json').read_text() == '{"start": 99}'
        assert copy.tensorboard_log == str(tmp_path / 'new-run' / 'tensorboard')
        assert copy.lr_schedule(1.0) == FEEL_SETTINGS['learning_rate']


class TestNamingARunToCopy:
    def test_finds_a_runs_latest_model_or_one_of_its_snapshots(self, tmp_path):
        (tmp_path / 'main' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'main' / 'model.zip').write_bytes(b'')
        (tmp_path / 'main' / 'checkpoints' / 'step-000000000010.zip').write_bytes(b'')
        assert source_path('main', tmp_path) == tmp_path / 'main' / 'model.zip'
        assert source_path('main/step-000000000010', tmp_path) == tmp_path / 'main' / 'checkpoints' / 'step-000000000010.zip'

    def test_refuses_a_snapshot_that_isnt_there(self, tmp_path):
        with pytest.raises(SystemExit, match='No run to start from'):
            source_path('main/step-missing', tmp_path)
