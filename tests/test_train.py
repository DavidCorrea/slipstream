import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure

from slipstream.driver_env import RETIRE_COST
from slipstream.env import REWARDS
from slipstream.train import (FEEL_CARS, FEEL_RIVALS, FEEL_SETTINGS, PASSING_FILE, TRAFFIC_REWARDS, Progress, copy_driver,
                              driver_env, race_rewards, resume_driver, source_path)


class TestTrainingADriverByFeel:
    def test_races_full_fields_where_places_are_worth_more(self):
        env = driver_env(races=1, seed=0)
        assert env.cars == FEEL_CARS == 20
        assert env.rivals == FEEL_RIVALS == 10 and env.num_envs == 10
        assert env.rewards['places'] == TRAFFIC_REWARDS['places'] > 1
        assert env.rewards['podium'] == TRAFFIC_REWARDS['podium'] > 1

    def test_the_clock_costs_more_than_in_the_original_rewards(self):
        from slipstream.env import REWARDS
        assert driver_env(races=1, seed=0).rewards['time'] == TRAFFIC_REWARDS['time'] > 2 * REWARDS['time']

    def test_resumes_at_the_current_learning_rate_not_the_saved_one(self, tmp_path):
        env = driver_env(races=1, cars=2, seed=0, rivals=0)
        saved = RecurrentPPO('MlpLstmPolicy', env, device='cpu', learning_rate=3e-4, n_steps=8, batch_size=16)
        saved.save(tmp_path / 'model')
        resumed = resume_driver(tmp_path / 'model.zip', env)
        assert resumed.lr_schedule(1.0) == FEEL_SETTINGS['learning_rate'] < 3e-4
        assert resumed.policy.optimizer.param_groups[0]['lr'] == FEEL_SETTINGS['learning_rate']

    def test_keeps_exploring_even_when_resumed_from_a_driver_saved_without_it(self, tmp_path):
        env = driver_env(races=1, cars=2, seed=0, rivals=0)
        RecurrentPPO('MlpLstmPolicy', env, device='cpu', ent_coef=0.0, n_steps=8, batch_size=16).save(tmp_path / 'model')
        assert resume_driver(tmp_path / 'model.zip', env).ent_coef == FEEL_SETTINGS['ent_coef'] > 0

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


class TestShapingFadesIn:
    def test_a_driver_by_feel_never_pays_for_its_cars_condition_once_shaping_is_full(self, tmp_path):
        env = driver_env(races=1, cars=2, seed=0, rivals=0)
        model = RecurrentPPO('MlpLstmPolicy', env, device='cpu', n_steps=8, batch_size=16)
        model.set_logger(configure(None, []))
        (tmp_path / 'shaping.json').write_text('{"start": 0}')
        progress = Progress(tmp_path, every=1_000_000)
        progress.init_callback(model)
        progress.num_timesteps = 10_000_000
        progress._set_shaping_strength()
        assert env.personality_strength == 1.0
        assert env.condition_strength == 0.0


class TestRacingForPasses:
    def test_a_passing_run_blames_the_car_behind_for_contact_and_pays_for_closing_in(self):
        env = driver_env(races=1, seed=0, rewards=race_rewards('passing'))
        assert env.rewards['contact_ahead'] < 1.0 and env.rewards['closing'] > 0
        assert env.rewards['time'] == TRAFFIC_REWARDS['time']

    def test_a_run_started_for_passes_keeps_racing_for_them_when_resumed_without_saying_so(self, tmp_path):
        assert race_rewards('passing', run=tmp_path) == race_rewards('passing')
        assert race_rewards(run=tmp_path) == race_rewards('passing')
        (tmp_path / 'other').mkdir()
        assert race_rewards(run=tmp_path / 'other') == TRAFFIC_REWARDS

    def test_a_run_started_for_passes_before_reward_styles_still_races_for_them(self, tmp_path):
        (tmp_path / PASSING_FILE).write_text('{}')
        assert race_rewards(run=tmp_path) == race_rewards('passing')


class TestRacingClean:
    def test_a_clean_run_pays_for_hard_hits_crashing_out_and_running_off(self):
        env = driver_env(races=1, seed=0, rewards=race_rewards('clean'))
        assert env.rewards['hard_hit'] > 0 and env.crash_cost > RETIRE_COST
        assert env.rewards['off_track'] > TRAFFIC_REWARDS.get('off_track', REWARDS['off_track'])
        assert env.rewards['closing'] == 0 and env.rewards['contact_ahead'] == 1.0

    def test_a_clean_run_keeps_racing_clean_when_resumed(self, tmp_path):
        race_rewards('clean', run=tmp_path)
        assert race_rewards(run=tmp_path) == race_rewards('clean')


class TestRacingFair:
    def test_a_fair_run_pays_for_hard_hits_only(self):
        env = driver_env(races=1, seed=0, rewards=race_rewards('fair'))
        assert env.rewards['hard_hit'] > 0 and env.crash_cost == RETIRE_COST
        assert env.rewards['off_track'] == TRAFFIC_REWARDS.get('off_track', REWARDS['off_track'])
