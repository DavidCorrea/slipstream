"""Trains the drivers with PPO (Stable-Baselines3), every car sharing one network.

    .venv/bin/python -m slipstream.train                  # run "main", resuming if it exists
    .venv/bin/python -m slipstream.train --run test --steps 2000000
    .venv/bin/python -m slipstream.train --from personality   # a new run that starts as a copy of another
    .venv/bin/python -m slipstream.train --from main/step-000005013504   # ...or of one of its snapshots
    .venv/bin/python -m slipstream.train --feel --run driver   # a driver that drives by feel, with a memory

Starting from another run copies every weight the two networks share (see warmstart.py), so the new run can
already drive and only has to learn what's new.

A run lives in runs/<name>/:
    model.zip                   the latest network (resumed from next time)
    checkpoints/step-*.zip      a snapshot at every benchmark, for watching how it learned
    history.jsonl               one line per benchmark: pace, place, off-track share, and race stats
    tensorboard/                curves for `tensorboard --logdir runs`

Ctrl+C saves the latest network before stopping.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import constant_fn

from .benchmark import benchmark
from .driver_env import DriverEnv
from .driver_imitation import collect_driving, imitate_driver
from .env import RaceVecEnv
from .strategy import ACTION_NAMES, DRIVING_ONLY
from .warmstart import warm_start

# Personality and condition both charge costs only a moving car pays, which makes sitting still look safest to a
# network that can't drive yet. So until the network can drive (at least this share of training cars finishing
# their races), every driver is rewarded just for driving well; from then on both fade in over SHAPING_RAMP
# decisions. (Starting them at a fixed step count failed when a run hadn't learned to drive by then.)
CAN_DRIVE = 0.5
SHAPING_RAMP = 4_000_000
# Exploration noise the strategy outputs start with (log standard deviation): much quieter than the driving
# outputs, so early learning isn't drowned in random pit calls and plans. The pit call gets a little more, so
# that stops still happen by chance about once in 70 decisions while the network finds out what they're worth.
STRATEGY_LOG_STD = -1.5
PIT_CALL_LOG_STD = -1.0

PPO_SETTINGS = dict(
    n_steps=512, batch_size=4096, n_epochs=5, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
)


# Drivers by feel (see driver_env.py) have a memory: an LSTM between what they sense and what they do.
# The driver by feel learned at 3e-4 until about 32M decisions, then its benchmarks went flat and noisy: steps that
# size kept knocking it off what it had learned. Smaller ones let it settle and keep improving.
FEEL_SETTINGS = dict(
    n_steps=256, batch_size=4096, n_epochs=5, learning_rate=1e-4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), lstm_hidden_size=128, log_std_init=-0.5),
)

# The driver by feel races full fields, and places count for more than in the original rewards: it learned to lap
# quickly on its own but kept losing places in traffic (its benchmark place stuck around 0.7, where 0.5 is level
# with the scripted driver). Three of every eight cars are scripted rivals: in fields made only of copies of
# itself (driver-traffic, 32M to 38M decisions) it learned to hold back instead, crashing half as much while
# dropping from 3% faster than the scripted driver to level, and further behind it in traffic.
FEEL_CARS = 8
FEEL_RIVALS = 3
TRAFFIC_REWARDS = {'podium': 2.0, 'places': 2.0}


def driver_env(races, seed, cars=FEEL_CARS, rivals=FEEL_RIVALS):
    return DriverEnv(races=races, cars=cars, seed=seed, rewards=TRAFFIC_REWARDS, rivals=rivals)


def resume_driver(path, env):
    """A saved driver by feel, carrying on at today's FEEL_SETTINGS learning rate rather than the one it was saved with."""
    rate = FEEL_SETTINGS['learning_rate']
    model = RecurrentPPO.load(path, env=env, device='cpu', custom_objects={'learning_rate': rate, 'lr_schedule': constant_fn(rate)})
    for group in model.policy.optimizer.param_groups:
        group['lr'] = rate
    return model


def copy_driver(source, env, run):
    """A new run that carries on from a saved driver by feel: same network, same count of decisions behind it (so
    its snapshots line up with the source's), logging to its own folder. It keeps the source's personality ramp,
    so personality stays at the strength the source had reached instead of ramping up from nothing again."""
    model = resume_driver(source, env)
    model.tensorboard_log = str(run / 'tensorboard')
    source_run = source.parent.parent if source.parent.name == 'checkpoints' else source.parent
    if (source_run / 'shaping.json').exists():
        (run / 'shaping.json').write_text((source_run / 'shaping.json').read_text())
    return model


def source_path(source, runs=Path('runs')):
    """The file for --from: a run's latest model ('main') or one of its snapshots ('main/step-N')."""
    run_name, _, snapshot = source.partition('/')
    path = runs / run_name / ('checkpoints' if snapshot else '') / f'{snapshot or "model"}.zip'
    if not path.exists():
        raise SystemExit(f'No run to start from at {path}')
    return path


class Progress(BaseCallback):
    """Logs race statistics every rollout, and every `every` steps benchmarks the network, saves a checkpoint
    and appends a line to history.jsonl."""

    def __init__(self, run: Path, every: int):
        super().__init__()
        self.run, self.every = run, every
        self.shaping_file = run / 'shaping.json'
        self.shaping_start = json.loads(self.shaping_file.read_text())['start'] if self.shaping_file.exists() else None
        self.races: list[dict] = []
        self.next_benchmark = None
        self.started = time.time()

    def _on_training_start(self):
        self.next_benchmark = (self.num_timesteps // self.every + 1) * self.every
        self._set_shaping_strength()

    def _set_shaping_strength(self):
        if self.shaping_start is None and self.races and np.mean([race['finished'] for race in self.races]) >= CAN_DRIVE:
            self.shaping_start = self.num_timesteps
            self.shaping_file.write_text(json.dumps({'start': self.shaping_start}))
            print(f'step {self.num_timesteps:,}: the drivers can drive; personality and condition start to count', flush=True)
        strength = 0.0 if self.shaping_start is None else min(1.0, (self.num_timesteps - self.shaping_start) / SHAPING_RAMP)
        self.training_env.set_attr('personality_strength', strength)
        self.training_env.set_attr('condition_strength', strength)
        self.logger.record('train/shaping_strength', strength)

    def _on_step(self):
        for info in self.locals['infos']:
            if 'race' in info:
                self.races.append(info['race'])
        return True

    def _on_rollout_end(self):
        if self.races:
            for key in self.races[0]:
                self.logger.record(f'race/{key}', float(np.mean([race[key] for race in self.races])))
        self._set_shaping_strength()
        if self.num_timesteps >= self.next_benchmark:
            self.next_benchmark += self.every
            self._checkpoint()

    def _checkpoint(self):
        scores = benchmark(self.model)
        for key, value in scores.items():
            self.logger.record(f'benchmark/{key}', value)
        entry = {'step': self.num_timesteps, 'time': round(time.time() - self.started), **scores}
        if self.races:
            entry.update({f'race_{key}': float(np.mean([race[key] for race in self.races])) for key in self.races[0]})
        self.races = []
        name = f'step-{self.num_timesteps:012d}'
        self.model.save(self.run / 'checkpoints' / name)
        self.model.save(self.run / 'model')
        with open(self.run / 'history.jsonl', 'a') as history:
            history.write(json.dumps({'checkpoint': name, **entry}) + '\n')
        print(f"step {self.num_timesteps:,}: pace {scores['pace']:.2f}x script, finished {scores['finished']:.0%}, "
              f"off track {scores['off_track']:.1%}, place vs script {scores['place']:.2f} (0 best, 1 worst), damage {scores['damage']:.2f}, "
              f"wet pace {scores['wet_pace']:.2f}x script (off track {scores['wet_off_track']:.1%})", flush=True)


def quiet_strategy(model):
    with torch.no_grad():
        model.policy.log_std[DRIVING_ONLY:] = STRATEGY_LOG_STD
        model.policy.log_std[ACTION_NAMES.index('pit')] = PIT_CALL_LOG_STD


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='main')
    parser.add_argument('--steps', type=int, default=100_000_000, help='total decisions to train for')
    parser.add_argument('--races', type=int, default=16, help='races played side by side')
    parser.add_argument('--cars', type=int, help=f'cars per race (default {FEEL_CARS} with --feel, else 6)')
    parser.add_argument('--benchmark-every', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--from', dest='source', help='start a new run as a copy of this run, or of run/step-N (ignored when resuming)')
    parser.add_argument('--rivals', type=int, default=FEEL_RIVALS, help='with --feel, scripted cars in each race (see env.RaceVecEnv)')
    parser.add_argument('--feel', action='store_true', help='train a driver that drives by feel, with a memory (see driver_env.py)')
    parser.add_argument('--imitate', type=int, default=0, metavar='DECISIONS',
                        help='with --feel, start a new run by copying the scripted driver over this many decisions (see driver_imitation.py)')
    arguments = parser.parse_args()

    torch.set_num_threads(4)
    run = Path('runs') / arguments.run
    (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
    saved = run / 'model.zip'
    if arguments.cars is None:
        arguments.cars = FEEL_CARS if arguments.feel else 6
    if arguments.feel:
        env = driver_env(races=arguments.races, cars=arguments.cars, seed=arguments.seed, rivals=arguments.rivals)
        if saved.exists():
            model = resume_driver(saved, env)
            print(f'Resuming {run} at step {model.num_timesteps:,}')
        elif arguments.source:
            source = source_path(arguments.source)
            model = copy_driver(source, env, run)
            print(f'Starting {run} as a copy of {source}, at step {model.num_timesteps:,}')
        else:
            model = RecurrentPPO('MlpLstmPolicy', env, device='cpu', seed=arguments.seed, tensorboard_log=str(run / 'tensorboard'), verbose=0,
                                 **FEEL_SETTINGS)
            if arguments.imitate:
                recording = DriverEnv(races=arguments.races, cars=arguments.cars, seed=arguments.seed + 1)
                observations, actions, starts = collect_driving(recording, arguments.imitate)
                error = imitate_driver(model.policy, observations, actions, starts)
                print(f'Starting {run}: a driver by feel, copying the scripted driver first ({observations.shape[0] * observations.shape[1]:,} decisions, error {error:.4f})')
            else:
                print(f'Starting {run}: a driver by feel')
    elif saved.exists():
        env = RaceVecEnv(races=arguments.races, cars=arguments.cars, seed=arguments.seed)
        model = PPO.load(saved, env=env, device='cpu')
        print(f'Resuming {run} at step {model.num_timesteps:,}')
    else:
        env = RaceVecEnv(races=arguments.races, cars=arguments.cars, seed=arguments.seed)
        model = PPO('MlpPolicy', env, device='cpu', seed=arguments.seed, tensorboard_log=str(run / 'tensorboard'), verbose=0, **PPO_SETTINGS)
        quiet_strategy(model)
        if arguments.source:
            source = source_path(arguments.source)
            run_name = arguments.source.partition('/')[0]
            warm_start(model, PPO.load(source, device='cpu'))
            quiet_strategy(model)
            if (Path('runs') / run_name / 'shaping.json').exists():
                # The copy already drives with personality and condition at full strength, so it keeps them there
                # instead of ramping them up from nothing again.
                (run / 'shaping.json').write_text(json.dumps({'start': -SHAPING_RAMP}))
            print(f'Starting {run} as a copy of {source}')
        else:
            print(f'Starting {run}')
    rivals = f', {arguments.rivals} of them scripted' if arguments.feel and arguments.rivals else ''
    print(f'{env.num_envs} cars per step ({arguments.races} races of {arguments.cars}{rivals}); benchmark every {arguments.benchmark_every:,} decisions')
    try:
        model.learn(total_timesteps=arguments.steps, callback=Progress(run, arguments.benchmark_every),
                    reset_num_timesteps=False, tb_log_name='ppo')
    except KeyboardInterrupt:
        print('Stopping')
    model.save(run / 'model')
    print(f'Saved {run / "model.zip"} at step {model.num_timesteps:,}')


if __name__ == '__main__':
    main()
