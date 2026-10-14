"""Trains the drivers with PPO (Stable-Baselines3), every car sharing one network.

    .venv/bin/python -m slipstream.train                  # run "main", resuming if it exists
    .venv/bin/python -m slipstream.train --run test --steps 2000000
    .venv/bin/python -m slipstream.train --from personality   # a new run that starts as a copy of another
    .venv/bin/python -m slipstream.train --from main/step-000005013504   # ...or of one of its snapshots
    .venv/bin/python -m slipstream.train --feel --run driver   # a driver that drives by feel, with a memory
    .venv/bin/python -m slipstream.train --racer --run racer --imitate 2000000 --workers 4   # a racing driver

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
import functools
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
from .numpy_network import export
from .parallel import ParallelRaces
from .racer_env import LEAGUE_SIZE, RacerEnv
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
# With nothing paying for exploration its steering and pedal noise kept shrinking (a standard deviation of 0.036 at
# 32M, 0.022 at 76M), until it could only polish the careful driving it already had and never tried a riskier pass
# that might pay. A little entropy bonus keeps it trying.
FEEL_SETTINGS = dict(
    n_steps=256, batch_size=4096, n_epochs=5, learning_rate=1e-4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.001, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), lstm_hidden_size=128, log_std_init=-0.5),
)

# The driver by feel races full fields, and places count for more than in the original rewards: it learned to lap
# quickly on its own but kept losing places in traffic (its benchmark place stuck around 0.7, where 0.5 is level
# with the scripted driver). Three of every eight cars are scripted rivals: in fields made only of copies of
# itself (driver-traffic, 32M to 38M decisions) it learned to hold back instead, crashing half as much while
# dropping from 3% faster than the scripted driver to level, and further behind it in traffic.
# The clock costs more too: at the original 0.005 a decision, being 5% quicker saved less over a race than a
# middling driver paid for cornering at the limit (personality's grip margin), so every run grew slower once it
# could drive. At 0.012 the time a faster lap saves outweighs that.
# The game races 20 cars, so training does too, and half of them are scripted: with 14 copies of itself among 20
# (driver-grid, from 36M decisions), places traded between copies cancelled out while each copy paid its own
# contact, so every run drifted back towards holding back.
FEEL_CARS = 20
FEEL_RIVALS = 10
TRAFFIC_REWARDS = {'podium': 2.0, 'places': 2.0, 'time': 0.012}


# Racing for passes (--passing): even with all that, drivers lapped quicker without winning more places. Here the
# car behind pays for contact and the car ahead only a fifth of it, as stewards would see it, so defending a line
# stops being as costly as diving into one; and closing in on the car ahead pays per metre (and dropping back costs
# the same), so setting up a pass is worth something before the place is won.
PASSING_REWARDS = {'contact_ahead': 0.2, 'closing': 0.008}
# Racing clean (--rewards clean): racing for passes won places, but in a field of itself driver-pass 90M had four
# times as many cars crash out as driver-race 82M and three times the time off the track. A hit's damage and a
# crash's lost race come mostly after the few seconds a driver looks ahead, so here a hard hit costs a quarter of
# a place's worth at the most, crashing out costs what a stuck car's retirement would at 2.5 times, and time off
# the tarmac three times as much.
CLEAN_REWARDS = {'hard_hit': 0.5, 'retire': 5.0, 'off_track': 0.015}
# Racing fair (--rewards fair): racing clean from 82M halved the crashes in a field of itself by 92M, then kept
# going until it lapped no quicker than the scripted driver and lost two thirds of its battles: a costly crash and
# costly grass taught it to back off rather than to drive well. Here only the hard hits cost; crashing out and
# running off cost what they do when racing in traffic.
FAIR_REWARDS = {'hard_hit': CLEAN_REWARDS['hard_hit']}
# Racing steady (--rewards steady): racing fair kept 82M's pace, but over 9M decisions crashed out of a field of
# itself as often as 82M or more (12-27% of cars against 93M's 6%): the hard-hit charge alone doesn't keep cars on
# the road. Here crashing out and running off cost about halfway between racing fair and racing clean.
STEADY_REWARDS = {**FAIR_REWARDS, 'retire': 3.5, 'off_track': 0.01}
REWARD_STYLES = {'traffic': TRAFFIC_REWARDS, 'passing': {**TRAFFIC_REWARDS, **PASSING_REWARDS},
                 'clean': {**TRAFFIC_REWARDS, **CLEAN_REWARDS}, 'fair': {**TRAFFIC_REWARDS, **FAIR_REWARDS},
                 'steady': {**TRAFFIC_REWARDS, **STEADY_REWARDS}}
STYLE_FILE = 'rewards.json'
# Polishing a driver that already races well (--learning-rate 3e-5): at 1e-4, runs from driver-clean 84M swung from
# snapshot to snapshot between crashing out of a field of itself (25% of its cars at driver-steady 98M) and holding
# back (4% at 88M, but losing more battles), the steps too big to settle on what was good. The run remembers it.
LEARNING_RATE_FILE = 'learning_rate.json'
PASSING_FILE = 'passing.json'   # how runs started for passes before there was more than one style


def race_rewards(style=None, run=None):
    """The rewards a driver by feel trains on (REWARD_STYLES). A run started with a style keeps it whenever it's
    resumed."""
    marker = run / STYLE_FILE if run is not None else None
    if style and marker is not None:
        marker.write_text(json.dumps({'style': style}))
    if not style and marker is not None:
        if marker.exists():
            style = json.loads(marker.read_text())['style']
        elif (run / PASSING_FILE).exists():
            style = 'passing'
    return REWARD_STYLES[style or 'traffic']


def run_learning_rate(rate=None, run=None, default=FEEL_SETTINGS['learning_rate']):
    """The learning rate a driver trains at: `default` (FEEL_SETTINGS's for a driver by feel), unless the run was
    started with its own, which it keeps whenever it's resumed."""
    marker = run / LEARNING_RATE_FILE if run is not None else None
    if rate and marker is not None:
        marker.write_text(json.dumps({'learning_rate': rate}))
    elif not rate and marker is not None and marker.exists():
        rate = json.loads(marker.read_text())['learning_rate']
    return rate or default


def driver_env(races, seed, cars=FEEL_CARS, rivals=FEEL_RIVALS, rewards=TRAFFIC_REWARDS):
    return DriverEnv(races=races, cars=cars, seed=seed, rewards=rewards, rivals=rivals)


# Racing drivers (racer_env.py) start from scratch, so they learn at the rate drivers by feel started at, and look
# further ahead: at 0.99 a reward counts about ten seconds ahead, at 0.995 about twenty, closer to how long a hit's
# damage or a penalty takes to tell in the race.
RACER_SETTINGS = {**FEEL_SETTINGS, 'learning_rate': 3e-4, 'gamma': 0.995}
LEAGUE_FOLDER = 'league'


def racer_env(races, seed, workers=1):
    """The racing drivers' training world, its races split over `workers` processes when there's more than one
    (parallel.py): the same races either way, just played on more CPU cores."""
    if workers <= 1:
        return RacerEnv(races=races, seed=seed)
    if races % workers:
        raise SystemExit(f'{races} races can\'t be split evenly over {workers} workers')
    return ParallelRaces([functools.partial(RacerEnv, races=races // workers, seed=seed + worker) for worker in range(workers)])


def resume_driver(path, env, rate=FEEL_SETTINGS['learning_rate']):
    """A saved driver by feel, carrying on at `rate` and today's FEEL_SETTINGS exploration rather than the ones it was
    saved with."""
    model = RecurrentPPO.load(path, env=env, device='cpu', custom_objects={
        'learning_rate': rate, 'lr_schedule': constant_fn(rate), 'ent_coef': FEEL_SETTINGS['ent_coef']})
    for group in model.policy.optimizer.param_groups:
        group['lr'] = rate
    return model


def copy_driver(source, env, run, rate=FEEL_SETTINGS['learning_rate']):
    """A new run that carries on from a saved driver by feel: same network, same count of decisions behind it (so
    its snapshots line up with the source's), logging to its own folder. It keeps the source's personality ramp,
    so personality stays at the strength the source had reached instead of ramping up from nothing again."""
    model = resume_driver(source, env, rate)
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
        # A driver by feel makes no stops, so its car's condition is never part of its reward (driver_env.py).
        if self.training_env.get_attr('pays_for_condition')[0]:
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
        if self.training_env.get_attr('has_league')[0]:
            # Racing drivers race their own earlier snapshots too (racer_env.py): this one joins them.
            snapshot = self.run / LEAGUE_FOLDER / f'{name}.npz'
            snapshot.parent.mkdir(exist_ok=True)
            export(self.model, snapshot)
            self.training_env.env_method('add_to_league', snapshot, indices=0)
        with open(self.run / 'history.jsonl', 'a') as history:
            history.write(json.dumps({'checkpoint': name, **entry}) + '\n')
        print(f"step {self.num_timesteps:,}: pace {scores['pace']:.2f}x script, finished {scores['finished']:.0%}, "
              f"off track {scores['off_track']:.1%}, place vs script {scores['place']:.2f} (0 best, 1 worst), damage {scores['damage']:.2f}, "
              f"wet pace {scores['wet_pace']:.2f}x script (off track {scores['wet_off_track']:.1%})", flush=True)


def racing_driver(arguments, run, saved):
    """The racing driver to train: the run's, resumed, or a new one, copying the scripted driver first if asked."""
    env = racer_env(arguments.races, arguments.seed, arguments.workers)
    rate = run_learning_rate(arguments.learning_rate, run, default=RACER_SETTINGS['learning_rate'])
    if saved.exists():
        model = resume_driver(saved, env, rate)
        league = sorted((run / LEAGUE_FOLDER).glob('step-*.npz'))
        for snapshot in league[-LEAGUE_SIZE:]:
            env.env_method('add_to_league', snapshot, indices=0)
        print(f'Resuming {run} at step {model.num_timesteps:,}, learning at {rate:g}, with {min(len(league), LEAGUE_SIZE)} snapshots in its league')
        return model
    model = RecurrentPPO('MlpLstmPolicy', env, device='cpu', seed=arguments.seed, tensorboard_log=str(run / 'tensorboard'), verbose=0,
                         **{**RACER_SETTINGS, 'learning_rate': rate})
    if arguments.imitate:
        recording = RacerEnv(races=arguments.races, seed=arguments.seed + 1)
        observations, actions, starts = collect_driving(recording, arguments.imitate)
        error = imitate_driver(model.policy, observations, actions, starts)
        print(f'Starting {run}: a racing driver, copying the scripted driver first ({observations.shape[0] * observations.shape[1]:,} decisions, error {error:.4f})')
    else:
        print(f'Starting {run}: a racing driver')
    return model


def quiet_strategy(model):
    with torch.no_grad():
        model.policy.log_std[DRIVING_ONLY:] = STRATEGY_LOG_STD
        model.policy.log_std[ACTION_NAMES.index('pit')] = PIT_CALL_LOG_STD


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='main')
    parser.add_argument('--steps', type=int, default=100_000_000, help='total decisions to train for')
    parser.add_argument('--races', type=int, help='races played side by side (default 32 with --racer, else 16)')
    parser.add_argument('--cars', type=int, help=f'cars per race (default {FEEL_CARS} with --feel, else 6)')
    parser.add_argument('--benchmark-every', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--from', dest='source', help='start a new run as a copy of this run, or of run/step-N (ignored when resuming)')
    parser.add_argument('--rivals', type=int, default=FEEL_RIVALS, help='with --feel, scripted cars in each race (see env.RaceVecEnv)')
    parser.add_argument('--feel', action='store_true', help='train a driver that drives by feel, with a memory (see driver_env.py)')
    parser.add_argument('--rewards', choices=sorted(REWARD_STYLES), help='with --feel, what to race for (REWARD_STYLES); the run remembers it')
    parser.add_argument('--learning-rate', type=float, help=f"with --feel, the run's learning rate (default {FEEL_SETTINGS['learning_rate']}); the run remembers it")
    parser.add_argument('--imitate', type=int, default=0, metavar='DECISIONS',
                        help='with --feel or --racer, start a new run by copying the scripted driver over this many decisions (see driver_imitation.py)')
    parser.add_argument('--racer', action='store_true', help='train a racing driver from scratch, in every rule of the game (see racer_env.py)')
    parser.add_argument('--workers', type=int, default=1, help='with --racer, processes to play the races on (more CPU cores, more fan)')
    arguments = parser.parse_args()

    torch.set_num_threads(4)
    run = Path('runs') / arguments.run
    (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
    saved = run / 'model.zip'
    if arguments.cars is None:
        arguments.cars = FEEL_CARS if arguments.feel else 6
    if arguments.races is None:
        # Racing drivers learn with five cars a race (racer_env.LEARNERS), so twice the races make as many cars a step.
        arguments.races = 32 if arguments.racer else 16
    if arguments.racer:
        model = racing_driver(arguments, run, saved)
    elif arguments.feel:
        env = driver_env(races=arguments.races, cars=arguments.cars, seed=arguments.seed, rivals=arguments.rivals,
                         rewards=race_rewards(arguments.rewards, run))
        if env.rewards['closing']:
            print('Racing for passes: the car behind pays for contact, and closing in on the car ahead pays', flush=True)
        if env.rewards['hard_hit']:
            print('Racing clean: hard hits, crashing out and running off the track cost more', flush=True)
        rate = run_learning_rate(arguments.learning_rate, run)
        if saved.exists():
            model = resume_driver(saved, env, rate)
            print(f'Resuming {run} at step {model.num_timesteps:,}, learning at {rate:g}')
        elif arguments.source:
            source = source_path(arguments.source)
            model = copy_driver(source, env, run, rate)
            print(f'Starting {run} as a copy of {source}, at step {model.num_timesteps:,}, learning at {rate:g}')
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
    if arguments.racer:
        env = model.get_env()
        print(f'{env.num_envs} cars per step ({arguments.races} races of 6 to 20 cars on {arguments.workers} worker(s)); '
              f'benchmark every {arguments.benchmark_every:,} decisions')
    else:
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
