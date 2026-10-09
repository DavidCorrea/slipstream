"""Trains the pit wall (see pitwall.py) with PPO, the driver network fixed.

    .venv/bin/python -m slipstream.train_pitwall                       # run "pitwall", driven by run "main"
    .venv/bin/python -m slipstream.train_pitwall --driver main --steps 400000
    .venv/bin/python -m slipstream.train_pitwall --traffic --run pitwall-traffic --from pitwall

Solo (the default) races every car alone, which learns quickly what a stop is worth. --traffic races them
together, so the pit wall also learns where a stop puts it among the others; start it from a solo run.

A pit-wall run lives in runs/<name>/, like a driver run, with driver.json naming the driver it learned with.
One decision is one lap, so a few hundred thousand decisions is a lot of racing.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .pitwall import PITWALL_ACTION_NAMES, PITWALL_OBSERVATION_NAMES
from .pitwall_env import PitWallEnv, TrafficRaces, benchmark, network_driver
from .imitation import collect, imitate
from .warmstart import carry_over

LOG_STD_INIT = -0.7
PPO_SETTINGS = dict(
    n_steps=16, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
    # A little entropy keeps the noise from collapsing: the first run's pit-call noise shrank from 0.47 to 0.31
    # within 650k decisions, after which it no longer tried the stops that would have shown it when they pay.
    clip_range=0.2, ent_coef=0.005, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=LOG_STD_INIT),
)


TRAFFIC_SETTINGS = dict(PPO_SETTINGS, n_steps=8, batch_size=128)


class TrafficSlots(VecEnv):
    """Only the shape of the traffic races' slots, for PPO to build its network and buffer around; the races
    themselves are played by TrafficPPO.collect_rollouts."""

    def __init__(self, slots):
        self.render_mode = None
        super().__init__(slots, spaces.Box(-np.inf, np.inf, (len(PITWALL_OBSERVATION_NAMES),), np.float32),
                         spaces.Box(-1.0, 1.0, (len(PITWALL_ACTION_NAMES),), np.float32))

    def reset(self):
        return np.zeros((self.num_envs, len(PITWALL_OBSERVATION_NAMES)), dtype=np.float32)

    def step_async(self, actions):
        raise NotImplementedError('Traffic races are played by TrafficPPO.collect_rollouts')

    def step_wait(self):
        raise NotImplementedError('Traffic races are played by TrafficPPO.collect_rollouts')

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class TrafficPPO(PPO):
    """PPO that learns from TrafficRaces. Cars reach their decision points at their own times, so instead of
    stepping every slot together, the races play tick by tick and each slot's decisions go into the rollout
    buffer in order, until every slot has n_steps of them. The PPO update itself is unchanged."""

    traffic: TrafficRaces = None

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        counts = np.zeros(self.n_envs, dtype=int)
        last_done = np.zeros(self.n_envs, dtype=bool)
        following = np.zeros((self.n_envs, len(PITWALL_OBSERVATION_NAMES)), dtype=np.float32)

        def decide(observations):
            with torch.no_grad():
                actions, values, log_probs = self.policy(obs_as_tensor(observations, self.device))
            return actions.cpu().numpy(), values.cpu().numpy().ravel(), log_probs.cpu().numpy()

        def record(slot, decision, reward, done, next_observation):
            row = counts[slot]
            if row >= n_rollout_steps:
                return
            rollout_buffer.observations[row, slot] = decision['observation']
            rollout_buffer.actions[row, slot] = decision['action']
            rollout_buffer.rewards[row, slot] = reward
            rollout_buffer.episode_starts[row, slot] = decision['start']
            rollout_buffer.values[row, slot] = decision['value']
            rollout_buffer.log_probs[row, slot] = decision['log_prob']
            counts[slot] += 1
            last_done[slot] = done
            if next_observation is not None:
                following[slot] = next_observation
            self.num_timesteps += 1

        while counts.min() < n_rollout_steps:
            ended = self.traffic.tick(decide, record)
            callback.update_locals({'infos': [{'race': summary} for summary in ended]})
            if not callback.on_step():
                return False
        with torch.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(following, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_done)
        rollout_buffer.pos, rollout_buffer.full = n_rollout_steps, True
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


class Progress(BaseCallback):
    def __init__(self, run: Path, every: int, driver):
        super().__init__()
        self.run, self.every, self.driver = run, every, driver
        self.races: list[dict] = []
        self.next_benchmark = None
        self.started = time.time()

    def _on_training_start(self):
        self.next_benchmark = (self.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        self.races += [info['race'] for info in self.locals['infos'] if 'race' in info]
        return True

    def _on_rollout_end(self):
        if self.races:
            for key in ('finished', 'stops'):
                self.logger.record(f'race/{key}', float(np.mean([race[key] for race in self.races])))
        if self.num_timesteps >= self.next_benchmark:
            self.next_benchmark += self.every
            self._checkpoint()

    def _checkpoint(self):
        scores = benchmark(self.model, self.driver)
        for key, value in scores.items():
            self.logger.record(f'benchmark/{key}', value)
        training = {f'race_{key}': float(np.mean([race[key] for race in self.races])) for key in ('finished', 'stops')} if self.races else {}
        self.races = []
        name = f'step-{self.num_timesteps:012d}'
        self.model.save(self.run / 'checkpoints' / name)
        self.model.save(self.run / 'model')
        entry = {'checkpoint': name, 'step': self.num_timesteps, 'time': round(time.time() - self.started), **scores, **training}
        with open(self.run / 'history.jsonl', 'a') as history:
            history.write(json.dumps(entry) + '\n')
        print(f"decision {self.num_timesteps:,}: {scores['pace']:.3f}x the scripted strategist, {scores['beats_no_stop']:.3f}x never stopping, "
              f"{scores['stops']:.1f} stops, finished {scores['finished']:.0%}, place in traffic {scores['traffic_place']:.2f} (0 best) "
              f"| training stops {training.get('race_stops', 0):.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='pitwall')
    parser.add_argument('--driver', default='weather', help='driver run whose latest network drives while the pit wall learns')
    parser.add_argument('--steps', type=int, default=1_000_000, help='pit decisions (laps) to train for')
    parser.add_argument('--groups', type=int, default=4, help='groups of ghost races run side by side')
    parser.add_argument('--cars', type=int, default=32, help='cars per group')
    parser.add_argument('--benchmark-every', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--traffic', action='store_true', help='race cars together instead of alone')
    parser.add_argument('--races', type=int, default=8, help='with --traffic: races played side by side')
    parser.add_argument('--from', dest='source', help='start a new run as a copy of this pit-wall run (ignored when resuming)')
    parser.add_argument('--imitate', type=int, default=0, metavar='DECISIONS',
                        help='start a new run by copying the scripted strategist over this many decisions (see imitation.py)')
    arguments = parser.parse_args()

    torch.set_num_threads(4)
    run = Path('runs') / arguments.run
    (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
    driver_file = Path('runs') / arguments.driver / 'model.zip'
    driver = PPO.load(driver_file, device='cpu')
    (run / 'driver.json').write_text(json.dumps({'driver': arguments.driver}))
    if arguments.traffic:
        cars = min(arguments.cars, 6)
        traffic = TrafficRaces(network_driver(driver), races=arguments.races, cars=cars, seed=arguments.seed)
        env, kind, settings = TrafficSlots(traffic.slots), TrafficPPO, TRAFFIC_SETTINGS
        layout = f'{arguments.races} races of {cars} cars, together'
    else:
        env = PitWallEnv(network_driver(driver), groups=arguments.groups, cars=arguments.cars, seed=arguments.seed)
        kind, settings = PPO, PPO_SETTINGS
        layout = f'{arguments.groups} groups of {arguments.cars} cars, each racing alone'
    saved = run / 'model.zip'
    if saved.exists():
        model = kind.load(saved, env=env, device='cpu')
        print(f'Resuming {run} at decision {model.num_timesteps:,}')
    else:
        model = kind('MlpPolicy', env, device='cpu', seed=arguments.seed, tensorboard_log=str(run / 'tensorboard'), verbose=0, **settings)
        if arguments.source:
            source = Path('runs') / arguments.source / 'model.zip'
            if not source.exists():
                raise SystemExit(f'No pit-wall run to start from at {source}')
            with torch.no_grad():
                model.policy.load_state_dict(carry_over(PPO.load(source, device='cpu').policy.state_dict(), model.policy.state_dict(),
                                                        names=PITWALL_ACTION_NAMES))
                # The copy keeps what the source learned but explores afresh: a source that settled on never
                # stopping had all but stopped trying anything else.
                model.policy.log_std.fill_(LOG_STD_INIT)
            print(f'Starting {run} as a copy of {source}, driven by {driver_file}')
        elif arguments.imitate:
            imitation_env = PitWallEnv(network_driver(driver), groups=arguments.groups, cars=arguments.cars, seed=arguments.seed + 1)
            observations, actions = collect(imitation_env, arguments.imitate)
            error = imitate(model.policy, observations, actions)
            print(f'Starting {run} as a copy of the scripted strategist ({len(observations):,} decisions, error {error:.4f}), '
                  f'driven by {driver_file}')
        else:
            print(f'Starting {run}, driven by {driver_file}')
    if arguments.traffic:
        model.traffic = traffic
    print(f'{env.num_envs} cars ({layout}); benchmark every {arguments.benchmark_every:,} decisions')
    try:
        model.learn(total_timesteps=arguments.steps, callback=Progress(run, arguments.benchmark_every, driver),
                    reset_num_timesteps=False, tb_log_name='ppo')
    except KeyboardInterrupt:
        print('Stopping')
    model.save(run / 'model')
    print(f'Saved {run / "model.zip"} at decision {model.num_timesteps:,}')


if __name__ == '__main__':
    main()
