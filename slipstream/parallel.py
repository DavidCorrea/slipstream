"""Training races spread over several processes, so a run uses more than one CPU core.

The races of a training step are independent until the network decides, so `ParallelRaces` splits them between
worker processes, each holding a vectorised environment of its own (like RacerEnv), and steps them all at once:
the actions go out, every worker plays its races' decisions, and the observations and rewards come back in order.
To the trainer it is one environment with every worker's cars.
"""
import multiprocessing

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def _work(connection, make_env):
    env = make_env()
    while True:
        command, argument = connection.recv()
        if command == 'step':
            connection.send(env.step(argument))
        elif command == 'reset':
            connection.send(env.reset())
        elif command == 'set_attr':
            env.set_attr(*argument)
            connection.send(None)
        elif command == 'get_attr':
            connection.send(env.get_attr(argument)[0])
        elif command == 'call':
            name, arguments = argument
            connection.send(getattr(env, name)(*arguments))
        elif command == 'close':
            connection.close()
            return
        else:
            raise ValueError(f'Unknown command {command!r} for a training worker')


class ParallelRaces(VecEnv):
    def __init__(self, makers):
        """`makers` builds each worker's environment (picklable, e.g. a functools.partial of RacerEnv)."""
        context = multiprocessing.get_context('spawn')
        self.connections, self.workers = [], []
        for make_env in makers:
            ours, theirs = context.Pipe()
            worker = context.Process(target=_work, args=(theirs, make_env), daemon=True)
            worker.start()
            theirs.close()
            self.connections.append(ours)
            self.workers.append(worker)
        sample = makers[0]()
        self.sizes = [sample.num_envs] * len(makers)
        super().__init__(sum(self.sizes), sample.observation_space, sample.action_space)
        self.actions = None

    def _ask_all(self, command, argument=None):
        for connection in self.connections:
            connection.send((command, argument))
        return [connection.recv() for connection in self.connections]

    def reset(self):
        return np.concatenate(self._ask_all('reset'))

    def step_async(self, actions):
        self.actions = np.asarray(actions)

    def step_wait(self):
        edges = np.cumsum([0] + self.sizes)
        for connection, start, end in zip(self.connections, edges[:-1], edges[1:]):
            connection.send(('step', self.actions[start:end]))
        results = [connection.recv() for connection in self.connections]
        observations, rewards, dones, infos = zip(*results)
        return np.concatenate(observations), np.concatenate(rewards), np.concatenate(dones), [info for part in infos for info in part]

    def call_all(self, name, *arguments):
        """Calls a method of every worker's environment (for example RacerEnv.add_to_league)."""
        return self._ask_all('call', (name, arguments))

    def close(self):
        for connection in self.connections:
            connection.send(('close', None))
        for worker in self.workers:
            worker.join(timeout=5)

    def get_attr(self, attr_name, indices=None):
        return [self._ask_all('get_attr', attr_name)[0]] * len(self._indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        self._ask_all('set_attr', (attr_name, value))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.call_all(method_name, *method_args)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else list(indices)
