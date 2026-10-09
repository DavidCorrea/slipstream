"""Starting the pit wall off as a copy of the scripted strategist, before it learns to beat it.

Left to find stops on its own, the pit wall learned never to stop, twice: a stop costs about twenty seconds on
the lap it happens, for certain, while what it buys comes laps later and differs from car to car and circuit to
circuit, so early in training every stop it tried looked like lost time. A network that already stops when the
strategist would gives its value estimate stops to learn from, and training can then work on making them better.
"""
import numpy as np
import torch

from .car import COMPOUNDS, FUEL_CAPACITY, SLICK_COUNT, centred
from .pitwall import PITWALL_ACTION_NAMES
from .strategy import WEATHER_TYRES, scripted_strategy

# Where in each output's range a choice sits: the middle of its band, so decoding gives it back.
YES, NO = 0.5, -0.5
THREE_WAY = np.array([-2 / 3, 0.0, 2 / 3])


def strategist_actions(race):
    """The scripted strategist's pit call and stop plan for every car, as pit-wall outputs."""
    call, plan = scripted_strategy(race)
    outputs = np.zeros((race.count, len(PITWALL_ACTION_NAMES)))
    column = PITWALL_ACTION_NAMES.index
    compound = np.asarray(plan.compound)
    is_slick = compound < SLICK_COUNT
    outputs[:, column('pit')] = np.where(call, YES, NO)
    outputs[:, column('compound')] = THREE_WAY[np.where(is_slick, compound, COMPOUNDS.index('medium'))]
    outputs[:, column('fuel')] = centred(np.minimum(plan.fuel, FUEL_CAPACITY) / FUEL_CAPACITY)
    outputs[:, column('wing')] = centred(plan.wing)
    outputs[:, column('engine')] = centred(plan.engine)
    outputs[:, column('repair')] = np.where(plan.repair, YES, NO)
    outputs[:, column('brakes')] = np.where(plan.brakes, YES, NO)
    weather_kind = np.select([compound == COMPOUNDS.index('intermediate'), compound == COMPOUNDS.index('wet')],
                             [WEATHER_TYRES.index('intermediate'), WEATHER_TYRES.index('wet')], WEATHER_TYRES.index('slick'))
    outputs[:, column('weather_tyres')] = THREE_WAY[weather_kind]
    return outputs


def collect(env, decisions):
    """Plays the pit wall's training races with the scripted strategist making every call, and returns each real
    decision (cars whose race is over only fill their slot) with what the strategist did."""
    observed, chosen = [], []
    observations = env.reset()
    while sum(len(batch) for batch in observed) < decisions:
        actions = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)
        deciding = np.zeros(env.num_envs, dtype=bool)
        for number, group in enumerate(env.groups):
            slots = slice(number * env.cars, (number + 1) * env.cars)
            actions[slots] = strategist_actions(group['race'])
            deciding[slots] = ~env._over(group)
        observed.append(observations[deciding])
        chosen.append(actions[deciding])
        observations, _, _, _ = env.step(actions)
    return np.concatenate(observed), np.concatenate(chosen)


def imitate(policy, observations, actions, epochs=80, batch_size=256, learning_rate=1e-3, seed=0):
    """Fits the policy's outputs to the given actions. Returns the final mean squared error."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.as_tensor(observations, dtype=torch.float32)
    targets = torch.as_tensor(actions, dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        for batch in torch.randperm(len(inputs), generator=generator).split(batch_size):
            mean = policy.get_distribution(inputs[batch]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return float(loss.detach())
