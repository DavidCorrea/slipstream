"""Starting a driver by feel off as a copy of the scripted driver, before it learns to drive better.

Learning to drive from nothing, with only its senses, a memory and its own reaction time to work with, a driver
by feel spent millions of decisions sliding off the road. The scripted driver can already get round: it reads the
track and the car exactly, which a driver by feel can't, but what it does is a fine first answer to what the
driver by feel senses. So we record the scripted driver racing (what each car's driver would have sensed, and what
the scripted driver did), fit the network's outputs to it, sequence by sequence so its memory learns too, and
then let reinforcement learning take over from there.
"""
import numpy as np
import torch

from .drivers import scripted_controls


def collect_driving(env, decisions, sequence_length=128):
    """Plays the env's races with the scripted driver at every wheel. Returns, cut into sequences of
    `sequence_length` decisions per car: what each driver sensed, what the scripted driver did (steer, pedal), and
    where a new race began inside a sequence."""
    observations = env.reset()
    sensed, did, began = [], [], []
    starting = np.ones(env.num_envs, dtype=bool)
    while len(sensed) * env.num_envs < decisions or len(sensed) % sequence_length:
        actions = np.zeros((env.num_envs, 2), dtype=np.float32)
        for number, race in enumerate(env.races):
            steer, throttle, brake = scripted_controls(race)
            slots = slice(number * env.cars, (number + 1) * env.cars)
            actions[slots, 0] = np.clip(steer, -1, 1)
            actions[slots, 1] = np.clip(throttle - brake, -1, 1)
        sensed.append(observations)
        did.append(actions)
        began.append(starting.copy())
        observations, _, dones, _ = env.step(actions)
        starting = dones
    # (time, car, ...) -> one row per car, then cut each car's history into sequences.
    by_car = lambda steps: np.stack(steps, axis=1).reshape(env.num_envs, -1, sequence_length, *steps[0].shape[1:])
    cut = lambda values: by_car(values).reshape(-1, sequence_length, *values[0].shape[1:])
    observations, actions, starts = cut(sensed), cut(did), cut(began)
    starts[:, 0] = True   # every sequence starts from an empty memory
    return observations.astype(np.float32), actions.astype(np.float32), starts


def imitate_driver(policy, observations, actions, starts, epochs=20, batch_sequences=32, learning_rate=1e-3, seed=0):
    """Fits a recurrent policy's outputs to the recorded actions, a batch of whole sequences at a time. Returns the
    mean squared error over everything afterwards."""
    sequences, length, size = observations.shape
    inputs = torch.as_tensor(observations)
    targets = torch.as_tensor(actions)
    restarts = torch.as_tensor(starts, dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    def outputs(batch):
        count = len(batch)
        hidden = policy.lstm_actor.hidden_size
        empty = (torch.zeros(policy.lstm_actor.num_layers, count, hidden), torch.zeros(policy.lstm_actor.num_layers, count, hidden))
        distribution, _ = policy.get_distribution(inputs[batch].reshape(-1, size), empty, restarts[batch].reshape(-1))
        return distribution.distribution.mean.reshape(count, length, -1)

    for _ in range(epochs):
        for batch in torch.randperm(sequences, generator=generator).split(batch_sequences):
            loss = torch.nn.functional.mse_loss(outputs(batch), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(outputs(torch.arange(sequences)), targets))
