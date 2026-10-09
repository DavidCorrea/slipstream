"""Starting a new run from an older one's network, so a new run doesn't have to learn to drive all over again.

Inputs and outputs are only ever added at the end (see observe.py and strategy.ACTION_NAMES), so the networks
line up: every weight the two share is copied, weights for new inputs start at zero (so a new input changes
nothing until training gives it a use), and new outputs start out saying what the older network would have
meant by saying nothing. The new network drives exactly like the old one on its first decision.
"""
import torch

from .strategy import ACTION_NAMES

# What each new output says before training changes it, in the network's -1 to 1 range: no pit call, a stop
# (if anyone calls one) for mediums and a full tank, wing and engine as they are, repairs yes, brakes no. The pit
# call sits below the call threshold but close enough that exploration crosses it about one lap in eleven: a
# network that never tries a stop can never learn when one pays. (At -1 with quiet noise, a run went 4 million
# decisions without one.) Weather tyres start on slicks, with exploration trying intermediates now and then.
FRESH_OUTPUTS = {'pit': -0.5, 'compound': 0.0, 'fuel': 1.0, 'wing': 0.0, 'engine': 0.0, 'repair': 1.0, 'brakes': -1.0,
                 'weather_tyres': -0.8}


def carry_over(source: dict, target: dict, names=ACTION_NAMES) -> dict:
    """A copy of the `target` network's weights with everything it shares with `source` taken from `source`.
    `names` lists the target's outputs (a pit wall's are its own), so new ones start at their fresh values."""
    result = {name: tensor.clone() for name, tensor in target.items()}
    for name, weights in result.items():
        old = source.get(name)
        if old is None:
            continue
        if old.shape == weights.shape:
            weights.copy_(old)
        elif name.endswith('.0.weight'):
            # First layer: one column per input. New inputs get zero weights.
            weights.zero_()
            weights[:, :old.shape[1]] = old
        elif name in ('action_net.weight', 'action_net.bias', 'log_std'):
            # One row per output. New outputs ignore the hidden layers and sit at their fresh value.
            weights[:old.shape[0]] = old
            if name == 'action_net.weight':
                weights[old.shape[0]:] = 0.0
            elif name == 'action_net.bias':
                for index in range(old.shape[0], weights.shape[0]):
                    weights[index] = FRESH_OUTPUTS[names[index]]
        else:
            raise ValueError(f'Cannot carry {name} over from shape {tuple(old.shape)} to {tuple(weights.shape)}')
    return result


def warm_start(model, source_model):
    """Loads `source_model`'s weights into `model`'s network, wherever they fit."""
    with torch.no_grad():
        model.policy.load_state_dict(carry_over(source_model.policy.state_dict(), model.policy.state_dict()))
