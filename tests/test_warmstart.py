import numpy as np
import pytest
import torch

from slipstream.strategy import ACTION_NAMES, decode
from slipstream.warmstart import FRESH_OUTPUTS, carry_over


def network(inputs, outputs, fill):
    torch.manual_seed(fill)
    return {
        'log_std': torch.randn(outputs),
        'mlp_extractor.policy_net.0.weight': torch.randn(8, inputs), 'mlp_extractor.policy_net.0.bias': torch.randn(8),
        'mlp_extractor.value_net.0.weight': torch.randn(8, inputs), 'mlp_extractor.value_net.0.bias': torch.randn(8),
        'action_net.weight': torch.randn(outputs, 8), 'action_net.bias': torch.randn(outputs),
        'value_net.weight': torch.randn(1, 8), 'value_net.bias': torch.randn(1),
    }


class TestCarryingAWeightsOver:
    def test_copies_everything_the_networks_share(self):
        old, new = network(5, 2, 1), network(7, len(ACTION_NAMES), 2)
        result = carry_over(old, new)
        assert torch.equal(result['mlp_extractor.policy_net.0.weight'][:, :5], old['mlp_extractor.policy_net.0.weight'])
        assert torch.equal(result['action_net.weight'][:2], old['action_net.weight'])
        assert torch.equal(result['value_net.weight'], old['value_net.weight'])
        assert torch.equal(result['log_std'][:2], old['log_std'])

    def test_lets_new_inputs_change_nothing_at_first(self):
        result = carry_over(network(5, 2, 1), network(7, len(ACTION_NAMES), 2))
        assert (result['mlp_extractor.policy_net.0.weight'][:, 5:] == 0).all()
        assert (result['mlp_extractor.value_net.0.weight'][:, 5:] == 0).all()

    def test_starts_new_outputs_at_their_fresh_values(self):
        result = carry_over(network(5, 2, 1), network(7, len(ACTION_NAMES), 2))
        assert (result['action_net.weight'][2:] == 0).all()
        assert [round(float(value), 3) for value in result['action_net.bias'][2:]] == [FRESH_OUTPUTS[name] for name in ACTION_NAMES[2:]]

    def test_names_a_pit_walls_new_outputs_by_its_own_list(self):
        from slipstream.pitwall import PITWALL_ACTION_NAMES
        result = carry_over(network(5, 7, 1), network(7, len(PITWALL_ACTION_NAMES), 2), names=PITWALL_ACTION_NAMES)
        assert round(float(result['action_net.bias'][7]), 3) == FRESH_OUTPUTS['weather_tyres']

    def test_refuses_layers_that_cannot_line_up(self):
        old = network(5, 2, 1)
        old['mlp_extractor.policy_net.2.weight'] = torch.randn(3, 3)
        new = network(7, len(ACTION_NAMES), 2)
        new['mlp_extractor.policy_net.2.weight'] = torch.randn(4, 4)
        with pytest.raises(ValueError, match='Cannot carry'):
            carry_over(old, new)

    def test_gives_a_car_with_fresh_outputs_no_pit_call_and_a_sensible_plan(self):
        actions = np.zeros((1, len(ACTION_NAMES)))
        actions[0, 2:] = [FRESH_OUTPUTS[name] for name in ACTION_NAMES[2:]]
        _, _, _, call, plan = decode(actions, 1)
        assert not call[0]
        assert plan.fuel[0] == 60.0 and plan.repair[0] and not plan.brakes[0]


class TestPitCalls:
    def test_say_come_in_above_zero(self):
        actions = np.zeros((3, len(ACTION_NAMES)))
        actions[:, 2] = [-0.4, 0.0, 0.3]
        assert decode(actions, 3)[3].tolist() == [False, False, True]
