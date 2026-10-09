"""Who controls each car in a race outside training: the network for some cars, the scripted driver for the rest.

Cars the network drives also take its pit calls and stop plans, if it makes them. Networks trained before pit
stops existed only drive, so their cars get the scripted strategist, which keeps them from running dry in long
races. Scripted cars always do. A pit wall network (see pitwall.py), when given, makes the pit decisions for
every car the network drives instead.
"""
import numpy as np

from .brains import NetworkDriver
from .drivers import scripted_controls
from .observe import for_network
from .pit import PitPlan
from .pitwall import pitwall_decisions, pitwall_observe
from .strategy import DRIVING_ONLY, decode, scripted_strategy


def makes_pit_calls(model):
    return model is not None and model.action_space.shape[0] > DRIVING_ONLY


def field_controls(race, driver: NetworkDriver | None, networked, personality=None, pitwall=None):
    """Steer, throttle, brake, pit calls and stop plans for every car. `driver` is the network at the wheel of
    the cars `networked` marks (scripted drivers take the rest); `personality` is one row of traits per car
    (neutral if not given)."""
    steer, throttle, brake = scripted_controls(race)
    call, plan = scripted_strategy(race)
    cars = np.flatnonzero(networked)
    if driver is not None and len(cars):
        # Every car is asked, so a recurrent driver's memory follows every car the same way.
        actions = driver.act(race, personality)[cars]
        network_steer, network_throttle, network_brake, network_call, network_plan = decode(actions, len(cars))
        steer[cars], throttle[cars], brake[cars] = network_steer, network_throttle, network_brake
        if makes_pit_calls(driver.model):
            _take_strategy(call, plan, cars, network_call, network_plan)
    # The pit wall calls the stops for the networked cars, or for every car when nobody else is a network.
    if pitwall is not None:
        strategised = cars if driver is not None else np.arange(race.count)
        wall_actions, _ = pitwall.predict(for_network(pitwall_observe(race, personality), pitwall)[strategised], deterministic=True)
        _take_strategy(call, plan, strategised, *pitwall_decisions(wall_actions))
    return steer, throttle, brake, call, plan


def _take_strategy(call, plan, cars, new_call, new_plan):
    call[cars] = new_call
    for name in PitPlan.__dataclass_fields__:
        getattr(plan, name)[cars] = getattr(new_plan, name)
