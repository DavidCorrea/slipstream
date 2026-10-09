"""Who a driver is: four traits between 0 and 1 that the network reads as inputs and that weight its rewards.

One network drives every car. In training, every driver gets random traits each race, and its rewards are
weighed by them, so the network learns how each kind of driver should race. At watch time you can move any
trait and the car changes how it drives straight away, with no retraining.

- aggression: how little contact bothers it. Low, it backs out of a fight; high, it leans on other cars.
- risk: how close to the grip limit it's willing to drive, and how little it minds running wide.
- overtaking: how much each place gained (or lost) matters to it, beyond simply driving fast.
- conservation: how much it spares its tyres and fuel, at a cost in pace.

Two more traits are about the person rather than the style (see human.py): `consistency`, how steady their hands and
feet are and how rarely they make a mistake, and `stamina`, how slowly they tire.
"""
import numpy as np

PERSONALITY_TRAITS = ('aggression', 'risk', 'overtaking', 'conservation')
HUMAN_TRAITS = ('consistency', 'stamina')
# Networks trained before the human traits read only the personality ones, which come first.
TRAITS = PERSONALITY_TRAITS + HUMAN_TRAITS
NEUTRAL = 0.5

# How strongly each trait reshapes the rewards (see trait_rewards).
WEIGHTS = {
    'contact': (1.0, 0.2),           # contact cost multiplier at aggression 0 and at 1
    'off_track': (1.5, 0.5),         # off-track cost multiplier at risk 0 and at 1
    'grip_margin': (0.55, 1.0),      # share of the grip limit it uses freely at risk 0 and at 1
    'grip_excess': 0.03,             # cost per decision for each whole grip limit used beyond that share
    'place': (0.05, 0.35),           # reward per place gained (cost per place lost) at overtaking 0 and at 1
    # Cost per unit of tyre wear and per kg of fuel burned, at conservation 1: together about half of what
    # driving flat out earns, so even the most careful driver still wants to go forward.
    'tyre_wear': 25.0,
    'fuel': 0.3,
}


def neutral(count):
    return np.full((count, len(TRAITS)), NEUTRAL)


def random_personalities(count, rng):
    return rng.uniform(0, 1, (count, len(TRAITS)))


def trait(personality, name):
    return personality[:, TRAITS.index(name)]


def blend(weights, amount):
    low, high = weights
    return low + (high - low) * amount
