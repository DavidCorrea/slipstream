import numpy as np

from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.env import REWARDS, RaceVecEnv
from slipstream.observe import NEARBY, OBSERVATION_NAMES, OBSERVATION_SIZE, observe
from slipstream.race import Race
from slipstream.track import generate_track


def driving(rows):
    """Full actions from (steer, pedal) rows, with no pit call and a neutral stop plan."""
    from slipstream.strategy import ACTION_NAMES
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    padded = np.zeros((len(rows), len(ACTION_NAMES)))
    padded[:, :2] = rows
    padded[:, 2] = -1.0
    return padded


def slot(name):
    return OBSERVATION_NAMES.index(name)


def scripted_actions(env):
    actions = []
    for race in env.races:
        steer, throttle, brake = scripted_controls(race)
        actions.append(np.stack([steer, throttle - brake], axis=1))
    return driving(np.concatenate(actions))


class TestObservations:
    def test_have_one_value_per_name_and_stay_finite(self):
        race = Race(generate_track(1), CarSpecs.uniform(6), laps=1)
        for _ in range(200):
            race.step(*scripted_controls(race))
        observation = observe(race)
        assert observation.shape == (6, OBSERVATION_SIZE) == (6, len(OBSERVATION_NAMES))
        assert np.isfinite(observation).all()

    def test_see_a_car_ahead_as_ahead_and_the_nearest_first(self):
        race = Race(generate_track(2), CarSpecs.uniform(3), laps=1)
        cars = race.cars
        cars.position[1] = cars.position[0] + cars.forward[0] * 20
        cars.position[2] = cars.position[0] + cars.forward[0] * 40
        observation = observe(race)[0]
        assert observation[slot('car.0.present')] == 1
        assert observation[slot('car.0.x')] > 0
        assert abs(observation[slot('car.0.y')]) < 0.05
        assert observation[slot('car.1.x')] > observation[slot('car.0.x')]

    def test_leave_empty_nearby_slots_at_zero_when_alone(self):
        observation = observe(Race(generate_track(3), CarSpecs.uniform(1), laps=1))[0]
        for car in range(NEARBY):
            for feature in ('present', 'x', 'y', 'vx', 'vy'):
                assert observation[slot(f'car.{car}.{feature}')] == 0

    def test_show_the_road_ahead_in_front_of_a_car_on_the_grid(self):
        observation = observe(Race(generate_track(4), CarSpecs.uniform(1), laps=1))[0]
        assert observation[slot('line.30.x')] > 0.2
        assert abs(observation[slot('heading_sin')]) < 0.1


class TestEnvironment:
    def test_returns_one_observation_per_car_in_every_race(self):
        env = RaceVecEnv(races=3, cars=4, seed=1)
        assert env.reset().shape == (12, OBSERVATION_SIZE)

    def test_rewards_driving_the_track_and_punishes_leaving_it(self):
        # One lap, so condition (damage from grid bumps, fuel) is worth nothing and only driving is measured.
        env = RaceVecEnv(races=2, cars=3, laps=1, seed=2)
        env.personality_strength = 0.0
        env.reset()
        on_the_road = sum(env.step(scripted_actions(env))[1].sum() for _ in range(100))
        env.reset()
        off = sum(env.step(driving(np.tile([1.0, 1.0], (6, 1))))[1].sum() for _ in range(300))
        assert on_the_road > 0
        assert off < on_the_road

    def test_pays_nothing_for_progress_made_off_the_tarmac(self):
        # One lap: too short for condition to be worth anything, so only track limits are being tested.
        env = RaceVecEnv(races=1, cars=1, laps=1, seed=5)
        env.reset()
        race = env.races[0]
        race.lateral[:] = race.track.width
        race.cars.position[0] = race.track.point_at(race.progress % race.track.length, race.track.width)[0]
        race.cars.velocity[0] = race.cars.forward[0] * 30
        reward = env.step(driving([[0.0, 1.0]]))[1][0]
        assert reward < 0

    def test_ends_every_car_of_a_finished_race_together_and_starts_a_new_circuit(self):
        env = RaceVecEnv(races=2, cars=3, seed=3)
        env.reset()
        first = env.races[0]
        first.tick = int(first.time_limit / 0.05)
        _, _, dones, infos = env.step(driving(np.zeros((6, 2))))
        assert dones.tolist() == [True] * 3 + [False] * 3
        assert all('terminal_observation' in info and 'episode' in info for info in infos[:3])
        assert 'race' in infos[0]
        assert env.races[0] is not first

    def test_pays_the_winner_more_for_finishing_than_the_last_car(self):
        assert REWARDS['podium'] > 0
        env = RaceVecEnv(races=1, cars=2, laps=1, seed=4)
        env.reset()
        race = env.races[0]
        race.progress[:] = race.track.length - 1.0
        race.progress[1] -= 0.5
        rewards = np.zeros(2)
        for _ in range(5):
            rewards += env.step(driving([[0.0, 1.0], [0.0, 1.0]]))[1]
        assert rewards[0] > rewards[1]


def personality(**values):
    from slipstream.personality import NEUTRAL, TRAITS
    return np.array([[values.get(name, NEUTRAL) for name in TRAITS]])


def one_car_env(seed=7):
    env = RaceVecEnv(races=1, cars=1, seed=seed)
    env.reset()
    return env, env.races[0]


def reward_for(trait_values, prepare, action=(0.0, 1.0), seed=7):
    env, race = one_car_env(seed)
    env.personalities[0] = personality(**trait_values)
    prepare(race)
    return env.step(driving([action]))[1][0]


class TestPersonality:
    def test_is_read_by_the_network_after_every_other_input(self):
        race = Race(generate_track(1), CarSpecs.uniform(2), laps=1)
        traits = np.array([[0.0, 1.0, 0.5, 0.25], [1.0, 0.0, 0.5, 0.75]])
        observation = observe(race, traits)
        first = OBSERVATION_NAMES.index('personality.aggression')
        assert OBSERVATION_NAMES[first:first + 4] == ['personality.aggression', 'personality.risk', 'personality.overtaking', 'personality.conservation']
        assert np.allclose(observation[:, first:first + 4], traits * 2 - 1)

    def test_is_cut_away_for_networks_trained_before_it_existed(self):
        from types import SimpleNamespace
        from slipstream.observe import for_network
        old = SimpleNamespace(observation_space=SimpleNamespace(shape=(OBSERVATION_SIZE - 4,)))
        observations = np.zeros((3, OBSERVATION_SIZE))
        assert for_network(observations, old).shape == (3, OBSERVATION_SIZE - 4)

    def test_makes_cautious_drivers_pay_for_driving_at_the_grip_limit(self):
        def at_the_limit(race):
            race.cars.velocity[0] = race.cars.forward[0] * 40
        cautious = reward_for({'risk': 0.0}, at_the_limit, action=(1.0, 1.0))
        risky = reward_for({'risk': 1.0}, at_the_limit, action=(1.0, 1.0))
        assert cautious < risky

    def test_makes_conserving_drivers_pay_for_tyres_and_fuel(self):
        def flat_out(race):
            race.cars.velocity[0] = race.cars.forward[0] * 30
        spendthrift = reward_for({'conservation': 0.0}, flat_out, action=(0.6, 1.0))
        saver = reward_for({'conservation': 1.0}, flat_out, action=(0.6, 1.0))
        assert saver < spendthrift

    def test_makes_contact_cost_an_aggressive_driver_less(self):
        def collide(env):
            race = env.races[0]
            race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * 2.0
            race.cars.velocity[0] = race.cars.forward[0] * 15
        rewards = {}
        for aggression in (0.0, 1.0):
            env = RaceVecEnv(races=1, cars=2, seed=8)
            env.reset()
            env.personalities[0] = np.repeat(personality(aggression=aggression), 2, axis=0)
            collide(env)
            rewards[aggression] = env.step(driving([[0.0, 0.0], [0.0, 0.0]]))[1][0]
        assert rewards[0.0] < rewards[1.0]

    def test_pays_eager_overtakers_more_for_a_place_gained(self):
        gains = {}
        for overtaking in (0.0, 1.0):
            env = RaceVecEnv(races=1, cars=2, seed=9)
            env.reset()
            env.personalities[0] = np.repeat(personality(overtaking=overtaking), 2, axis=0)
            race = env.races[0]
            race.progress[:] = [10.0, 10.5]
            race.cars.position[0] = race.track.point_at(np.array([12.0]))[0]
            race.cars.position[1] = race.track.point_at(np.array([8.0]), np.array([4.0]))[0]
            gains[overtaking] = env.step(driving(np.zeros((2, 2))))[1][0]
        assert gains[1.0] > gains[0.0]

    def test_pays_more_for_a_place_gained_when_places_are_worth_more(self):
        gains = {}
        for places in (1.0, 2.0):
            env = RaceVecEnv(races=1, cars=2, seed=9, rewards={'places': places})
            env.reset()
            race = env.races[0]
            race.progress[:] = [10.0, 10.5]
            race.cars.position[0] = race.track.point_at(np.array([12.0]))[0]
            race.cars.position[1] = race.track.point_at(np.array([8.0]), np.array([4.0]))[0]
            gains[places] = env.step(driving(np.zeros((2, 2))))[1][0]
        assert gains[2.0] > gains[1.0]


class TestPersonalityStrength:
    def test_at_zero_rewards_every_personality_the_same(self):
        rewards = []
        for traits in ({'risk': 0.0, 'conservation': 1.0}, {'risk': 1.0, 'conservation': 0.0}):
            env, race = one_car_env(seed=10)
            env.personality_strength = 0.0
            env.personalities[0] = personality(**traits)
            race.cars.velocity[0] = race.cars.forward[0] * 40
            rewards.append(env.step(driving([[1.0, 1.0]]))[1][0])
        assert rewards[0] == rewards[1]


class TestConditionShaping:
    def test_pays_nothing_for_condition_that_doesnt_change(self):
        env = RaceVecEnv(races=1, cars=1, laps=10, seed=11)
        env.reset()
        race = env.races[0]
        race.cars.fuel[0] = 5.0   # far short of what ten laps need, so the condition value is well below zero
        env.condition[0] = __import__('slipstream.strategy', fromlist=['condition_value']).condition_value(race)
        assert env.condition[0][0] < -1
        reward = env.step(driving([[0.0, 0.0]]))[1][0]
        assert reward <= 0

    def test_never_rewards_a_car_for_running_out_of_time(self):
        env = RaceVecEnv(races=1, cars=1, laps=10, seed=12)
        env.reset()
        race = env.races[0]
        race.cars.fuel[0] = 5.0
        race.tick = int(race.time_limit / 0.05) - 1
        env.condition[0] = __import__('slipstream.strategy', fromlist=['condition_value']).condition_value(race)
        _, rewards, dones, _ = env.step(driving([[0.0, 0.0]]))
        assert dones[0]
        assert rewards[0] <= 0

    def test_switched_off_pays_nothing_for_a_change_in_condition(self):
        rewards = []
        for wear in (0.0, 0.9):
            env = RaceVecEnv(races=1, cars=1, laps=10, seed=13)
            env.condition_strength = 0.0
            env.personality_strength = 0.0
            env.reset()
            env.races[0].cars.tyre_wear[0] = wear
            rewards.append(env.step(driving([[0.0, 0.0]]))[1][0])
        assert rewards[0] == rewards[1]


class TestConservation:
    def test_never_pays_for_a_refill_or_fresh_tyres(self):
        env, race = one_car_env(seed=14)
        env.personalities[0] = personality(conservation=1.0)
        env.condition_strength = 0.0
        race.cars.fuel[0] = 10.0
        race.cars.tyre_wear[0] = 0.8
        before = env.step(driving([[0.0, 0.0]]))[1][0]
        env, race = one_car_env(seed=14)
        env.personalities[0] = personality(conservation=1.0)
        race.cars.fuel[0] = 10.0
        race.cars.tyre_wear[0] = 0.8
        # The crew refills the tank and fits new tyres during the next tick.
        original_step = race.step
        def serviced(*arguments, **keywords):
            events = original_step(*arguments, **keywords)
            race.cars.fuel[0], race.cars.tyre_wear[0] = 60.0, 0.0
            return events
        race.step = serviced
        env.condition_strength = 0.0
        after = env.step(driving([[0.0, 0.0]]))[1][0]
        assert after <= before + 1e-6
