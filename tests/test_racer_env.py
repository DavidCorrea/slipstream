import numpy as np

from slipstream.racer_env import FIELD, LAPS, LEARNERS, RACER_REWARDS, STILL_SPEED, RacerEnv
from slipstream import pit
from slipstream.senses import RACER_SENSE_NAMES
from tests.test_pit_lane_driven import put_in_lane
from slipstream.strategy import RACER_ACTIONS


def env(races=4, seed=0):
    racing = RacerEnv(races=races, seed=seed)
    racing.personality_strength = 1.0
    return racing


class TestTheTrainingWorld:
    def test_races_have_the_games_rules_and_every_size_of_field(self):
        racing = env(races=12)
        racing.reset()
        sizes = {race.count for race in racing.races}
        assert min(sizes) >= FIELD[0] and max(sizes) <= FIELD[1] and len(sizes) > 3
        for race in racing.races:
            assert race.stewards is not None and race.marshals is not None and not race.assisted
            assert LAPS[0] <= race.laps <= LAPS[1]
        assert racing.num_envs == 12 * LEARNERS
        assert racing.observation_space.shape == (len(RACER_SENSE_NAMES),)
        assert racing.action_space.shape == (len(RACER_ACTIONS),)

    def test_steps_whole_races_through_to_new_ones(self):
        racing = env(races=2)
        observations = racing.reset()
        assert observations.shape == (2 * LEARNERS, len(RACER_SENSE_NAMES))
        actions = np.zeros((racing.num_envs, len(RACER_ACTIONS)))
        actions[:, 1] = 0.6
        for _ in range(200):
            observations, rewards, dones, infos = racing.step(actions)
        assert np.isfinite(rewards).all() and np.isfinite(observations).all()


def decisions(racing, controls, count):
    race, mine = racing.races[0], racing.learner_cars[0]
    return sum(racing._drive(race, racing.personalities[0], controls, mine) for _ in range(count))


def braking(race):
    controls = np.zeros((race.count, len(RACER_ACTIONS)))
    controls[:, 1] = -1.0
    return controls


class TestWhatARacingDriverIsPaidFor:
    def test_nothing_racing_costs_extra_until_the_network_can_drive(self):
        quiet, jumping = env(races=1), env(races=1)
        for racing in (quiet, jumping):
            racing.personality_strength = 0.0
            racing.reset()
        car = quiet.learner_cars[0][0]
        early = braking(jumping.races[0])
        early[car, 1] = 1.0
        assert jumping.races[0].stewards is not None
        assert decisions(jumping, early, 20)[car] > decisions(quiet, braking(quiet.races[0]), 20)[car] - RACER_REWARDS['penalty'] * 9

    def test_a_penalty_costs_what_that_much_race_time_is_worth(self):
        quiet, jumping = env(races=1), env(races=1)
        quiet.reset(), jumping.reset()
        car = quiet.learner_cars[0][0]
        early = braking(jumping.races[0])
        early[car, 1] = 1.0
        # Moving off the grid before the lights is a jump start (the lights take five seconds at least).
        assert decisions(jumping, early, 20)[car] < decisions(quiet, braking(quiet.races[0]), 20)[car] - RACER_REWARDS['penalty'] * 9

    def test_sitting_still_after_the_lights_costs_on_top_of_the_time_lost(self):
        charged, free = env(races=1), RacerEnv(races=1, rewards={'still': 0.0})
        free.personality_strength = 1.0
        for racing in (charged, free):
            racing.reset()
            racing.races[0].marshals.lights_out_at = racing.races[0].time
        car = charged.learner_cars[0][0]
        extra = decisions(free, braking(free.races[0]), 5)[car] - decisions(charged, braking(charged.races[0]), 5)[car]
        assert np.isclose(extra, RACER_REWARDS['still'] * 5)

    def test_waiting_for_the_lights_costs_nothing_extra(self):
        racing = env(races=1)
        racing.reset()
        race, car = racing.races[0], racing.learner_cars[0][0]
        assert not race.marshals.started(race.time) and race.cars.speed[car] < STILL_SPEED
        assert decisions(racing, braking(race), 1)[car] > -RACER_REWARDS['still']

    def test_damage_costs_more_with_more_race_to_go(self):
        paid = {}
        for laps in (2, 20):
            racing = env(races=1)
            racing.reset()
            race, car = racing.races[0], racing.learner_cars[0][0]
            race.laps = laps
            race.cars.velocity[car] = race.cars.forward[car] * 30.0
            paid[laps] = decisions(racing, braking(race), 3)[car]
            assert race.cars.damage[car] > 0
        assert paid[20] < paid[2]


class TestTheLeague:
    def test_earlier_snapshots_drive_some_of_the_rivals(self, tmp_path):
        from sb3_contrib import RecurrentPPO

        from slipstream.numpy_network import export
        from slipstream.train import RACER_SETTINGS

        racing = env(races=6, seed=3)
        export(RecurrentPPO('MlpLstmPolicy', racing, device='cpu', **RACER_SETTINGS), tmp_path / 'snapshot.npz')
        racing.add_to_league(tmp_path / 'snapshot.npz')
        racing.reset()
        actions = np.zeros((racing.num_envs, len(RACER_ACTIONS)))
        for _ in range(3):
            racing.step(actions)
        seated = [racing.league_cars[id(race)] for race in racing.races]
        assert any(len(cars) for cars in seated)
        for race, cars, mine in zip(racing.races, seated, racing.learner_cars):
            assert not set(cars) & set(mine)
            if not racing.self_races[id(race)]:
                assert len(cars) <= (race.count - LEARNERS) // 2

    def test_some_races_are_against_nobody_but_itself(self, tmp_path):
        racing = league_env(tmp_path, races=12)
        assert any(racing.self_races.values()) and not all(racing.self_races.values())
        for race, mine in zip(racing.races, racing.learner_cars):
            if racing.self_races[id(race)]:
                assert sorted([*racing.league_cars[id(race)], *mine]) == list(range(race.count))

    def test_with_no_league_yet_every_race_has_scripted_rivals(self):
        racing = env(races=12, seed=3)
        racing.reset()
        racing.step(np.zeros((racing.num_envs, len(RACER_ACTIONS))))
        assert not any(len(racing.league_cars[id(race)]) for race in racing.races)


class TestStartDrills:
    def test_some_races_start_the_learners_from_the_back_of_the_grid(self):
        racing = env(races=12, seed=1)
        racing.reset()
        drilled = [racing.start_drills[id(race)] for race in racing.races]
        assert any(drilled) and not all(drilled)
        for race, mine, at_the_back in zip(racing.races, racing.learner_cars, drilled):
            if at_the_back:
                assert list(mine) == list(range(race.count - LEARNERS, race.count))
                assert race.progress[mine].max() < np.delete(race.progress, mine).min()


def league_env(tmp_path, races, seed=3):
    from sb3_contrib import RecurrentPPO

    from slipstream.numpy_network import export
    from slipstream.train import RACER_SETTINGS

    racing = env(races=races, seed=seed)
    export(RecurrentPPO('MlpLstmPolicy', racing, device='cpu', **RACER_SETTINGS), tmp_path / 'snapshot.npz')
    racing.add_to_league(tmp_path / 'snapshot.npz')
    racing.reset()
    racing.step(np.zeros((racing.num_envs, len(RACER_ACTIONS))))
    return racing


class TestPitStopsArePaidForWhatTheyAre:
    def test_a_called_stop_is_worth_more_than_ignoring_the_call(self):
        # Bringing the car in when called pays for turning in and for the stop; the time in the lane isn't charged.
        # Driving past the entry costs as much as the stop pays. (When a stop cost about 3 in time and paid 0.5,
        # the racing driver learned to drive past every call.)
        assert RACER_REWARDS['lane_entry'] + RACER_REWARDS['stop'] >= 3 * RACER_REWARDS['time'] * 10

    def test_turning_into_the_lane_when_called_pays(self):
        paid = {}
        for called in (True, False):
            racing = env(races=1)
            racing.reset()
            race, car = racing.races[0], racing.learner_cars[0][0]
            race.marshals.lights_out_at = race.time
            if called:
                race.pit_state[car] = pit.CALLED
            put_in_lane(race, car, 5.0, 12.0, lateral=race.pit.centre(5.0))
            paid[called] = decisions(racing, braking(race), 1)[car]
            assert race.in_lane[car]
        assert paid[True] - paid[False] >= RACER_REWARDS['lane_entry'] * 0.9

    def test_the_time_in_the_lane_on_a_called_stop_is_not_charged(self):
        racing = env(races=1)
        racing.reset()
        race, car = racing.races[0], racing.learner_cars[0][0]
        race.marshals.lights_out_at = race.time
        race.pit_state[car] = pit.IN_LANE
        race.lane_called[car] = True
        put_in_lane(race, car, 80.0, 0.0)
        race.pit_state[car] = pit.IN_LANE
        race.lane_called[car] = True
        paid = decisions(racing, braking(race), 1)[car]
        assert paid > -RACER_REWARDS['time'] / 2

    def test_more_races_are_pit_drills(self):
        from slipstream.racer_env import PIT_DRILL
        assert PIT_DRILL >= 0.4
