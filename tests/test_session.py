import json

import torch

import pytest

from slipstream.session import SCRIPTED, RaceSession, brain_path, list_brains


class TestBrains:
    def test_always_include_the_scripted_driver_first(self, tmp_path):
        assert list_brains(tmp_path)[0]['id'] == SCRIPTED

    def test_list_every_run_snapshot_newest_first(self, tmp_path):
        for step in (1000, 3000, 2000):
            (tmp_path / 'main' / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (tmp_path / 'main' / 'checkpoints' / f'step-{step:012d}.zip').write_bytes(b'')
        steps = [brain['step'] for brain in list_brains(tmp_path)[1:]]
        assert steps == [3000, 2000, 1000]

    def test_list_and_find_networks_exported_to_run_with_numpy(self, tmp_path):
        (tmp_path / 'main' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'main' / 'checkpoints' / 'step-000000005000.npz').write_bytes(b'')
        assert [brain['id'] for brain in list_brains(tmp_path)[1:]] == ['main/step-000000005000']
        assert brain_path('main/step-000000005000', tmp_path).suffix == '.npz'

    def test_refuse_paths_outside_the_runs_folder(self, tmp_path):
        (tmp_path / 'main' / 'checkpoints').mkdir(parents=True)
        with pytest.raises(ValueError):
            brain_path('main/../../../etc/passwd', tmp_path)
        with pytest.raises(ValueError):
            brain_path('main/step-missing', tmp_path)


class TestSessions:
    def test_describe_the_circuit_and_where_every_car_lines_up(self):
        intro = RaceSession(SCRIPTED, seed=5, cars=4, laps=2).intro()
        assert intro['type'] == 'race' and intro['seed'] == 5
        assert len(intro['cars']) == len(intro['grid']) == 4
        assert len(intro['track']['points']) == len(intro['track']['normals'])
        json.dumps(intro)

    def test_send_frames_that_move_the_cars_and_serialise_cleanly(self):
        session = RaceSession(SCRIPTED, seed=5, cars=4, laps=2)
        first = session.advance(1)
        later = session.advance(40)
        assert later['time'] > first['time']
        assert later['cars']['x'] != first['cars']['x']
        assert sorted(later['order']) == [0, 1, 2, 3]
        json.dumps(later)

    def test_send_frames_that_serialise_cleanly_after_the_winner_finishes(self):
        session = RaceSession(SCRIPTED, seed=6, cars=2, laps=1)
        frame = None
        while frame is None or not frame['done']:
            frame = session.advance(200)
            json.dumps(frame)

    def test_report_the_finish_and_stop_advancing_once_the_race_is_over(self):
        session = RaceSession(SCRIPTED, seed=6, cars=2, laps=1)
        finished = []
        frame = None
        while frame is None or not frame['done']:
            frame = session.advance(200)
            finished += frame['events']['finished']
        assert sorted(finished) == [0, 1]
        assert session.advance(10)['time'] == frame['time']


class TestTheCast:
    def test_draws_a_different_grid_each_race_and_the_same_one_for_the_same_circuit(self):
        orders = {tuple(car['name'] for car in RaceSession(SCRIPTED, seed=seed, cars=6, laps=1).intro()['cars']) for seed in range(6)}
        assert len(orders) > 1
        again = [RaceSession(SCRIPTED, seed=9, cars=6, laps=1).intro()['cars'] for _ in range(2)]
        assert again[0] == again[1]

    def test_every_driver_races_with_their_own_character(self):
        from slipstream.personality import TRAITS
        session = RaceSession(SCRIPTED, seed=5, cars=8, laps=1)
        for car, driver in enumerate(session.lineup):
            assert list(session.personality[car]) == [driver.character[name] for name in TRAITS]
        assert len({tuple(row) for row in session.personality}) == 8

    def test_rivals_cars_vary_a_little_from_race_to_race(self):
        from slipstream.cast import RACE_DAY
        def vegas_grip(seed):
            session = RaceSession(SCRIPTED, seed=seed, cars=8, laps=1)
            return session.stats[[driver.key for driver in session.lineup].index('vega')]['grip']
        grips = {vegas_grip(seed) for seed in range(4)}
        assert len(grips) > 1 and max(grips) - min(grips) <= 2 * RACE_DAY

    def test_your_driver_always_races_even_in_a_small_field(self):
        for seed in range(5):
            session = RaceSession(SCRIPTED, seed=seed, cars=2, laps=1, yours='haddad')
            assert session.lineup[session.yours].key == 'haddad'

    def test_drivers_you_edited_keep_exactly_what_you_set(self):
        edited = {'sato': {'grip': 0.1, 'risk': 0.95}}
        session = RaceSession(SCRIPTED, seed=3, cars=20, laps=1, edited=edited)
        sato = [driver.key for driver in session.lineup].index('sato')
        assert session.stats[sato]['grip'] == 0.1 and session.stats[sato]['risk'] == 0.95


class TestTheGridMenu:
    def test_changes_any_cars_specs_and_personality_from_the_sliders(self):
        from slipstream.car import SPEC_DEFAULTS
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1)
        before = session.race.specs.top_speed[0]
        session.set_stats({'top_speed': 1.0, 'aggression': 0.9}, car=1)
        assert session.race.specs.top_speed[1] > SPEC_DEFAULTS['top_speed'] * 1.2
        assert session.race.specs.top_speed[0] == before
        assert session.personality[1, 0] == 0.9

    def test_changes_your_car_when_no_car_is_named(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1, yours=None)
        session.set_stats({'risk': 0.95})
        assert session.stats[session.yours]['risk'] == 0.95

    def test_refuses_unknown_stats_cars_and_values_outside_the_slider(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1)
        with pytest.raises(ValueError, match='Unknown stat'):
            session.set_stats({'nitro': 1.0})
        with pytest.raises(ValueError, match='from 0 to 1'):
            session.set_stats({'grip': 3})
        with pytest.raises(ValueError, match='from 0 to 1'):
            session.set_stats({'grip': 'max'})
        with pytest.raises(ValueError, match='Car must be'):
            session.set_stats({'grip': 0.5}, car=7)

    def test_resets_a_driver_to_their_usual_self(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1)
        session.set_stats({'risk': 0.0, 'grip': 0.0}, car=2)
        session.reset_stats(2)
        assert session.stats[2] == session.lineup[2].character

    def test_taking_over_another_car_leaves_everyone_as_they_were(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1, yours=None)
        stats = [dict(car) for car in session.stats]
        session.set_yours(2)
        assert session.yours == 2 and session.stats == stats
        with pytest.raises(ValueError):
            session.set_yours(7)


class TestRenamingDrivers:
    def test_races_drivers_under_the_names_you_gave_them(self):
        session = RaceSession(SCRIPTED, seed=5, cars=20, laps=1, names={'vega': 'Da Silva'})
        names = [car['name'] for car in session.intro()['cars']]
        assert 'Da Silva' in names and 'Vega' not in names

    def test_renames_a_driver_on_the_grid_tidying_the_spaces(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1)
        session.rename(1, '  Da  Costa ')
        assert session.intro()['cars'][1]['name'] == 'Da Costa'

    def test_refuses_names_that_are_empty_too_long_taken_or_not_a_name(self):
        session = RaceSession(SCRIPTED, seed=5, cars=8, laps=1)
        taken = session.intro()['cars'][0]['name']
        for name, problem in (('   ', 'empty'), ('A' * 17, 'at most 16'), (taken.lower(), 'already'), ('<b>X</b>', 'letters'), (7, 'text')):
            with pytest.raises(ValueError, match=problem):
                session.rename(1, name)

    def test_refuses_renaming_once_the_race_is_under_way(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=1)
        session.advance(10)
        with pytest.raises(ValueError, match='before the start'):
            session.rename(0, 'Da Silva')


class TestYourPitStrategy:
    def test_refuses_unknown_plans_and_strategies(self):
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=3)
        with pytest.raises(ValueError, match='Strategy'):
            session.set_pit(strategy='random')
        with pytest.raises(ValueError, match='Compound'):
            session.set_pit(plan={'compound': 'ultrasoft'})
        with pytest.raises(ValueError, match='from 0 to 1'):
            session.set_pit(plan={'fuel': 2})
        with pytest.raises(ValueError, match='true or false'):
            session.set_pit(plan={'repair': 'yes'})

    def test_stops_your_car_when_you_call_it_in_and_fits_your_plan(self):
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=4)
        session.set_yours(1)
        session.set_pit(strategy='mine', plan={'compound': 'hard', 'wing': 0.9}, box=True)
        mine = []
        for _ in range(400):
            frame = session.advance(20)
            mine += [stop for stop in frame['events']['pitStopped'] if stop['car'] == 1]
            if 1 in frame['events']['pitReleased']:
                break
        assert len(mine) == 1
        assert mine[0]['compound'] == 'hard' and mine[0]['jobs']['wing'] > 0
        assert frame['cars']['compound'][1] == 'hard'
        assert not session.box_called


class TestPitWalls:
    def test_are_listed_apart_from_drivers(self, tmp_path):
        from slipstream.session import DRIVER_DECIDES, SCRIPTED_STRATEGIST, list_pitwalls
        for run, pitwall in (('main', False), ('wall', True)):
            (tmp_path / run / 'checkpoints').mkdir(parents=True)
            (tmp_path / run / 'checkpoints' / 'step-000000001000.zip').write_bytes(b'')
            if pitwall:
                (tmp_path / run / 'driver.json').write_text('{"driver": "main"}')
        assert [brain['run'] for brain in list_brains(tmp_path)[1:]] == ['main']
        walls = list_pitwalls(tmp_path)
        assert [wall['id'] for wall in walls[:2]] == [DRIVER_DECIDES, SCRIPTED_STRATEGIST]
        assert [wall['run'] for wall in walls[2:]] == ['wall']

    def test_can_hand_strategy_to_the_scripted_strategist(self):
        from slipstream.session import SCRIPTED_STRATEGIST
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=3, pitwall_id=SCRIPTED_STRATEGIST)
        frame = session.advance(10)
        assert session.intro()['pitwall'] == SCRIPTED_STRATEGIST and frame['type'] == 'frame'

    def test_let_a_pit_wall_network_call_the_stops_even_for_scripted_drivers(self, tmp_path):
        from stable_baselines3 import PPO
        from slipstream.drivers import scripted_controls
        from slipstream.pitwall_env import PitWallEnv
        env = PitWallEnv(lambda race, personality: scripted_controls(race), groups=1, cars=2, laps=3)
        wall = PPO('MlpPolicy', env, n_steps=4, batch_size=8, device='cpu')
        with torch.no_grad():
            wall.policy.action_net.bias[0] = 5.0   # always calls a stop
            wall.policy.action_net.weight[0] = 0.0
        (tmp_path / 'wall' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'wall' / 'driver.json').write_text('{}')
        wall.save(tmp_path / 'wall' / 'checkpoints' / 'step-000000000100')
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=3, runs=tmp_path, pitwall_id='wall/step-000000000100')
        stops = 0
        for _ in range(300):
            stops += len(session.advance(20)['events']['pitStopped'])
            if stops:
                break
        assert stops


class TestShowingTheRace:
    def test_frames_carry_temperatures_damage_and_the_air(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=2)
        frame = session.advance(40)
        for key in ('tyreTemp', 'brakeTemp', 'engineTemp', 'punctured', 'retired', 'wingDamage', 'suspensionDamage', 'draft', 'dirtyAir'):
            assert len(frame['cars'][key]) == 3
        assert 'debris' in frame and 'mistakes' in frame['events']
        json.dumps(frame)

    def test_the_track_surface_goes_out_about_once_a_second_and_decodes(self):
        import base64
        import numpy as np
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=2)
        frames = [session.advance(1) for _ in range(25)]
        maps = [frame['surface'] for frame in frames if frame['surface']]
        assert 1 <= len(maps) <= 2
        surface = maps[0]
        rubber = np.frombuffer(base64.b64decode(surface['rubber']), dtype=np.uint8)
        assert rubber.size == surface['rows'] * surface['lanes']


class TestYourRacerIsSetBeforeTheStart:
    def test_takes_driver_and_car_changes_on_the_grid(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=2)
        session.set_stats({'risk': 0.9})
        assert session.stats[session.yours]['risk'] == 0.9

    def test_refuses_them_once_the_lights_have_gone_out(self):
        import pytest
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=2)
        session.advance(100)
        with pytest.raises(ValueError, match='before the start'):
            session.set_stats({'risk': 0.9})


class TestAFullGrid:
    def test_twenty_different_drivers_in_twenty_different_colours_can_race(self):
        session = RaceSession(SCRIPTED, seed=5, cars=20, laps=1)
        cars = session.intro()['cars']
        assert len({car['name'] for car in cars}) == len({car['color'] for car in cars}) == len({car['number'] for car in cars}) == 20
