import json

import pytest
from stable_baselines3 import PPO

from slipstream.env import RaceVecEnv
from slipstream.publish import ENGINE_FILES, ROOT, browser_modules, choose_default_driver, publish


class TestPublishingTheSite:
    def test_the_page_downloads_every_module_the_browser_imports(self):
        listed = json.loads((ROOT / ENGINE_FILES).read_text())
        assert sorted(listed['modules']) == browser_modules()

    def test_the_page_downloads_every_published_brain_and_nothing_missing(self):
        listed = json.loads((ROOT / ENGINE_FILES).read_text())
        assert listed['brains'] and all((ROOT / path).is_file() for path in listed['brains'])

    def test_the_page_knows_how_much_it_downloads_so_it_can_show_progress(self):
        listed = json.loads((ROOT / ENGINE_FILES).read_text())
        assert listed['bytes'] == sum((ROOT / path).stat().st_size for path in listed['modules'] + listed['brains'])

    def test_exports_snapshots_with_what_a_pit_wall_run_says_about_its_driver(self, tmp_path):
        runs = tmp_path / 'runs'
        (runs / 'wall' / 'checkpoints').mkdir(parents=True)
        (runs / 'wall' / 'driver.json').write_text('{"driver": "fast"}')
        PPO('MlpPolicy', RaceVecEnv(races=1, cars=2), device='cpu').save(runs / 'wall' / 'checkpoints' / 'step-000000000100')
        published = publish(['wall/step-000000000100'], runs=runs, destination=tmp_path / 'brains')
        assert (tmp_path / 'brains' / 'wall' / 'checkpoints' / 'step-000000000100.npz').is_file()
        assert (tmp_path / 'brains' / 'wall' / 'driver.json').read_text() == '{"driver": "fast"}'
        assert sorted(path.name for path in published) == ['driver.json', 'step-000000000100.npz']

    def test_names_the_driver_the_page_starts_with(self, tmp_path):
        (tmp_path / 'polish' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'polish' / 'checkpoints' / 'step-000000000100.npz').write_bytes(b'')
        choose_default_driver('polish/step-000000000100', destination=tmp_path)
        assert (tmp_path / 'default-driver.txt').read_text() == 'polish/step-000000000100\n'

    def test_refuses_a_default_driver_that_isnt_published(self, tmp_path):
        with pytest.raises(ValueError, match='polish/step-000000000100'):
            choose_default_driver('polish/step-000000000100', destination=tmp_path)
