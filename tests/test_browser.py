import subprocess
import sys

TRAINING_LIBRARIES = ('torch', 'gymnasium', 'stable_baselines3', 'sb3_contrib', 'aiohttp')


def imports_without_training_libraries(module):
    """Imports `module` in a fresh interpreter that refuses the training libraries, as the browser has none."""
    script = f'''
import sys
class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {TRAINING_LIBRARIES!r}:
            raise ImportError(f'needs {{name}}')
sys.meta_path.insert(0, Refuse())
import {module}
'''
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)


class TestRunningInTheBrowser:
    def test_a_race_session_needs_none_of_the_training_libraries(self):
        result = imports_without_training_libraries('slipstream.session')
        assert result.returncode == 0, result.stderr


class TestRacingInTheBrowser:
    def test_plays_a_race_from_the_pages_messages_and_hands_back_its_frames(self, tmp_path):
        import asyncio
        import json
        from slipstream.browser import BrowserRaces
        races = BrowserRaces(tmp_path)

        async def play():
            await races.receive(json.dumps({'type': 'start', 'cars': 3, 'laps': 1}))
            await races.receive(json.dumps({'type': 'go'}))
            for _ in range(3):
                await races.tick()
        asyncio.run(play())
        kinds = [json.loads(message)['type'] for message in races.collect()]
        assert kinds[0] == 'race' and kinds[1:] == ['frame'] * 3
        assert races.collect() == []

    def test_lists_the_published_brains(self, tmp_path):
        import json
        from slipstream.browser import BrowserRaces
        (tmp_path / 'driver-rivals' / 'checkpoints').mkdir(parents=True)
        (tmp_path / 'driver-rivals' / 'checkpoints' / 'step-000036016128.npz').write_bytes(b'')
        listed = json.loads(BrowserRaces(tmp_path).brains())
        assert [brain['id'] for brain in listed['drivers']] == ['scripted', 'driver-rivals/step-000036016128']
        assert listed['pitwalls'][0]['id'] == 'driver'

    def test_the_browser_side_needs_none_of_the_training_libraries(self):
        result = imports_without_training_libraries('slipstream.browser')
        assert result.returncode == 0, result.stderr
