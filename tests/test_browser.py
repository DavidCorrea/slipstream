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
