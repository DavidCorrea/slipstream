"""Gets the site ready to publish: the trained networks it races with, and the list of files the page downloads.

    .venv/bin/python -m slipstream.publish driver-rivals/step-000036016128 pitwall-feel/step-000000251904
    .venv/bin/python -m slipstream.publish            # only refreshes the list, after changing the simulation

The published site has no server: the page runs the simulation itself (browser.py in a web worker). Static
hosting can't list a folder, so ENGINE_FILES names everything the worker fetches: the Python modules the browser
side imports and the networks in brains/, exported to run with numpy (numpy_network.py). Training runs stay
private; only the snapshots named here are published.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .session import RUNS, brain_path, is_pitwall_run

ROOT = Path(__file__).resolve().parent.parent
BRAINS = ROOT / 'brains'
ENGINE_FILES = Path('web') / 'engine-files.json'


def browser_modules():
    """The slipstream modules the browser imports, as paths from the project's root. Worked out in a fresh
    interpreter, so only what browser.py really pulls in counts."""
    script = ('import sys, slipstream.browser; '
              'print("\\n".join(sorted(module.__file__ for name, module in sys.modules.items() if name.split(".")[0] == "slipstream")))')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, cwd=ROOT, check=True)
    return sorted(str(Path(path).resolve().relative_to(ROOT)) for path in result.stdout.split())


def publish(snapshots, runs=RUNS, destination=BRAINS):
    """Exports snapshots ('run/step-N') into `destination` laid out like runs/, with each pit-wall run's note of
    the driver it trained with. Returns the files written."""
    from .brains import load_network
    from .numpy_network import export
    written = []
    for snapshot in snapshots:
        run, step = snapshot.split('/', 1)
        target = destination / run / 'checkpoints' / f'{step}.npz'
        export(load_network(brain_path(snapshot, runs)), target)
        written.append(target)
        if is_pitwall_run(runs / run):
            shutil.copyfile(runs / run / 'driver.json', destination / run / 'driver.json')
            written.append(destination / run / 'driver.json')
    return written


def write_engine_files():
    brains = sorted(str(path.relative_to(ROOT)) for path in BRAINS.rglob('*') if path.is_file())
    (ROOT / ENGINE_FILES).write_text(json.dumps({'modules': browser_modules(), 'brains': brains}, indent=2) + '\n')
    return brains


def main():
    parser = argparse.ArgumentParser(description='Export networks for the published site and list the files it needs')
    parser.add_argument('snapshots', nargs='*', help="snapshots to publish, like 'driver-rivals/step-000036016128'")
    arguments = parser.parse_args()
    for path in publish(arguments.snapshots):
        print(f'published {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.2f} MB)')
    brains = write_engine_files()
    print(f'{ENGINE_FILES}: {len(browser_modules())} modules, {len(brains)} brain files')


if __name__ == '__main__':
    main()
