"""A second opinion on every snapshot: the benchmark again, on circuits no snapshot was ever chosen on.

    .venv/bin/python -m slipstream.check --run driver-rivals      # keeps checking new snapshots as they're saved

The benchmark that runs during training scores each snapshot on twelve circuits, and picking the best of dozens of
snapshots on the same circuits favours the lucky ones: on 2026-10-09 a snapshot that led on them looked like
being beaten by every run that followed it, and only fresh circuits showed one of those runs was level with it
and driving more cleanly. This scores each snapshot once on CHECK_SEEDS, in its own process so training doesn't
wait, and appends a line to the run's check.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

from .benchmark import benchmark
from .brains import load_network

CHECK_SEEDS = tuple(range(2001, 2025))
CHECKED_SCORES = ('pace', 'place', 'damage', 'off_track', 'finished')
WAIT_SECONDS = 60


def unchecked(run):
    """The run's snapshots without a line in check.jsonl yet, oldest first."""
    log = run / 'check.jsonl'
    done = {json.loads(line)['checkpoint'] for line in log.read_text().splitlines()} if log.exists() else set()
    return sorted(path for path in (run / 'checkpoints').glob('step-*.zip') if path.stem not in done)


def record(run, checkpoint, scores):
    entry = {'checkpoint': checkpoint, 'step': int(checkpoint.split('-')[1]), **{name: scores[name] for name in CHECKED_SCORES}}
    with open(run / 'check.jsonl', 'a') as log:
        log.write(json.dumps(entry) + '\n')
    return entry


def main():
    parser = argparse.ArgumentParser(description='Score every snapshot of a driver run on circuits it was never chosen on')
    parser.add_argument('--run', required=True)
    parser.add_argument('--since', type=int, default=0, help='skip snapshots before this many decisions')
    arguments = parser.parse_args()
    run = Path('runs') / arguments.run
    if not (run / 'checkpoints').is_dir():
        raise SystemExit(f'No run with snapshots at {run}')
    while True:
        for path in unchecked(run):
            if int(path.stem.split('-')[1]) < arguments.since:
                continue
            entry = record(run, path.stem, benchmark(load_network(path), seeds=CHECK_SEEDS))
            print(f"checked {entry['step']:,}: pace {entry['pace']:.3f}x script, place vs script {entry['place']:.2f}, "
                  f"damage {entry['damage']:.2f}, off track {entry['off_track']:.1%}", flush=True)
        time.sleep(WAIT_SECONDS)


if __name__ == '__main__':
    main()
