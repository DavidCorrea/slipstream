"""A second opinion on every snapshot: the benchmark again, on circuits no snapshot was ever chosen on.

    .venv/bin/python -m slipstream.check --run driver-rivals      # keeps checking new snapshots as they're saved

The benchmark that runs during training scores each snapshot on twelve circuits, and picking the best of dozens of
snapshots on the same circuits favours the lucky ones: on 2026-10-09 a snapshot that led on them looked like
being beaten by every run that followed it, and only fresh circuits showed one of those runs was level with it
and driving more cleanly. This scores each snapshot once on CHECK_SEEDS, in its own process so training doesn't
wait, and appends a line to the run's check.jsonl.

It also races each snapshot against itself across a whole field among the scenery (benchmark.field_race), the race
the viewer watches: beating scripted rivals says nothing about whether a field of the driver wrecks itself.
"""
import argparse
import json
import time
from pathlib import Path

from .benchmark import benchmark, field_race
from .brains import load_network

CHECK_SEEDS = tuple(range(2001, 2025))
CHECKED_SCORES = ('pace', 'place', 'damage', 'off_track', 'finished', 'field_out', 'field_off_track', 'field_hits')
# Racing drivers' field races also say how much penalty time they collect and why cars drop out.
RACER_SCORES = ('field_penalties', 'field_out_barrier', 'field_out_barrier_after_contact', 'field_out_stuck', 'field_out_engine')
WAIT_SECONDS = 60
# Place moves by about 0.05 from one snapshot to the next with no real change, so a run is judged by the average of
# its latest few rather than by its single best, which is as much luck as driving.
AVERAGED = 3


def unchecked(run):
    """The run's snapshots without a line in check.jsonl yet, oldest first."""
    log = run / 'check.jsonl'
    done = {json.loads(line)['checkpoint'] for line in log.read_text().splitlines()} if log.exists() else set()
    return sorted(path for path in (run / 'checkpoints').glob('step-*.zip') if path.stem not in done)


def record(run, checkpoint, scores):
    entry = {'checkpoint': checkpoint, 'step': int(checkpoint.split('-')[1]),
             **{name: scores[name] for name in CHECKED_SCORES + RACER_SCORES if name in scores}}
    with open(run / 'check.jsonl', 'a') as log:
        log.write(json.dumps(entry) + '\n')
    return entry


def recent_average(run, count=AVERAGED):
    """Each score averaged over the run's latest `count` checked snapshots that have it (snapshots checked before
    the field race was scored don't)."""
    lines = [json.loads(line) for line in (run / 'check.jsonl').read_text().splitlines()]
    latest = sorted(lines, key=lambda entry: entry['step'])[-count:]
    averages = {}
    for name in CHECKED_SCORES:
        values = [entry[name] for entry in latest if name in entry]
        if values:
            averages[name] = sum(values) / len(values)
    return averages


def main():
    parser = argparse.ArgumentParser(description='Score every snapshot of a driver run on circuits it was never chosen on')
    parser.add_argument('--run', required=True)
    parser.add_argument('--since', type=int, default=0, help='skip snapshots before this many decisions')
    arguments = parser.parse_args()
    run = Path('runs') / arguments.run
    if not run.is_dir():
        raise SystemExit(f'No run at {run}')
    while True:
        # A run just started has no snapshots yet; its first comes after a million decisions or so.
        for path in unchecked(run) if (run / 'checkpoints').is_dir() else []:
            if int(path.stem.split('-')[1]) < arguments.since:
                continue
            network = load_network(path)
            entry = record(run, path.stem, benchmark(network, seeds=CHECK_SEEDS) | field_race(network))
            print(f"checked {entry['step']:,}: pace {entry['pace']:.3f}x script, place vs script {entry['place']:.2f}, "
                  f"damage {entry['damage']:.2f}, off track {entry['off_track']:.1%}", flush=True)
            print(f"  in a field of itself: out {entry['field_out']:.1%}, off track {entry['field_off_track']:.1%}, "
                  f"hard hits {entry['field_hits']:.1f} a car", flush=True)
            if 'field_penalties' in entry:
                print(f"  penalties {entry['field_penalties']:.1f} s a car; out into a barrier {entry['field_out_barrier']:.1%} alone, "
                      f"{entry['field_out_barrier_after_contact']:.1%} after contact, stuck {entry['field_out_stuck']:.1%}", flush=True)
            average = recent_average(run)
            print(f"  latest {AVERAGED} on average: pace {average['pace']:.3f}x, place {average['place']:.2f}, "
                  f"damage {average['damage']:.2f}, field out {average['field_out']:.1%}", flush=True)
        time.sleep(WAIT_SECONDS)


if __name__ == '__main__':
    main()
