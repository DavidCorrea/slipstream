"""The promotion gate: whether a snapshot's exam (exam.py) earns it a place on the site over the published driver.

A challenger passes when:
- no must-pass test the published driver passes fails for it (a step back is never worth it),
- no score gets worse by more than its noise margin, and
- something gets better: a must-pass test it newly passes, or a score better by more than its margin.

Must-pass tests both fail are listed as known failures: they don't block (the published driver fails them too) but
they stay in sight. The decision is the script's, with every reason printed, so nobody has to judge it by eye.

    .venv/bin/python -m slipstream.gate runs/racer-2/exam/step-000084032800.json    # against the published driver
    .venv/bin/python -m slipstream.gate challenger.json --against champion.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

from .exam import exam_path

DEFAULT_DRIVER = Path('brains') / 'default-driver.txt'
# How much a score may move between two runs of the same driver before it counts, by test name: what one exam run's
# noise can do, from repeated runs (a share of the value, and at least this much in the test's own unit).
MARGINS = {
    'dry pace against the scripted driver': (0.0, 0.02),
    'wet pace against the scripted driver': (0.0, 0.03),
    'pace on a drying track against the scripted driver': (0.0, 0.03),
    'penalty seconds per car in a field of its copies': (0.1, 1.0),
    'cars stuck in a field of its copies': (0.0, 0.04),
    'hard hits per car in a field of its copies': (0.1, 0.2),
    'place against scripted rivals (0 first, 1 last)': (0.0, 0.05),
}
DEFAULT_MARGIN = (0.1, 0.01)


def margin(name, value):
    share, floor = MARGINS.get(name, DEFAULT_MARGIN)
    return max(abs(value) * share, floor)


def decide(challenger, champion):
    """(passed, reasons, improvements, known failures) for two exams' results (lists of result dicts)."""
    champions = {result['name']: result for result in champion}
    reasons, improvements, known = [], [], []
    for result in challenger:
        name, theirs = result['name'], champions.get(result['name'])
        if result['kind'] == 'must':
            if result['passed'] and theirs and not theirs['passed']:
                improvements.append(f'now passes: {name}')
            elif not result['passed'] and theirs and theirs['passed']:
                reasons.append(f'fails what the published driver passes: {name} ({result["detail"]})')
            elif not result['passed']:
                known.append(f'{name} ({result["detail"]})')
            continue
        if not theirs or not math.isfinite(result['value']) or not math.isfinite(theirs['value']):
            continue
        change = result['value'] - theirs['value']
        if result['better'] == 'lower':
            change = -change
        allowed = margin(name, theirs['value'])
        if change < -allowed:
            reasons.append(f'worse: {name} {theirs["value"]:.3f} -> {result["value"]:.3f} (margin {allowed:.3f})')
        elif change > allowed:
            improvements.append(f'better: {name} {theirs["value"]:.3f} -> {result["value"]:.3f}')
    if not reasons and not improvements:
        reasons.append('nothing is better than the published driver by more than the noise')
    return not reasons, reasons, improvements, known


def published_exam():
    """The exam file of the driver the site starts with (its snapshot in runs/)."""
    brain = DEFAULT_DRIVER.read_text().strip()
    return exam_path(brain)


def main():
    parser = argparse.ArgumentParser(description="Decide whether a snapshot's exam earns it a place over the published driver")
    parser.add_argument('challenger', help='the exam file of the snapshot to promote')
    parser.add_argument('--against', help="the exam file to beat (the published driver's by default)")
    arguments = parser.parse_args()
    against = Path(arguments.against) if arguments.against else published_exam()
    if not against.exists():
        sys.exit(f'No exam of the published driver at {against}: run slipstream.exam on it first')
    challenger = json.loads(Path(arguments.challenger).read_text())
    champion = json.loads(against.read_text())
    if challenger['quick'] != champion['quick']:
        sys.exit(f'Compare exams of the same tier: {arguments.challenger} is {"quick" if challenger["quick"] else "full"}, {against} is not')
    passed, reasons, improvements, known = decide(challenger['results'], champion['results'])
    print(f"{'PASS' if passed else 'FAIL'}: {challenger['brain']} against {champion['brain']}")
    for heading, lines in (('Blocking', reasons), ('Better', improvements), ('Known failures (both fail)', known)):
        if lines:
            print(f'{heading}:')
            for line in lines:
                print(f'  - {line}')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
