import json

from slipstream.check import CHECKED_SCORES, record, unchecked


def snapshot(run, step):
    (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints' / f'step-{step:012d}.zip').write_bytes(b'')


class TestCheckingSnapshots:
    def test_lists_snapshots_not_yet_checked_oldest_first(self, tmp_path):
        for step in (3000, 1000, 2000):
            snapshot(tmp_path, step)
        record(tmp_path, 'step-000000002000', {'pace': 1.0, 'place': 0.5, 'damage': 0.2, 'off_track': 0.01, 'finished': 1.0, 'wet_pace': 1.0})
        assert [path.stem for path in unchecked(tmp_path)] == ['step-000000001000', 'step-000000003000']

    def test_records_the_step_and_only_the_scores_that_judge_a_driver(self, tmp_path):
        snapshot(tmp_path, 1000)
        record(tmp_path, 'step-000000001000', {'pace': 1.04, 'place': 0.5, 'damage': 0.3, 'off_track': 0.0, 'finished': 1.0, 'risk_pace': 1.1})
        line = json.loads((tmp_path / 'check.jsonl').read_text())
        assert line['step'] == 1000 and set(line) == {'checkpoint', 'step', *CHECKED_SCORES}
