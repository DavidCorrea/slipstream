import json

import pytest

from slipstream.check import CHECKED_SCORES, recent_average, record, unchecked


def snapshot(run, step):
    (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints' / f'step-{step:012d}.zip').write_bytes(b'')


class TestCheckingSnapshots:
    def test_lists_snapshots_not_yet_checked_oldest_first(self, tmp_path):
        for step in (3000, 1000, 2000):
            snapshot(tmp_path, step)
        record(tmp_path, 'step-000000002000', {'pace': 1.0, 'place': 0.5, 'damage': 0.2, 'off_track': 0.01, 'finished': 1.0, 'wet_pace': 1.0, 'field_out': 0.1, 'field_off_track': 0.02, 'field_hits': 2.0})
        assert [path.stem for path in unchecked(tmp_path)] == ['step-000000001000', 'step-000000003000']

    def test_records_the_step_and_only_the_scores_that_judge_a_driver(self, tmp_path):
        snapshot(tmp_path, 1000)
        record(tmp_path, 'step-000000001000', {'pace': 1.04, 'place': 0.5, 'damage': 0.3, 'off_track': 0.0, 'finished': 1.0, 'risk_pace': 1.1, 'field_out': 0.1, 'field_off_track': 0.02, 'field_hits': 2.0})
        line = json.loads((tmp_path / 'check.jsonl').read_text())
        assert line['step'] == 1000 and set(line) == {'checkpoint', 'step', *CHECKED_SCORES}

    def test_averages_the_latest_snapshots_so_one_lucky_score_doesnt_decide(self, tmp_path):
        for step, place in ((1000, 0.6), (2000, 0.3), (3000, 0.45), (4000, 0.45)):
            snapshot(tmp_path, step)
            record(tmp_path, f'step-{step:012d}', {'pace': 1.0, 'place': place, 'damage': 0.2, 'off_track': 0.01, 'finished': 1.0, 'field_out': 0.1, 'field_off_track': 0.02, 'field_hits': 2.0})
        assert recent_average(tmp_path)['place'] == pytest.approx(0.4)
        assert recent_average(tmp_path)['pace'] == pytest.approx(1.0)

    def test_averages_the_field_race_over_only_the_snapshots_checked_with_it(self, tmp_path):
        (tmp_path / 'check.jsonl').write_text(json.dumps({'checkpoint': 'step-000000001000', 'step': 1000, 'pace': 1.0, 'place': 0.5,
                                                          'damage': 0.2, 'off_track': 0.01, 'finished': 1.0}) + '\n')
        snapshot(tmp_path, 2000)
        record(tmp_path, 'step-000000002000', {'pace': 1.2, 'place': 0.3, 'damage': 0.2, 'off_track': 0.01, 'finished': 1.0,
                                               'field_out': 0.25, 'field_off_track': 0.05, 'field_hits': 3.0})
        average = recent_average(tmp_path)
        assert average['pace'] == pytest.approx(1.1) and average['field_out'] == pytest.approx(0.25)
