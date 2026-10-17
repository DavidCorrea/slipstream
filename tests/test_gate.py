from slipstream.gate import decide


def must(name, passed, detail=''):
    return {'name': name, 'kind': 'must', 'passed': passed, 'value': float(passed), 'better': None, 'detail': detail}


def scored(name, value, better='higher'):
    return {'name': name, 'kind': 'scored', 'passed': None, 'value': value, 'better': better, 'detail': ''}


class TestThePromotionGate:
    def test_a_challenger_that_fails_what_the_published_driver_passes_is_refused(self):
        passed, reasons, _, _ = decide([must('starts', False), scored('pace', 2.0)], [must('starts', True), scored('pace', 1.0)])
        assert not passed and any('starts' in reason for reason in reasons)

    def test_a_score_worse_beyond_its_margin_is_refused(self):
        passed, reasons, _, _ = decide([scored('hits', 3.0, 'lower')], [scored('hits', 2.0, 'lower')])
        assert not passed and any('worse: hits' in reason for reason in reasons)

    def test_a_change_within_the_noise_is_neither_better_nor_worse(self):
        passed, reasons, improvements, _ = decide([scored('pace', 1.005)], [scored('pace', 1.0)])
        assert not passed and not improvements and 'noise' in reasons[0]

    def test_newly_passing_a_must_test_is_an_improvement(self):
        passed, _, improvements, _ = decide([must('pit stops', True)], [must('pit stops', False)])
        assert passed and improvements == ['now passes: pit stops']

    def test_what_both_fail_is_listed_but_does_not_block(self):
        passed, reasons, _, known = decide([must('pit stops', False, '0/8'), scored('pace', 1.5)], [must('pit stops', False), scored('pace', 1.0)])
        assert passed and not reasons and known == ['pit stops (0/8)']
