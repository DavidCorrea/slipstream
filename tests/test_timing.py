import json

import numpy as np

from slipstream.session import SCRIPTED, RaceSession
from slipstream.timing import Timing

LENGTH = 300.0


def drive(timing, speeds, seconds, start=-10.0, tick=0.05):
    """Moves every car at its own constant speed (m/s) and returns all the timing events."""
    progress = np.full(len(speeds), start)
    events = []
    for step in range(int(round(seconds / tick))):
        before = progress.copy()
        progress = progress + np.asarray(speeds) * tick
        events += timing.record(step * tick, (step + 1) * tick, before, progress)
    return events


class TestSectors:
    def test_time_each_third_of_the_lap_from_the_moment_the_car_crosses_it(self):
        events = drive(Timing(LENGTH, 1), [10.0], 35.0)
        sectors = [event for event in events if event['kind'] == 'sector']
        # Crosses the line at 1 s, then every 10 s a sector boundary.
        assert [event['sector'] for event in sectors[:3]] == [0, 1, 2]
        assert all(abs(event['time'] - 10.0) < 1e-6 for event in sectors)

    def test_say_nothing_before_the_first_line_crossing(self):
        assert drive(Timing(LENGTH, 1), [10.0], 0.9) == []

    def test_rate_the_first_ever_time_in_a_sector_as_the_overall_best(self):
        events = drive(Timing(LENGTH, 1), [10.0], 12.0)
        assert events[0]['rating'] == 'overall'


class TestLaps:
    def test_time_a_lap_from_line_to_line(self):
        events = drive(Timing(LENGTH, 1), [10.0], 32.0)
        laps = [event for event in events if event['kind'] == 'lap']
        assert len(laps) == 1 and abs(laps[0]['time'] - 30.0) < 1e-6

    def test_rate_a_slower_car_by_its_own_best_and_the_faster_one_overall(self):
        timing = Timing(LENGTH, 2)
        events = drive(timing, [10.0, 12.0], 65.0)
        laps = [event for event in events if event['kind'] == 'lap']
        faster = [event for event in laps if event['car'] == 1]
        slower = [event for event in laps if event['car'] == 0]
        assert faster[0]['rating'] == 'overall'
        assert slower[0]['rating'] == 'personal'
        # The same pace again beats nobody, not even itself.
        assert slower[1]['rating'] == 'none'

    def test_keep_each_cars_last_and_best_lap(self):
        timing = Timing(LENGTH, 2)
        drive(timing, [10.0, 12.0], 65.0)
        assert abs(timing.best_lap[1] - 25.0) < 1e-6
        assert abs(timing.last_lap[0] - 30.0) < 1e-6


class TestFastestLapGhost:
    def test_hand_over_the_path_of_a_new_fastest_lap_once(self):
        timing = Timing(LENGTH, 1)
        progress = -1.0
        handed = []
        for step in range(700):
            before = progress
            progress += 0.5
            timing.record(step * 0.05, (step + 1) * 0.05, np.array([before]), np.array([progress]))
            timing.sample((step + 1) * 0.05, np.array([[progress, 0.0]]), np.array([0.0]))
            if timing.fastest_trace is not None:
                handed.append(timing.take_fastest_trace())
        assert len(handed) == 1
        trace = handed[0]
        assert trace['car'] == 0 and abs(trace['time'] - 30.0) < 1e-6
        # Times count from the start of that lap, and the path covers it.
        assert trace['points'][0][0] < 0.2 and trace['points'][-1][0] > 29.5


class TestRaceFrames:
    def test_carry_timing_events_and_lap_times_that_serialise(self):
        session = RaceSession(SCRIPTED, seed=5, cars=2, laps=2)
        frames = [session.advance(20) for _ in range(150)]
        timing = [event for frame in frames for event in frame['events']['timing']]
        assert any(event['kind'] == 'lap' for event in timing)
        assert any(frame['events']['fastestLap'] for frame in frames)
        assert any(time is not None for time in frames[-1]['cars']['bestLap'])
        assert all(start is None or start <= frames[-1]['time'] for start in frames[-1]['cars']['lapStart'])
        json.dumps(frames[-1])
