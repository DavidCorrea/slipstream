"""Lap and sector times, like a timing screen: every car's time through each third of the lap, rated against its
own best and everyone's, and the path of the fastest lap so far, for the viewer's ghost car.

Times are measured from the moment a car crosses a boundary, found by interpolating within the tick, so they
don't jump by a whole tick depending on where a crossing happened to land. Nothing is timed until a car first
crosses the line: the run from the grid isn't a lap.
"""
import numpy as np

SECTORS = 3
SAMPLE_EVERY = 0.1   # seconds between points on a lap's recorded path
BETTER_BY = 1e-6     # matching a best time doesn't beat it


class Timing:
    def __init__(self, track_length, count):
        self.track_length = track_length
        self.sector_length = track_length / SECTORS
        self.count = count
        self.boundary_time = np.full(count, np.nan)          # when each car last crossed a sector boundary
        # The furthest boundary each car has crossed: a car that spins back over one and crosses it again
        # hasn't started a new sector.
        self.furthest = np.full(count, -1)
        self.lap_start = np.full(count, np.nan)
        self.last_lap = np.full(count, np.nan)
        self.best_lap = np.full(count, np.nan)
        self.best_sectors = np.full((count, SECTORS), np.nan)
        self.overall_best_lap = np.inf
        self.overall_best_sectors = np.full(SECTORS, np.inf)
        self.paths = [[] for _ in range(count)]
        self.fastest_trace = None

    def record(self, time_before, time_after, progress_before, progress_after):
        """Timing events for every boundary crossed between two moments, in the order they happened."""
        events = []
        boundary_before = np.floor(progress_before / self.sector_length)
        boundary_after = np.floor(progress_after / self.sector_length)
        for car in np.flatnonzero(boundary_after > boundary_before):
            for boundary in range(int(boundary_before[car]) + 1, int(boundary_after[car]) + 1):
                if boundary < 0 or boundary <= self.furthest[car]:
                    continue
                self.furthest[car] = boundary
                share = (boundary * self.sector_length - progress_before[car]) / (progress_after[car] - progress_before[car])
                events += self._cross(int(car), boundary, time_before + (time_after - time_before) * share)
        return events

    def sample(self, time, positions, headings):
        """Adds where every car is to the path of its current lap, every SAMPLE_EVERY seconds."""
        for car in range(self.count):
            if np.isnan(self.lap_start[car]):
                continue
            path = self.paths[car]
            since_start = time - self.lap_start[car]
            if path and since_start - path[-1][0] < SAMPLE_EVERY:
                continue
            path.append([round(float(since_start), 2), round(float(positions[car][0]), 2), round(float(positions[car][1]), 2),
                         round(float(headings[car]), 3)])

    def take_fastest_trace(self):
        """The newest fastest lap's path, once: later calls give None until another lap beats it."""
        trace, self.fastest_trace = self.fastest_trace, None
        return trace

    def _cross(self, car, boundary, time):
        sector = (boundary - 1) % SECTORS
        crossed_line = boundary % SECTORS == 0
        events = []
        if not np.isnan(self.boundary_time[car]):
            sector_time = time - self.boundary_time[car]
            rating = self._rate(sector_time, self.best_sectors[car, sector], self.overall_best_sectors[sector])
            if rating == 'overall':
                self.overall_best_sectors[sector] = sector_time
            if rating != 'none':
                self.best_sectors[car, sector] = sector_time
            events.append({'car': car, 'kind': 'sector', 'sector': sector, 'time': round(sector_time, 3), 'rating': rating})
        self.boundary_time[car] = time
        if not crossed_line:
            return events
        if not np.isnan(self.lap_start[car]):
            events.append(self._finish_lap(car, time))
        self.lap_start[car] = time
        self.paths[car] = []
        return events

    def _finish_lap(self, car, time):
        lap_time = time - self.lap_start[car]
        rating = self._rate(lap_time, self.best_lap[car], self.overall_best_lap)
        self.last_lap[car] = lap_time
        if rating != 'none':
            self.best_lap[car] = lap_time
        if rating == 'overall':
            self.overall_best_lap = lap_time
            self.fastest_trace = {'car': car, 'time': round(lap_time, 3), 'points': self.paths[car]}
        return {'car': car, 'kind': 'lap', 'time': round(lap_time, 3), 'rating': rating}

    @staticmethod
    def _rate(time, personal_best, overall_best):
        if time < overall_best - BETTER_BY:
            return 'overall'
        if np.isnan(personal_best) or time < personal_best - BETTER_BY:
            return 'personal'
        return 'none'
