"""The track surface over a race: rubber, marbles and the drying line.

The circuit is divided into cells, a few metres long and a fraction of the track wide. Cars lay rubber where they
drive, so the racing line grips more as the race goes on; the rubber their tyres shed gathers off the line as
marbles, which grip less, so going off-line to pass or defend costs. After rain the line cars take dries first,
and fresh rain washes that dry line away again.
"""
import numpy as np

CELL_LENGTH = 4.0      # metres
LANES = 8              # cells across the track
RUBBER = {'rate': 0.2, 'grip': 0.05, 'wash': 0.02}          # per second a car is on a cell; grip at full rubber; rain washing it off
MARBLES = {'per_wear': 40.0, 'grip': 0.15, 'sweep': 2.0}    # per unit of wear shed; grip lost on full marbles; cleared by a car driving through
DRY_LINE = {'rate': 0.5, 'effect': 0.6, 'rewet': 0.05}      # per second a car is on a cell; share of the wetness it takes away; rain undoing it


class Surface:
    def __init__(self, track):
        self.track = track
        rows = int(np.ceil(track.length / CELL_LENGTH))
        self.rubber = np.zeros((rows, LANES))
        self.marbles = np.zeros((rows, LANES))
        self.dryness = np.zeros((rows, LANES))

    def _cells(self, distance, lateral):
        """Row and lane of each point, and whether it's on the tarmac at all."""
        row = (np.floor(np.asarray(distance) / CELL_LENGTH).astype(int)) % len(self.rubber)
        half = self.track.width / 2
        on_tarmac = np.abs(lateral) < half
        lane = np.clip(np.floor((np.asarray(lateral) + half) / self.track.width * LANES).astype(int), 0, LANES - 1)
        return row, lane, on_tarmac

    def update(self, distance, lateral, on_track, wear_shed, rain, seconds):
        """One tick: cars on the tarmac lay rubber, sweep and dry their cells and shed marbles off the line; rain
        washes the rubber and re-wets the dry line everywhere."""
        row, lane, on_tarmac = self._cells(distance, lateral)
        driving = on_track & on_tarmac
        row, lane, shed = row[driving], lane[driving], np.asarray(wear_shed)[driving]
        np.add.at(self.rubber, (row, lane), RUBBER['rate'] * seconds)
        np.add.at(self.dryness, (row, lane), DRY_LINE['rate'] * seconds * (1 - rain))
        self.marbles[row, lane] *= max(0.0, 1 - MARBLES['sweep'] * seconds)
        # What tyres shed is flung to the edges of the track, off the line.
        for edge in (0, LANES - 1):
            np.add.at(self.marbles, (row, np.full_like(row, edge)), MARBLES['per_wear'] * shed / 2)
        self.rubber *= 1 - RUBBER['wash'] * rain * seconds
        self.dryness -= DRY_LINE['rewet'] * rain * seconds
        np.clip(self.rubber, 0, 1, out=self.rubber)
        np.clip(self.marbles, 0, 1, out=self.marbles)
        np.clip(self.dryness, 0, 1, out=self.dryness)

    def grip(self, distance, lateral):
        """How much grip the surface gives at each point: 1 for plain tarmac or off it, more on rubber, less on
        marbles."""
        row, lane, on_tarmac = self._cells(distance, lateral)
        grip = 1 + RUBBER['grip'] * self.rubber[row, lane] - MARBLES['grip'] * self.marbles[row, lane]
        return np.where(on_tarmac, grip, 1.0)

    def wetness(self, distance, lateral, wetness):
        """How wet the track is at each point, given how wet it is overall: drier on a dry line."""
        row, lane, on_tarmac = self._cells(distance, lateral)
        return np.where(on_tarmac, wetness * (1 - DRY_LINE['effect'] * self.dryness[row, lane]), wetness)
