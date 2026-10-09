"""Fixed tests of a trained driver against the scripted one, on circuits that never change.

- Time trial: the trained driver alone, then the scripted driver alone, on the same circuit. `pace` is the
  scripted time over the trained time, so above 1 means faster than the script.
- Mixed race: three trained and three scripted cars on one grid, alternating slots. `place` is the trained
  cars' average finishing position, 0 for first and 1 for last, and `damage` their average damage at the flag.
- Long race (for networks that make pit calls): the network alone over LONG_LAPS laps, where a tank won't last
  and tyres go off, against the scripted driver with its strategist. `long_pace` is the script's time over the
  network's (0 if the network didn't finish, say because it ran dry), and `stops` how often the network pitted.
- Wet time trial: the same as the time trial on a soaked track, both cars starting on full wets. `wet_pace` is
  the scripted time over the trained time and `wet_off_track` the trained car's share of time off the tarmac.
- Trait effects (for networks that read personality): solo time trials at the two ends of a trait, everything
  else neutral. `risk_pace` is the cautious driver's time over the risky one's, so above 1 means risk makes it
  faster. `conservation_saving` is the pusher's tyre wear plus fuel used over the saver's, so above 1 means the
  saver really saves. Both sit at 1 for a network that ignores personality.
"""
import numpy as np

from . import personality as traits
from .brains import NetworkDriver
from .car import CarSpecs, FUEL_CAPACITY
from .field import field_controls, makes_pit_calls
from .human import Driving
from .observe import OBSERVATION_NAMES
from .race import Race
from .track import generate_track
from .weather import Weather

# Twelve circuits (six wet): with six, one snapshot's score swung by about 3% from luck alone, more than the gains
# a driver near the scripted one's pace makes from one snapshot to the next.
SEEDS = tuple(range(1001, 1013))
TRAIT_SEEDS = (1001, 1004)
LAPS = 2
LONG_SEEDS = (1002,)
WET_SEEDS = (1001, 1003, 1005, 1007, 1009, 1011)
LONG_LAPS = 8
STALLED_SECONDS = 15.0


def benchmark_race(track, cars, laps, weather=None):
    """A race on a benchmark circuit. Every car, trained or scripted, races with human hands and feet (see
    human.py), as drivers train with them; chance is seeded from the circuit, so every snapshot faces the same race."""
    return Race(track, CarSpecs.uniform(cars), laps, weather=weather, rng=np.random.default_rng(track.seed),
                driving=Driving(cars, np.random.default_rng(track.seed + 1)))


def run_race(race, model, trained, personality=None):
    """Plays a race to the end, or until every trained car is stuck (STALLED_SECONDS without gaining ground: a
    network that can't drive yet would otherwise sit out every race's whole time limit). Cars where `trained` is
    True follow the model, the rest the script. Trained cars drive with a neutral personality unless one is given,
    so runs are compared on driving alone."""
    driver = NetworkDriver(model, race.count) if model is not None else None
    furthest, gained_at = race.progress.copy(), np.zeros(race.count)
    while not race.done:
        race.step(*field_controls(race, driver, trained, personality))
        gaining = race.progress > furthest + 1.0
        furthest = np.where(gaining, race.progress, furthest)
        gained_at = np.where(gaining, race.time, gained_at)
        stuck = race.finished | (race.time - gained_at > STALLED_SECONDS)
        if stuck[trained].all() and not race.finished[trained].all():
            break
    return race


def benchmark(model, seeds=SEEDS):
    paces, finished, off_track, places, damage = [], [], [], [], []
    for seed in seeds:
        track = generate_track(seed)
        solo = run_race(benchmark_race(track, 1, LAPS), model, np.array([True]))
        script = run_race(benchmark_race(track, 1, LAPS), None, np.array([False]))
        finished.append(float(solo.finished[0]))
        paces.append(script.finish_time[0] / solo.finish_time[0] if solo.finished[0] else 0.0)
        off_track.append(solo.off_track_ticks[0] / solo.tick)

        trained = np.arange(6) % 2 == 0
        mixed = run_race(benchmark_race(track, 6, LAPS), model, trained)
        place = np.empty(6)
        place[mixed.standings()] = np.arange(6) / 5
        places.append(place[trained].mean())
        damage.append(mixed.cars.damage[trained].mean())
    scores = {
        'pace': float(np.mean(paces)),
        'finished': float(np.mean(finished)),
        'off_track': float(np.mean(off_track)),
        'place': float(np.mean(places)),
        'damage': float(np.mean(damage)),
    }
    scores.update(wet_time_trial(model))
    if makes_pit_calls(model):
        scores.update(long_race(model))
    if model.observation_space.shape[0] >= OBSERVATION_NAMES.index('personality.conservation') + 1:
        scores.update(trait_effects(model))
    return scores


def wet_time_trial(model, seeds=WET_SEEDS):
    paces, off_track = [], []
    for seed in seeds:
        track = generate_track(seed)
        wet = lambda: benchmark_race(track, 1, LAPS, Weather.for_race(seed, track.length, LAPS, 'wet'))
        solo = run_race(wet(), model, np.array([True]))
        script = run_race(wet(), None, np.array([False]))
        paces.append(script.finish_time[0] / solo.finish_time[0] if solo.finished[0] else 0.0)
        off_track.append(solo.off_track_ticks[0] / solo.tick)
    return {'wet_pace': float(np.mean(paces)), 'wet_off_track': float(np.mean(off_track))}


def long_race(model, seeds=LONG_SEEDS):
    paces, stops = [], []
    for seed in seeds:
        track = generate_track(seed)
        network = run_race(benchmark_race(track, 1, LONG_LAPS), model, np.array([True]))
        script = run_race(benchmark_race(track, 1, LONG_LAPS), None, np.array([False]))
        paces.append(script.finish_time[0] / network.finish_time[0] if network.finished[0] else 0.0)
        stops.append(network.stops[0])
    return {'long_pace': float(np.mean(paces)), 'stops': float(np.mean(stops))}


def solo_with(model, track, **trait_values):
    personality = traits.neutral(1)
    for name, value in trait_values.items():
        personality[0, traits.TRAITS.index(name)] = value
    return run_race(benchmark_race(track, 1, LAPS), model, np.array([True]), personality)


def trait_effects(model, seeds=TRAIT_SEEDS):
    paces, savings = [], []
    for seed in seeds:
        track = generate_track(seed)
        cautious, risky = solo_with(model, track, risk=0.0), solo_with(model, track, risk=1.0)
        if cautious.finished[0] and risky.finished[0]:
            paces.append(cautious.finish_time[0] / risky.finish_time[0])
        used = lambda race: race.tyre_used[0] + race.fuel_used[0] / FUEL_CAPACITY
        pusher, saver = solo_with(model, track, conservation=0.0), solo_with(model, track, conservation=1.0)
        savings.append(used(pusher) / max(used(saver), 1e-6))
    return {'risk_pace': float(np.mean(paces)) if paces else 0.0, 'conservation_saving': float(np.mean(savings))}
