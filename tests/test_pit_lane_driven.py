import numpy as np

from slipstream import pit
from slipstream.car import CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.pit import PitLane, PitPlan
from slipstream.race import Race
from slipstream.scenery import place
from slipstream.stewards import PIT_LINE, PIT_SPEEDING
from slipstream.track import generate_track


def driven_race(seed=3, cars=1, laps=3, props=True):
    race = Race(generate_track(seed), CarSpecs.uniform(cars), laps, stewards=True, assisted=False)
    if props:
        race.place_props(place(race.track, race.pit, seed, 'countryside'))
    return race


def drive_until(race, condition, call=True, limit=8000, limiter=None):
    events = []
    for _ in range(limit):
        if condition(race):
            return events
        events.append(race.step(*scripted_controls(race), pit_call=np.full(race.count, call), plan=PitPlan.standard(race.count),
                                limiter=limiter))
    raise AssertionError(f'condition never met; pit state {race.pit_state}, lane along {race.lane_along}')


def put_in_lane(race, car, along, speed, lateral=None):
    lane, track = race.pit, race.track
    distance = lane.entry + along
    race.cars.position[car] = track.point_at(distance, lane.fast_lane if lateral is None else lateral)
    race.cars.heading[car] = track.heading_at(distance)
    race.cars.velocity[car] = race.cars.forward[car] * speed
    race.progress[car] = race.track.length + distance % race.track.length
    race.track_index, _, race.lateral = track.locate(race.cars.position)


class TestDrivingTheLane:
    def test_a_called_car_drives_in_stops_on_its_marks_and_rejoins_without_a_penalty(self):
        race = driven_race()
        race.cars.tyre_wear[0] = 0.75
        events = drive_until(race, lambda race: race.pit_state[0] == pit.STOPPED)
        assert any(tick.pit_entered for tick in events)
        assert abs(race.lane_along[0] - race.pit.boxes[0]) <= pit.BOX_MARKS['reach_along']
        events += drive_until(race, lambda race: race.stops[0] == 1 and race.pit_state[0] == pit.RACING, call=False)
        assert any(tick.pit_exited for tick in events)
        assert race.cars.tyre_wear[0] < 0.1
        assert not [penalty for tick in events for penalty in tick.penalties]
        assert not [hit for tick in events for hit in tick.prop_hits]

    def test_finishes_a_race_with_stops_counting_every_lap(self):
        race = driven_race(laps=2)
        events = drive_until(race, lambda race: race.done, limit=20000)
        assert race.finished[0]
        assert race.stops[0] >= 1
        assert sum(len(tick.laps) for tick in events) == 2

    def test_a_car_reaches_the_last_box_of_a_twenty_car_lane_and_rejoins(self):
        for seed in (3, 5, 8):
            race = driven_race(seed=seed, props=False)
            race.pit = PitLane(race.track, 20)
            race.pit.boxes = race.pit.boxes[-1:]
            race.place_props(place(race.track, race.pit, seed, 'countryside'))
            events = drive_until(race, lambda race: race.stops[0] == 1 and race.pit_state[0] == pit.RACING)
            assert not [hit for tick in events for hit in tick.prop_hits], f'seed {seed}'
            assert not [penalty for tick in events for penalty in tick.penalties], f'seed {seed}'

    def test_two_cars_called_together_queue_and_both_stop(self):
        race = driven_race(seed=3, cars=2)
        lane_contacts = []
        for _ in range(12000):
            if (race.stops == 1).all() and (race.pit_state == pit.RACING).all():
                break
            events = race.step(*scripted_controls(race), pit_call=np.ones(2, dtype=bool), plan=PitPlan.standard(2))
            lane_contacts += [contact for contact in events.contacts if race.in_lane[[contact.first, contact.second]].any()]
        assert (race.stops == 1).all()
        assert lane_contacts == []

    def test_a_car_that_wasnt_called_drives_straight_through(self):
        race = driven_race()
        put_in_lane(race, 0, 5.0, 20.0, lateral=race.pit.side * (race.track.width / 2 + 1.0))
        events = drive_until(race, lambda race: race.in_lane[0], call=False, limit=50)
        events += drive_until(race, lambda race: not race.in_lane[0], call=False)
        assert race.stops[0] == 0
        assert any(tick.pit_exited for tick in events)


class TestTheRules:
    def test_the_limiter_holds_a_car_at_the_speed_limit(self):
        race = driven_race(props=False)
        top = 0.0
        for _ in range(400):
            race.step(np.zeros(1), np.ones(1), np.zeros(1), limiter=np.ones(1, dtype=bool))
            top = max(top, race.cars.speed[0])
        assert pit.SPEED_LIMIT - 1.0 < top <= pit.SPEED_LIMIT

    def test_speeding_in_the_lane_costs_time_once_a_visit(self):
        race = driven_race(props=False)
        put_in_lane(race, 0, 60.0, 30.0)
        race.pit_state[0] = pit.IN_LANE
        penalties = []
        for _ in range(10):
            penalties += race.step(np.zeros(1), np.ones(1), np.zeros(1)).penalties
        assert [(penalty.reason, penalty.seconds) for penalty in penalties] == [('pit speeding', PIT_SPEEDING)]

    def test_the_only_way_in_is_through_the_mouth(self):
        race = driven_race(props=False)
        put_in_lane(race, 0, race.pit.mouth + 10.0, 20.0)
        events = race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        assert not race.in_lane[0] and not race.on_track[0]
        assert events.penalties == []

    def test_leaving_the_lane_over_the_white_line_costs_time(self):
        race = driven_race(props=False)
        put_in_lane(race, 0, 5.0, 15.0, lateral=race.pit.side * (race.track.width / 2 + 1.0))
        race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        assert race.in_lane[0]
        put_in_lane(race, 0, 60.0, 15.0, lateral=race.pit.side * (race.track.width / 2 - 1.0))
        events = race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        assert not race.in_lane[0]
        assert [(penalty.reason, penalty.seconds) for penalty in events.penalties] == [('pit line', PIT_LINE)]

    def test_a_car_short_of_its_marks_is_dragged_onto_them_for_a_price(self):
        exact, short = driven_race(props=False), driven_race(props=False)
        for race, off in ((exact, 0.0), (short, 2.0)):
            race.pit_state[0], race.lane_called[0] = pit.IN_LANE, True
            box = race.pit.boxes[0] - off
            put_in_lane(race, 0, box - 0.05, 0.0, lateral=race.pit.working_lane)
            race.step(np.zeros(1), np.zeros(1), np.ones(1), plan=PitPlan.standard(1))
        assert exact.pit_state[0] == short.pit_state[0] == pit.STOPPED
        assert short.service_left[0] - exact.service_left[0] == pit.BOX_MARKS['drag']

    def test_assisted_races_still_drive_cars_through_the_lane(self):
        race = Race(generate_track(3), CarSpecs.uniform(1), 3)
        drive_until(race, lambda race: race.pit_state[0] == pit.STOPPED)
        assert race.lane_along[0] == race.pit.boxes[0]
