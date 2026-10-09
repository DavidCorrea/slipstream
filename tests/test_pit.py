import numpy as np

from slipstream import pit
from slipstream.car import COMPOUNDS, FUEL_CAPACITY, CarSpecs
from slipstream.drivers import scripted_controls
from slipstream.pit import PitPlan, service_times
from slipstream.race import Race
from slipstream.track import SPACING, generate_track


def race_with_one_car(seed=3, laps=3):
    return Race(generate_track(seed), CarSpecs.uniform(1), laps)


def drive_until(race, condition, plan=None, call=True, limit=6000):
    events = []
    for _ in range(limit):
        if condition(race):
            return events
        events.append(race.step(*scripted_controls(race), pit_call=np.array([call]), plan=plan))
    raise AssertionError('condition never met')


class TestPitStops:
    def test_take_a_called_car_through_the_lane_stop_it_in_its_box_and_send_it_back_out(self):
        race = race_with_one_car()
        race.cars.tyre_wear[0] = 0.8
        race.cars.fuel[0] = 10.0
        plan = PitPlan.standard(1)
        plan.compound[0] = COMPOUNDS.index('soft')
        stopped = drive_until(race, lambda race: race.pit_state[0] == pit.STOPPED, plan)
        assert race.lane_along[0] == race.pit.boxes[0]
        assert any(tick.pit_entered for tick in stopped)
        drive_until(race, lambda race: race.pit_state[0] == pit.RACING, plan, call=False)
        assert race.cars.tyre_wear[0] == 0.0
        assert race.cars.compound[0] == COMPOUNDS.index('soft')
        assert race.cars.fuel[0] == FUEL_CAPACITY
        assert race.stops[0] == 1
        assert race.on_track[0]

    def test_never_pit_a_car_that_doesnt_ask(self):
        race = race_with_one_car()
        drive_until(race, lambda race: race.lap_of()[0] >= 1, call=False)
        assert race.stops[0] == 0

    def test_decide_once_a_lap_at_the_decision_point(self):
        race = race_with_one_car(laps=4)
        # Call only for a moment as the car reaches the decision point, then withdraw: the stop still happens.
        drive_until(race, lambda race: race.pit.in_window(race.progress % race.track.length)[0] and race.progress[0] > 0, call=False)
        race.step(*scripted_controls(race), pit_call=np.array([True]))
        assert race.pit_state[0] == pit.CALLED
        drive_until(race, lambda race: race.pit_state[0] == pit.STOPPED, call=False)

    def test_wait_for_the_next_lap_when_called_after_the_decision_point(self):
        race = race_with_one_car(laps=4)
        drive_until(race, lambda race: race.pit.in_window(race.progress % race.track.length)[0] and race.progress[0] > 0, call=False)
        race.step(*scripted_controls(race), pit_call=np.array([False]))
        lap = race.lap_of()[0]
        drive_until(race, lambda race: race.in_lane[0] or race.lap_of()[0] > lap, call=True)
        assert not race.in_lane[0]
        drive_until(race, lambda race: race.pit_state[0] == pit.STOPPED, call=True)

    def test_keep_counting_laps_through_the_pit_lane(self):
        race = race_with_one_car(laps=2)
        events = drive_until(race, lambda race: race.done, call=True, limit=20000)
        assert race.finished[0]
        assert race.stops[0] >= 1
        assert sum(len(tick.laps) for tick in events) == 2

    def test_stand_still_for_as_long_as_the_jobs_take(self):
        race = race_with_one_car()
        race.cars.fuel[0] = 0.0
        race.cars.damage[0] = 0.5
        plan = PitPlan.standard(1)
        plan.brakes[0] = True
        jobs, standing = service_times(plan, race.cars, 0)
        assert jobs['fuel'] == round(FUEL_CAPACITY / pit.SERVICE['fuel_per_second'], 2)
        assert standing == pit.SERVICE['base'] + jobs['fuel'] + jobs['repair'] + jobs['brakes']

    def test_keep_cars_in_the_lane_out_of_contact_with_the_track(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=2)
        race.pit_state[0] = pit.IN_LANE
        race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * 1.0
        assert race._resolve_contacts() == []


class TestDamage:
    def bump(self, speed):
        race = Race(generate_track(0), CarSpecs.uniform(2), laps=1)
        # Nose to tail, just touching: front wing 2.8 m ahead of one car's centre, rear 2.2 m behind the other's.
        race.cars.heading[1] = race.cars.heading[0]
        race.cars.position[1] = race.cars.position[0] + race.cars.forward[0] * 4.9
        race.cars.velocity[0] = race.cars.forward[0] * speed
        race.step(np.zeros(2), np.zeros(2), np.zeros(2))
        return race.cars.damage

    def test_comes_from_real_impacts(self):
        assert (self.bump(20) > 0).all()

    def test_spares_cars_that_only_rub(self):
        assert (self.bump(0.8) == 0).all()


class TestPitEntry:
    def test_notices_a_car_crossing_it_even_when_it_covers_more_centre_line_than_its_speed_says(self):
        # On the inside of a bend a car gains more centre-line distance per tick than speed x time; a crossing
        # predicted from speed alone could slip through the gap and the car would never be let into the lane.
        race = race_with_one_car(laps=3)
        track, lane = race.track, race.pit
        place = lambda distance: (track.point_at(np.array([distance % track.length]))[0], distance)
        race.progress[0] = track.length - (track.length - lane.entry) - 1.0
        race.cars.position[0], _ = place(lane.entry - 1.0)
        race.track_index[0] = track.locate(race.cars.position)[0][0]
        race.pit_state[0] = pit.CALLED
        race.cars.velocity[0] = 0.0
        race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        race.cars.position[0], _ = place(lane.entry + 2.0)
        race.cars.velocity[0] = 0.0
        race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        assert race.in_lane[0]

    def test_ignores_a_called_car_passing_the_entry_far_off_the_track(self):
        # A car spinning across the grass is nowhere near the lane: it can't turn in, wherever along the circuit it
        # happens to be.
        race = race_with_one_car(laps=3)
        track, lane = race.track, race.pit
        # 60 m to the right of the entry on this circuit is open grass, well clear of any other part of the track.
        place = lambda distance: track.point_at(np.array([distance % track.length]), np.array([-60.0]))[0]
        race.progress[0] = lane.entry - 1.0
        race.cars.position[0] = place(lane.entry - 1.0)
        race.track_index[0] = int(lane.entry // SPACING)
        race.pit_state[0] = pit.CALLED
        for distance in (lane.entry - 0.5, lane.entry + 2.0, lane.entry + 3.0):
            race.cars.position[0] = place(distance)
            race.cars.velocity[0] = 0.0
            race.step(np.zeros(1), np.zeros(1), np.zeros(1))
        assert abs(race.lateral[0]) > 50 and not race.in_lane[0] and race.pit_state[0] == pit.CALLED

    def test_never_counts_sliding_backwards_as_passing_it(self):
        # Going backwards along the loop isn't nearly a whole lap forwards. (Read that way, a car spinning backwards
        # "passed" the entry, was let into the lane far beyond its end, and looped in and out of it forever.)
        lane = race_with_one_car().pit
        before = lane.entry - 100.0
        assert not lane.entered(np.array([before]), np.array([before - 1.0]))[0]
        assert lane.entered(np.array([lane.entry - 1.0]), np.array([lane.entry + 1.0]))[0]


class TestTheLaneLikeARealOne:
    def race_in_the_lane(self, cars=2):
        """`cars` cars called in and placed nose to tail just before the lane entry."""
        race = Race(generate_track(3), CarSpecs.uniform(cars), laps=3)
        lane = race.pit
        for car in range(cars):
            distance = lane.entry - 2.0 - car * 6.0
            race.progress[car] = distance
            race.cars.position[car] = race.track.point_at(np.array([distance]))[0]
            race.cars.heading[car] = race.track.heading_at(np.array([distance]))[0]
            race.cars.velocity[car] = race.cars.forward[car] * 40.0
        race.track_index = race.track.locate(race.cars.position)[0]
        race.pit_state[:] = pit.CALLED
        return race

    def test_cars_called_in_together_queue_rather_than_overlap(self):
        from slipstream.race import FOOTPRINT_OFFSET, _footprint_overlap
        race = self.race_in_the_lane(cars=3)
        cars = race.cars
        for _ in range(600):
            race.step(np.zeros(3), np.full(3, 0.3), np.zeros(3), np.zeros(3, dtype=bool))
            lane_cars = np.flatnonzero(race.in_lane)
            centres = cars.position + cars.forward * FOOTPRINT_OFFSET
            for first in lane_cars:
                for second in lane_cars:
                    if first < second:
                        assert _footprint_overlap(centres[first], cars.forward[first], cars.left[first],
                                                  centres[second], cars.forward[second], cars.left[second]) is None

    def test_a_car_in_its_box_stands_in_the_working_lane_out_of_the_way(self):
        race = self.race_in_the_lane(cars=2)
        lane = race.pit
        for _ in range(600):
            race.step(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool))
            if race.pit_state[0] == pit.STOPPED and race.in_lane[1] and race.lane_along[1] < lane.boxes[0]:
                break
        lateral = lambda car: abs(race.track.locate(race.cars.position[[car]])[2][0])
        assert race.pit_state[0] == pit.STOPPED
        assert lateral(0) > lateral(1) + 2.0

    def test_a_car_heading_for_a_later_box_drives_past_one_being_worked_on(self):
        race = self.race_in_the_lane(cars=2)
        lane = race.pit
        lane.boxes[1] = lane.boxes[0] + 2 * pit.BOX_SPACING
        slowest_past = np.inf
        for _ in range(900):
            race.step(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool))
            if race.pit_state[0] == pit.STOPPED and abs(race.lane_along[1] - lane.boxes[0]) < 3:
                slowest_past = min(slowest_past, race.lane_speed[1])
        assert slowest_past > 10.0

    def test_a_car_leaving_its_box_waits_for_a_gap_in_the_fast_lane(self):
        race = self.race_in_the_lane(cars=2)
        lane = race.pit
        lane.boxes[1] = lane.boxes[0] + 3 * pit.BOX_SPACING
        # Car 0 has just been released from its box; car 1 is coming down the fast lane right behind it.
        race.pit_state[:] = pit.IN_LANE
        race.lane_along[0], race.lane_speed[0], race.lane_shift[0] = lane.boxes[0], 0.0, 1.0
        race.service_jobs[0] = {'jobs': {}, 'plan': PitPlan.standard(2)}
        race.lane_along[1], race.lane_speed[1], race.lane_shift[1] = lane.boxes[0] - 8.0, pit.SPEED_LIMIT, 0.0
        held_until = None
        for tick in range(80):
            race.step(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool))
            if held_until is None and race.lane_along[0] > lane.boxes[0] + 0.5:
                held_until = race.lane_along[1]
        assert held_until is not None and held_until > lane.boxes[0] + 5.0


class TestAFullField:
    def test_every_box_fits_in_the_lane_with_room_to_pull_out_and_rejoin(self):
        from slipstream.pit import PULL_OUT, RAMP, PitLane
        track = generate_track(7)
        for count in (2, 8, 14, 20):
            lane = PitLane(track, count)
            assert lane.boxes[-1] + PULL_OUT <= lane.length - RAMP

    def test_reports_a_stop_in_the_last_box_of_a_twenty_car_field(self):
        race = Race(generate_track(7), CarSpecs.uniform(20), laps=3)
        last = 19
        race.progress[last] = race.pit.entry - 5.0
        stopped = []
        calls = np.zeros(20, dtype=bool)
        calls[last] = True
        for _ in range(2000):
            events = race.step(np.zeros(20), np.full(20, 0.4), np.zeros(20), calls, PitPlan.standard(20))
            stopped += [car for car, _, _ in events.pit_stopped]
            if stopped:
                assert race.service_jobs[last] is not None
                break
        assert stopped == [last]
