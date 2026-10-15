import numpy as np

from slipstream.car import CarSpecs
from slipstream.drivers import BACK_UP, scripted_controls
from slipstream.pit import PitPlan
from slipstream.race import Race
from slipstream.scenery import BREAKABLE, SOFT, SOLID, Prop, place
from slipstream.track import generate_track


def race_heading_into(hardness, speed=40.0, kind='wall', seed=3):
    """One car on the start straight at `speed`, with a prop right across the track 25 m ahead."""
    track = generate_track(seed)
    race = Race(track, CarSpecs.uniform(1), laps=3)
    distance = float(race.progress[0] % track.length) + 25.0
    x, y = track.point_at(distance)
    tangent = track.tangents[int(distance / 2.0) % track.size]
    race.place_props([Prop(kind, x, y, hardness, half_length=1.0, half_width=track.width, angle=float(np.arctan2(tangent[1], tangent[0])))])
    race.cars.velocity[0] = race.cars.forward[0] * speed
    # How far ahead of the car the prop stands, in progress (the car starts behind the line).
    return race, float(race.progress[0]) + 25.0


def drive(race, seconds, throttle=1.0):
    hits, crashed = [], []
    for _ in range(int(seconds / 0.05)):
        events = race.step(np.zeros(race.count), np.full(race.count, throttle), np.zeros(race.count), np.zeros(race.count, dtype=bool), PitPlan.standard(race.count))
        hits += events.prop_hits
        crashed += events.crashed
    return hits, crashed


class TestHittingProps:
    def test_a_solid_prop_stops_a_car_instead_of_letting_it_through(self):
        race, distance = race_heading_into(SOLID, speed=20.0)
        hits, _ = drive(race, 3.0)
        assert hits and hits[0].prop == 0
        assert race.progress[0] < distance
        assert race.cars.damage[0] > 0

    def test_a_breakable_prop_breaks_and_only_slows_the_car(self):
        race, distance = race_heading_into(BREAKABLE, speed=30.0, kind='board')
        hits, _ = drive(race, 2.0, throttle=0.0)
        assert [hit.broke for hit in hits] == [True]
        assert race.props.broken[0]
        assert race.progress[0] > distance + 5
        assert race.cars.damage[0] < 0.1

    def test_a_tyre_wall_spares_the_car_more_than_a_wall(self):
        damage = {}
        for hardness in (SOLID, SOFT):
            race, _ = race_heading_into(hardness, speed=25.0)
            drive(race, 2.0, throttle=0.0)
            damage[hardness] = race.cars.damage[0]
        assert 0 < damage[SOFT] < damage[SOLID]

    def test_a_very_hard_hit_puts_the_car_out(self):
        race, _ = race_heading_into(SOLID, speed=70.0)
        _, crashed = drive(race, 1.5)
        assert crashed == [0] and race.retired[0]

    def test_a_car_stuck_against_a_prop_retires_after_a_while(self):
        race, _ = race_heading_into(SOLID, speed=8.0)
        _, crashed = drive(race, 25.0, throttle=0.4)
        assert race.retired[0] and not race.finished[0]


    def test_a_car_nosed_into_a_wall_beside_the_track_backs_out_and_drives_on(self):
        track = generate_track(3)
        race = Race(track, CarSpecs.uniform(1), laps=3)
        start = float(race.progress[0])
        index = int((race.progress[0] % track.length) / 2.0) % track.size
        tangent, normal = track.tangents[index], track.normals[index]
        # A long wall along the left edge, and the car pointing straight at it, almost stopped.
        wall = track.points[index] + normal * (track.width / 2 + 2.0) + tangent * 40
        race.place_props([Prop('wall', wall[0], wall[1], SOLID, half_length=80, half_width=0.3, angle=float(np.arctan2(tangent[1], tangent[0])))])
        race.cars.position[0] = track.points[index] + normal * (track.width / 2 - 1.0)
        race.cars.heading[0] = float(np.arctan2(normal[1], normal[0]))
        race.cars.velocity[0] = normal * 2.0
        # Driven as a driver would, steering back for the track once it's free of the wall.
        for _ in range(int(12.0 / 0.05)):
            race.step(*scripted_controls(race))
        assert not race.retired[0] and race.progress[0] > start + 40


    def test_a_car_off_the_track_stopped_by_a_tree_in_front_turns_back_to_the_track(self):
        track = generate_track(3)
        race = Race(track, CarSpecs.uniform(1), laps=3)
        start = float(race.progress[0])
        index = int((race.progress[0] % track.length) / 2.0) % track.size
        tangent, normal = track.tangents[index], track.normals[index]
        # Off the left edge, pointing along the track, with a tree just ahead of its nose.
        race.cars.position[0] = track.points[index] + normal * (track.width / 2 + 3.0)
        race.cars.heading[0] = float(np.arctan2(tangent[1], tangent[0]))
        tree = race.cars.position[0] + tangent * 4.0
        race.place_props([Prop('tree', tree[0], tree[1], SOLID, radius=1.0)])
        race.cars.velocity[0] = tangent * 2.0
        for _ in range(int(12.0 / 0.05)):
            race.step(*scripted_controls(race))
        assert not race.retired[0] and race.progress[0] > start + 40


class TestClearRacing:
    def test_a_clean_lap_hits_nothing(self):
        track = generate_track(5)
        race = Race(track, CarSpecs.uniform(1), laps=1)
        race.place_props(place(track, race.pit, 5, 'city'))
        hits = []
        while not race.done:
            events = race.step(*scripted_controls(race))
            hits += events.prop_hits
        assert race.finished[0] and hits == []

    def test_races_without_props_have_none_to_hit(self):
        race = Race(generate_track(3), CarSpecs.uniform(2), laps=1)
        events = race.step(*scripted_controls(race))
        assert race.props is None and events.prop_hits == [] and events.crashed == []


class TestBackingOut:
    def stopped_car(self):
        race = Race(generate_track(3), CarSpecs.uniform(1), laps=3)
        race.cars.stopped_for[0] = 2.0
        return race

    def test_a_scripted_driver_stopped_for_a_while_selects_reverse(self):
        race = self.stopped_car()
        _, throttle, brake = scripted_controls(race)
        assert throttle[0] == 1.0 and brake[0] == 1.0

    def test_in_reverse_it_backs_up_steering_the_other_way_then_brakes_to_go_forward_again(self):
        race = self.stopped_car()
        steer_forward, _, _ = scripted_controls(race)
        race.cars.heading[0] += 1.2                      # pointing well off the way ahead
        race.cars.reversing[0], race.cars.reversed_for[0] = True, 0.5
        steer, throttle, brake = scripted_controls(race)
        assert throttle[0] == 1.0 and brake[0] == 0.0 and steer[0] == 1.0   # nose wants right: reverse steers left
        race.cars.reversed_for[0] = BACK_UP['longest'] + 0.5
        _, throttle, brake = scripted_controls(race)
        assert throttle[0] == 0.0 and brake[0] == 1.0

    def test_a_blocked_scripted_car_really_drives_backwards(self):
        race = self.stopped_car()
        for _ in range(int(3.0 / 0.05)):
            race.step(*scripted_controls(race))
            if race.cars.reversing[0] and race.cars.speed[0] < -1.0:
                return
        raise AssertionError('never backed up')

    def test_a_scripted_driver_with_the_road_clear_never_reverses(self):
        race = Race(generate_track(5), CarSpecs.uniform(1), laps=1)
        while not race.done:
            race.step(*scripted_controls(race))
            assert not race.cars.reversing[0]
