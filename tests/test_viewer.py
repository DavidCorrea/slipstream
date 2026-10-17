import asyncio

from slipstream.viewer import Viewer


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


def run(*steps):
    viewer = Viewer(FakeSocket())

    async def play():
        for step in steps:
            await step(viewer)
    asyncio.run(play())
    return viewer


class TestTheGrid:
    def test_a_new_race_waits_on_the_grid_until_you_start_it(self):
        async def start(viewer):
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 1})
            for _ in range(10):
                await viewer.tick()
        viewer = run(start)
        kinds = [message['type'] for message in viewer.socket.sent]
        assert kinds == ['race'] and viewer.session.race.tick == 0

    def test_your_driver_and_car_can_be_set_on_the_grid_then_the_race_goes(self):
        async def setup(viewer):
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 1})
            await viewer.handle({'type': 'yours', 'car': 2})
            await viewer.handle({'type': 'stats', 'stats': {'risk': 0.9, 'grip': 0.8}})
            await viewer.handle({'type': 'go'})
            for _ in range(5):
                await viewer.tick()
        viewer = run(setup)
        assert viewer.session.yours == 2 and viewer.session.stats[2]['risk'] == 0.9
        assert any(message['type'] == 'frame' for message in viewer.socket.sent) and viewer.session.race.tick > 0


class TestTheChampionship:
    def test_a_repainted_car_keeps_its_colour_in_the_next_race(self):
        async def repaint(viewer):
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 1})
            await viewer.handle({'type': 'recolor', 'car': 1, 'color': '#123456'})
            await viewer.handle({'type': 'start'})
        viewer = run(repaint)
        key = viewer.socket.sent[0]['cars'][1]['key']
        repainted = [car for car in viewer.socket.sent[-1]['cars'] if car['key'] == key]
        assert viewer.colors == {key: '#123456'}
        assert not repainted or repainted[0]['color'] == '#123456'

    def test_your_driver_and_your_edits_carry_over_to_the_next_race(self):
        async def two_races(viewer):
            await viewer.handle({'type': 'start', 'cars': 4, 'laps': 1})
            viewer.chosen = viewer.session.lineup[3].key
            await viewer.handle({'type': 'yours', 'car': 3})
            await viewer.handle({'type': 'stats', 'car': 0, 'stats': {'grip': 0.05}})
            viewer.rival = viewer.session.lineup[0].key
            await viewer.handle({'type': 'start'})
        viewer = run(two_races)
        keys = [driver.key for driver in viewer.session.lineup]
        assert keys[viewer.session.yours] == viewer.chosen
        if viewer.rival in keys:
            assert viewer.session.stats[keys.index(viewer.rival)]['grip'] == 0.05

    def test_a_reset_driver_goes_back_to_varying_with_the_race_day(self):
        async def reset(viewer):
            await viewer.handle({'type': 'start', 'cars': 4, 'laps': 1})
            await viewer.handle({'type': 'stats', 'car': 1, 'stats': {'grip': 0.05}})
            await viewer.handle({'type': 'reset', 'car': 1})
        viewer = run(reset)
        assert viewer.edited == {}

    def test_a_renamed_driver_keeps_their_new_name_and_their_edits(self):
        async def rename(viewer):
            await viewer.handle({'type': 'start', 'cars': 20, 'laps': 1})
            viewer.renamed = viewer.session.lineup[2].key
            await viewer.handle({'type': 'stats', 'car': 2, 'stats': {'grip': 0.05}})
            await viewer.handle({'type': 'rename', 'car': 2, 'name': 'Da Silva'})
            await viewer.handle({'type': 'start'})
        viewer = run(rename)
        car = [driver.key for driver in viewer.session.lineup].index(viewer.renamed)
        assert viewer.session.intro()['cars'][car]['name'] == 'Da Silva'
        assert viewer.session.stats[car]['grip'] == 0.05


class TestMessagesFromTheTab:
    def test_answers_a_message_it_cannot_use_with_an_error_naming_it(self):
        async def garbled(viewer):
            await viewer.receive('{"type": "start", "cars": "lots"')
        viewer = run(garbled)
        errors = [message for message in viewer.socket.sent if message['type'] == 'error']
        assert len(errors) == 1 and 'Could not use message' in errors[0]['message']


class TestRaceSizes:
    def test_allows_a_formula_one_sized_field_and_race(self):
        async def big(viewer):
            await viewer.handle({'type': 'start', 'cars': 20, 'laps': 60})
        viewer = run(big)
        assert viewer.session.race.count == 20 and viewer.session.race.laps == 60

    def test_keeps_sizes_within_limits(self):
        async def huge(viewer):
            await viewer.handle({'type': 'start', 'cars': 50, 'laps': 500})
        viewer = run(huge)
        assert viewer.session.race.count == 20 and viewer.session.race.laps == 70


class TestLocations:
    def test_races_where_the_tab_asked_and_keeps_it_for_the_next_race(self):
        async def city(viewer):
            await viewer.handle({'type': 'start', 'cars': 2, 'laps': 1, 'location': 'city'})
            await viewer.handle({'type': 'start'})
        viewer = run(city)
        assert viewer.session.location == 'city'

    def test_leaves_it_to_the_circuit_when_asked_to(self):
        from slipstream.scenery import location_for
        async def any_place(viewer):
            await viewer.handle({'type': 'start', 'cars': 2, 'laps': 1, 'location': 'random', 'seed': 9})
        viewer = run(any_place)
        assert viewer.session.location == location_for(9)


class TestPlaybackSpeeds:
    def test_plays_from_one_to_sixteen_times_real_time(self):
        async def fastest(viewer):
            await viewer.handle({'type': 'speed', 'value': 16})
        assert run(fastest).speed == 16

    def test_ignores_a_speed_it_doesnt_offer(self):
        async def silly(viewer):
            await viewer.handle({'type': 'speed', 'value': 1000})
        assert run(silly).speed == 1.0


class TestRadioOrders:
    def test_your_orders_reach_your_driver_during_the_race(self):
        async def race(viewer):
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 2})
            await viewer.handle({'type': 'go'})
            await viewer.handle({'type': 'radio', 'pace': 'push', 'approach': 'attack'})
            await viewer.handle({'type': 'pit', 'stayOut': True})
            await viewer.tick()
        viewer = run(race)
        assert viewer.session.radio_orders == {'pace': 'push', 'approach': 'attack'}
        assert viewer.session.stay_out_lap == 0

    def test_an_order_it_doesnt_know_is_answered_with_an_error(self):
        async def race(viewer):
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 2})
            await viewer.receive('{"type": "radio", "pace": "warp"}')
        viewer = run(race)
        assert any(message['type'] == 'error' and 'warp' in message['message'] for message in viewer.socket.sent)


class TestRaceFormats:
    def test_races_in_the_format_the_tab_asked_for_and_keeps_it(self):
        async def start(viewer):
            await viewer.handle({'type': 'start', 'cars': 4, 'laps': 6, 'format': 'timed', 'minutes': 20})
            await viewer.handle({'type': 'start'})
        viewer = run(start)
        assert viewer.socket.sent[-1]['format'] == {'kind': 'timed', 'minutes': 20}

    def test_refuses_a_format_it_doesnt_have(self):
        async def start(viewer):
            await viewer.receive('{"type": "start", "format": "demolition derby"}')
        viewer = run(start)
        assert any(message['type'] == 'error' and 'demolition' in message['message'] for message in viewer.socket.sent)


class TestAfterTheFlag:
    def test_a_finished_race_stays_for_the_report_until_the_tab_asks_for_the_next(self):
        async def race(viewer):
            await viewer.handle({'type': 'start', 'cars': 2, 'laps': 1})
            await viewer.handle({'type': 'go'})
            viewer.session.race.finish_time[:] = viewer.session.race.time
            for _ in range(400):
                await viewer.tick()
        viewer = run(race)
        assert [message['type'] for message in viewer.socket.sent].count('race') == 1
