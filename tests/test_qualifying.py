import asyncio

from slipstream.session import SCRIPTED, RaceSession
from slipstream.viewer import Viewer


def grid(session):
    return [car['key'] for car in session.intro()['cars']]


class TestQualifying:
    def test_every_driver_sets_a_lap_and_the_fastest_comes_first(self):
        session = RaceSession(SCRIPTED, seed=5, cars=4, laps=3)
        results = session.qualify()
        assert sorted(key for key, _ in results) == sorted(grid(session))
        times = [time for _, time in results]
        assert all(time is not None for time in times) and times == sorted(times)

    def test_a_faster_car_qualifies_ahead(self):
        session = RaceSession(SCRIPTED, seed=5, cars=3, laps=3)
        session.set_stats({'top_speed': 1.0, 'acceleration': 1.0, 'grip': 1.0}, 2)
        session.set_stats({'top_speed': 0.0, 'acceleration': 0.0, 'grip': 0.0}, 0)
        order = [key for key, _ in session.qualify()]
        assert order.index(grid(session)[2]) < order.index(grid(session)[0])

    def test_the_race_lines_up_in_qualifying_order_with_everyone_as_they_were(self):
        session = RaceSession(SCRIPTED, seed=5, cars=4, laps=3, names={'vega': 'Da Silva'}, colors={'vega': '#00ff88'})
        session.set_yours(grid(session).index('vega'))
        session.set_stats({'risk': 0.9}, session.yours)
        before = {key: stats for key, stats in zip(grid(session), session.stats)}
        order = [key for key, _ in session.qualify()]
        regridded = session.regrid(order)
        assert grid(regridded) == order
        assert regridded.lineup[regridded.yours].key == 'vega'
        assert {key: stats for key, stats in zip(grid(regridded), regridded.stats)} == before
        vega = regridded.intro()['cars'][regridded.yours]
        assert (vega['name'], vega['color'], vega['edited']) == ('Da Silva', '#00ff88', True)


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


class TestQualifyingFromTheGrid:
    def test_the_tab_gets_the_race_again_in_qualifying_order_with_the_times(self):
        viewer = Viewer(FakeSocket())

        async def play():
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 2})
            await viewer.handle({'type': 'qualify'})
        asyncio.run(play())
        intro = viewer.socket.sent[-1]
        assert intro['type'] == 'race' and [result['key'] for result in intro['qualifying']] == [car['key'] for car in intro['cars']]
        assert viewer.on_grid

    def test_a_reverse_grid_starts_the_fastest_at_the_back(self):
        viewer = Viewer(FakeSocket())

        async def play():
            await viewer.handle({'type': 'start', 'cars': 3, 'laps': 2, 'format': 'reverse'})
            await viewer.handle({'type': 'qualify'})
        asyncio.run(play())
        intro = viewer.socket.sent[-1]
        assert [result['key'] for result in intro['qualifying']] == [car['key'] for car in intro['cars']][::-1]
