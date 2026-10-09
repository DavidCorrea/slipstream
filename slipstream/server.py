"""The viewer's server: serves the page, lists the brains, and plays races to each open tab over a websocket.

    .venv/bin/python -m slipstream.server            # then open http://127.0.0.1:8765

Each tab gets its own race. The race runs in real time (or faster) and a frame goes out 20 times a second.
The tab sends back what it wants: a new race with a given brain, a playback speed, pause.
"""
import argparse
import asyncio
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from .session import list_brains, list_pitwalls
from .viewer import FRAMES_PER_SECOND, Viewer

WEB = Path(__file__).resolve().parent.parent / 'web'


async def index(request):
    return web.FileResponse(WEB / 'index.html')


async def brains(request):
    return web.json_response({'drivers': list_brains(), 'pitwalls': list_pitwalls()})


async def race_socket(request):
    socket = web.WebSocketResponse(heartbeat=20)
    await socket.prepare(request)
    viewer = Viewer(socket)

    async def play():
        while not socket.closed:
            started = asyncio.get_running_loop().time()
            await viewer.tick()
            await asyncio.sleep(max(0.0, 1 / FRAMES_PER_SECOND - (asyncio.get_running_loop().time() - started)))

    player = asyncio.create_task(play())
    try:
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            await viewer.receive(message.data)
    finally:
        player.cancel()
    return socket


@web.middleware
async def always_fresh(request, handler):
    """Makes browsers check for a newer page and scripts on every load. Without it they reuse cached copies by
    guesswork, and a refresh after the viewer changes can keep running the old scripts."""
    response = await handler(request)
    if request.path == '/' or request.path.startswith('/web/'):
        response.headers['Cache-Control'] = 'no-cache'
    return response


def create_app():
    app = web.Application(middlewares=[always_fresh])
    app.add_routes([
        web.get('/', index),
        web.get('/api/brains', brains),
        web.get('/ws', race_socket),
        web.static('/web', WEB),
    ])
    return app


def main():
    parser = argparse.ArgumentParser(description='Slipstream viewer')
    parser.add_argument('--port', type=int, default=8765)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f'Slipstream viewer at http://127.0.0.1:{arguments.port}')
    web.run_app(create_app(), host='127.0.0.1', port=arguments.port, print=None)


if __name__ == '__main__':
    main()
