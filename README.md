# Slipstream

A racing game where every driver is a neural network. On the grid you pick your car and set up any driver: their personality (aggression, risk, overtaking, how much they spare their tyres and fuel, consistency, stamina) and their car (top speed, grip, acceleration, braking, handling). One network drives every car and reads those stats as inputs, so each driver races their own way. During the race you're your car's pit wall.

**Play it online: https://davidcorrea.github.io/slipstream/** (Chrome, Edge, Firefox or Safari; the first visit downloads about 17 MB).

## Status

The race simulation, training, the viewer, your racer's stats, pit stops, weather and a long list of realism (below) are in place. The viewer has a TV director, lap and sector timing with a fastest-lap ghost, a race feed with team radio, and a commentator. A new driver that drives by feel, with a memory, is training from scratch (run `driver`, see *The driver*); the pit wall retrains on top of it once it can drive.

## Setup

Requires Python 3.12 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Watching

```sh
.venv/bin/python -m slipstream.server                # then open http://127.0.0.1:8765
```

Run locally, the page races on the server and lists every snapshot of every run as training saves it.

## The published site

The site at https://davidcorrea.github.io/slipstream/ is this repository served as it is by GitHub Pages, with no server behind it. When the page finds no server (`api/brains`), it runs the simulation itself: Pyodide (Python and numpy compiled to WebAssembly) runs the same race code in a web worker (`web/engine-worker.js`, `slipstream/browser.py`), and three.js draws it as usual.

- **Networks** race with numpy alone (`slipstream/numpy_network.py`): a trained network is exported to its actor's weights (0.8 MB for a driver), which give the same actions as the original to within 1e-6.
- **Which brains are published** is up to you: training runs (`runs/`) stay private, and only the snapshots you publish go into `brains/`:

  ```sh
  .venv/bin/python -m slipstream.publish driver-rivals/step-000036016128 pitwall-feel/step-000000251904
  .venv/bin/python -m slipstream.publish     # after changing the simulation: refreshes web/engine-files.json
  ```

  Then commit and push; Pages updates within a minute or two. A test fails if `web/engine-files.json` misses a module the browser needs.
- **The race code stays free of the training libraries** (PyTorch, Stable-Baselines3, Gymnasium, aiohttp), which the browser doesn't have; a test checks it. Training environments live in `env.py`, `driver_env.py` and `pitwall_env.py`.
- **To try the published version locally**, serve the folder as plain files: `python3 -m http.server` and open http://127.0.0.1:8000.

The viewer plays races live in the browser, in isometric 3D. Every race opens **on the grid**: a menu showing the field in grid order, where you pick the car you drive, set up any driver (personality, consistency, stamina) and their car (top speed, acceleration, braking, grip, handling), and choose the race. The cars wait until you press **Start race**; drivers and cars are fixed from then on. Training doesn't need to be stopped: every snapshot it saves shows up in the **Brain** list within half a minute, so you can watch the same circuit driven by the network at any stage of its training.

| Control | What it does |
|---|---|
| Brain | Who drives every car: the scripted driver, or any snapshot of any driver run |
| Pit wall | Who calls the stops: the driver network's own calls, the scripted strategist, or any snapshot of a pit-wall run |
| Weather | Up to the circuit (its seed decides), or a forecast: dry, rain on the way, a passing shower, drying out, wet throughout |
| Location | Up to the circuit, or countryside, desert, alpine, coast, city, or the city at night (scenery only: the race is the same) |
| Cars, Laps | Field size (2 to 20, a Formula 1 grid) and race length (1 to 70 laps); **Apply** puts them on the grid on the same circuit. A big field in the published site's browser simulation runs at about 2× at most at the start of a race (the local server keeps up at 8×) |
| Another circuit, New race | A new generated circuit, back on the grid (changing brain, pit wall or location keeps the circuit) |
| 0.5× to 8×, Pause (Space) | Playback speed. 1× is twice real time (0.5× is real time, 8× is 16×): with less grip than a Formula 1 car and shorter circuits, a race at real time looked slow, and at twice it looks like one on television. Race times and lap times are still real |
| Follow, TV, Circuit (F, T, O) | Follow the selected car, watch it like a broadcast (see below), or see the whole circuit |
| Ghost (G) | Off by default: a see-through car driving the fastest lap so far, in step with the selected car's lap |
| Replays (R, Esc) | After a serious crash (a hit far harder than a fight's rubbing, or a car losing a big piece at once) the race holds while it's replayed in slow motion, letterboxed, the camera close on the cars and the commentator introducing it, about once or twice a race; R replays the last 10 s of the selected car, Esc skips back to the race, and Settings turns the automatic ones off (`web/replay.js`) |
| Commentary | Off, captions, or captions and the browser's voice |
| Voices | In Settings: **Natural** or **Browser** voices, and one of that kind for the commentator, your driver and your race engineer, each with ▶ to hear it first (remembered in this browser). Natural voices are Kokoro (an open 82M-parameter speech model) on your graphics card: offered once on the grid where the browser has WebGPU and a GPU, a 325 MB download kept by the browser, and speaking a line about half a second after it's written. Team radio comes over a radio (telephone band, a little overdrive, a click), and its lines are generated in the background so they play at once |
| Load AI commentator | Downloads a small language model once (about 0.9 GB) and runs it on your GPU, for commentary it writes itself |
| Click a car or a name, 1–8 | Select a car to follow and see its telemetry |
| Q / E, scroll | Turn the view a quarter turn, zoom |
| On the grid | Click a driver to set them up or rename them, **Drive** to make their car yours, **Back to their usual self** to undo your changes; Start race |
| ★ Pit wall (your car) | During the race: leave the stops to the AI or make your own plan and box |

On screen:
- **Every circuit has a corner with character** (`slipstream/track.py`): a sharp turn (a straight run into a corner that turns 90 degrees or more within 45 m, at the 22 m minimum radius) or a chicane (a quick left-right laid into a straight, two opposite bends within 45 m), and often both. The main straight keeps 160 m clear for the grid. Plain generated circuits only had one about half the time; now a layout without one is drawn again. The driver copes on them without retraining (1.05× the scripted driver on 24 new circuits, as before).
- **Everything around the track is real** (`slipstream/scenery.py`, `slipstream/obstacles.py`): the sim places every prop (trees, rocks, cacti, bales, buildings, lamps, palms, sponsor and braking boards, marshal posts, camera towers, the sponsor bridge, armco, street walls, tyre walls, the gantry, grandstands, the pit wall and garages) and the viewer draws exactly that list. A car that reaches one hits it: solid ones bounce it off, taking speed and doing damage by how hard it hit; tyre walls give more and harm less; boards, signs, lamps, cacti and bales break, slow the car a little and stay broken. A hit harder than about 100 km/h head on puts a car out, and so does getting less than 25 m further in 20 s (a car nosed into something backs off and turns round first, as a driver would in reverse). Only cars near the edge of the track are checked, so it costs little. Training races have no props, so the drivers train and benchmark as before. The pit wall stands only beside the straight: a big field's long pit lane runs on into the first corner, where cars running wide piled into it.
- **Locations** (`web/scenery.js`): countryside fields and hay bales; desert sand, boulders, cacti and mesas; alpine pines, snow-capped mountains and a lake; a coast with the sea, a beach, palms and a marina; a city street circuit between walls, low blocks by the track and towers behind; and the city at night, with floodlit pools of light, lit windows, and cars running headlights and glowing tail lights. Each has its own sky and light.
- **Trackside, everywhere:** sponsor boards and a sponsor bridge over a straight, 3-2-1 braking boards before the big stops, marshal posts with flags flapping in the race's wind, TV camera towers, and armco on the outside of the fast stretches.
- **Cars:** a second livery colour (stripes and helmet), race numbers on the nose and sidepods, mirrors, an airbox, a floor and diffuser, and a rear light that flashes in the wet.
- **Circuit:** tarmac with racing-line rubber, curbs on corners, gravel traps and tyre walls on the outside of tight bends, a start gantry whose lights count down, grid boxes, grandstands with a crowd that never sits still, and trees.
- **Cars:** wheels that spin and steer, bodies that lean in corners and dip under braking, brake lights and a glowing exhaust.
- **Effects:** tyre smoke, dust off the grass, sparks on a real knock, and confetti for the winner. The race leaves its history on the circuit: skid marks where cars slid, lock-up streaks where they braked hard, ruts in the grass and gravel, and debris from crashes, which stays where it fell.
- **Damage:** a damaged car loses a front-wing endplate, then the other, then the flap, and its rear wing bends; each lost part is left on the track. A repair puts everything back.
- **Look:** Classic, or Pixel (P): the same 3D scene drawn at a quarter resolution with a short palette, a light dither and dark outlines, like a 16-bit racer.
- **Weather:** rain falling around the camera, a grey sky and dimmer sun, tarmac that darkens and shines as it gets wet, and spray off every car; tyres show green for intermediates and blue for full wets.
- **HUD:** a standings tower with gaps, position changes, the weather and how wet the track is, and who holds the fastest lap; live telemetry for the selected car: speed, throttle, brake, tyres, fuel, steering, the running lap time, last and best laps, and sector times coloured like a timing screen (purple the best anyone has done, green a personal best, yellow neither).
- **Race feed:** race control (overtakes, contact, mistakes, stops, fastest laps, rain), and team radio between drivers and their engineers. Click an entry to follow that car.

**TV mode** (`web/director.js`) is a director cutting between cameras like a race broadcast. It follows the story, cutting to an overtake, contact, a stop or rain arriving, and between stories it watches the closest fight on track. Shots: a trackside camera ahead of the cars zooming in as they come past, the helicopter circling a fight, a chase camera, the onboard T-cam, the pit lane, the start and the finish line, with a TV graphic naming the shot.

**The commentator** (`web/broadcaster.js`) says one line at a time about the biggest moment going, and fills quiet spells with the order and the gaps. Lines come from templates until you load the AI commentator: Llama 3.2 1B through WebLLM, running in a web worker on your GPU, nothing sent anywhere. It's given only the facts of the moment, and a line that names a driver or number not in them, or guesses a driver's gender, is thrown away for the template one. With voice on, your own car's team radio is spoken too, in a different voice. Every race moment comes from one place (`web/race-events.js`), so the feed, the director and the commentator always agree.

The server (`slipstream/server.py`) runs each tab's race in real time and sends 20 frames a second; the page (`web/`, three.js) draws smoothly between frames.

## Your racer

The field is a cast of twenty drivers (`slipstream/cast.py`), the same people every race, each with their own character and their own car: Vega fast, fearless and hard on tyres; Okafor the tyre whisperer; Lindqvist metronomic to the flag; Moreau always hunting the move; Sato smooth, in the car with the most grip; Ferreira brilliant until fatigue sets in; Kowalski the rookie, quick in a straight line and erratic; Haddad the cautious veteran who never tires; and twelve more, from Novak the late-braker and Mbeki the slipstream artist to Brennan, a podium or the gravel. Their traits don't change, but their cars vary a little from race to race (set-up, engine, tyre batch), and the grid is drawn afresh every race. When the field is smaller than twenty, some sit the race out. All of it follows the circuit's seed, so the same circuit brings back the same day. Each driver keeps their number wherever they start.

One car is yours, marked with a gold chevron on track and a star in the standings. On the grid you can set up any driver, not just yours. Any driver can be renamed (letters, spaces and . ' -, up to 16 characters, and not anyone else's name); the commentator uses the new name, matching whole words only, and won't let names from its prompt's examples slip into a line. Like a championship, your driver, everyone you've edited and every new name carry over from race to race (by each driver's fixed key, so a renamed driver keeps their edits): an edited driver keeps exactly what you set, and the others turn up as they usually are, give or take the race day.

**Personality** (`slipstream/personality.py`) is four traits from 0 to 1:

| Trait | Low | High | What it changes in training |
|---|---|---|---|
| Aggression | Avoids contact | Leans on other cars | Contact costs 1× at 0 down to 0.2× at 1 |
| Risk | Keeps a margin | Drives on the limit | Cornering beyond 55% (at 0) to 100% (at 1) of the grip limit costs reward; running wide costs 1.5× down to 0.5× |
| Overtaking | Holds station | Hunts places | Each place gained pays 0.05 to 0.35, and each place lost costs the same |
| Conservation | Pushes | Saves tyres and fuel | Tyre wear and fuel burned cost reward, scaled by the trait |

**Car** sliders set top speed, acceleration, braking and grip from 25% below to 25% above the defaults: the same range training cars are drawn from.

How one network can be every driver: each car in every training race gets random traits and a random car. Both are inputs to the network, and the traits reweigh that car's rewards, so the network learns what each kind of driver should do. Moving a slider changes the inputs, so the car changes how it drives on the spot, without retraining. Personality only starts to count once the network can drive (half the training cars finishing their races), then fades in over 4 million decisions: costs that only a moving car pays would otherwise make sitting still look safest to a network that can't drive yet.

Networks trained before personality existed (the `basic` run) still drive in the viewer; they read only the inputs they know, so only the car sliders affect them.

## Pit stops

A car's setup and condition all change how it drives, and a stop can change every one of them (`slipstream/car.py`, `slipstream/pit.py`):

| Part | What it does | At a stop |
|---|---|---|
| Tyres | Softs grip 6% more and wear 1.8× faster; hards grip 5% less and wear 0.55× as fast. All lose grip as they wear, slowly at first and then sharply: mediums go off after four or five laps | A new set of any compound, 2.4 s |
| Fuel | 60 kg is about five laps; a full tank is heavier and slower; an empty one stops the car | Topped up to the planned amount at 12 kg/s |
| Wing | More wing: up to 8% more cornering grip and 7% less top speed | Reset to a new angle, 1.5 s |
| Engine mode | Push gives 8% more power and a little top speed for 35% more fuel; lean saves 25% for less of both | Changed, 1 s |
| Damage | Every real impact (rubbing doesn't count) costs grip, top speed and acceleration | Repaired, 10 s for a wrecked car |
| Brakes | Pads wear under braking and stop the car less hard | New pads, 4 s |

Tyres, fuel, wing and engine are done side by side, so the longest counts; repairs and brakes come on top, after 2 s of jacks and wheel guns. With the drive down the lane at 80 km/h, a stop costs about 9 seconds. With the scripted driver and its strategist: nobody stops in a 5-lap race, a car that doesn't stop runs dry in an 8-lap one, and 10 laps takes one or two stops depending on the circuit.

**Who decides:** the network has outputs for calling a stop and for the plan (compound, fuel, wing, engine, repair, brakes). The call is made once a lap, like a pit wall's "box this lap": the first decision after a point 150 m before the pit entry commits the car, in or not, and the car is told when that decision is due. A car that's been called in drives itself through the lane, so nobody has to learn to steer a pit lane. (An earlier version let any decision call a stop; only the one that happened to fall at the entry mattered, and a run learned nothing about stopping in 7 million decisions.) Races in training run 3 to 10 laps, so the network learns when stopping pays and when it doesn't. A stop's pay-off comes over laps, far beyond the ten seconds a driver looks ahead, so each car carries a value for its condition (`slipstream/strategy.py`): the grip its tyres will lose over the rest of the race (counting the wear still to come at its compound's usual rate), damage and worn brakes for every lap left, and fuel short of the flag. With five laps to go, new tyres for 60%-worn ones are worth about twice what the stop costs; with two to go they aren't worth it. Training rewards every change in that value as it happens, so a stop that fixes the car earns its worth straight away. Like personality, condition only counts once the network can drive: crashing and wasting fuel are costs only a moving car pays, and before then they made parking on the grid look safest.

**Your racer's pit wall:** leave strategy to the AI, or switch to **My plan**: pick the compound, fuel, wing, engine mode, repairs and brakes, and press **Box this lap**. The car comes in the next time it reaches the pit entry, or a lap later if it has already passed the decision point.

**The lane works like a real one** (`slipstream/pit.py`): a fast lane nearest the track for driving through and a working lane along the garages, divided by a dashed line. A car moves over to the working lane just before its box and back out after its stop, so cars being worked on never block the ones driving past. Nobody overtakes under the speed limit: cars queue nose to tail, braking hard if they come in on top of the car ahead, and never get closer than a car length. A car leaving its box waits for a gap in the fast lane before pulling out.

**In the viewer:** a pit lane with a wall and catch fence, a garage per team, and a crew of eight per car. When a car stops, they run out, jack it up, swap all four wheels (the new ones in the new compound's colour), refuel through a hose, lift the rear wing off and back at its new angle, open the engine cover, swap the nose for repairs, and the lollipop goes green when they're done. Every part shows on the car as it races: tyre stripes in the compound's colour that grey with wear, wing planes steeper with more wing, a bigger and hotter exhaust flame the harder the engine mode, a drooping front wing, darkened paint and smoke from damage, glowing brake discs, and a car that sits lower on a full tank.

Networks trained before pit stops existed still race: the scripted strategist makes their calls.

### The pit wall

The driver network learned to drive well but never learned when to pit: its pit call is one decision in about four hundred, and the rest drowned it out. So pit decisions now have their own network, the **pit wall** (`slipstream/pitwall.py`), like a real team's strategist sitting apart from the driver.

- **What it decides:** once a lap, at the decision point, whether to come in and the plan for the stop.
- **What it sees:** tyre wear and compound, fuel and the fuel needed to finish, damage, brakes, wing and engine, laps left and race length, stops so far, the driver's personality and the car's specs, and how wet the track is and how hard it's raining.
- **How it learns:** the driver network drives (fixed), and every training step is one pit decision followed by a lap. Cars race alone, as ghosts on a shared circuit, many at once; each one waits at its decision point until the rest of its group has reached theirs, so a batch of decisions is made together. The reward is minus the race time, so it learns whatever finishes soonest; a car that runs dry or out of time pays for the race it didn't finish.
- **Benchmark:** long races (8 and 10 laps) on fixed circuits, dry and in changing weather, the driver with the pit wall against the driver with the scripted strategist and against never stopping.

```sh
.venv/bin/python -m slipstream.train_pitwall                                        # run "pitwall": cars racing alone
.venv/bin/python -m slipstream.train_pitwall --traffic --run pitwall-traffic --from pitwall   # then in traffic
```

**Solo** races are quick to learn from: every car races alone, so a stop's worth is plain to see. **Traffic** races put six cars on the track together: the pit wall also sees its place and the gaps to the cars ahead and behind, and on top of race time it's rewarded for its finishing place (up to half a minute's worth for a win), so jumping a rival through the stops (an undercut) can pay. In traffic nobody waits at the decision point: each car asks the pit wall the moment it gets there, every decision goes into the training batch in order, and the PPO update itself is the library's. A traffic run starts as a copy of a solo one. Its benchmark adds `traffic_place`: the average place of three cars the pit wall calls for, racing three the scripted strategist calls for (0 means it took the top places).

## Weather

Each race gets a forecast from its seed (`slipstream/weather.py`): half are dry, the rest bring rain on the way, a passing shower, a track drying from a wet start, or rain throughout. Rain soaks the track in about a minute; once it stops, a soaked track takes about seven minutes to dry. How wet it is decides which tyres are fastest:

| Tyres | Best when | Dry | Damp | Soaked |
|---|---|---|---|---|
| Slicks (soft, medium, hard) | under about a third wet | 1.06 / 1.00 / 0.95 | 0.75 | 0.53 |
| Intermediates | a third to 70% wet | 0.90 | 0.88 | 0.74 |
| Full wets | over 70% wet | 0.84 | 0.82 | 0.83 |

(Grip relative to a dry medium.) Wet-weather tyres overheat and wear fast once the track is drier than they need, and water cools every tyre. Cars start on whatever suits the grid, laps are slower in the wet so a tank lasts fewer of them, and on wet grass there's almost no grip at all. Nothing else is a rule: when to change tyres is for the strategists. The scripted strategist changes when the track is clearly past a crossover; the networks see how wet it is, how hard it's raining and which tyres they're on, and the pit wall's tyre choice gained intermediates and full wets.

## Realism

What a race does to a car, all in the physics (`slipstream/car.py`, `aero.py`, `surface.py`, `race.py`, `human.py`) and all on screen:

| | In the race | In the viewer |
|---|---|---|
| Slipstream and dirty air | Close behind a car: less drag on the straights (up to 7% more top speed), but less grip in corners and less engine cooling | Air streaming past a car in a tow; "Tow" and "Dirty air" in its telemetry; tows in the feed |
| Tyre temperature | Each compound grips best in its window; cold out of the pits (blankets only get them part way), overheated when slid or worked hard, cooled by rain and a cold day, warmed by a hot one; overheating also wears them | Tyre temperature in °C, blue when cold, orange or red when hot |
| Brakes | Heat with braking, cool with airflow, and fade when hot | Discs glow with their real temperature; brake °C in telemetry |
| Engine | Hotter pushing, at full throttle and in dirty air; a hot engine can fail, and its car is out | Engine °C; a failure bursts into flame, smokes, and the car is "OUT" |
| Handling | Every car has a balance: understeer runs wide, oversteer rotates past its grip and slides | "understeers" or "oversteers" in its setup line |
| Local damage | A hit to the nose breaks the front wing (less front grip, more understeer); a hit to the side bends the suspension (the car pulls to one side) | Front wing halves break off, a wheel splays out |
| Debris and punctures | A hard hit leaves debris; running over it can cut a tyre: slow, and it has to come in | Debris lies where it fell until a car runs it over; a flat tyre sags and its rim throws sparks |
| The surface | Cars lay rubber on the racing line (more grip as the race goes on); worn rubber collects off-line as marbles (less grip); after rain a dry line forms where cars drive | The racing line darkens, marbles gather at the edges, the dry line shows lighter in the wet |
| The day | Each race has its own air temperature and wind: a headwind costs top speed on one straight, a tailwind adds it on another | Temperature and wind in the weather line; rain leans in the wind |
| The driver as a person | Inputs arrive 0.15 s late; hands and feet get less steady under pressure (a car close ahead or behind) and with fatigue; now and then a real mistake. Two new traits shape it: consistency and stamina | Mistakes in the feed; consistency and stamina sliders |
| What a driver can see | Mirrors with blind spots; spray off a wet track and smoke from a damaged car hide the cars further up the road; heavy rain shortens the view | |

## The driver

The newest drivers drive by feel (`slipstream/senses.py`, `driver_env.py`). A real driver isn't told the car's top speed, how much grip the tyres have left, or how wet the track is: they feel how the car slides, turns, brakes and accelerates, see the road and the cars they can see, and get told a few things on the radio and dash (tyres fitted, fuel, engine mode, laps to go, box this lap). That's all these drivers get. They have a memory (an LSTM, sb3-contrib's RecurrentPPO), and every training race hands them a different car (specs ±35%, any handling balance, a random setup) in whatever weather the day brings, so to drive well they have to feel out each car and the conditions as they go. They only steer and work the pedal: the team (the scripted strategist in training, the pit wall in races) calls the stops, and the pit wall keeps the full telemetry a real team has.

Your racer splits the same way. Before the race you pick the driver (personality, consistency, stamina) and the car (specs, handling); during the race you're the pit wall.

```sh
.venv/bin/python -m slipstream.train --feel --run driver     # trains (or resumes) the driver by feel
```

The driver by feel (run `driver-copy`) learned at a rate of 3e-4 until about 32M decisions, then went flat: it lapped about as fast as the scripted driver on its own but kept losing places in traffic. Its 32M snapshot was the best (re-scored on the larger benchmark below: 1.03× the scripted driver dry and wet, against 0.95× and 1.01× at 40M), so the run `driver-traffic` carries on from there (`--from driver-copy/step-000032022528`; `--from` works for drivers by feel too, keeping the step count and the personality ramp). That run raced only copies of itself and learned to hold back (by 38M it crashed half as much but was no faster than the scripted driver and further behind it in traffic), so `driver-rivals` starts again from the 32M snapshot with three of every eight cars driven by the scripted driver (`--rivals`, `FEEL_RIVALS`): the network drives and learns from its own five, and races all eight. Every snapshot also gets a second opinion on 24 circuits no snapshot was ever chosen on (`python -m slipstream.check --run driver-rivals`, written to the run's `check.jsonl`): on the twelve benchmark circuits, choosing the best of dozens of snapshots favours lucky ones, and only the fresh circuits showed the 32M snapshot (1.05× the scripted driver, place 0.53) being matched by `driver-rivals` at 36M (1.05×, 0.52, with a fifth less damage and a third of the time off track) while `driver-traffic` really had declined (1.02×, 0.65). Both train at 1e-4 (smaller steps, so it settles instead of being knocked off what it learned), in races of 8 cars instead of 6, with places worth twice as much (`TRAFFIC_REWARDS` in `slipstream/train.py`: 2 for the win at the flag, and double what its personality pays per place gained or lost on the way). A resumed run always takes the current settings.

## Training

```sh
.venv/bin/python -m slipstream.train                 # trains run "main", resuming it if it exists
.venv/bin/tensorboard --logdir runs                  # curves at http://localhost:6006
.venv/bin/python -m slipstream.train --run next --from main   # a new run that starts as a copy of "main"
```

A new run can start as a copy of an older one (`--from`): every weight the two networks share is copied, new inputs start with zero weights so they change nothing, and new outputs start at sensible defaults (no pit call; a stop, if called, for mediums and a full tank). It drives exactly like the old one on its first decision and only has to learn what's new. The current `main` run started this way from `personality`.

Ctrl+C saves the latest network before stopping. A run lives in `runs/<name>/`:

- `model.zip`: the latest network, which training resumes from.
- `checkpoints/step-*.zip`: a snapshot every million decisions, for watching how it learned.
- `history.jsonl`: one line per snapshot with the benchmark scores and race statistics.
- `tensorboard/`: training curves.

How it works:

- **One network drives every car.** Stable-Baselines3's PPO trains it on 16 races of 6 cars at once, each on a new circuit, so every step is 96 decisions. A driver decides 10 times a second: how much to steer, and one pedal that is throttle forward and brake back.
- **What a driver sees** (`slipstream/observe.py`), all in its own frame:
  - its speed, sideways slip, how far it is from the centre line and which way it points relative to the road;
  - tyre wear and fuel;
  - the track's curvature at 11 distances up to 160 m ahead, and where the centre line goes;
  - the 4 nearest cars within 60 m: where they are and how fast they're closing;
  - its race position, how much race is left, and the gaps to the cars ahead and behind;
  - its car's specs.
- **Rewards** (`REWARDS` in `slipstream/env.py`): 0.01 per metre of progress (about 12 a lap), a small cost for every tenth of a second still racing, so standing still always loses, a small cost for every tenth of a second off the tarmac and for every contact, and on taking the flag 1 for finishing plus up to 1 more for the place. Progress made off the tarmac earns nothing (track limits). The first run had a larger off-track cost that outweighed the progress a crawling car makes, and it learned to sit on the grid with the brake on. The second run had no track limits, and it learned to cut corners across the grass.
- **Benchmarks** (`slipstream/benchmark.py`): every snapshot is tested on twelve fixed circuits, six of them in the wet (six and three until 40M decisions of `driver-copy`; with that few, a snapshot's score swung about 3% on luck alone, so scores from before and after aren't directly comparable).
  - **Time trial:** `pace` is the scripted driver's time divided by the network's, so above 1 means faster than the script.
  - **Mixed race:** three network cars against three scripted ones. `place` is the network cars' average finishing position, 0 for first and 1 for last.
  - **Long race:** 8 laps, where a tank won't last and tyres go off: `long_pace` compares the network with the scripted driver and its strategist, and `stops` counts the network's stops.
  - **Wet time trial:** the same on a soaked track, both on full wets: `wet_pace` and `wet_off_track`.
  - **Trait effects:** solo laps at the two ends of a trait. `risk_pace` is above 1 when a risky driver laps faster than a cautious one, and `conservation_saving` is above 1 when a saver uses less tyre and fuel than a pusher. Both sit at 1 for a network that ignores personality.

## The simulation

- **Tracks** (`slipstream/track.py`): every race gets a new closed circuit from its seed. A smooth spline runs through randomly placed points, some pulled in toward the middle for hairpins and esses, and the longest gap is always made into a straight. Corners tighter than a 22 m radius are eased open rather than thrown away, and layouts that are too short, too long or pass too close to themselves are drawn again. The start line sits 60% of the way down the straightest stretch, so the grid is on a straight.
- **Cars** (`slipstream/car.py`): all cars update together, in ticks of 0.05 s. The tyres have one grip budget shared between the pedals and the steering (the friction circle), so braking hard leaves less grip for turning, and asking for too much turn makes the car run wide. Grip falls as tyres wear and on the grass, and a full tank makes the car slower to speed up and to stop.
- **Races** (`slipstream/race.py`): cars start on a staggered two-column grid. For contact each car is its footprint as drawn, a rectangle from front wing to rear wing and as wide as its wheels, turned with the car; contact pushes cars apart and trades speed between them. Progress is measured along the centre line, so a car finishes when it has covered the set number of laps, and reversing over the line never counts as a lap.
- **Scripted driver** (`slipstream/drivers.py`): steers toward a point ahead on the centre line, and plans its speed by working backward from the corners ahead. At each point it only brakes with the grip that point's cornering leaves over. It has no racecraft. It proves the physics can be driven, and it will be the fixed opponent trained drivers are measured against.
