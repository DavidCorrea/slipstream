// Turns the stream of frames into the moments a broadcast talks about: overtakes, battles, contact, mistakes,
// pit calls and stops, fastest laps, the last lap and the flag, plus the team radio that goes with them.
// The feed, the TV director and the commentator all read the same moments, so they never disagree.
//
// A moment is { kind, time, lap, priority (1 quiet to 5 drop everything), car, other?, ...details, radio? }.
// `radio` is { car, speaker: 'driver' | 'engineer', text } when someone on a team would say something.

const IN_LANE = 2;
const PASS_REPEAT_SECONDS = 4;       // the same two cars swapping again this soon is one fight, not news
const BATTLE_GAP_SECONDS = 0.6;
const BATTLE_REPEAT_SECONDS = 30;
// Clipping the grass on a wide line happens all the time; a car that stays off this long has made a mistake.
const OFF_TRACK_SECONDS = 1.2;
const OFF_TRACK_REPEAT_SECONDS = 30;
const DAMAGE_WORTH_A_CALL = 0.3;
const TYRES_GONE = 0.65;
const FUEL_CRITICAL = 0.1;
const LIGHTS_OUT = 1.9;              // when the start gantry goes green (see circuit.js)
const TOW = 0.6;                     // how deep in another car's slipstream counts as a tow
const TOW_REPEAT_SECONDS = 40;
const MISTAKE_REPEAT_SECONDS = 20;

export function createRaceEvents() {
  let race = null, previous = null;
  let started = false, finalLapCalled = false, winnerCalled = false;
  const lastPass = new Map(), lastBattle = new Map(), lastOffTrack = new Map(), offSince = new Map(), lastTow = new Map(), lastMistake = new Map();
  const warned = { tyres: new Set(), fuel: new Set(), damage: new Set() };

  function reset(intro) {
    race = intro;
    previous = null;
    started = finalLapCalled = winnerCalled = false;
    lastPass.clear(); lastBattle.clear(); lastOffTrack.clear(); offSince.clear(); lastTow.clear(); lastMistake.clear();
    Object.values(warned).forEach(cars => cars.clear());
  }

  // Seconds between a car and the one ahead of it, from the distance between them at the chaser's speed.
  function gapAhead(frame, car) {
    const place = frame.order.indexOf(car);
    if (place <= 0) return Infinity;
    const ahead = frame.order[place - 1];
    return (frame.cars.progress[ahead] - frame.cars.progress[car]) / Math.max(frame.cars.speed[car], 15);
  }

  function read(frame) {
    if (!race) return [];
    const cars = frame.cars, time = frame.time, moments = [];
    const lap = Math.min(cars.lap[frame.order[0]] + 1, race.laps);
    const add = moment => moments.push({ time, lap, ...moment });
    const racing = car => cars.finishTime[car] === null && cars.pit[car] < IN_LANE;

    // The weather turning: rain arriving or stopping.
    if (previous && started) {
      const wasRaining = previous.weather.rain > 0, raining = frame.weather.rain > 0;
      if (raining && !wasRaining) {
        const leader = frame.order[0];
        add({ kind: 'rainStarts', priority: 4, car: leader, wetness: frame.weather.wetness,
              radio: radio('rain', leader) });
      }
      if (!raining && wasRaining) add({ kind: 'rainStops', priority: 3, car: frame.order[0], wetness: frame.weather.wetness });
    }

    if (!started && time >= LIGHTS_OUT) {
      started = true;
      add({ kind: 'start', priority: 5, car: frame.order[0] });
    }

    if (previous && started) {
      // Overtakes: everyone a car was behind last frame and is ahead of now, on track, not through the pits.
      frame.order.forEach((car, place) => {
        const before = previous.order.indexOf(car);
        if (before <= place || !racing(car)) return;
        for (const other of previous.order.slice(0, before)) {
          if (frame.order.indexOf(other) < place || !racing(other) || previous.cars.pit[other] >= IN_LANE) continue;
          const pair = [car, other].sort().join('-');
          const recent = time - (lastPass.get(pair) ?? -Infinity) < PASS_REPEAT_SECONDS;
          lastPass.set(pair, time);
          if (recent) continue;
          add({ kind: 'overtake', priority: place === 0 ? 5 : place < 3 ? 4 : 3, car, other, place: place + 1 });
        }
      });
    }

    // Battles: a car within striking distance of the one ahead, said once in a while per pair.
    if (started) {
      frame.order.forEach((car, place) => {
        if (place === 0 || !racing(car) || !racing(frame.order[place - 1])) return;
        if (gapAhead(frame, car) > BATTLE_GAP_SECONDS) return;
        const ahead = frame.order[place - 1], pair = `${ahead}-${car}`;
        if (time - (lastBattle.get(pair) ?? -Infinity) < BATTLE_REPEAT_SECONDS) return;
        lastBattle.set(pair, time);
        add({ kind: 'battle', priority: place < 3 ? 3 : 2, car, other: ahead, place: place + 1 });
      });
    }

    // Contact: the two cars nearest to where it happened.
    for (const [x, y, impulse] of frame.events.contacts) {
      if (impulse < 1.5) continue;
      const nearest = cars.x.map((carX, id) => [Math.hypot(carX - x, cars.y[id] - y), id]).sort((a, b) => a[0] - b[0]);
      const [car, other] = [nearest[0][1], nearest[1]?.[1]];
      add({ kind: 'contact', priority: impulse > 4 ? 5 : 3, car, other, impulse });
    }

    for (let car = 0; car < cars.x.length; car++) {
      if (!cars.offTrack[car] || !racing(car)) offSince.delete(car);
      else if (!offSince.has(car)) offSince.set(car, time);
      const offFor = offSince.has(car) ? time - offSince.get(car) : 0;
      if (started && offFor >= OFF_TRACK_SECONDS && cars.speed[car] > 5) {
        if (time - (lastOffTrack.get(car) ?? -Infinity) > OFF_TRACK_REPEAT_SECONDS) {
          lastOffTrack.set(car, time);
          add({ kind: 'offTrack', priority: frame.order.indexOf(car) < 3 ? 3 : 2, car, place: frame.order.indexOf(car) + 1 });
        }
      }
      if (cars.damage[car] > DAMAGE_WORTH_A_CALL && !warned.damage.has(car)) {
        warned.damage.add(car);
        add({ kind: 'damage', priority: 3, car, radio: radio('damage', car) });
      }
      if (cars.tyreWear[car] > TYRES_GONE && !warned.tyres.has(car) && cars.finishTime[car] === null) {
        warned.tyres.add(car);
        add({ kind: 'tyresGone', priority: 2, car, compound: cars.compound[car], radio: radio('tyres', car) });
      }
      if (cars.fuel[car] < FUEL_CRITICAL && !warned.fuel.has(car) && cars.finishTime[car] === null) {
        warned.fuel.add(car);
        add({ kind: 'lowFuel', priority: 2, car, radio: radio('fuel', car) });
      }
      // A stop gets called: the pit wall tells the driver.
      if (previous && cars.pit[car] === 1 && previous.cars.pit[car] === 0) {
        add({ kind: 'pitCall', priority: 2, car, radio: radio('box', car) });
      }
      // Worn tyres and repairs get reset by a stop, so they can be warned about again.
      if (cars.tyreWear[car] < 0.2) warned.tyres.delete(car);
      if (cars.damage[car] < 0.1) warned.damage.delete(car);
      if (cars.fuel[car] > 0.3) warned.fuel.delete(car);
    }

    for (const car of frame.events.punctures ?? []) {
      add({ kind: 'puncture', priority: 4, car, place: frame.order.indexOf(car) + 1, radio: radio('puncture', car) });
    }
    for (const car of frame.events.engineFailures ?? []) {
      add({ kind: 'engineFailure', priority: 5, car, place: frame.order.indexOf(car) + 1, radio: radio('engine', car) });
    }
    for (const car of frame.events.mistakes ?? []) {
      if (time - (lastMistake.get(car) ?? -Infinity) < MISTAKE_REPEAT_SECONDS || !racing(car)) continue;
      lastMistake.set(car, time);
      add({ kind: 'mistake', priority: frame.order.indexOf(car) < 3 ? 3 : 2, car, place: frame.order.indexOf(car) + 1 });
    }
    // A tow: a car pulled along in the slipstream of the one ahead, closing on it.
    if (started) {
      frame.order.forEach((car, place) => {
        if (place === 0 || !racing(car) || (cars.draft?.[car] ?? 0) < TOW) return;
        if (time - (lastTow.get(car) ?? -Infinity) < TOW_REPEAT_SECONDS) return;
        lastTow.set(car, time);
        add({ kind: 'tow', priority: place < 3 ? 3 : 2, car, other: frame.order[place - 1], place: place + 1 });
      });
    }

    for (const stop of frame.events.pitStopped) {
      add({ kind: 'pitStop', priority: frame.order.indexOf(stop.car) < 3 ? 4 : 3, car: stop.car, compound: stop.compound,
            standing: stop.standing, jobs: Object.entries(stop.jobs).filter(([, seconds]) => seconds > 0).map(([job]) => job) });
    }
    for (const car of frame.events.pitExited) {
      add({ kind: 'pitExit', priority: 2, car, place: frame.order.indexOf(car) + 1 });
    }

    for (const timing of frame.events.timing) {
      if (timing.kind === 'lap' && timing.rating === 'overall' && cars.lap[timing.car] > 1) {
        add({ kind: 'fastestLap', priority: 3, car: timing.car, lapTime: timing.time,
              radio: radio('fastest', timing.car) });
      }
    }

    if (!finalLapCalled && cars.lap[frame.order[0]] === race.laps - 1 && race.laps > 1) {
      finalLapCalled = true;
      add({ kind: 'finalLap', priority: 4, car: frame.order[0], other: frame.order[1] });
    }
    for (const car of frame.events.finished) {
      const place = frame.order.indexOf(car) + 1;
      if (place === 1 && !winnerCalled) {
        winnerCalled = true;
        add({ kind: 'win', priority: 5, car, radio: radio('win', car) });
      } else {
        add({ kind: 'finish', priority: place <= 3 ? 3 : 1, car, place });
      }
    }

    previous = frame;
    return moments;
  }

  return { reset, read, gapAhead };
}

// Team radio, by who says it. Fixed lines with no names in them, so the natural voices can have them ready before
// they're needed (see speech.js).
const RADIO = {
  box: { speaker: 'engineer', lines: ['Box, box.', 'Box this lap, box this lap.', 'Pit this lap, we are ready for you.', 'Box, box. Confirm.'] },
  tyres: { speaker: 'driver', lines: ['These tyres are gone!', 'I have no grip at the rear.', 'The tyres are finished, mate.', 'Rears are dropping off a cliff.'] },
  fuel: { speaker: 'engineer', lines: ['Fuel is critical, lift and coast.', 'We are very low on fuel. Save it.', 'Fuel target minus, manage it.'] },
  damage: { speaker: 'driver', lines: ['I have damage, front wing is broken!', 'Something is wrong with the car.', 'Someone hit me! I have damage.'] },
  fastest: { speaker: 'engineer', lines: ['Fastest lap, great job.', 'That is purple, fastest lap.', 'Mega lap, fastest of the race.'] },
  win: { speaker: 'engineer', lines: ['YES! You won it! Get in there!', 'P1! What a drive!', 'Race winner! Brilliant job, brilliant job.'] },
  rain: { speaker: 'driver', lines: ['It is raining here. Rain, rain.', 'Spots of rain at the last corner.', 'It is getting slippery, it is raining.'] },
  puncture: { speaker: 'driver', lines: ['Puncture! I have a puncture!', 'Tyre is going down, rear tyre!', 'I think I have a puncture, the car is all over the place.'] },
  engine: { speaker: 'driver', lines: ['I have lost power! Engine is gone.', 'No power, no power! Stopping the car.', 'Engine failure. I am out.'] },
};

function radio(kind, car) {
  return { car, speaker: RADIO[kind].speaker, text: pick(RADIO[kind].lines) };
}

// Every radio line there is, with who says it.
export function radioLines() {
  return Object.values(RADIO).flatMap(({ speaker, lines }) => lines.map(text => ({ speaker, text })));
}

export function pick(lines) {
  return lines[Math.floor(Math.random() * lines.length)];
}
