// Action replays. The viewer keeps the last few seconds of frames; when a serious crash happens it lets the race
// run on a moment for the aftermath, then plays the crash again in slow motion, close on the cars involved.
//
// Serious means a hit far harder than the rubbing of a fight (HARD_HIT, about the top 0.5% of contacts), against
// another car or a prop, a car losing a big piece of itself at once (DAMAGE_JUMP of wing, suspension or body damage
// within a second), or a car crashing out. With at
// most one replay every COOLDOWN_SECONDS of racing, that's one or two an 8-lap race, and nine races in ten have
// one (measured over ten races with driver-rivals; a harder bar left four in ten without).
const KEEP_SECONDS = 15;
const MANUAL_SECONDS = 10;
const MANUAL_IMPACT_BEFORE_END = 3;
const BEFORE_SECONDS = 4;        // a crash's replay starts this long before the hit
const AFTERMATH_SECONDS = 2.5;   // and comes once the race has run on this long after it
const HARD_HIT = 8;
const DAMAGE_JUMP = 0.2;
const DAMAGE_WINDOW_SECONDS = 1;
const COOLDOWN_SECONDS = 45;
const DAMAGE_KEYS = ['damage', 'wingDamage', 'suspensionDamage'];
// What blends smoothly between two frames; everything else is taken from the later one.
export const NUMERIC = ['x', 'y', 'speed', 'steer', 'throttle', 'brake'];

export function createReplay() {
  let frames = [], lastCrash = -Infinity, waiting = null;

  // The worst of each car's damage, the moment it's this frame.
  const worstDamage = frame => frame.cars.x.map((_, car) => Math.max(...DAMAGE_KEYS.map(key => frame.cars[key]?.[car] ?? 0)));

  function seriousCrash(frame) {
    // Hits between cars, and against props (a wall, a building, a tree: anything solid), count alike.
    const hits = [...frame.events.contacts, ...(frame.events.propHits ?? []).filter(hit => !hit.broke).map(hit => [hit.x, hit.y, hit.impulse])];
    const hardest = hits.reduce((best, contact) => (contact[2] > best[2] ? contact : best), [0, 0, 0]);
    const earlier = frames.find(old => old.time >= frame.time - DAMAGE_WINDOW_SECONDS);
    const before = earlier ? worstDamage(earlier) : null, now = worstDamage(frame);
    const out = new Set(frame.events.crashed ?? []);
    const broken = now.map((value, car) => out.has(car) || (before !== null && value - before[car] >= DAMAGE_JUMP));
    if (hardest[2] < HARD_HIT && !broken.includes(true)) return null;
    // The cars to watch: those that broke, else the two nearest the hardest hit.
    const involved = broken.flatMap((isBroken, car) => (isBroken ? [car] : []));
    const cars = involved.length ? involved : nearest(frame, hardest[0], hardest[1], 2);
    return { time: frame.time, cars };
  }

  return {
    reset() {
      frames = [];
      lastCrash = -Infinity;
      waiting = null;
    },

    // Keeps the frame, and returns a crash once its aftermath has played out and it's time for its replay.
    record(frame, automatic) {
      frames.push(frame);
      while (frames.length && frames[0].time < frame.time - KEEP_SECONDS) frames.shift();
      if (automatic && !waiting && frame.time - lastCrash >= COOLDOWN_SECONDS) {
        const crash = seriousCrash(frame);
        if (crash) {
          waiting = crash;
          lastCrash = crash.time;
        }
      }
      if (!waiting || frame.time < waiting.time + AFTERMATH_SECONDS) return null;
      const crash = waiting;
      waiting = null;
      return { ...crash, frames: frames.filter(old => old.time >= crash.time - BEFORE_SECONDS) };
    },

    // The last few seconds, for a replay asked for, watching `car`.
    lastMoments(car) {
      const end = frames[frames.length - 1]?.time ?? 0;
      // No crash to centre on: the last few seconds play as the slow part.
      return { time: end - MANUAL_IMPACT_BEFORE_END, cars: [car], frames: frames.filter(old => old.time >= end - MANUAL_SECONDS), asked: true };
    },
  };
}

// The race between two frames, `amount` of the way from `previous` to `current`.
export function blend(previous, current, amount) {
  const cars = { ...current.cars };
  for (const key of NUMERIC) cars[key] = current.cars[key].map((value, id) => previous.cars[key][id] + (value - previous.cars[key][id]) * amount);
  cars.heading = current.cars.heading.map((value, id) => {
    const from = previous.cars.heading[id];
    const turn = Math.atan2(Math.sin(value - from), Math.cos(value - from));
    return from + turn * amount;
  });
  return { ...current, time: previous.time + (current.time - previous.time) * amount, cars };
}

// The race at `time` in recorded frames, and the frames passed since `since` (for their sparks).
export function replayAt(frames, time, since) {
  const next = frames.findIndex(frame => frame.time >= time);
  const index = next === -1 ? frames.length - 1 : Math.max(next, 1);
  const previous = frames[index - 1], current = frames[index];
  const amount = current.time > previous.time ? Math.min(1, Math.max(0, (time - previous.time) / (current.time - previous.time))) : 1;
  return { frame: blend(previous, current, amount), passed: frames.filter(frame => frame.time > since && frame.time <= time) };
}

function nearest(frame, x, y, count) {
  return frame.cars.x.map((carX, id) => [Math.hypot(carX - x, frame.cars.y[id] - y), id])
    .sort((first, second) => first[0] - second[0]).slice(0, count).map(([, id]) => id);
}
