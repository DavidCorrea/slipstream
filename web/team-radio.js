// Team radio to your driver during the race: the pace (push, normal, save, or left to the pit wall), the approach
// (attack, defend, bring it home, or their own call) and the pit call (box this lap, stay out). Racing drivers
// listen to the radio and read their own traits as they drive (see slipstream/session.py's radio orders), so these
// change how your driver races without anything retrained. Each order goes out in the engineer's words and your
// driver answers a moment later, in the feed and, with voices on, out loud.
import { radio } from './race-events.js';

const $ = id => document.getElementById(id);
const REPLY_MS = 1400;
const PACE_LINES = { auto: 'order.autoPace', push: 'order.push', normal: 'order.normal', save: 'order.save' };
const PACE_REPLIES = { auto: 'reply.copy', push: 'reply.push', normal: 'reply.normal', save: 'reply.save' };
const APPROACH_LINES = { auto: 'order.autoApproach', attack: 'order.attack', defend: 'order.defend', bring_it_home: 'order.bring_it_home' };
const APPROACH_REPLIES = { auto: 'reply.copy', attack: 'reply.attack', defend: 'reply.defend', bring_it_home: 'reply.bring_it_home' };

// `raceTime` says how far into the race it is, for the feed's timestamps.
export function createTeamRadio({ send, onSay, raceTime }) {
  const panel = $('team-radio');
  let race = null, yours = 0, boxCalled = false;

  // The order in the engineer's words now, the driver's answer a moment later.
  function say(order, reply) {
    onSay([{ kind: 'radioOrder', priority: 0, car: yours, time: raceTime(), radio: radio(order, yours) }]);
    setTimeout(() => onSay([{ kind: 'radioOrder', priority: 0, car: yours, time: raceTime(), radio: radio(reply, yours) }]), REPLY_MS);
  }

  panel.querySelectorAll('[data-pace]').forEach(button => button.addEventListener('click', () => {
    const pace = button.dataset.pace;
    send({ type: 'radio', pace });
    say(PACE_LINES[pace], PACE_REPLIES[pace]);
  }));
  panel.querySelectorAll('[data-approach]').forEach(button => button.addEventListener('click', () => {
    const approach = button.dataset.approach;
    send({ type: 'radio', approach });
    say(APPROACH_LINES[approach], APPROACH_REPLIES[approach]);
  }));
  $('radio-box').addEventListener('click', () => {
    send({ type: 'pit', box: !boxCalled });
    if (boxCalled) say('order.cancelBox', 'reply.stayOut');
    else say('box', 'reply.box');
  });
  $('radio-stay-out').addEventListener('click', () => {
    send({ type: 'pit', stayOut: true });
    say('order.stayOut', 'reply.stayOut');
  });

  const highlight = (attribute, value) => panel.querySelectorAll(`[data-${attribute}]`)
    .forEach(button => button.classList.toggle('active', button.dataset[attribute] === value));

  return {
    setRace(intro) {
      race = intro;
      yours = intro.yours;
    },

    // Every frame: the buttons show the orders standing and your car's pit call (`car` is yours).
    update(frame, car) {
      if (!race) return;
      yours = car;
      const orders = frame.yourRadio ?? { pace: 'auto', approach: 'auto' };
      highlight('pace', orders.pace);
      highlight('approach', orders.approach);
      boxCalled = frame.yourPit?.boxCalled ?? false;
      const box = $('radio-box');
      box.classList.toggle('called', boxCalled);
      box.textContent = boxCalled ? 'Boxing · cancel' : 'Box this lap';
      $('radio-stay-out').classList.toggle('active', !!frame.yourPit?.stayingOut);
      const out = frame.cars.retired?.[yours] || frame.cars.finishTime[yours] !== null;
      panel.classList.toggle('disabled', !!out);
      const place = frame.order.indexOf(yours) + 1;
      const where = frame.cars.retired?.[yours] ? 'out' : frame.cars.finishTime[yours] !== null ? `finished P${place}` : `P${place}`;
      $('car-drawer-title').textContent = `${race.cars[yours].name} · ${where}`;
    },
  };
}

