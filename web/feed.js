// The race feed: team radio, race control and timing messages, newest on top, the way a timing app shows them.
const $ = id => document.getElementById(id);
const KEEP = 5;

export function createFeed({ onSelect }) {
  const list = $('feed-list');
  let race = null;

  const name = car => race.cars[car].name;
  const chip = car => `<span class="chip" style="background:${race.cars[car].color}"></span>`;
  const time = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

  // What race control or the timing screen says about a moment, or null when the radio says it all. A third
  // element of false means the moment isn't about one car (the weather), so it gets no car's colour.
  function headline(moment) {
    const { car, other } = moment;
    switch (moment.kind) {
      case 'start': return ['control', 'Lights out: the race is on'];
      case 'overtake': return ['control', `${name(car)} passes ${name(other)} for P${moment.place}`];
      case 'contact': return ['control', `Contact: ${name(car)} and ${name(other)}`];
      case 'offTrack': return ['control', `${name(car)} runs off the track`];
      case 'pitStop': return ['pit', `${name(car)} stops: ${moment.compound} tyres, ${moment.standing.toFixed(1)} s`];
      case 'pitExit': return ['pit', `${name(car)} rejoins in P${moment.place}`];
      case 'fastestLap': return ['timing', `Fastest lap: ${name(car)} ${formatLap(moment.lapTime)}`];
      case 'finalLap': return ['control', 'Final lap'];
      case 'puncture': return ['control', `${name(car)} has a puncture`];
      case 'engineFailure': return ['control', `${name(car)} retires: engine failure`];
      case 'mistake': return ['control', `A mistake from ${name(car)}`];
      case 'tow': return ['control', `${name(car)} in the tow of ${name(other)}`];
      case 'rainStarts': return ['control', 'Rain is falling on the circuit', false];
      case 'rainStops': return ['control', `The rain has stopped; the track is ${Math.round(moment.wetness * 100)}% wet`, false];
      case 'win': return ['control', `${name(car)} wins the race`];
      case 'finish': return moment.place <= 3 ? ['control', `${name(car)} finishes P${moment.place}`] : null;
      default: return null;
    }
  }

  function entry(kind, car, html, at) {
    const item = document.createElement('li');
    item.className = kind;
    item.innerHTML = `<span class="when">${time(at)}</span>${car === undefined ? '<span></span>' : chip(car)}<span class="what">${html}</span>`;
    if (car !== undefined) item.addEventListener('click', () => onSelect(car));
    list.prepend(item);
    while (list.children.length > KEEP) list.lastChild.remove();
  }

  return {
    reset(intro) {
      race = intro;
      list.replaceChildren();
    },
    add(moments) {
      if (!race) return;
      for (const moment of moments) {
        const said = headline(moment);
        if (said) entry(said[0], said[2] === false ? undefined : moment.car, said[1], moment.time);
        if (moment.radio) {
          const { car, speaker, text } = moment.radio;
          entry('radio', car, `<b>${speaker === 'driver' ? name(car) : `${name(car)}'s engineer`}:</b> “${text}”`, moment.time);
        }
      }
    },
  };
}

export function formatLap(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(3).padStart(6, '0')}`;
}
