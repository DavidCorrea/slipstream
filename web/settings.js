// Settings the commentary remembers in this browser (its mode, the AI commentator, the voices), under one key.
const SETTINGS_KEY = 'slipstream.commentary';

export function loadSetting(name, fallback) {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
    return name in saved ? saved[name] : fallback;
  } catch {
    return fallback;
  }
}

export function saveSetting(name, value) {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...saved, [name]: value }));
  } catch {
    // Private windows and blocked storage just don't remember the setting.
  }
}
