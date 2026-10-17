// Pop-up menus: a button with data-menu opens the menu with that id (closing any other), and a choice in it, a click
// anywhere else or Esc closes it. They keep the controls that are used now and then (speed, camera, the rest) out of
// sight until they're wanted.
export function createMenus() {
  const menus = [...document.querySelectorAll('.menu')];
  const close = () => {
    menus.forEach(menu => menu.classList.add('hidden'));
    document.querySelectorAll('[data-menu]').forEach(button => button.setAttribute('aria-expanded', 'false'));
  };
  document.querySelectorAll('[data-menu]').forEach(button => button.addEventListener('click', event => {
    event.stopPropagation();
    const menu = document.getElementById(button.dataset.menu);
    const opening = menu.classList.contains('hidden');
    close();
    if (!opening) return;
    menu.classList.remove('hidden');
    button.setAttribute('aria-expanded', 'true');
  }));
  // A choice closes its menu once it has done its job (the choice's own listener runs first).
  menus.forEach(menu => menu.addEventListener('click', event => { if (event.target.closest('button')) setTimeout(close); }));
  document.addEventListener('click', event => { if (!event.target.closest('.menu')) close(); });
  window.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
  return {
    get open() { return menus.some(menu => !menu.classList.contains('hidden')); },
    close,
  };
}
