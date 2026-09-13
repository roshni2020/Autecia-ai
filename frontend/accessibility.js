// Native controls, persistent accessibility preferences, and keyboard choices.
(() => {
  const root = document.documentElement;
  const controls = [['largeText','large-text'],['highContrast','high-contrast'],['reduceMotion','still-ui']];
  for (const [id, cls] of controls) {
    const button = document.getElementById(id);
    const key = 'echoloop_' + cls;
    let enabled = cls === 'still-ui' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    try { const saved = localStorage.getItem(key); if (saved !== null) enabled = saved === '1'; } catch {}
    const apply = value => { root.classList.toggle(cls, value); button.setAttribute('aria-pressed', String(value)); };
    apply(enabled);
    button.addEventListener('click', () => { enabled = !root.classList.contains(cls); apply(enabled); try { localStorage.setItem(key, enabled ? '1' : '0'); } catch {} });
  }
  const choices = document.getElementById('cands');
  choices.setAttribute('role','radiogroup');
  choices.addEventListener('keydown', event => {
    if (!['ArrowDown','ArrowUp','ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    const buttons = [...choices.querySelectorAll('[role="radio"]')];
    const index = buttons.indexOf(document.activeElement);
    if (index < 0 || !buttons.length) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (['ArrowDown','ArrowRight'].includes(event.key) ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].click();
    choices.querySelectorAll('[role="radio"]')[next]?.focus();
  });
  // Keep selection focus after the existing renderer replaces the buttons.
  const selection = new MutationObserver(() => {
    choices.querySelectorAll('[role="radio"]').forEach(b => { b.tabIndex = b.getAttribute('aria-checked') === 'true' ? 0 : -1; });
  });
  selection.observe(choices,{childList:true});
  document.querySelectorAll('.nav-item').forEach(button => {
    button.setAttribute('aria-label',button.title);
    button.setAttribute('aria-controls','tab-' + button.dataset.tab);
  });
  const updateNavigation = () => document.querySelectorAll('.nav-item').forEach(b => { if (b.classList.contains('active')) b.setAttribute('aria-current','page'); else b.removeAttribute('aria-current'); });
  new MutationObserver(updateNavigation).observe(document.querySelector('nav'),{subtree:true,attributes:true,attributeFilter:['class']});
  updateNavigation();
  document.getElementById('confirmedText').setAttribute('aria-live','polite');
  const dialog = document.getElementById('editDialog');
  dialog.querySelector('h2').id = 'editTitle';
  dialog.setAttribute('aria-labelledby','editTitle');
})();
