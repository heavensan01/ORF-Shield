// mobile menu
const burger = document.querySelector('.burger'), links = document.querySelector('.nav-links');
if (burger) burger.addEventListener('click', () => links.classList.toggle('open'));

// flash messages: close button + auto dismiss
document.querySelectorAll('.flash').forEach(f => {
  f.querySelector('button').addEventListener('click', () => f.remove());
  setTimeout(() => f.remove(), 6000);
});

// scroll reveal
const io = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
}), { threshold: .12 });
document.querySelectorAll('.reveal').forEach(el => io.observe(el));

// show / hide password
document.querySelectorAll('.pw button').forEach(b => b.addEventListener('click', () => {
  const i = b.parentElement.querySelector('input');
  i.type = i.type === 'password' ? 'text' : 'password';
  b.textContent = i.type === 'password' ? 'Show' : 'Hide';
}));

// animated probability gauge
document.querySelectorAll('.gauge .fg').forEach(c => {
  const r = c.r.baseVal.value, len = 2 * Math.PI * r, pct = parseFloat(c.dataset.pct) / 100;
  c.style.strokeDasharray = len;
  c.style.strokeDashoffset = len;
  requestAnimationFrame(() => requestAnimationFrame(() => c.style.strokeDashoffset = len * (1 - pct)));
});

// detector page helpers
const ta = document.getElementById('jobtext');
if (ta) {
  const count = document.getElementById('count');
  const upd = () => {
    const t = ta.value.trim();
    count.textContent = `${t ? t.split(/\s+/).length : 0} words · ${ta.value.length} chars`;
  };
  ta.addEventListener('input', upd); upd();
  document.querySelectorAll('[data-sample]').forEach(b => b.addEventListener('click', () => {
    ta.value = document.getElementById(b.dataset.sample).textContent.trim(); upd(); ta.focus();
  }));
  document.getElementById('clear').addEventListener('click', () => { ta.value = ''; upd(); ta.focus(); });
  document.getElementById('detect-form').addEventListener('submit', () => {
    const btn = document.getElementById('detect-btn');
    btn.disabled = true; btn.textContent = 'Analysing…';
  });
}
