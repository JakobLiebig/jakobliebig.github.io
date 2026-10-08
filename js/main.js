// Reveal init and widget lifecycle.
//
// Widgets are classic scripts (not ES modules) so the deck also works from file://.
// Each file in js/widgets/ is loaded after this one and registers itself:
//
//   Widgets.register('W3', {
//     init(el, data) { ...; return { start() {}, stop() {} }; }
//   });
//
// A slide hosts a widget with <div class="widget" data-widget="W3" data-src="data/toy/forward.json" ...>.
// Config comes from data-* attributes, so one widget can appear on several slides.
// Data is loaded lazily the first time the widget's slide is shown.

(function () {
  const registry = {};
  const instances = new WeakMap(); // el -> { promise, instance }

  window.Widgets = {
    register(id, def) {
      registry[id] = def;
      // Reveal can report "ready" before every widget script has run. Widgets of this kind that are
      // already active (their slide is showing) and still waiting get started now.
      document.querySelectorAll(`.widget[data-widget="${id}"]`).forEach((el) => {
        if (el.dataset.active === '1' && !instances.has(el)) {
          ensure(el).then((w) => { if (w && el.dataset.active === '1') w.start(); });
        }
      });
    },
  };

  async function loadData(el) {
    const src = el.dataset.src;
    if (!src) return null;
    const res = await fetch(src);
    if (!res.ok) throw new Error(`${src}: ${res.status}`);
    return src.endsWith('.json') ? res.json() : res.blob();
  }

  function ensure(el) {
    let entry = instances.get(el);
    if (entry) return entry.promise;
    const def = registry[el.dataset.widget];
    if (!def) return Promise.resolve(null); // not built yet: placeholder stays
    entry = {};
    entry.promise = loadData(el)
      .then((data) => {
        el.classList.remove('placeholder');
        entry.instance = def.init(el, data);
        return entry.instance;
      })
      .catch((err) => { console.error(`Widget ${el.dataset.widget}:`, err); return null; });
    instances.set(el, entry);
    return entry.promise;
  }

  function widgetsIn(slide) {
    return slide ? slide.querySelectorAll('.widget[data-widget]') : [];
  }

  function start(slide) {
    widgetsIn(slide).forEach((el) => {
      el.dataset.active = '1';
      ensure(el).then((w) => { if (w && el.dataset.active === '1') w.start(); });
    });
  }

  function stop(slide) {
    widgetsIn(slide).forEach((el) => {
      el.dataset.active = '';
      const entry = instances.get(el);
      if (entry && entry.instance) entry.instance.stop();
    });
  }

  const params = new URLSearchParams(location.search);
  const scrollView = params.get('view') === 'scroll';

  // Fullscreen button, fixed in the bottom-left corner on every slide (not in PDF export).
  // Safari still needs the webkit-prefixed API.
  if (!params.has('print-pdf')) {
    const root = document.documentElement;
    const isFull = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
    const btn = document.createElement('button');
    btn.className = 'fullscreen-btn';
    btn.type = 'button';
    const ENTER = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
    const EXIT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>';
    const sync = () => {
      btn.innerHTML = isFull() ? EXIT : ENTER;
      btn.title = btn.ariaLabel = isFull() ? 'Exit fullscreen' : 'Fullscreen';
    };
    btn.addEventListener('click', () => {
      if (isFull()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      else (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
      btn.blur();  // keep the arrow keys for reveal
    });
    document.addEventListener('fullscreenchange', sync);
    document.addEventListener('webkitfullscreenchange', sync);
    sync();
    document.body.appendChild(btn);
  }

  Reveal.initialize({
    hash: true,
    slideNumber: 'c/t',
    transition: 'fade',
    controls: scrollView,
    progress: true,
    center: true,
    width: 1280,
    height: 720,
    margin: 0.06,
    katex: {
      local: 'vendor/katex',
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    },
    plugins: [RevealMath.KaTeX],
  });

  if (scrollView) {
    // In scroll view every slide is on the page; run widgets while they are visible.
    Reveal.on('ready', () => {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
          const el = e.target;
          if (e.isIntersecting) {
            el.dataset.active = '1';
            ensure(el).then((w) => { if (w && el.dataset.active === '1') w.start(); });
          } else {
            el.dataset.active = '';
            const entry = instances.get(el);
            if (entry && entry.instance) entry.instance.stop();
          }
        });
      }, { threshold: 0.25 });
      document.querySelectorAll('.widget[data-widget]').forEach((el) => io.observe(el));
    });
  } else {
    Reveal.on('ready', (e) => start(e.currentSlide));
    Reveal.on('slidechanged', (e) => {
      stop(e.previousSlide);
      start(e.currentSlide);
    });
  }
})();
