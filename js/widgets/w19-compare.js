// W19: three ways from noise to data, side by side.
//
// Each panel: a mixture of Gaussians (data, left) and one Gaussian (noise, right), and the paths
// that carry the same noise samples to the data. Everything is computed exactly for this mixture:
//   DDPM          reverse VP SDE, 100 Euler-Maruyama steps, exact score
//   DDIM          deterministic DDIM (eta = 0), 25 steps of the 100-step schedule, exact epsilon
//   Flow matching marginal flow-matching ODE, dx/dt = E[eps - x0 | x_t] for x_t = (1 - t) x0 + t eps
// The slide opens with the setup only: in every panel the data, the noise and the start points.
// Fragments with data-w19-stage="1" / "2" / "3" then draw the paths of DDPM, DDIM and flow matching.
//
// With data-src="data/toy/compare.json" the paths come from trained models instead (see the
// simpleDiffusions repo, HANDOFF.md). Schema: { xrange, yrange, data: [{mx, my, s}], noise: {mx, my, s},
// panels: [{ title, sub, paths: [[[x, y], ...], ...] }] }, paths running from noise to data.

(function () {
  // World coordinates: data on the left, noise on the right.
  const XR0 = [-3.6, 3.6], YR0 = [-2.6, 2.6];
  const DATA0 = [
    { mx: -2.4, my: -1.5, s: 0.28 },
    { mx: -2.0, my: 0.1, s: 0.3 },
    { mx: -2.6, my: 1.55, s: 0.26 },
  ];
  const NOISE0 = { mx: 2.2, my: 0, s: 0.75 };
  // The analytic paths below use these defaults.
  const DATA = DATA0, NOISE = NOISE0;
  const N_PATHS = 36;

  const C_DATA = [242, 153, 74];
  const C_NOISE = [138, 155, 180];
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  function rng(seed) {
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return () => {
      const u1 = Math.max(u(), 1e-12), u2 = u();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
  }

  // Log-weights of a Gaussian mixture at x, normalized to probabilities.
  function posterior(x, y, comps) {
    const lw = comps.map((c) => Math.log(c.pi) - Math.log(c.v) - ((x - c.mx) ** 2 + (y - c.my) ** 2) / (2 * c.v));
    const m = Math.max(...lw);
    const w = lw.map((l) => Math.exp(l - m));
    const z = w.reduce((a, b) => a + b, 0);
    return w.map((v) => v / z);
  }

  // ---- Reverse VP SDE, relative to the noise center, with stationary std NOISE.s ----
  // beta(t) = 0.1 + 19.9 t, alphaBar(t) = exp(-(0.1 t + 9.95 t^2)).
  const betaAt = (t) => 0.1 + 19.9 * t;
  const abAt = (t) => Math.exp(-(0.1 * t + 9.95 * t * t));
  function sdePaths(starts, steps, seed) {
    const g = rng(seed);
    const s2 = NOISE.s * NOISE.s;
    const dt = 1 / steps;
    return starts.map(([sx, sy]) => {
      let x = sx - NOISE.mx, y = sy - NOISE.my;
      const path = [[sx, sy]];
      for (let k = steps; k >= 1; k--) {
        const t = k / steps, b = betaAt(t), ab = abAt(t), a = Math.sqrt(ab);
        // Marginal p_t: component k is N(sqrt(ab) mu_k', (ab s_k^2 + (1 - ab) s_n^2) I).
        const comps = DATA.map((c) => ({
          mx: a * (c.mx - NOISE.mx), my: a * (c.my - NOISE.my),
          v: ab * c.s * c.s + (1 - ab) * s2, pi: 1 / DATA.length,
        }));
        const w = posterior(x, y, comps);
        let gx = 0, gy = 0;
        comps.forEach((c, i) => { gx -= (w[i] * (x - c.mx)) / c.v; gy -= (w[i] * (y - c.my)) / c.v; });
        x += (0.5 * b * x + s2 * b * gx) * dt + NOISE.s * Math.sqrt(b * dt) * g();
        y += (0.5 * b * y + s2 * b * gy) * dt + NOISE.s * Math.sqrt(b * dt) * g();
        path.push([x + NOISE.mx, y + NOISE.my]);
      }
      return path;
    });
  }

  // ---- DDIM (eta = 0): k = 100, 100 - 100 / steps, ..., 0 with alphaBar(k / 100), exact epsilon ----
  // Normalized coordinates z = (x - noise mean) / noise std, so the noise is N(0, I).
  function ddimPaths(starts, steps) {
    const T = 100, stride = T / steps, sn = NOISE.s;
    const ab = (k) => (k === 0 ? 1 : abAt(k / T));
    return starts.map(([sx, sy]) => {
      let x = (sx - NOISE.mx) / sn, y = (sy - NOISE.my) / sn;
      const path = [[sx, sy]];
      for (let k = T; k > 0; k -= stride) {
        const a = ab(k), ap = ab(k - stride), r = Math.sqrt(a);
        const comps = DATA.map((c) => ({
          mx: r * (c.mx - NOISE.mx) / sn, my: r * (c.my - NOISE.my) / sn,
          v: a * (c.s / sn) ** 2 + 1 - a, pi: 1 / DATA.length,
        }));
        const w = posterior(x, y, comps);
        // eps* = -sqrt(1 - aB) * score.
        let ex = 0, ey = 0;
        comps.forEach((c, i) => { ex += (w[i] * (x - c.mx)) / c.v; ey += (w[i] * (y - c.my)) / c.v; });
        ex *= Math.sqrt(1 - a); ey *= Math.sqrt(1 - a);
        const x0 = (x - Math.sqrt(1 - a) * ex) / r, y0 = (y - Math.sqrt(1 - a) * ey) / r;
        x = Math.sqrt(ap) * x0 + Math.sqrt(1 - ap) * ex;
        y = Math.sqrt(ap) * y0 + Math.sqrt(1 - ap) * ey;
        path.push([NOISE.mx + sn * x, NOISE.my + sn * y]);
      }
      return path;
    });
  }

  // ---- Flow matching: exact marginal velocity for x_t = (1 - t) x0 + t eps ----
  function fmVelocity(x, y, t) {
    const sn2 = NOISE.s * NOISE.s;
    const comps = DATA.map((c) => {
      const V = (1 - t) ** 2 * c.s * c.s + t * t * sn2;
      return { c, V, mx: (1 - t) * c.mx + t * NOISE.mx, my: (1 - t) * c.my + t * NOISE.my, v: V, pi: 1 / DATA.length };
    });
    const w = posterior(x, y, comps);
    let ux = 0, uy = 0;
    comps.forEach((k, i) => {
      // E[x0 | x] and E[eps | x] for this component (jointly Gaussian).
      const fx = (1 - t) * k.c.s * k.c.s / k.V, fe = t * sn2 / k.V;
      const ex0 = k.c.mx + fx * (x - k.mx), ey0 = k.c.my + fx * (y - k.my);
      const exe = NOISE.mx + fe * (x - k.mx), eye = NOISE.my + fe * (y - k.my);
      ux += w[i] * (exe - ex0); uy += w[i] * (eye - ey0);
    });
    return [ux, uy];
  }
  function fmPaths(starts, steps) {
    const dt = 1 / steps;
    return starts.map(([sx, sy]) => {
      let x = sx, y = sy;
      const path = [[x, y]];
      for (let k = steps; k >= 1; k--) {
        // RK4 backwards in time, from t = 1 (noise) to t = 0 (data).
        const t = k / steps;
        const f = (px, py, tt) => fmVelocity(px, py, Math.min(1, Math.max(1e-4, tt)));
        const k1 = f(x, y, t);
        const k2 = f(x - 0.5 * dt * k1[0], y - 0.5 * dt * k1[1], t - 0.5 * dt);
        const k3 = f(x - 0.5 * dt * k2[0], y - 0.5 * dt * k2[1], t - 0.5 * dt);
        const k4 = f(x - dt * k3[0], y - dt * k3[1], t - dt);
        x -= (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        y -= (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        path.push([x, y]);
      }
      return path;
    });
  }

  Widgets.register('W19', {
    init(el, json) {
      // Trained paths from JSON if given, otherwise the exact analytic ones.
      const trained = json && Array.isArray(json.panels);
      const still = el.dataset.still === '1';
      const XR = trained ? json.xrange : XR0, YR = trained ? json.yrange : YR0;
      const DATA = trained ? json.data : DATA0, NOISE = trained ? json.noise : NOISE0;
      const canvas = document.createElement('canvas');
      el.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const W = el.clientWidth || 1100, H = el.clientHeight || 340;
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      canvas.width = Math.round(W * ratio); canvas.height = Math.round(H * ratio);

      // Same noise samples for every panel.
      const g = rng(2024);
      const starts = [];
      while (starts.length < N_PATHS) {
        const p = [NOISE.mx + NOISE.s * g(), NOISE.my + NOISE.s * g()];
        if (p[1] > YR[0] + 0.2 && p[1] < YR[1] - 0.2 && p[0] < XR[1] - 0.2) starts.push(p);
      }
      const panels = trained ? json.panels.map((p) => ({ ...p })) : [
        { title: 'DDPM', sub: '100 steps', paths: sdePaths(starts, 100, 11) },
        { title: 'DDIM', sub: 'same network, 25 deterministic steps', paths: ddimPaths(starts, 25) },
        { title: 'Flow matching', sub: 'ODE, nearly straight', paths: fmPaths(starts, 60) },
      ];

      // Layout: three panels in a row, title above each.
      // The still (overview) shows titles only: the panels are too narrow for the subtitles.
      const gap = 24, headH = still ? 28 : 46;
      const pw = (W - 2 * gap) / 3;
      const scale = Math.min(pw / (XR[1] - XR[0]), (H - headH - 6) / (YR[1] - YR[0]));
      const ph = (YR[1] - YR[0]) * scale;

      function blob(px, py, ox, oy, s, col, alpha) {
        const cx = ox + (px - XR[0]) * scale, cy = oy + (YR[1] - py) * scale, r = 2.6 * s * scale;
        const gr = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        gr.addColorStop(0, rgb(col, alpha));
        gr.addColorStop(1, rgb(col, 0));
        ctx.fillStyle = gr;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
      }

      const shown = [0, 0, 0];   // draw progress per panel, 0..1
      function draw() {
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, W, H);
        panels.forEach((p, i) => {
          const ox = i * (pw + gap) + (pw - (XR[1] - XR[0]) * scale) / 2, oy = headH;
          const X = (x) => ox + (x - XR[0]) * scale, Y = (y) => oy + (YR[1] - y) * scale;
          // Panel background and the two distributions.
          ctx.fillStyle = '#171920';
          ctx.beginPath(); ctx.roundRect(ox, oy, (XR[1] - XR[0]) * scale, ph, 10); ctx.fill();
          ctx.save();
          ctx.beginPath(); ctx.roundRect(ox, oy, (XR[1] - XR[0]) * scale, ph, 10); ctx.clip();
          for (const c of DATA) blob(c.mx, c.my, ox, oy, c.s, C_DATA, 0.55);
          blob(NOISE.mx, NOISE.my, ox, oy, NOISE.s, C_NOISE, 0.35);
          // Before its turn: just the start points on the noise side.
          if (shown[i] <= 0) {
            ctx.fillStyle = '#fff';
            for (const path of p.paths) { ctx.beginPath(); ctx.arc(X(path[0][0]), Y(path[0][1]), 2.6, 0, Math.PI * 2); ctx.fill(); }
          }
          // Paths, revealed from the noise end.
          if (shown[i] > 0) for (const path of p.paths) {
            const n = Math.max(1, Math.round(shown[i] * (path.length - 1)));
            const [ax, ay] = path[0], [bx, by] = path[path.length - 1];
            const gr = ctx.createLinearGradient(X(ax), Y(ay), X(bx), Y(by));
            gr.addColorStop(0, rgb(C_NOISE, 0.75));
            gr.addColorStop(1, rgb(C_DATA, 0.9));
            ctx.strokeStyle = gr; ctx.lineWidth = 1.2; ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(X(ax), Y(ay));
            for (let k = 1; k <= n; k++) ctx.lineTo(X(path[k][0]), Y(path[k][1]));
            ctx.stroke();
            const [ex, ey] = path[n];
            ctx.fillStyle = shown[i] >= 1 ? rgb(C_DATA) : '#fff';
            ctx.beginPath(); ctx.arc(X(ex), Y(ey), 2.2, 0, Math.PI * 2); ctx.fill();
          }
          ctx.restore();
          // Titles.
          ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
          ctx.fillStyle = '#e8e9ee'; ctx.font = '600 20px Inter, system-ui, sans-serif';
          ctx.fillText(p.title, ox, 22);
          ctx.fillStyle = '#8b8fa0'; ctx.font = '15px Inter, system-ui, sans-serif';
          if (!still) ctx.fillText(p.sub, ox, 40);
        });
      }

      // Animation: each panel draws itself when its turn comes.
      const DRAW_MS = 2600;
      let raf = 0;
      const anim = [null, null, null];
      function tick(now) {
        raf = 0;
        let more = false;
        anim.forEach((t0, i) => {
          if (t0 === null) return;
          shown[i] = Math.max(0, Math.min(1, (now - t0) / DRAW_MS));
          if (shown[i] < 1) more = true; else anim[i] = null;
        });
        draw();
        if (more) raf = requestAnimationFrame(tick);
      }
      function play(i) {
        if (shown[i] >= 1 || anim[i] !== null) return;
        anim[i] = performance.now();
        if (!raf) raf = requestAnimationFrame(tick);
      }

      const slide = el.closest('section');
      const stageFromDom = () => {
        let st = 0;
        if (slide) slide.querySelectorAll('.fragment.visible[data-w19-stage]').forEach((f) => { st = Math.max(st, +f.dataset.w19Stage); });
        return st;
      };
      function sync(animated) {
        const st = stageFromDom();
        for (let i = 0; i < 3; i++) {
          if (i < st) { if (animated) play(i); else { shown[i] = 1; anim[i] = null; } }
          else { shown[i] = 0; anim[i] = null; }
        }
        draw();
      }
      if (window.Reveal) {
        const onFragment = () => { if (el.dataset.active === '1' && !still) sync(true); };
        Reveal.on('fragmentshown', onFragment);
        Reveal.on('fragmenthidden', onFragment);
      }

      draw();
      return {
        start() {
          // Still mode (data-still, e.g. the overview slide): all panels fully drawn, no animation.
          if (still) { shown.fill(1); anim.fill(null); draw(); return; }
          // Arriving at the slide's start: setup only, nothing plays until the first click.
          // Coming back from a later slide: show what's revealed as is.
          if (stageFromDom() > 0) { sync(false); return; }
          shown.fill(0); anim.fill(null);
          draw();
        },
        stop() { if (raf) cancelAnimationFrame(raf); raf = 0; },
      };
    },
  });
})();
