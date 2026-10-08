// W21: latent diffusion, live: flow matching in the 2D latent space of the VAE from W20.
//
// Left: the latent plane ([-4, 4]^2, same window as the W20 map) with the encoded digits as faint
// clouds (the VAE distribution), N(0, I) as dashed circles, and the learned velocity field at the
// current time t as arrows. Pick a start point by clicking (or "new noise"), pick a condition
// (one of 0-9) and the guidance weight w, then press play or drag the time slider: the point follows -v from
// t = 1 (noise) to t = 0 (data). Right: the W20 decoder turns the current point into an image.
//
// Data: data/latent/flow.json from simpleDiffusions/latent.py (README there). fields[label][k][row][col]
// = v(z, t_k) with z_1 = grid[col], z_2 = grid[row], t_k = 1.0, 0.9, ..., 0.0. Between grid points
// and times we interpolate (bilinear in z, linear in t) and integrate with RK4; this reproduces the
// exported paths to about 0.02-0.07 latent units.

(function () {
  const SRC = 'data/latent/flow.json';
  const STEPS = 40;          // RK4 steps for a path
  const PLAY_MS = 3200;
  const FALLBACK_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];

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

  function trapezoid() {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 220');
    svg.setAttribute('class', 'w20-trap dec');
    const poly = document.createElementNS(ns, 'polygon');
    poly.setAttribute('points', '0,80 120,20 120,200 0,140');
    svg.appendChild(poly);
    const t1 = document.createElementNS(ns, 'text');
    t1.setAttribute('x', 60); t1.setAttribute('y', 106); t1.setAttribute('class', 'w20-trap-title'); t1.textContent = 'Decoder';
    const t2 = document.createElementNS(ns, 'text');
    t2.setAttribute('x', 60); t2.setAttribute('y', 128); t2.setAttribute('class', 'w20-trap-sub'); t2.textContent = 'z → 784';
    svg.append(t1, t2);
    return svg;
  }

  Widgets.register('W21', {
    init(el, F) {
      el.classList.add('w21');
      el.innerHTML = '';
      const R = F.range, G = F.grid, NG = G.length, T = F.times, NT = T.length;

      // ---- Layout ----
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const PW = 470, PH = 340;   // 3D view of the latent space
      const plane = document.createElement('canvas');
      plane.width = PW * ratio; plane.height = PH * ratio; plane.className = 'w21-plane';
      plane.style.width = `${PW}px`; plane.style.height = `${PH}px`;
      const ctx = plane.getContext('2d');

      const chips = document.createElement('div'); chips.className = 'w21-chips';
      const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
      let cond = '3';
      labels.forEach((l) => {
        const b = document.createElement('button'); b.type = 'button'; b.className = 'w21-chip';
        b.textContent = l === 'all' ? 'any' : l; b.dataset.label = l;
        b.addEventListener('click', () => { cond = l; b.blur(); syncChips(); recompute(); });
        chips.appendChild(b);
      });
      const syncChips = () => [...chips.children].forEach((b) => b.classList.toggle('active', b.dataset.label === cond));

      // Layout: [latent plane] -> decoder -> [image], centered; below it one horizontal control bar.
      const section = (title, ...kids) => {
        const d = document.createElement('div'); d.className = 'w21-sec';
        const h = document.createElement('div'); h.className = 'w21-sec-title'; h.textContent = title;
        d.append(h, ...kids);
        return d;
      };
      const rowOf = (...kids) => { const d = document.createElement('div'); d.className = 'w21-row'; d.append(...kids); return d; };

      // Time: a slider from noise (t = 1) to data (t = 0), plus the two buttons.
      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = '0'; slider.max = '1'; slider.step = '0.001'; slider.value = '0'; slider.setAttribute('aria-label', 'time');
      const tLabel = document.createElement('span'); tLabel.className = 'w21-t';
      const playBtn = document.createElement('button'); playBtn.type = 'button'; playBtn.className = 'w21-btn w21-play'; playBtn.textContent = '▶ generate';
      const noiseBtn = document.createElement('button'); noiseBtn.type = 'button'; noiseBtn.className = 'w21-btn w21-noise'; noiseBtn.textContent = 'sample noise';

      // Classifier-free guidance: v_w = v_any + w (v_c - v_any). w = 2 keeps paths on the data and
      // lands about 85% on the requested digit (w = 1: about 73%).
      let w = +(el.dataset.w || 2);
      const wSlider = document.createElement('input');
      wSlider.type = 'range'; wSlider.min = '0'; wSlider.max = '6'; wSlider.step = '0.1'; wSlider.setAttribute('aria-label', 'guidance');
      const wVal = document.createElement('span'); wVal.className = 'w21-t w21-wval';
      wSlider.value = String(w); wVal.textContent = `w = ${w.toFixed(1)}`;
      wSlider.addEventListener('input', () => { w = +wSlider.value; wVal.textContent = `w = ${w.toFixed(1)}`; recompute(); });
      wSlider.addEventListener('pointerup', () => wSlider.blur());

      // 2D / 3D toggle: 3D shows p_t as a surface, 2D a top view with the (guided) vector field.
      let mode3d = true;
      const toggle = document.createElement('div'); toggle.className = 'w21-toggle';
      toggle.innerHTML = '<button type="button" class="w21-btn" data-v="2d">2D</button><button type="button" class="w21-btn" data-v="3d">3D</button>';
      const syncToggle = () => [...toggle.children].forEach((b) => b.classList.toggle('active', (b.dataset.v === '3d') === mode3d));
      toggle.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; b.blur(); mode3d = b.dataset.v === '3d'; syncToggle(); draw(); });
      syncToggle();

      const outCanvas = document.createElement('canvas'); outCanvas.width = outCanvas.height = 28; outCanvas.className = 'w20-digit';
      const panel = document.createElement('div'); panel.className = 'w21-panel';
      panel.setAttribute('data-prevent-swipe', '');
      panel.append(
        section('Digit', chips),
        section('Guidance', rowOf(wSlider, wVal)),
        section('Time', rowOf(slider, tLabel), rowOf(playBtn, noiseBtn)),
        section('View', toggle));
      const vis = document.createElement('div'); vis.className = 'w21-vis';
      vis.append(plane, trapezoid(), outCanvas);
      el.append(vis, panel);

      // ---- Field interpolation and integration ----
      // Guided field: v_any + w (v_c - v_any).
      function fieldAt(label, x, y, t) {
        const vu = rawField('all', x, y, t), vc = rawField(label, x, y, t);
        return [vu[0] + w * (vc[0] - vu[0]), vu[1] + w * (vc[1] - vu[1])];
      }
      function rawField(label, x, y, t) {
        const Fl = F.fields[label];
        const ti = Math.max(0, Math.min(NT - 1 - 1e-9, (1 - t) * (NT - 1)));
        const k0 = Math.floor(ti), a = ti - k0;
        const gx = Math.max(0, Math.min(NG - 1 - 1e-9, ((x + R) / (2 * R)) * (NG - 1)));
        const gy = Math.max(0, Math.min(NG - 1 - 1e-9, ((y + R) / (2 * R)) * (NG - 1)));
        const c0 = Math.floor(gx), r0 = Math.floor(gy), fx = gx - c0, fy = gy - r0;
        const v = [0, 0];
        for (const [k, w] of [[k0, 1 - a], [k0 + 1, a]]) {
          if (w === 0 || k >= NT) continue;
          const M = Fl[k];
          for (let d = 0; d < 2; d++) {
            v[d] += w * (M[r0][c0][d] * (1 - fx) * (1 - fy) + M[r0][c0 + 1][d] * fx * (1 - fy)
              + M[r0 + 1][c0][d] * (1 - fx) * fy + M[r0 + 1][c0 + 1][d] * fx * fy);
          }
        }
        return v;
      }
      function integrate(label, z0) {
        const dt = 1 / STEPS, path = [z0.slice()];
        let [x, y] = z0, t = 1;
        for (let i = 0; i < STEPS; i++) {
          const k1 = fieldAt(label, x, y, t);
          const k2 = fieldAt(label, x - dt / 2 * k1[0], y - dt / 2 * k1[1], t - dt / 2);
          const k3 = fieldAt(label, x - dt / 2 * k2[0], y - dt / 2 * k2[1], t - dt / 2);
          const k4 = fieldAt(label, x - dt * k3[0], y - dt * k3[1], t - dt);
          x -= dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
          y -= dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
          t -= dt;
          path.push([x, y]);
        }
        return path;
      }

      // ---- State ----
      const MAX_R = 2;          // start points: noise within 2 sigma
      const g = rng(42);
      let start = [g() * 0.9, g() * 0.9];
      let path = null, p = 0;   // p in [0, 1]: progress from noise (t = 1) to data (t = 0)
      let VAE = null;           // decoder + latents once loaded
      let base = null;          // cached background
      const posAt = (q) => {
        const f = Math.max(0, Math.min(1, q)) * STEPS, i = Math.min(STEPS - 1, Math.floor(f)), r = f - i;
        return [path[i][0] + (path[i + 1][0] - path[i][0]) * r, path[i][1] + (path[i + 1][1] - path[i][1]) * r];
      };

      // ---- 3D view: the surface is p_t, the distribution of z_t = (1 - t) z_0 + t eps with z_0 the
      // encoded test digits of the chosen label (a kernel estimate, cached per label on a grid at
      // t = 0, 0.1, ..., 1). At t = 1 it is N(0, I), at t = 0 the digit's cluster. ----
      const E3 = 2.6, NGR = 56, NTK = 11, HT = 1.3, BW = 0.2;
      const densCache = {};
      function densityFor(label) {
        if (densCache[label]) return densCache[label];
        const pts = [];
        VAE.latents.points.forEach((pt, i) => { if (String(VAE.latents.labels[i]) === label) pts.push(pt); });
        const sub = pts.filter((_, i) => i % Math.max(1, Math.ceil(pts.length / 450)) === 0);
        const out = [];
        for (let k = 0; k < NTK; k++) {
          const t = k / (NTK - 1), a = 1 - t, v = (a * BW) ** 2 + t * t;
          const D = new Float32Array((NGR + 1) * (NGR + 1));
          let mx = 0;
          for (let j = 0; j <= NGR; j++) for (let i = 0; i <= NGR; i++) {
            const x = -E3 + (2 * E3 * i) / NGR, y = -E3 + (2 * E3 * j) / NGR;
            let sum = 0;
            for (const [zx, zy] of sub) sum += Math.exp(-((x - a * zx) ** 2 + (y - a * zy) ** 2) / (2 * v));
            D[j * (NGR + 1) + i] = sum; mx = Math.max(mx, sum);
          }
          for (let q = 0; q < D.length; q++) D[q] /= mx;
          out.push(D);
        }
        return (densCache[label] = out);
      }
      const gauss = (x, y) => Math.exp(-(x * x + y * y) / 2);
      function heightAt(x, y, t) {
        if (!VAE) return HT * gauss(x, y);
        const Ds = densityFor(cond);
        const tk = Math.max(0, Math.min(NTK - 1 - 1e-9, t * (NTK - 1))), k0 = Math.floor(tk), a = tk - k0;
        const gx = Math.max(0, Math.min(NGR - 1e-9, ((x + E3) / (2 * E3)) * NGR));
        const gy = Math.max(0, Math.min(NGR - 1e-9, ((y + E3) / (2 * E3)) * NGR));
        const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j, W1 = NGR + 1;
        const at = (D) => D[j * W1 + i] * (1 - fx) * (1 - fy) + D[j * W1 + i + 1] * fx * (1 - fy) + D[(j + 1) * W1 + i] * (1 - fx) * fy + D[(j + 1) * W1 + i + 1] * fx * fy;
        return HT * (at(Ds[k0]) * (1 - a) + at(Ds[Math.min(NTK - 1, k0 + 1)]) * a);
      }

      const AZ3 = -0.3, EL3 = 0.62, DIST3 = 12;
      const ca = Math.cos(AZ3), sa = Math.sin(AZ3), ce = Math.cos(EL3), se = Math.sin(EL3);
      const view = (x, y, z) => {
        const xr = x * ca - y * sa, yr = x * sa + y * ca;
        const up = z * ce + yr * se, depth = yr * ce - z * se, f = DIST3 / (DIST3 + depth);
        return [xr * f, up * f, depth];
      };
      // Fixed fit: the corners at z = 0 and the center at full height.
      const fit = (() => {
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (const [x, y, z] of [[-E3, -E3, 0], [E3, -E3, 0], [E3, E3, 0], [-E3, E3, 0], [0, 0, HT], [-E3, E3, HT], [E3, E3, HT]]) {
          const v = view(x, y, z); x0 = Math.min(x0, v[0]); x1 = Math.max(x1, v[0]); y0 = Math.min(y0, v[1]); y1 = Math.max(y1, v[1]);
        }
        const sc = Math.min((PW - 12) / (x1 - x0), (PH - 12) / (y1 - y0));
        return { sc, ox: PW / 2 - ((x0 + x1) / 2) * sc, oy: PH / 2 + ((y0 + y1) / 2) * sc };
      })();
      const toS = (v) => [fit.ox + v[0] * fit.sc, fit.oy - v[1] * fit.sc];
      const onSurf = (x, y, t) => toS(view(x, y, heightAt(x, y, t) + 0.03));

      function drawSurface(t) {
        const verts = [];
        for (let i = 0; i <= NGR; i++) {
          const row = [];
          for (let j = 0; j <= NGR; j++) {
            const x = -E3 + (2 * E3 * i) / NGR, y = -E3 + (2 * E3 * j) / NGR, z = heightAt(x, y, t);
            row.push({ x, y, z, v: view(x, y, z) });
          }
          verts.push(row);
        }
        const quads = [];
        for (let i = 0; i < NGR; i++) for (let j = 0; j < NGR; j++) {
          const q = [verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]];
          quads.push({ q, d: (q[0].v[2] + q[1].v[2] + q[2].v[2] + q[3].v[2]) / 4, i, j });
        }
        quads.sort((p1, p2) => p2.d - p1.d);
        const Lv = [-0.5, -0.6, 0.8], ln = Math.hypot(...Lv);
        const CL = [30, 35, 47], CN = [138, 155, 180], CD = [242, 153, 74];
        const mixc = (A, B, u) => A.map((v, k) => Math.round(v + (B[k] - v) * u));
        for (const { q, i, j } of quads) {
          const [a, b, c, d] = q;
          const ux = c.x - a.x, uy = c.y - a.y, uz = c.z - a.z, wx = d.x - b.x, wy = d.y - b.y, wz = d.z - b.z;
          let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
          const nn = Math.hypot(nx, ny, nz) || 1; nx /= nn; ny /= nn; nz /= nn; if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
          const lam = Math.max(0, (nx * Lv[0] + ny * Lv[1] + nz * Lv[2]) / ln);
          const h = (a.z + b.z + c.z + d.z) / (4 * HT);
          const base3 = mixc(CL, mixc(CN, CD, Math.min(1, h * 1.4)), Math.min(1, 0.25 + h * 1.6));
          const col = `rgb(${base3.map((v) => Math.min(255, Math.round(v * (0.55 + 0.55 * lam)))).join(',')})`;
          const pts = q.map((pp) => toS(pp.v));
          ctx.beginPath(); ctx.moveTo(...pts[0]); for (let k = 1; k < 4; k++) ctx.lineTo(...pts[k]); ctx.closePath();
          ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 0.8; ctx.fill(); ctx.stroke();
          if (i % 7 === 0 || j % 7 === 0) {
            ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.6; ctx.beginPath();
            if (i % 7 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[3]); }
            if (j % 7 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); }
            ctx.stroke();
          }
        }
      }

      // 2D top view: a square in the middle of the canvas, p_t as brightness, the guided field as arrows.
      const SQ = PH - 8, sx0 = (PW - SQ) / 2, sy0 = 4;
      const to2 = (x, y) => [sx0 + ((x + E3) / (2 * E3)) * SQ, sy0 + SQ - ((y + E3) / (2 * E3)) * SQ];
      function draw2D(t) {
        const N2 = 64, cell = SQ / N2;
        for (let i = 0; i < N2; i++) for (let j = 0; j < N2; j++) {
          const x = -E3 + (2 * E3 * (i + 0.5)) / N2, y = -E3 + (2 * E3 * (j + 0.5)) / N2;
          const h = heightAt(x, y, t) / HT;
          const c = [30 + 212 * h, 35 + 118 * h, 47 + 27 * h].map((v) => Math.round(v));
          ctx.fillStyle = `rgb(${c.join(',')})`;
          ctx.fillRect(sx0 + i * cell, sy0 + SQ - (j + 1) * cell, cell + 0.6, cell + 0.6);
        }
        const step = (2 * E3) / 12;
        for (let x = -E3 + step / 2; x < E3; x += step) for (let y = -E3 + step / 2; y < E3; y += step) {
          const v = fieldAt(cond, x, y, t), m = Math.hypot(v[0], v[1]);
          if (m < 1e-6) continue;
          const ux = -v[0] / m, uy = -v[1] / m, L = cell * 3.2, [ax, ay] = to2(x, y);
          const bx = ax + ux * L, by = ay - uy * L, hd = 5;
          ctx.strokeStyle = ctx.fillStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.3;
          ctx.beginPath(); ctx.moveTo(ax - ux * L / 2, ay + uy * L / 2); ctx.lineTo(bx - ux * L / 2, by + uy * L / 2); ctx.stroke();
          const tx = bx - ux * L / 2, ty = by + uy * L / 2;
          ctx.beginPath(); ctx.moveTo(tx, ty);
          ctx.lineTo(tx - ux * hd + uy * hd * 0.55, ty + uy * hd + ux * hd * 0.55);
          ctx.lineTo(tx - ux * hd - uy * hd * 0.55, ty + uy * hd - ux * hd * 0.55);
          ctx.closePath(); ctx.fill();
        }
      }

      function draw() {
        const t = 1 - p;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, plane.width, plane.height);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        const proj = mode3d ? (x, y) => onSurf(x, y, t) : (x, y) => to2(x, y);
        if (mode3d) drawSurface(t); else draw2D(t);
        // The path so far (on the current surface) and the point.
        if (path) {
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.2; ctx.lineJoin = 'round';
          ctx.beginPath();
          const n = p * STEPS;
          for (let i = 0; i <= Math.min(STEPS, Math.floor(n)); i++) {
            const [x, y] = path[i], [sx, sy] = proj(x, y);
            if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
          }
          const [cx, cy] = posAt(p), [px, py] = proj(cx, cy);
          ctx.lineTo(px, py); ctx.stroke();
          ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fillStyle = '#0e0f13'; ctx.fill();
          ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
        }
        tLabel.textContent = `t = ${t.toFixed(2)}`;
        slider.value = String(p);
        decode();
      }

      function decode() {
        if (!VAE || !path) return;
        const [x, y] = posAt(p);
        const out = tf.tidy(() => VAE.dec.predict(tf.tensor([[x, y]], [1, 2])).dataSync());
        const c = outCanvas.getContext('2d'), img = c.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
          const v = Math.max(0, Math.min(255, out[i] * 255));
          img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
        }
        c.putImageData(img, 0, 0);
      }

      function recompute() { path = integrate(cond, start); p = 0; stopPlay(); draw(); }

      // ---- Interaction ----
      let raf = 0, t0 = 0, from = 0;
      function stopPlay() { if (raf) cancelAnimationFrame(raf); raf = 0; }
      function tick(now) {
        // The frame timestamp can be a bit earlier than the click: clamp to [0, 1].
        const q = Math.max(0, Math.min(1, (now - t0) / PLAY_MS));
        p = from + (1 - from) * q;
        draw();
        raf = q < 1 ? requestAnimationFrame(tick) : 0;
      }
      playBtn.addEventListener('click', (e) => {
        e.currentTarget.blur();
        if (p >= 1) p = 0;
        from = p; t0 = performance.now(); stopPlay(); raf = requestAnimationFrame(tick);
      });
      // Start points are noise: draws from N(0, I) within 2 sigma. Far outside, real noise almost never
      // lands, the model has barely seen that region and its paths end in odd places.
      noiseBtn.addEventListener('click', (e) => {
        e.currentTarget.blur();
        do { start = [g(), g()]; } while (Math.hypot(start[0], start[1]) > MAX_R);
        recompute();
      });
      slider.addEventListener('input', () => { stopPlay(); p = +slider.value; draw(); });
      slider.addEventListener('pointerup', () => slider.blur());

      syncChips();
      recompute();
      (window.W20Models ? window.W20Models() : Promise.reject(new Error('W20 not loaded'))).then((m) => {
        VAE = m; draw();
        if (window.Reveal && Reveal.layout) Reveal.layout();
      }).catch((err) => console.error('W21:', err));

      return { start() { draw(); }, stop() { stopPlay(); } };
    },
  });
})();
