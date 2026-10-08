// W23: conditioning, one noise point and ten labels.
//
// From the same start point in the 2D latent space of the W20 VAE, the latent flow (flow.json) is
// integrated once per digit label c = 0..9 with classifier-free guidance
//   v_w = v_any + w (v_c - v_any)
// (w = 1 is the plain conditional field, w = 0 the unconditional one). Each path ends in a code that
// the W20 decoder turns into an image. A counter shows how many end points sit among real codes of
// the requested digit (label of the 10 nearest encoded test digits).
//
// data-w: guidance weight (default 1). data-slider="1": show a slider for w (0..6).
// data-still="1": compact, no labels (overview slide). data-start="x,y": start point.

(function () {
  const STEPS = 40;
  const FALLBACK_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];

  Widgets.register('W23', {
    init(el, F) {
      el.innerHTML = '';
      el.classList.add('w23');
      const R = F.range, NG = F.grid.length, NT = F.times.length;
      const still = el.dataset.still === '1';
      const start = (el.dataset.start || '0.05,-0.25').split(',').map(Number);
      let w = +(el.dataset.w || 1);
      const colors = (window.d3 && d3.schemeTableau10) || FALLBACK_COLORS;

      // ---- Layout ----
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const S = still ? (+el.dataset.size || 140) : 300;
      const plane = document.createElement('canvas');
      plane.width = plane.height = S * ratio; plane.style.width = plane.style.height = `${S}px`;
      plane.className = 'w23-plane';
      const ctx = plane.getContext('2d');
      const thumbs = document.createElement('div'); thumbs.className = 'w23-thumbs';
      const tiles = [];
      for (let k = 0; k < 10; k++) {
        const fig = document.createElement('div'); fig.className = 'w23-tile';
        const cv = document.createElement('canvas'); cv.width = cv.height = 28;
        cv.style.borderColor = colors[k];
        const cap = document.createElement('span'); cap.textContent = `c = ${k}`;
        fig.append(cv); if (!still) fig.append(cap);
        thumbs.append(fig); tiles.push({ cv, fig });
      }
      const right = document.createElement('div'); right.className = 'w23-right';
      const score = document.createElement('div'); score.className = 'w23-score';
      right.append(thumbs);
      if (!still) right.append(score);
      let slider = null, wLabel = null;
      if (el.dataset.slider === '1') {
        const bar = document.createElement('div'); bar.className = 'w23-slider';
        bar.setAttribute('data-prevent-swipe', '');
        bar.innerHTML = '<span>guidance w</span><input type="range" min="0" max="6" step="0.1"><span class="w23-w"></span>';
        slider = bar.querySelector('input'); wLabel = bar.querySelector('.w23-w');
        slider.value = String(w);
        right.append(bar);
      }
      el.append(plane, right);

      // ---- Field: bilinear in z, linear in t, guided ----
      function grid(label, k, x, y) {
        const M = F.fields[label][k];
        const gx = Math.max(0, Math.min(NG - 1 - 1e-9, ((x + R) / (2 * R)) * (NG - 1)));
        const gy = Math.max(0, Math.min(NG - 1 - 1e-9, ((y + R) / (2 * R)) * (NG - 1)));
        const c0 = Math.floor(gx), r0 = Math.floor(gy), fx = gx - c0, fy = gy - r0;
        return [0, 1].map((d) => M[r0][c0][d] * (1 - fx) * (1 - fy) + M[r0][c0 + 1][d] * fx * (1 - fy)
          + M[r0 + 1][c0][d] * (1 - fx) * fy + M[r0 + 1][c0 + 1][d] * fx * fy);
      }
      function field(label, x, y, t) {
        const ti = Math.max(0, Math.min(NT - 1 - 1e-9, (1 - t) * (NT - 1)));
        const k0 = Math.floor(ti), a = ti - k0, k1 = Math.min(NT - 1, k0 + 1);
        const at = (lb) => { const p = grid(lb, k0, x, y), q = grid(lb, k1, x, y); return [p[0] * (1 - a) + q[0] * a, p[1] * (1 - a) + q[1] * a]; };
        const vu = at('all'), vc = at(label);
        return [vu[0] + w * (vc[0] - vu[0]), vu[1] + w * (vc[1] - vu[1])];
      }
      function integrate(label) {
        const dt = 1 / STEPS, path = [start.slice()];
        let [x, y] = start, t = 1;
        for (let i = 0; i < STEPS; i++) {
          const k1 = field(label, x, y, t);
          const k2 = field(label, x - dt / 2 * k1[0], y - dt / 2 * k1[1], t - dt / 2);
          const k3 = field(label, x - dt / 2 * k2[0], y - dt / 2 * k2[1], t - dt / 2);
          const k4 = field(label, x - dt * k3[0], y - dt * k3[1], t - dt);
          x -= dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
          y -= dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
          t -= dt;
          path.push([x, y]);
        }
        return path;
      }

      // ---- Drawing ----
      let VAE = null, base = null;
      const toPx = (v) => ((v + R) / (2 * R)) * S;
      function buildBase() {
        base = document.createElement('canvas'); base.width = base.height = S * ratio;
        const c = base.getContext('2d');
        c.setTransform(ratio, 0, 0, ratio, 0, 0);
        c.fillStyle = '#171920'; c.beginPath(); c.roundRect(0, 0, S, S, 6); c.fill();
        if (VAE) {
          const { points, labels } = VAE.latents;
          c.globalAlpha = 0.22;
          for (let i = 0; i < points.length; i++) {
            const [a, b] = points[i];
            if (Math.abs(a) > R || Math.abs(b) > R) continue;
            c.fillStyle = colors[labels[i] % 10];
            c.fillRect(toPx(a) - 0.9, S - toPx(b) - 0.9, 1.8, 1.8);
          }
          c.globalAlpha = 1;
        }
      }
      // Label of the 10 nearest encoded test digits.
      function knnLabel(x, y) {
        const { points, labels } = VAE.latents;
        const best = [];
        for (let i = 0; i < points.length; i++) {
          const d = (points[i][0] - x) ** 2 + (points[i][1] - y) ** 2;
          if (best.length < 10 || d < best[best.length - 1][0]) {
            best.push([d, labels[i]]); best.sort((p, q) => p[0] - q[0]); if (best.length > 10) best.pop();
          }
        }
        const counts = new Array(10).fill(0); best.forEach(([, l]) => counts[l]++);
        return counts.indexOf(Math.max(...counts));
      }

      function render() {
        const paths = Array.from({ length: 10 }, (_, k) => integrate(String(k)));
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, plane.width, plane.height);
        if (base) ctx.drawImage(base, 0, 0);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        paths.forEach((p, k) => {
          ctx.strokeStyle = colors[k]; ctx.lineWidth = still ? 1.6 : 2.2; ctx.lineJoin = 'round';
          ctx.beginPath();
          p.forEach(([x, y], i) => (i ? ctx.lineTo(toPx(x), S - toPx(y)) : ctx.moveTo(toPx(x), S - toPx(y))));
          ctx.stroke();
          const [ex, ey] = p[STEPS];
          ctx.beginPath(); ctx.arc(toPx(ex), S - toPx(ey), still ? 3 : 4.5, 0, Math.PI * 2);
          ctx.fillStyle = colors[k]; ctx.fill(); ctx.strokeStyle = '#0e0f13'; ctx.lineWidth = 1.5; ctx.stroke();
        });
        ctx.beginPath(); ctx.arc(toPx(start[0]), S - toPx(start[1]), still ? 4 : 6, 0, Math.PI * 2);
        ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#0e0f13'; ctx.lineWidth = 2; ctx.stroke();
        if (wLabel) { wLabel.textContent = w.toFixed(1); slider.value = String(w); }
        if (!VAE) return;
        let hits = 0;
        paths.forEach((p, k) => {
          const [ex, ey] = p[STEPS];
          const out = tf.tidy(() => VAE.dec.predict(tf.tensor([[ex, ey]], [1, 2])).dataSync());
          const c = tiles[k].cv.getContext('2d'), img = c.createImageData(28, 28);
          for (let i = 0; i < 784; i++) {
            const v = Math.max(0, Math.min(255, out[i] * 255));
            img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
          }
          c.putImageData(img, 0, 0);
          const ok = knnLabel(ex, ey) === k;
          hits += ok;
          tiles[k].fig.classList.toggle('miss', !ok);
        });
        score.textContent = `${hits} of 10 land on their digit`;
      }

      if (slider) {
        slider.addEventListener('input', () => { w = +slider.value; render(); });
        slider.addEventListener('pointerup', () => slider.blur());
      }
      buildBase(); render();
      if (window.W20Models) window.W20Models().then((m) => { VAE = m; buildBase(); render(); if (window.Reveal && Reveal.layout) Reveal.layout(); }).catch((e) => console.error('W23:', e));
      return { start() {}, stop() {} };
    },
  });
})();
