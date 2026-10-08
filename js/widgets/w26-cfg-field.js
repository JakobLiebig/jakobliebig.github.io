// W26: classifier-free guidance as three vector fields in the 2D latent space.
//
// At every grid point, from the trained latent flow (data/latent/flow.json) at time data-t:
//   green  unconditional flow   v(z, t, ∅)
//   red    conditional flow     v(z, t, c)          (c = data-digit)
//   blue   guided flow          v(∅) + w (v(c) - v(∅))
// Arrows show the direction a point moves (-v, toward the data), scaled per grid point.
// A slider sets w. Faint grey dots: the encoded test digits, brighter ones: digit c.

(function () {
  const GREEN = '#6cc070', RED = '#ef6b6b', BLUE = '#6fa8ff';

  Widgets.register('W26', {
    init(el, F) {
      el.innerHTML = '';
      el.classList.add('w26');
      const R = F.range, NG = F.grid.length, NT = F.times.length;
      const cond = el.dataset.digit || '3';
      const t = +(el.dataset.t || 0.6);
      let w = +(el.dataset.w || 2);

      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      // Zoomed window: center data-center, half-width data-view (latent units).
      const S = 380, VIEW = +(el.dataset.view || 1.2);
      const [CX, CY] = (el.dataset.center || '0.4,-0.5').split(',').map(Number);
      const cv = document.createElement('canvas');
      cv.width = cv.height = S * ratio; cv.style.width = cv.style.height = `${S}px`; cv.className = 'w26-plane';
      const ctx = cv.getContext('2d');

      const legend = document.createElement('div'); legend.className = 'w26-legend';
      legend.innerHTML = `<span><i style="background:${GREEN}"></i>unconditional</span>`
        + `<span><i style="background:${RED}"></i>conditional (digit ${cond})</span>`
        + `<span><i style="background:${BLUE}"></i>guided</span>`;
      const bar = document.createElement('div'); bar.className = 'w26-slider';
      bar.setAttribute('data-prevent-swipe', '');
      bar.innerHTML = '<span>guidance w</span><input type="range" min="0" max="5" step="0.1"><span class="w26-w"></span>';
      const slider = bar.querySelector('input'), wLabel = bar.querySelector('.w26-w');
      slider.value = String(w);
      el.append(cv, legend, bar);

      // Field at (x, y): bilinear in z, linear in t.
      function grid(label, k, x, y) {
        const M = F.fields[label][k];
        const gx = Math.max(0, Math.min(NG - 1 - 1e-9, ((x + R) / (2 * R)) * (NG - 1)));
        const gy = Math.max(0, Math.min(NG - 1 - 1e-9, ((y + R) / (2 * R)) * (NG - 1)));
        const c0 = Math.floor(gx), r0 = Math.floor(gy), fx = gx - c0, fy = gy - r0;
        return [0, 1].map((d) => M[r0][c0][d] * (1 - fx) * (1 - fy) + M[r0][c0 + 1][d] * fx * (1 - fy)
          + M[r0 + 1][c0][d] * (1 - fx) * fy + M[r0 + 1][c0 + 1][d] * fx * fy);
      }
      function field(label, x, y) {
        const ti = Math.max(0, Math.min(NT - 1 - 1e-9, (1 - t) * (NT - 1)));
        const k0 = Math.floor(ti), a = ti - k0, k1 = Math.min(NT - 1, k0 + 1);
        const p = grid(label, k0, x, y), q = grid(label, k1, x, y);
        return [p[0] * (1 - a) + q[0] * a, p[1] * (1 - a) + q[1] * a];
      }

      const toPxX = (v) => ((v - CX + VIEW) / (2 * VIEW)) * S, toPxY = (v) => ((v - CY + VIEW) / (2 * VIEW)) * S;
      const STEP = (2 * VIEW) / 5, CELL = (STEP / (2 * VIEW)) * S;
      const pts = [];
      for (let x = CX - VIEW + STEP / 2; x < CX + VIEW; x += STEP) for (let y = CY - VIEW + STEP / 2; y < CY + VIEW; y += STEP) {
        const vu = field('all', x, y), vc = field(cond, x, y);
        pts.push({ x, y, u: [-vu[0], -vu[1]], c: [-vc[0], -vc[1]] });
      }
      // Scale per grid point: the longer of green and red fills about 0.75 of a cell, blue uses the
      // same scale (so the geometry v(c) - v(∅) is exact at each point; sizes differ across points).
      pts.forEach((p) => { p.k = (0.75 * CELL) / (Math.max(Math.hypot(...p.u), Math.hypot(...p.c)) || 1); });

      function arrow(x0, y0, vx, vy, k, color, width) {
        let dx = vx * k, dy = -vy * k;
        const len = Math.hypot(dx, dy);
        if (len < 1.5) return;
        const cap = 1.15 * CELL;                      // keep huge guided arrows inside their cell's area
        if (len > cap) { dx *= cap / len; dy *= cap / len; }
        const L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L, hd = Math.min(13, L * 0.35);
        const x1 = x0 + dx, y1 = y0 + dy;
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * hd * 0.7, y1 - uy * hd * 0.7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - ux * hd - uy * hd * 0.55, y1 - uy * hd + ux * hd * 0.55);
        ctx.lineTo(x1 - ux * hd + uy * hd * 0.55, y1 - uy * hd - ux * hd * 0.55);
        ctx.closePath(); ctx.fill(); ctx.lineCap = 'butt';
      }

      let VAE = null;
      function draw() {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, cv.width, cv.height);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.fillStyle = '#171920'; ctx.beginPath(); ctx.roundRect(0, 0, S, S, 8); ctx.fill();
        if (VAE) {
          const { points, labels } = VAE.latents;
          for (let i = 0; i < points.length; i++) {
            const [a, b] = points[i];
            if (Math.abs(a - CX) > VIEW || Math.abs(b - CY) > VIEW) continue;
            ctx.fillStyle = String(labels[i]) === cond ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.07)';
            ctx.fillRect(toPxX(a) - 1.5, S - toPxY(b) - 1.5, 3, 3);
          }
        }
        for (const p of pts) {
          const x0 = toPxX(p.x), y0 = S - toPxY(p.y);
          const g = [p.u[0] + w * (p.c[0] - p.u[0]), p.u[1] + w * (p.c[1] - p.u[1])];
          // Widest underneath, so overlapping arrows stay visible (w = 0: blue under green, w = 1: under red).
          arrow(x0, y0, ...g, p.k, BLUE, 7);
          arrow(x0, y0, ...p.c, p.k, RED, 3.5);
          arrow(x0, y0, ...p.u, p.k, GREEN, 2.5);
          ctx.beginPath(); ctx.arc(x0, y0, 3, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fill();
        }
        wLabel.textContent = w.toFixed(1);
      }

      slider.addEventListener('input', () => { w = +slider.value; draw(); });
      slider.addEventListener('pointerup', () => slider.blur());
      draw();
      if (window.W20Models) window.W20Models().then((m) => { VAE = m; draw(); }).catch((e) => console.error('W26:', e));
      return { start() { draw(); }, stop() {} };
    },
  });
})();
