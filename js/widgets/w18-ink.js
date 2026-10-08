// W18: a drop of ink dissolving in water, as particles.
//
// Each particle follows the same kind of SDE as the diffusion slides: a deterministic part
// (the water slowly carries the ink down and around: sinking + a gentle divergence-free swirl)
// plus random kicks (Brownian motion from water molecules). The cloud's density obeys the
// Fokker-Planck equation. Paths are precomputed at fixed checkpoints, so the slider can scrub
// time in both directions. A few particles show their trails.

(function () {
  const N = 4000;
  const T_END = 12;         // simulated seconds
  const K = 120;            // checkpoints
  const SUB = 5;            // Euler-Maruyama substeps per checkpoint
  const SIGMA = 10;         // Brownian strength, px / sqrt(s)
  const SINK = 16;          // px / s
  const SWIRL = 22;         // px / s
  const TRAILS = [7, 401, 1203];

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

  Widgets.register('W18', {
    init(el) {
      const canvas = document.createElement('canvas');
      el.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const W = el.clientWidth || 1100, H = el.clientHeight || 500;
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      canvas.width = Math.round(W * ratio); canvas.height = Math.round(H * ratio);

      // The tank, leaving room for the slider at the bottom.
      const tank = { x: W * 0.2, y: 14, w: W * 0.6, h: H - 80 };
      const drop = [tank.x + tank.w / 2, tank.y + 50];

      // Water flow: sinking plus a swirl from the stream function psi = sin(k1 x) sin(k2 y).
      const k1 = (2 * Math.PI) / tank.w, k2 = Math.PI / tank.h;
      const flow = (x, y) => {
        const X = x - tank.x, Y = y - tank.y;
        return [
          SWIRL * Math.sin(k1 * X) * Math.cos(k2 * Y),
          -SWIRL * (k1 / k2) * Math.cos(k1 * X) * Math.sin(k2 * Y) + SINK,
        ];
      };

      // Precompute positions at every checkpoint.
      const pos = new Float32Array((K + 1) * N * 2);
      {
        const g = rng(77);
        const dt = T_END / K / SUB, sq = Math.sqrt(dt) * SIGMA;
        const xs = new Float32Array(N), ys = new Float32Array(N);
        for (let i = 0; i < N; i++) { xs[i] = drop[0] + 7 * g(); ys[i] = drop[1] + 7 * g(); }
        const lo = [tank.x + 4, tank.y + 4], hi = [tank.x + tank.w - 4, tank.y + tank.h - 4];
        const reflect = (v, a, b) => (v < a ? 2 * a - v : v > b ? 2 * b - v : v);
        for (let k = 0; k <= K; k++) {
          for (let i = 0; i < N; i++) { pos[(k * N + i) * 2] = xs[i]; pos[(k * N + i) * 2 + 1] = ys[i]; }
          if (k === K) break;
          for (let sub = 0; sub < SUB; sub++) {
            for (let i = 0; i < N; i++) {
              const [u, v] = flow(xs[i], ys[i]);
              xs[i] = reflect(xs[i] + u * dt + sq * g(), lo[0], hi[0]);
              ys[i] = reflect(ys[i] + v * dt + sq * g(), lo[1], hi[1]);
            }
          }
        }
      }
      const P = (k, i) => [pos[(k * N + i) * 2], pos[(k * N + i) * 2 + 1]];
      const at = (t, i) => {
        const f = Math.max(0, Math.min(K, (t / T_END) * K));
        const k = Math.min(K - 1, Math.floor(f)), r = f - k;
        const [ax, ay] = P(k, i), [bx, by] = P(k + 1, i);
        return [ax + (bx - ax) * r, ay + (by - ay) * r];
      };

      // A soft orange dot, drawn additively so dense ink glows and thin ink fades.
      const sprite = document.createElement('canvas');
      const SR = Math.round(9 * ratio);
      sprite.width = sprite.height = SR * 2;
      {
        const s = sprite.getContext('2d');
        const gr = s.createRadialGradient(SR, SR, 0, SR, SR, SR);
        gr.addColorStop(0, 'rgba(242,153,74,1)');
        gr.addColorStop(1, 'rgba(242,153,74,0)');
        s.fillStyle = gr; s.fillRect(0, 0, SR * 2, SR * 2);
      }

      let t = 0;
      function draw() {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        // Water
        ctx.fillStyle = '#141824';
        ctx.strokeStyle = 'rgba(138,155,180,0.45)'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.roundRect(tank.x, tank.y, tank.w, tank.h, 14); ctx.fill(); ctx.stroke();
        ctx.save();
        ctx.beginPath(); ctx.roundRect(tank.x, tank.y, tank.w, tank.h, 14); ctx.clip();
        // Ink
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = 0.07;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        for (let i = 0; i < N; i++) {
          const [x, y] = at(t, i);
          ctx.drawImage(sprite, x * ratio - SR, y * ratio - SR);
        }
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
        // A few particles: their jagged paths (drift + kicks).
        const kNow = Math.round((t / T_END) * K);
        for (const i of TRAILS) {
          ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.4; ctx.lineJoin = 'round';
          ctx.beginPath();
          for (let k = 0; k <= kNow; k++) { const [x, y] = P(k, i); if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
          const [x, y] = at(t, i);
          ctx.lineTo(x, y); ctx.stroke();
          ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
        }
        ctx.restore();
      }

      // Slider + autoplay on entering the slide.
      const bar = document.createElement('div');
      bar.className = 'w15-controls';
      bar.setAttribute('data-prevent-swipe', '');
      bar.innerHTML = `<span class="w15-end">drop</span><input type="range" min="0" max="${T_END}" step="0.01" value="0" aria-label="time"><span class="w15-end">dissolved</span>`;
      el.appendChild(bar);
      const input = bar.querySelector('input');
      let raf = 0, t0 = 0, from = 0;
      const PLAY_MS = 9000;
      function tick(now) {
        const p = Math.min(1, (now - t0) / PLAY_MS);
        t = from + (T_END - from) * p;
        input.value = String(t);
        draw();
        raf = p < 1 ? requestAnimationFrame(tick) : 0;
      }
      input.addEventListener('input', () => {
        if (raf) { cancelAnimationFrame(raf); raf = 0; }
        t = +input.value; draw();
      });
      input.addEventListener('pointerup', () => input.blur());

      draw();
      return {
        start() {
          if (t <= 0 && !raf) { from = 0; t0 = performance.now() + 400; raf = requestAnimationFrame(tick); }
          else draw();
        },
        stop() { if (raf) cancelAnimationFrame(raf); raf = 0; },
      };
    },
  });
})();
