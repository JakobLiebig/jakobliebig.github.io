// W28: Brownian particles drifting behind the opening image.
//
// A canvas filling the slide, behind the image. Every particle follows dx = f dt + sigma dW: a slow, gently
// varying drift (the water) plus Gaussian kicks (the molecules). Like real Brownian motion, the
// kicks shrink with particle size (diffusion ~ 1 / radius), so small particles jiggle more.
// Particles that leave the slide come back on the other side. Runs only while the slide shows.

(function () {
  const N = 220;
  const DRIFT = [7, -4];   // px per second, the water's slow current
  const SIGMA = 26;        // kick strength for the smallest particle, px per sqrt(second)

  function gauss() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  Widgets.register('W28', {
    init(el) {
      const cv = document.createElement('canvas');
      cv.className = 'w28-canvas';
      el.appendChild(cv);
      const ctx = cv.getContext('2d');
      let W = 0, H = 0, ratio = 1, raf = 0, last = 0, t = 0;
      let parts = [];

      function resize() {
        // Layout size in slide coordinates (Reveal scales the whole slide afterwards).
        W = el.offsetWidth; H = el.offsetHeight;
        ratio = Math.min(2, window.devicePixelRatio || 1);
        cv.width = Math.round(W * ratio); cv.height = Math.round(H * ratio);
        if (!parts.length && W) {
          parts = Array.from({ length: N }, () => {
            const r = 0.8 + 2.4 * Math.random() ** 2;   // mostly small, a few large
            return { x: Math.random() * W, y: Math.random() * H, r, a: 0.25 + 0.5 * Math.random() };
          });
        }
      }

      function step(dt) {
        t += dt;
        const fx = DRIFT[0] * (1 + 0.5 * Math.sin(0.23 * t)), fy = DRIFT[1] * (1 + 0.5 * Math.cos(0.17 * t));
        const sq = Math.sqrt(dt);
        for (const p of parts) {
          const s = SIGMA / Math.sqrt(p.r);
          p.x += fx * dt + s * sq * gauss();
          p.y += fy * dt + s * sq * gauss();
          if (p.x < -5) p.x += W + 10; else if (p.x > W + 5) p.x -= W + 10;
          if (p.y < -5) p.y += H + 10; else if (p.y > H + 5) p.y -= H + 10;
        }
      }

      function draw() {
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, W, H);
        for (const p of parts) {
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 236, 210, ${0.12 * p.a})`; ctx.fill();
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 244, 230, ${p.a})`; ctx.fill();
        }
      }

      function frame(now) {
        const dt = Math.min(0.05, (now - last) / 1000 || 0);
        last = now;
        step(dt); draw();
        raf = requestAnimationFrame(frame);
      }

      resize();

      return {
        start() {
          resize();
          cancelAnimationFrame(raf);
          last = performance.now();
          raf = requestAnimationFrame(frame);
        },
        stop() { cancelAnimationFrame(raf); raf = 0; },
      };
    },
  });
})();
