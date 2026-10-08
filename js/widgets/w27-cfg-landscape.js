// W27: classifier-free guidance on the dog/cat landscape of the conditioning slide (W25).
//
// A mixture of two Gaussians, a dog peak and a cat peak, as a 3D surface. At a few points:
//   green  unconditional flow: toward the weighted average of both peaks, i.e. up to the pass
//   red    conditional flow for "a picture of a cat": toward the cat peak
//   blue   guided flow: green + w (red - green)
// A slider sets w. Uses W25's surface renderer (window.W25).

(function () {
  const GREEN = '#6cc070', RED = '#ef6b6b', BLUE = '#6fa8ff';

  Widgets.register('W27', {
    init(el) {
      el.innerHTML = '';
      el.classList.add('w27');
      const { surface, loadImage, DOG, CAT } = window.W25;
      let w = +(el.dataset.w || 2);
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const W = 640, H = 330;
      const cv = document.createElement('canvas');
      cv.width = W * ratio; cv.height = H * ratio; cv.style.width = `${W}px`; cv.style.height = `${H}px`;
      const legend = document.createElement('div'); legend.className = 'w26-legend';
      legend.innerHTML = `<span><i style="background:${GREEN}"></i>unconditional</span>`
        + `<span><i style="background:${RED}"></i>conditional (“a picture of a cat”)</span>`
        + `<span><i style="background:${BLUE}"></i>guided</span>`;
      const bar = document.createElement('div'); bar.className = 'w26-slider';
      bar.setAttribute('data-prevent-swipe', '');
      bar.innerHTML = '<span>guidance w</span><input type="range" min="0" max="3" step="0.05"><span class="w26-w"></span>';
      const slider = bar.querySelector('input'), wLabel = bar.querySelector('.w26-w');
      slider.value = String(w);
      el.append(cv, legend);
      // The slider goes below the formula (which is part of the slide, after this widget).
      const formula = el.closest('section') && el.closest('section').querySelector('.cfg-formula');
      if (formula) formula.after(bar); else el.append(bar);

      // One start point in front, where middle and cat lie in clearly different directions.
      const STARTS = [[-0.55, -2.0]];
      // The landscape at intermediate noise: each peak blurred from 0.5 to SB, so the two merge into a
      // ridge with a saddle (a pass) in the middle. The middle is lower than both peaks: a dog/cat mix
      // is not a likely image, but from the front the pass is the way uphill.
      const SB = 0.8;
      const comps = [{ m: DOG, s: SB, w: 0.5 }, { m: CAT, s: SB, w: 0.5 }];
      // Unconditional flow: toward E[x0 | x_t], the responsibility-weighted average of both peaks.
      const posteriorMean = (x, y) => {
        const r = [DOG, CAT].map((m) => Math.exp(-((x - m[0]) ** 2 + (y - m[1]) ** 2) / (2 * SB * SB)));
        const z = r[0] + r[1];
        return [(r[0] * DOG[0] + r[1] * CAT[0]) / z, (r[0] * DOG[1] + r[1] * CAT[1]) / z];
      };
      let imgs = {};

      function arrow3(ctx, P, from, vec, color, width) {
        const K = 0.62;                                 // world length per unit of vec
        const to = [from[0] + vec[0] * K, from[1] + vec[1] * K];
        const [ax, ay] = P(...from, 0.03), [bx, by] = P(...to, 0.03);
        const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
        if (L < 3) return;
        const ux = dx / L, uy = dy / L, hd = Math.min(24, L * 0.3);
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'rgba(14,15,19,0.8)'; ctx.lineWidth = width + 3;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx - ux * hd * 0.7, by - uy * hd * 0.7); ctx.stroke();
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx - ux * hd * 0.7, by - uy * hd * 0.7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx, by);
        ctx.lineTo(bx - ux * hd - uy * hd * 0.55, by - uy * hd + ux * hd * 0.55);
        ctx.lineTo(bx - ux * hd + uy * hd * 0.55, by - uy * hd - ux * hd * 0.55);
        ctx.closePath(); ctx.fill(); ctx.lineCap = 'butt';
      }

      function draw() {
        surface(cv, W, H, ratio, comps, {
          cards: [
            { at: DOG, img: imgs.dog, likely: false, label: '', side: 'left' },
            { at: CAT, img: imgs.cat, likely: true, label: '', side: 'right' },
          ],
          draw(ctx, P) {
            for (const s of STARTS) {
              const pm = posteriorMean(...s);
              const u = [pm[0] - s[0], pm[1] - s[1]], c = [CAT[0] - s[0], CAT[1] - s[1]];
              const g = [u[0] + w * (c[0] - u[0]), u[1] + w * (c[1] - u[1])];
              // Widest underneath, so overlapping arrows stay visible.
              arrow3(ctx, P, s, g, BLUE, 10);
              arrow3(ctx, P, s, c, RED, 6);
              arrow3(ctx, P, s, u, GREEN, 4);
              const [px, py] = P(...s, 0.03);
              ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
            }
          },
        });
        wLabel.textContent = w.toFixed(2);
      }
      slider.addEventListener('input', () => { w = +slider.value; draw(); });
      slider.addEventListener('pointerup', () => slider.blur());
      draw();
      Promise.all([loadImage('media/peaks/pug.jpg'), loadImage('media/peaks/cat.jpg')]).then(([dog, cat]) => { imgs = { dog, cat }; draw(); });
      return { start() { draw(); }, stop() {} };
    },
  });
})();
