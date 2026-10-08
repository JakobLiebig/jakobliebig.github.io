// W22: a still of latent diffusion for the overview slide.
//
// Noise in N(0, I) -> one path of the trained latent flow (data/latent/flow.json, digit
// data-digit, path data-path) -> decoder -> the decoded image. The latent plane shows the encoded
// digits as faint clouds and N(0, I) as dashed circles, like W21. Uses the W20 models for the
// latents and the decoder.

(function () {
  const FALLBACK_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];

  Widgets.register('W22', {
    init(el, F) {
      el.innerHTML = '';
      const R = F.range;
      const digit = el.dataset.digit || '0';
      const path = F.paths[digit][+(el.dataset.path || 0)];
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const W = el.clientWidth || 560, H = el.clientHeight || 190;
      const cv = document.createElement('canvas');
      cv.width = W * ratio; cv.height = H * ratio;
      cv.style.width = `${W}px`; cv.style.height = `${H}px`;
      el.appendChild(cv);
      const c = cv.getContext('2d');

      // Layout: [plane] -> [decoder] -> [image], with small labels underneath.
      const P = H, px0 = 0, py0 = 0;                 // plane square
      const decX = P + 28, decW = 80;                 // decoder trapezoid
      const imgX = decX + decW + 22, IMG = P;          // decoded image
      const toX = (v) => px0 + ((v + R) / (2 * R)) * P;
      const toY = (v) => py0 + P - ((v + R) / (2 * R)) * P;

      function draw(VAE) {
        c.setTransform(ratio, 0, 0, ratio, 0, 0);
        c.clearRect(0, 0, W, H);
        c.fillStyle = '#171920';
        c.beginPath(); c.roundRect(px0, py0, P, P, 6); c.fill();
        if (VAE) {
          const colors = (window.d3 && d3.schemeTableau10) || FALLBACK_COLORS;
          const { points, labels } = VAE.latents;
          c.globalAlpha = 0.3;
          for (let i = 0; i < points.length; i += 2) {
            const [a, b] = points[i];
            if (Math.abs(a) > R || Math.abs(b) > R) continue;
            c.fillStyle = colors[labels[i] % 10];
            c.fillRect(toX(a) - 0.8, toY(b) - 0.8, 1.6, 1.6);
          }
          c.globalAlpha = 1;
        }
        c.strokeStyle = 'rgba(138,155,180,0.75)'; c.setLineDash([4, 4]); c.lineWidth = 1.2;
        for (const r of [1, 2]) { c.beginPath(); c.arc(toX(0), toY(0), (r / (2 * R)) * P, 0, Math.PI * 2); c.stroke(); }
        c.setLineDash([]);
        // The path: noise (blue-grey ring) -> latent (orange dot).
        c.strokeStyle = '#fff'; c.lineWidth = 2; c.lineJoin = 'round';
        c.beginPath();
        path.forEach(([x, y], i) => (i ? c.lineTo(toX(x), toY(y)) : c.moveTo(toX(x), toY(y))));
        c.stroke();
        const [sx, sy] = path[0], [ex, ey] = path[path.length - 1];
        c.strokeStyle = 'rgb(138,155,180)'; c.lineWidth = 2;
        c.beginPath(); c.arc(toX(sx), toY(sy), 5, 0, Math.PI * 2); c.stroke();
        c.fillStyle = '#f2994a'; c.beginPath(); c.arc(toX(ex), toY(ey), 4.5, 0, Math.PI * 2); c.fill();

        // Arrow plane -> decoder, then the decoder trapezoid (narrow -> wide).
        const mid = P / 2;
        c.strokeStyle = 'rgba(232,233,238,0.6)'; c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(P + 6, mid); c.lineTo(decX - 4, mid); c.stroke();
        c.fillStyle = 'rgba(242,153,74,0.16)'; c.strokeStyle = 'rgba(242,153,74,0.6)';
        c.beginPath();
        c.moveTo(decX, mid - P * 0.12); c.lineTo(decX + decW, mid - P * 0.42);
        c.lineTo(decX + decW, mid + P * 0.42); c.lineTo(decX, mid + P * 0.12); c.closePath();
        c.fill(); c.stroke();
        c.fillStyle = '#e8e9ee'; c.font = '600 13px Inter, system-ui, sans-serif'; c.textAlign = 'center';
        c.fillText('Decoder', decX + decW / 2, mid + 4);

        // Decoded image of the end point.
        c.fillStyle = '#000'; c.fillRect(imgX, py0, IMG, IMG);
        if (VAE) {
          const out = tf.tidy(() => VAE.dec.predict(tf.tensor([[ex, ey]], [1, 2])).dataSync());
          const small = document.createElement('canvas'); small.width = small.height = 28;
          const sc = small.getContext('2d'), img = sc.createImageData(28, 28);
          for (let i = 0; i < 784; i++) {
            const v = Math.max(0, Math.min(255, out[i] * 255));
            img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
          }
          sc.putImageData(img, 0, 0);
          c.imageSmoothingEnabled = false;
          c.drawImage(small, imgX, py0, IMG, IMG);
        }
      }

      draw(null);
      if (window.W20Models) window.W20Models().then(draw).catch((e) => console.error('W22:', e));
      return { start() {}, stop() {} };
    },
  });
})();
