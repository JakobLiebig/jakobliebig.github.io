// W24: latent diffusion in one picture: N(0, I) -> latent space -> image space.
//
// One example from the trained latent flow (data/latent/flow.json, digit data-digit, path data-path):
// its noise point on a 3D Gaussian (left), its flowed code on the 3D density of the encoded MNIST
// test digits (middle; histogram of the W20 latents, blurred), and the W20 decoder's image of that
// code (right).

(function () {
  const C_DATA = [242, 153, 74], C_NOISE = [138, 155, 180], C_LOW = [30, 35, 47];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const E = 3.4, G = 56;                     // surface over [-E, E]^2, G x G quads
  const AZ = -0.3, EL = 0.62, DIST = 12;

  // Draw a 3D surface z = hf(x, y) (heights in [0, 1]) with one marked point.
  function surface(cv, W, H, ratio, hf, point) {
    const ctx = cv.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const HT = 1.3;
    const height = (x, y) => HT * hf(x, y);
    const ca = Math.cos(AZ), sa = Math.sin(AZ), ce = Math.cos(EL), se = Math.sin(EL);
    const view = (x, y, z) => {
      const xr = x * ca - y * sa, yr = x * sa + y * ca;
      const up = z * ce + yr * se, depth = yr * ce - z * se, f = DIST / (DIST + depth);
      return [xr * f, up * f, depth];
    };
    const verts = [];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i <= G; i++) {
      const row = [];
      for (let j = 0; j <= G; j++) {
        const x = -E + (2 * E * i) / G, y = -E + (2 * E * j) / G, z = height(x, y), v = view(x, y, z);
        x0 = Math.min(x0, v[0]); x1 = Math.max(x1, v[0]); y0 = Math.min(y0, v[1]); y1 = Math.max(y1, v[1]);
        row.push({ x, y, z, v });
      }
      verts.push(row);
    }
    const pad = 6;
    const sc = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    const ox = W / 2 - ((x0 + x1) / 2) * sc, oy = H / 2 + ((y0 + y1) / 2) * sc;
    const S = (v) => [ox + v[0] * sc, oy - v[1] * sc];
    const quads = [];
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const q = [verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]];
      quads.push({ q, d: (q[0].v[2] + q[1].v[2] + q[2].v[2] + q[3].v[2]) / 4, i, j });
    }
    quads.sort((a, b) => b.d - a.d);
    const L = [-0.5, -0.6, 0.8], ln = Math.hypot(...L);
    for (const { q, i, j } of quads) {
      const [a, b, c, d] = q;
      const ux = c.x - a.x, uy = c.y - a.y, uz = c.z - a.z, wx = d.x - b.x, wy = d.y - b.y, wz = d.z - b.z;
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const nn = Math.hypot(nx, ny, nz) || 1; nx /= nn; ny /= nn; nz /= nn; if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
      const lam = Math.max(0, (nx * L[0] + ny * L[1] + nz * L[2]) / ln);
      const h = (a.z + b.z + c.z + d.z) / (4 * HT);
      const base = mix(C_LOW, mix(C_NOISE, C_DATA, Math.min(1, h * 1.4)), Math.min(1, 0.25 + h * 1.6));
      const col = rgb(base.map((v) => Math.min(255, v * (0.55 + 0.55 * lam))));
      const pts = q.map((p) => S(p.v));
      ctx.beginPath(); ctx.moveTo(...pts[0]); for (let k = 1; k < 4; k++) ctx.lineTo(...pts[k]); ctx.closePath();
      ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 0.8; ctx.fill(); ctx.stroke();
      if (i % 7 === 0 || j % 7 === 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.6; ctx.beginPath();
        if (i % 7 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[3]); }
        if (j % 7 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); }
        ctx.stroke();
      }
    }
    if (point) {
      const [px, py] = S(view(point[0], point[1], height(point[0], point[1]) + 0.03));
      ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fillStyle = '#0e0f13'; ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
    }
  }

  // Density of the encoded digits on a grid: 2D histogram, then a separable Gaussian blur.
  function latentDensity(points) {
    const N = 80, h = new Float64Array(N * N);
    for (const [a, b] of points) {
      const i = Math.floor(((a + E) / (2 * E)) * N), j = Math.floor(((b + E) / (2 * E)) * N);
      if (i >= 0 && i < N && j >= 0 && j < N) h[j * N + i] += 1;
    }
    const sig = 2.2, R = 7, k = [];
    for (let d = -R; d <= R; d++) k.push(Math.exp(-(d * d) / (2 * sig * sig)));
    const tmp = new Float64Array(N * N), out = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let s = 0; for (let d = -R; d <= R; d++) { const ii = i + d; if (ii >= 0 && ii < N) s += h[j * N + ii] * k[d + R]; } tmp[j * N + i] = s;
    }
    let mx = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let s = 0; for (let d = -R; d <= R; d++) { const jj = j + d; if (jj >= 0 && jj < N) s += tmp[jj * N + i] * k[d + R]; } out[j * N + i] = s; mx = Math.max(mx, s);
    }
    return (x, y) => {
      const gx = Math.max(0, Math.min(N - 1.001, ((x + E) / (2 * E)) * N - 0.5));
      const gy = Math.max(0, Math.min(N - 1.001, ((y + E) / (2 * E)) * N - 0.5));
      const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j;
      const v = out[j * N + i] * (1 - fx) * (1 - fy) + out[j * N + i + 1] * fx * (1 - fy) + out[(j + 1) * N + i] * (1 - fx) * fy + out[(j + 1) * N + i + 1] * fx * fy;
      return v / mx;
    };
  }

  Widgets.register('W24', {
    init(el, F) {
      el.innerHTML = '';
      el.classList.add('w24');
      const path = F.paths[el.dataset.digit || '0'][+(el.dataset.path || 3)];
      const z0 = path[0], z1 = path[path.length - 1];
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const PW = 300, PH = 210;
      const mkCv = (w, h, cls) => { const c = document.createElement('canvas'); c.width = w * ratio; c.height = h * ratio; c.style.width = `${w}px`; c.style.height = `${h}px`; c.className = cls; return c; };
      const panel = (title, ...kids) => {
        const col = document.createElement('div'); col.className = 'w24-col';
        const lab = document.createElement('div'); lab.className = 'w20-label'; lab.textContent = title;
        col.append(lab, ...kids);
        return col;
      };
      const arrow = (txt) => { const a = document.createElement('div'); a.className = 'w24-arrow'; a.innerHTML = `<span>${txt}</span><div>→</div>`; return a; };

      const noiseCv = mkCv(PW, PH, 'w24-plane'), latCv = mkCv(PW, PH, 'w24-plane');
      const img = document.createElement('canvas'); img.width = img.height = 28; img.className = 'w24-image';
      el.append(panel('Noise', noiseCv), arrow('flow matching'), panel('Latent space', latCv), arrow('decoder'), panel('Image space', img));

      const gauss = (x, y) => Math.exp(-(x * x + y * y) / 2);
      surface(noiseCv, PW, PH, ratio, gauss, z0);
      surface(latCv, PW, PH, ratio, gauss, null);     // placeholder until the latents are loaded

      if (window.W20Models) window.W20Models().then((VAE) => {
        surface(latCv, PW, PH, ratio, latentDensity(VAE.latents.points), z1);
        const out = tf.tidy(() => VAE.dec.predict(tf.tensor([z1], [1, 2])).dataSync());
        const c = img.getContext('2d'), im = c.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
          const v = Math.max(0, Math.min(255, out[i] * 255));
          im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255;
        }
        c.putImageData(im, 0, 0);
      }).catch((e) => console.error('W24:', e));
      return { start() {}, stop() {} };
    },
  });
})();
