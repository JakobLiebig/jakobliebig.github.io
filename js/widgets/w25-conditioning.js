// W25: conditioning as a picture. Two prompts, one model, two different target distributions.
//
//   [noise, "A picture of a dog"] ->         -> [dog peak high (likely), cat peak low (unlikely)]
//                                     Model
//   [noise, "A picture of a cat"] ->         -> [cat peak high (likely), dog peak low (unlikely)]
//
// All four distributions are small 3D surfaces in the style of the landscape slides. On the target
// distributions, arrows show where the flow pushes: mostly toward the likely peak. Photos (Unsplash,
// see media/peaks/CREDITS.md) mark the two peaks.

(function () {
  const C_DATA = [242, 153, 74], C_NOISE = [138, 155, 180], C_LOW = [30, 35, 47];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const EX = 3, EY = 2.1, GX = 60, GY = 42;
  const AZ = -0.25, EL = 0.55, DIST = 12;
  const DOG = [-1.45, 0.35], CAT = [1.5, -0.25];

  function loadImage(src) {
    return new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = src; });
  }

  // comps: [{ m: [x, y], s, w }]
  function density(comps, x, y) {
    let p = 0;
    for (const c of comps) p += (c.w / (c.s * c.s)) * Math.exp(-((x - c.m[0]) ** 2 + (y - c.m[1]) ** 2) / (2 * c.s * c.s));
    return p;
  }

  // Draw one 3D surface into a canvas. opts: { arrows: bool, cards: [{ at: [x, y], img, label }] }
  function surface(cv, W, H, ratio, comps, opts = {}) {
    const ctx = cv.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, W, H);
    let pmax = 0;
    for (const c of comps) pmax = Math.max(pmax, density(comps, ...c.m));
    const HT = 1.25;
    const height = (x, y) => (HT * density(comps, x, y)) / pmax;
    const ca = Math.cos(AZ), sa = Math.sin(AZ), ce = Math.cos(EL), se = Math.sin(EL);
    const view = (x, y, z) => {
      const xr = x * ca - y * sa, yr = x * sa + y * ca;
      const up = z * ce + yr * se, depth = yr * ce - z * se, f = DIST / (DIST + depth);
      return [xr * f, up * f, depth];
    };
    // Fit, leaving room on top for the photo cards.
    const verts = [];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i <= GX; i++) {
      const row = [];
      for (let j = 0; j <= GY; j++) {
        const x = -EX + (2 * EX * i) / GX, y = -EY + (2 * EY * j) / GY, z = height(x, y);
        const v = view(x, y, z);
        x0 = Math.min(x0, v[0]); x1 = Math.max(x1, v[0]); y0 = Math.min(y0, v[1]); y1 = Math.max(y1, v[1]);
        row.push({ x, y, z, v });
      }
      verts.push(row);
    }
    const top = opts.cards ? 74 : 8, pad = 6;
    const sc = Math.min((W - 2 * pad) / (x1 - x0), (H - top - pad) / (y1 - y0));
    const ox = W / 2 - ((x0 + x1) / 2) * sc, oy = top + (H - top - pad - (y1 - y0) * sc) / 2 + y1 * sc;
    const S = (v) => [ox + v[0] * sc, oy - v[1] * sc];
    const P = (x, y, dz = 0.02) => S(view(x, y, height(x, y) + dz));

    const quads = [];
    for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
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
      if (i % 6 === 0 || j % 6 === 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.6; ctx.beginPath();
        if (i % 6 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[3]); }
        if (j % 6 === 0) { ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); }
        ctx.stroke();
      }
    }

    // Arrows: the flow toward the likely peak, a regular grid of small arrows over the whole plane,
    // all pointing straight at it (none right on the peak).
    if (opts.arrowsTo) {
      const [tx, ty] = opts.arrowsTo;
      const step = 0.6;
      for (let x = -EX + 0.3; x <= EX - 0.29; x += step) for (let y = -EY + 0.3; y <= EY - 0.29; y += step) {
        const dist = Math.hypot(tx - x, ty - y);
        // Skip only the very top of the likely peak; flanks and the other bump get arrows too.
        if (dist < 0.3) continue;
        // Also skip arrows just behind the peak: the hill would hide them, but they'd be drawn on top.
        if (dist < 1.6 && view(x, y, 0)[2] > view(tx, ty, 0)[2] - 0.1) continue;
        const ux0 = (tx - x) / dist, uy0 = (ty - y) / dist;
        // Direction from the surface, length fixed on screen, so arrows on steep slopes aren't stretched.
        const [cx, cy] = P(x, y), [qx, qy] = P(x + ux0 * 0.05, y + uy0 * 0.05);
        const l = Math.hypot(qx - cx, qy - cy) || 1, ux = (qx - cx) / l, uy = (qy - cy) / l, half = 8, hd = 4.5;
        const ax = cx - ux * half, ay = cy - uy * half, bx = cx + ux * half, by = cy + uy * half;
        const shaft = () => { ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx - ux * hd * 0.7, by - uy * hd * 0.7); };
        const head = () => { ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - ux * hd - uy * hd * 0.6, by - uy * hd + ux * hd * 0.6); ctx.lineTo(bx - ux * hd + uy * hd * 0.6, by - uy * hd - ux * hd * 0.6); ctx.closePath(); };
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'rgba(14,15,19,0.7)'; ctx.lineWidth = 3.2; shaft(); ctx.stroke(); head(); ctx.lineWidth = 2; ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,0.92)'; ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.lineWidth = 1.4; shaft(); ctx.stroke(); head(); ctx.fill();
        ctx.lineCap = 'butt';
      }
    }

    // Extra drawing on top of the surface (used by W27): gets the context and the projection.
    if (opts.draw) opts.draw(ctx, P, height);

    // Photo cards above the peaks, with a stem and a label.
    for (const card of opts.cards || []) {
      const [px, py] = P(...card.at, 0);
      const size = 58, cy = Math.max(size / 2 + 2, py - 18 - size / 2);
      ctx.strokeStyle = 'rgba(232,233,238,0.7)'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, cy + size / 2); ctx.stroke();
      if (card.img) {
        const side = Math.min(card.img.naturalWidth, card.img.naturalHeight);
        ctx.save(); ctx.beginPath(); ctx.roundRect(px - size / 2, cy - size / 2, size, size, 5); ctx.clip();
        ctx.drawImage(card.img, (card.img.naturalWidth - side) / 2, (card.img.naturalHeight - side) / 2, side, side, px - size / 2, cy - size / 2, size, size);
        ctx.restore();
      }
      ctx.strokeStyle = card.likely ? rgb(C_DATA) : 'rgba(138,155,180,0.9)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(px - size / 2, cy - size / 2, size, size, 5); ctx.stroke();
      ctx.font = '600 13px Inter, system-ui, sans-serif'; ctx.textAlign = card.side === 'left' ? 'right' : 'left';
      ctx.fillStyle = card.likely ? rgb(C_DATA) : 'rgba(170,180,200,0.95)';
      ctx.fillText(card.label, card.side === 'left' ? px - size / 2 - 6 : px + size / 2 + 6, cy + 4);
    }
  }

  // Shared with W27 (classifier-free guidance on the same dog/cat landscape).
  window.W25 = { surface, loadImage, DOG, CAT };

  Widgets.register('W25', {
    init(el) {
      el.innerHTML = '';
      el.classList.add('w25');
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const IW = 210, IH = 120, OW = 420, OH = 228;
      const mk = (w, h) => { const c = document.createElement('canvas'); c.width = w * ratio; c.height = h * ratio; c.style.width = `${w}px`; c.style.height = `${h}px`; return c; };
      const cell = (cls, ...kids) => { const d = document.createElement('div'); d.className = cls; d.append(...kids); return d; };
      const text = (cls, t) => { const d = document.createElement('div'); d.className = cls; d.textContent = t; return d; };

      const rows = [
        { prompt: 'A picture of a dog', likely: 'dog' },
        { prompt: 'A picture of a cat', likely: 'cat' },
      ].map((r) => ({ ...r, inCv: mk(IW, IH), outCv: mk(OW, OH) }));

      const model = cell('w25-model', text('w25-model-title', 'Model'));
      const grid = cell('w25-grid');
      rows.forEach((r, i) => {
        grid.append(
          cell(`w25-in r${i}`, r.inCv, text('w25-prompt', `“${r.prompt}”`)),
          cell(`w25-arrow r${i} a1`, text('', '→')),
        );
        if (i === 0) grid.append(model);
        grid.append(cell(`w25-arrow r${i} a2`, text('', '→')), cell(`w25-out r${i}`, r.outCv));
      });
      el.append(grid);
      // The slide was laid out while the widget was still empty: center it again.
      if (window.Reveal && Reveal.layout) Reveal.layout();

      const noise = [{ m: [0, 0], s: 0.9, w: 1 }];
      function draw(imgs) {
        rows.forEach((r) => {
          surface(r.inCv, IW, IH, ratio, noise);
          const dogW = r.likely === 'dog' ? 0.82 : 0.18;
          const comps = [{ m: DOG, s: 0.5, w: dogW }, { m: CAT, s: 0.5, w: 1 - dogW }];
          surface(r.outCv, OW, OH, ratio, comps, {
            arrowsTo: r.likely === 'dog' ? DOG : CAT,
            cards: [
              { at: DOG, img: imgs.dog, likely: r.likely === 'dog', label: r.likely === 'dog' ? 'likely' : 'unlikely', side: 'left' },
              { at: CAT, img: imgs.cat, likely: r.likely === 'cat', label: r.likely === 'cat' ? 'likely' : 'unlikely', side: 'right' },
            ],
          });
        });
      }
      draw({});
      Promise.all([loadImage('media/peaks/pug.jpg'), loadImage('media/peaks/cat.jpg')])
        .then(([dog, cat]) => draw({ dog, cat }));
      return { start() {}, stop() {} };
    },
  });
})();
