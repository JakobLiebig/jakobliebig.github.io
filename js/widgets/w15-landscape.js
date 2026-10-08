// W15: p(x) as a 3D landscape (mixture of Gaussians).
//
// data-mode="surface"  just the landscape p_0
// data-mode="marked"   peaks marked with their images, one valley point with a noise image
// data-mode="diffuse"  only the peaks; a slider runs the DDPM forward process in T = 100 steps.
//                      Each image rides on its dot and gets noisier along the walk.
//                      The landscape is the marginal p_t: every peak slides toward the center and
//                      widens until the whole surface is one N(0, I). Each point walks downhill
//                      while its image gets noisier.
// data-mode="reverse"  the camera tilts to a top-down view, then steps through the reverse
//                      process, driven by fragments in the same slide (data-w15-stage="1".."5"):
//                        1 zoom in: arrow back along the zigzag (this step's noise?)
//                        2 a second zigzag from the same image lands on the same x_t
//                        3 zoom in: its last step came from elsewhere, so the step noise can't
//                          be predicted
//                        4 both share one way back to the image: the entire noise eps, fixed by
//                          x_t and x_0, is what the network predicts
//                        5 DDPM sampling: small step toward the estimated image, predict again,
//                          repeat; the landscape grows back from N(0, I) into the peaks
//                      data-focus="3" picks which peak's walk is explained.
//
// Peak images default to media/peaks/*.jpg; data-images="a.jpg,b.jpg,..." overrides them.
// Labels are KaTeX, laid over the canvas. The surface is cached whenever nothing moves.

(function () {
  // ---- Scene ---------------------------------------------------------------

  // Mixture components: mean, std, weight. Peaks sit around the rim, the valley in the middle.
  const PEAKS = [
    { mx: -3.00, my: -0.60, s: 0.40, w: 1.00 },
    { mx: -1.30, my:  1.30, s: 0.38, w: 0.85 },
    { mx:  1.40, my: -1.20, s: 0.36, w: 0.75 },
    { mx:  3.00, my:  0.90, s: 0.42, w: 1.10 },
  ];
  const IMAGES = ['media/peaks/puppy.jpg', 'media/peaks/pug.jpg', 'media/peaks/lioness.jpg', 'media/peaks/fawn.jpg'];
  const VALLEY = { x: 0.0, y: 0.0 };
  const EXT_X = 4.2;     // surface covers [-EXT_X, EXT_X] x [-EXT_Y, EXT_Y]
  const EXT_Y = 2.6;
  const GRID_X = 100;    // quads along x
  const GRID_Y = 62;     // quads along y
  const HEIGHT = 1.4;    // world height of the tallest peak of p_0
  const AZ = -0.2;       // camera azimuth (rad)
  const EL = 0.38;       // camera elevation (rad)
  const DIST = 13;       // perspective distance

  const T = 100;         // DDPM steps
  const IMG = 96;        // image resolution (pixels per side)

  // Colors follow the deck's code: orange = data, blue-grey = noise.
  const C_DATA = [242, 153, 74];
  const C_NOISE = [138, 155, 180];
  const C_LOW = [30, 35, 47];

  // Linear beta schedule (as in DDPM, just fewer steps): ends at alphaBar ~ 0.02.
  const beta = [0];
  const alphaBar = [1];
  for (let k = 1; k <= T; k++) {
    const b = 0.005 + ((0.073 - 0.005) * (k - 1)) / (T - 1);
    beta.push(b);
    alphaBar.push(alphaBar[k - 1] * (1 - b));
  }
  // alphaBar at a fractional step, for smooth slider motion.
  function abAt(s) {
    if (s <= 0) return 1;
    if (s >= T) return alphaBar[T];
    const k = Math.floor(s);
    return alphaBar[k] + (alphaBar[k + 1] - alphaBar[k]) * (s - k);
  }

  // A landscape: mixture components and the marginal p_t of the forward process, where component k
  // becomes N(sqrt(ab) mu_k, (ab s_k^2 + 1 - ab) I). ab = 1 is the data, ab -> 0 is N(0, I).
  // Heights are scaled so the tallest peak of p_0 is HEIGHT. With normalize, every p_t is scaled
  // to its own maximum instead, so a moving, widening distribution stays visible as it spreads.
  function makeScene(components, { normalize = false } = {}) {
    const wsum = components.reduce((a, k) => a + k.w, 0);
    const comps = components.map((k) => ({ ...k, pi: k.w / wsum }));
    function density(x, y, ab = 1) {
      const a = Math.sqrt(ab);
      let p = 0;
      for (const k of comps) {
        const v = ab * k.s * k.s + 1 - ab;
        const dx = x - a * k.mx, dy = y - a * k.my;
        p += (k.pi / (2 * Math.PI * v)) * Math.exp(-(dx * dx + dy * dy) / (2 * v));
      }
      return p;
    }
    // Max of p_t, taken at the component means (exact for one component).
    const maxAt = (ab) => {
      const a = Math.sqrt(ab);
      return Math.max(...comps.map((k) => density(a * k.mx, a * k.my, ab)));
    };
    const pmax = maxAt(1);
    const maxCache = new Map();
    const scaleAt = (ab) => {
      if (!normalize) return pmax;
      let m = maxCache.get(ab);
      if (m === undefined) { m = maxAt(ab); if (maxCache.size > 400) maxCache.clear(); maxCache.set(ab, m); }
      return m;
    };
    const height = (x, y, ab = 1) => (HEIGHT * density(x, y, ab)) / scaleAt(ab);
    // Grid vertices of the surface for a given alphaBar, cached per ab.
    const cache = new Map();
    function vertsFor(ab) {
      let v = cache.get(ab);
      if (v) return v;
      v = [];
      for (let i = 0; i <= GRID_X; i++) {
        const row = [];
        for (let j = 0; j <= GRID_Y; j++) {
          const x = -EXT_X + (2 * EXT_X * i) / GRID_X, y = -EXT_Y + (2 * EXT_Y * j) / GRID_Y;
          row.push({ x, y, z: height(x, y, ab) });
        }
        v.push(row);
      }
      if (cache.size > 40) cache.clear();
      cache.set(ab, v);
      return v;
    }
    return { comps, density, height, vertsFor };
  }
  // The four-peak landscape of the DDPM slides, and one off-center peak for the SDE slides.
  const MAIN = makeScene(PEAKS);
  const SINGLE = makeScene([{ mx: 2.4, my: 0.7, s: 0.3, w: 1 }], { normalize: true });
  // A thin winding ridge for the latent-diffusion slide: all the probability sits on a 1D curve
  // (the "manifold"), the rest of the plane is empty.
  const manifoldCurve = (u) => [-3.7 + 7.4 * u, 1.35 * Math.sin(2 * Math.PI * 0.85 * u + 0.5) - 0.25];
  const MANIFOLD = makeScene(Array.from({ length: 48 }, (_, i) => {
    const [mx, my] = manifoldCurve(i / 47);
    return { mx, my, s: 0.17, w: 1 };
  }));
  // One centered Gaussian for the score slide.
  const SCORE = makeScene([{ mx: 0, my: 0, s: 0.9, w: 1 }]);
  // One Gaussian for the long-path slide: the image is its peak, the noise point lies on the flat.
  const FLOW_PK = { mx: 2.0, my: 0.6, s: 0.55, w: 1 };
  const FLOW = makeScene([FLOW_PK]);
  const FLOW_START = [-1.6, -1.0];
  // The deterministic path (DDIM, eta = 0), mocked for the picture: a smooth, gently curved path
  // from the noise point to the peak in FLOW_DDIM steps, its steps getting denser toward the image.
  // (The exact DDIM path for one Gaussian is nearly straight and stops 2 std short of the center.)
  const FLOW_DDIM = 25;
  const FLOW_DDIM_PATH = (() => {
    const [ax, ay] = FLOW_START, bx = FLOW_PK.mx, by = FLOW_PK.my;
    const dx = bx - ax, dy = by - ay, l = Math.hypot(dx, dy);
    // Quadratic Bezier, control point 1.0 to the side of the midpoint (bulge 0.5).
    const cx = (ax + bx) / 2 - (dy / l) * 1.0, cy = (ay + by) / 2 + (dx / l) * 1.0;
    const path = [];
    for (let i = 0; i <= FLOW_DDIM; i++) {
      const u = 1 - (1 - i / FLOW_DDIM) ** 2;
      path.push([(1 - u) ** 2 * ax + 2 * u * (1 - u) * cx + u * u * bx, (1 - u) ** 2 * ay + 2 * u * (1 - u) * cy + u * u * by]);
    }
    return path;
  })();
  // The random reverse path from the noise point to the peak: a DDPM posterior bridge, seeded.
  let flowPathCache = null;
  function flowPath() {
    if (flowPathCache) return flowPathCache;
    for (let attempt = 0; attempt < 200 && !flowPathCache; attempt++) flowPathCache = bridge(FLOW_START, FLOW_PK, rng(8100 + attempt));
    return flowPathCache || [FLOW_START, [FLOW_PK.mx, FLOW_PK.my]];
  }
  const height = MAIN.height;

  // ---- Randomness (seeded, so walks and noise are the same every time) ----------

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

  const inside = (x, y) => Math.abs(x) < EXT_X - 0.15 && Math.abs(y) < EXT_Y - 0.15;
  const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

  // Forward DDPM walks x_k = sqrt(1 - beta_k) x_{k-1} + sqrt(beta_k) eps, one per peak.
  // Plain DDPM: each point rides its peak as the peak melts and slides toward N(0, I).
  // Among many seeds we keep the one that climbs least (so it reads as walking
  // downhill), ending inside the noise blob, away from its peak and from the other end points.
  const WALKS = (() => {
    const ends = [];
    return PEAKS.map((pk, idx) => {
      let best = null;
      for (let attempt = 0; attempt < 4000; attempt++) {
        const g = rng(1000 * (idx + 1) + attempt);
        let x = pk.mx, y = pk.my, h = height(x, y, 1), up = 0, ok = true;
        const path = [[x, y]];
        for (let k = 1; k <= T && ok; k++) {
          x = Math.sqrt(1 - beta[k]) * x + Math.sqrt(beta[k]) * g();
          y = Math.sqrt(1 - beta[k]) * y + Math.sqrt(beta[k]) * g();
          ok = inside(x, y);
          const hk = height(x, y, alphaBar[k]);
          up += Math.max(0, hk - h);
          h = hk;
          path.push([x, y]);
        }
        if (!ok) continue;
        const r = Math.hypot(x, y);
        if (r < 0.5 || r > 2.0 || Math.hypot(x - pk.mx, y - pk.my) < 1.8) continue;
        // Images sit above the dots, so end points should be apart left to right on screen.
        const across = (px, py) => px * Math.cos(AZ) - py * Math.sin(AZ);
        const crowded = ends.filter(([ex, ey]) => Math.abs(across(ex, ey) - across(x, y)) < 0.9).length;
        const score = up + 10 * crowded;
        if (!best || score < best.score) best = { path, score };
      }
      if (!best) return [[pk.mx, pk.my]];
      ends.push(best.path[T]);
      return best.path;
    });
  })();

  // One draw from the DDPM posterior q(x_{k-1} | x_k, x_0), walking from x_T back to x_0.
  // This is a bridge: every draw starts at x_T and ends exactly on x_0. Each step moves part of
  // the way toward x_0, plus a little fresh noise. path[i] is step T - i; null if it leaves the map.
  function bridge(start, peak, g) {
    let [x, y] = start;
    const path = [[x, y]];
    for (let k = T; k >= 1; k--) {
      const ab = alphaBar[k], abp = alphaBar[k - 1], b = beta[k];
      const c0 = (Math.sqrt(abp) * b) / (1 - ab);
      const ct = (Math.sqrt(1 - b) * (1 - abp)) / (1 - ab);
      const sd = Math.sqrt(((1 - abp) / (1 - ab)) * b);
      x = c0 * peak.mx + ct * x + sd * g();
      y = c0 * peak.my + ct * y + sd * g();
      if (!inside(x, y)) return null;
      path.push([x, y]);
    }
    return path;
  }

  // A second forward path from the same image to the same x_T, in forward order (path[k] = x_k).
  // Same endpoints, different zigzag: its last step comes from somewhere else, so "the noise of
  // this step" is ambiguous, while the entire noise (fixed by x_T and x_0) is the same.
  // We keep a seed whose last step is ordinary in size but points elsewhere, and whose route is
  // visibly different.
  function secondPath(fwd, peak, idx) {
    let best = null;
    for (let attempt = 0; attempt < 600; attempt++) {
      const b = bridge(fwd[T], peak, rng(7000 + 100 * idx + attempt));
      if (!b) continue;
      const path = b.slice().reverse();
      const [px, py] = fwd[T - 1], [qx, qy] = path[T - 1], [ex, ey] = fwd[T];
      const lastGap = Math.hypot(px - qx, py - qy);
      // Angle between the two last steps; we want them pointing clearly apart.
      const c = ((px - ex) * (qx - ex) + (py - ey) * (qy - ey)) / ((Math.hypot(px - ex, py - ey) * Math.hypot(qx - ex, qy - ey)) || 1);
      let sep = 0;
      for (let k = 0; k <= T; k += 5) sep += Math.hypot(path[k][0] - fwd[k][0], path[k][1] - fwd[k][1]);
      sep /= T / 5 + 1;
      // A typical step, not an outlier: about the size of the first path's last step.
      const lastLen = Math.hypot(qx - ex, qy - ey);
      if (lastGap < 0.25 || c > -0.2 || lastLen > 0.4) continue;
      const score = Math.min(sep, 0.9) + Math.min(lastGap, 0.5);
      if (!best || score > best.score) best = { path, score };
    }
    return best ? best.path : fwd;
  }

  // DDPM reverse sampling with a perfect noise prediction (it knows x_0). Prefer a seed that
  // climbs at every step. path[i] is step T - i; the last point is exactly the peak.
  function ddpmSample(start, peak, idx) {
    let fallback = null;
    for (let attempt = 0; attempt < 300; attempt++) {
      const path = bridge(start, peak, rng(9000 + 100 * idx + attempt));
      if (!path) continue;
      let climbs = true;
      for (let i = 1; i <= T && climbs; i++) {
        climbs = height(...path[i], alphaBar[T - i]) >= height(...path[i - 1], alphaBar[T - i + 1]);
      }
      if (climbs) return path;
      if (!fallback) fallback = path;
    }
    return fallback || [start, [peak.mx, peak.my]];
  }

  // ---- SDE particles (single-peak scene) ---------------------------------------

  // Forward: samples of the peak, each stepped with x_k = sqrt(1 - beta_k) x_{k-1} + sqrt(beta_k) eps,
  // i.e. Euler-Maruyama for the VP SDE dx = -1/2 beta x dt + sqrt(beta) dW. Per step this is a
  // deterministic pull toward 0 plus a random kick.
  // Reverse: start from each forward end point (a sample of ~N(0, I)) and run the reverse SDE
  // dx = [f - g^2 score] dt + g dW_bar backwards with the exact score of the marginal p_t,
  // which for one Gaussian component is -(x - m_t) / v_t. path[i] is step T - i.
  const SDE_N = 180;
  let sdeCache = null;
  function sdeParticles() {
    if (sdeCache) return sdeCache;
    const pk = SINGLE.comps[0];
    const g = rng(4242);
    const fwd = [];
    while (fwd.length < SDE_N) {
      let x = pk.mx + pk.s * g(), y = pk.my + pk.s * g(), ok = inside(x, y);
      const path = [[x, y]];
      for (let k = 1; k <= T && ok; k++) {
        x = Math.sqrt(1 - beta[k]) * x + Math.sqrt(beta[k]) * g();
        y = Math.sqrt(1 - beta[k]) * y + Math.sqrt(beta[k]) * g();
        ok = inside(x, y);
        path.push([x, y]);
      }
      if (ok) fwd.push(path);
    }
    const rev = fwd.map((f) => {
      for (let attempt = 0; attempt < 20; attempt++) {
        let [x, y] = f[T], ok = true;
        const path = [[x, y]];
        for (let k = T; k >= 1 && ok; k--) {
          const ab = alphaBar[k], v = ab * pk.s * pk.s + 1 - ab, a = Math.sqrt(ab);
          const sx = -(x - a * pk.mx) / v, sy = -(y - a * pk.my) / v;
          x += 0.5 * beta[k] * x + beta[k] * sx + Math.sqrt(beta[k]) * g();
          y += 0.5 * beta[k] * y + beta[k] * sy + Math.sqrt(beta[k]) * g();
          ok = inside(x, y);
          path.push([x, y]);
        }
        if (ok) return path;
      }
      return f.slice().reverse();
    });
    sdeCache = { fwd, rev, peak: pk };
    return sdeCache;
  }

  // ---- Images ----------------------------------------------------------------

  function blankCanvas() {
    const cv = document.createElement('canvas');
    cv.width = cv.height = IMG;
    return cv;
  }

  // Image pixels as floats in [-1, 1], the way DDPM sees them.
  function pixels(cv) {
    const d = cv.getContext('2d').getImageData(0, 0, IMG, IMG).data;
    const out = new Float32Array(IMG * IMG * 3);
    for (let i = 0, j = 0; i < d.length; i += 4) {
      out[j++] = d[i] / 127.5 - 1; out[j++] = d[i + 1] / 127.5 - 1; out[j++] = d[i + 2] / 127.5 - 1;
    }
    return out;
  }

  function loadImage(src) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => { console.warn(`W15: could not load ${src}`); resolve(null); };
      im.src = src;
    });
  }

  // A card holds one image x0 and fixed noise eps; render(ab) draws sqrt(ab) x0 + sqrt(1 - ab) eps.
  function makeCard(x0, seed) {
    const g = rng(seed);
    const eps = new Float32Array(IMG * IMG * 3).map(() => g());
    const cv = blankCanvas();
    const ctx = cv.getContext('2d');
    const out = ctx.createImageData(IMG, IMG);
    let last = -1;
    return {
      cv,
      // lin = false: DDPM noising with alphaBar = level. lin = true: flow matching interpolation
      // (1 - t) x0 + t eps with t = level.
      render(level, lin = false) {
        const key = `${level}|${lin}`;
        if (key === last) return;
        last = key;
        const a = lin ? 1 - level : Math.sqrt(level), b = lin ? level : Math.sqrt(1 - level);
        const d = out.data;
        for (let i = 0, j = 0; i < d.length; i += 4) {
          for (let c = 0; c < 3; c++, j++) {
            const v = x0 ? a * x0[j] + b * eps[j] : eps[j];
            d[i + c] = Math.max(0, Math.min(255, (v + 1) * 127.5));
          }
          d[i + 3] = 255;
        }
        ctx.putImageData(out, 0, 0);
      },
    };
  }

  // ---- Camera ------------------------------------------------------------------

  // World (x, y, z) -> camera space [screen x, screen y (up), depth (far = larger)].
  function camera(az, el) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    return (x, y, z) => {
      const xr = x * ca - y * sa;
      const yr = x * sa + y * ca;
      const up = z * ce + yr * se;
      const depth = yr * ce - z * se;
      const f = DIST / (DIST + depth);
      return [xr * f, up * f, depth];
    };
  }
  const SIDE = { az: AZ, el: EL };
  const TOP = { az: 0, el: Math.PI / 2 };

  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const rgb = (c, alpha = 1) => `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;

  const LIGHT = (() => { const l = [-0.5, -0.6, 0.8]; const n = Math.hypot(...l); return l.map((c) => c / n); })();

  function quadColor(a, b, c, d) {
    const ux = c.x - a.x, uy = c.y - a.y, uz = c.z - a.z;
    const wx = d.x - b.x, wy = d.y - b.y, wz = d.z - b.z;
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const nn = Math.hypot(nx, ny, nz) || 1; nx /= nn; ny /= nn; nz /= nn;
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    const lam = Math.max(0, nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]);
    const h = (a.z + b.z + c.z + d.z) / (4 * HEIGHT);
    const base = mix(C_LOW, mix(C_NOISE, C_DATA, Math.min(1, h * 1.4)), Math.min(1, 0.25 + h * 1.6));
    return rgb(base.map((v) => Math.min(255, v * (0.55 + 0.55 * lam))));
  }

  // ---- Widget ------------------------------------------------------------------

  Widgets.register('W15', {
    init(el) {
      const mode = el.dataset.mode || 'surface';
      const showMarkers = mode !== 'surface';
      const showValley = mode === 'marked';
      const diffuse = mode === 'diffuse';
      const reverse = mode === 'reverse';
      const sde = mode === 'sde';                // single peak, forward SDE
      const sdeRev = mode === 'sde-reverse';     // single peak, reverse SDE
      const single = sde || sdeRev;
      const scoreMode = mode === 'score';        // one Gaussian with its score field
      const flowMode = mode === 'flow';
      const manifoldMode = mode === 'manifold';  // thin ridge: images live on a manifold          // long reverse path vs. a straight line (top view)
      // Which landscape this instance shows. Shadows the module-level helpers of the main scene.
      const { height, vertsFor } = manifoldMode ? MANIFOLD : scoreMode ? SCORE : flowMode ? FLOW : single ? SINGLE : MAIN;

      el.classList.add('w15');
      const canvas = document.createElement('canvas');
      el.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const tagLayer = document.createElement('div');
      tagLayer.className = 'w15-tags';
      el.appendChild(tagLayer);

      const W = el.clientWidth || 1100;
      const H = el.clientHeight || 500;
      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      canvas.width = Math.round(W * ratio);
      canvas.height = Math.round(H * ratio);

      // Layout in CSS px. In the side view, room for cards and slider is reserved in every mode,
      // so the surface sits at exactly the same place on all slides and the fades are seamless.
      const CARD = Math.round(H * 0.15);
      const LIFT = Math.round(H * 0.055);
      const pad = 12;

      // ---- View: camera blend u (0 = side, 1 = top), then a 2D zoom on top ----
      // The score slide looks from higher up, so the far slope of the hill (and its arrows) shows.
      const side = scoreMode ? { az: AZ, el: 0.75 } : manifoldMode ? { az: AZ, el: 0.8 } : flowMode ? { az: AZ, el: 0.55 } : SIDE;
      let u = 0, view, base;
      let zoom = { z: 1, cx: W / 2, cy: H / 2 };
      const ZOOM0 = { z: 1, cx: W / 2, cy: H / 2 };
      function setCamera(v) {
        u = v;
        const e = ease(v);
        view = camera(side.az + (TOP.az - side.az) * e, side.el + (TOP.el - side.el) * e);
        // Fit p_0 (the tallest landscape), so the frame doesn't jump as the peaks melt.
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const row of vertsFor(1)) for (const p of row) {
          const q = view(p.x, p.y, p.z);
          if (q[0] < minX) minX = q[0]; if (q[0] > maxX) maxX = q[0];
          if (q[1] < minY) minY = q[1]; if (q[1] > maxY) maxY = q[1];
        }
        // The single-peak slides have no image cards, so they need less room on top.
        const top = pad + (scoreMode ? 0 : manifoldMode ? 30 : single ? 40 : CARD + LIFT) * (1 - e);
        // The single-peak slides keep their slider in the top view.
        const bottom = scoreMode || flowMode || manifoldMode ? pad : single ? 52 : 52 + (pad - 52) * e;
        const scale = Math.min((W - 2 * pad) / (maxX - minX), (H - top - bottom) / (maxY - minY));
        const slack = H - top - bottom - (maxY - minY) * scale;
        const ox = W / 2 - ((minX + maxX) / 2) * scale;
        const oy = top + slack / 2 + maxY * scale;
        base = (q) => [ox + q[0] * scale, oy - q[1] * scale];
      }
      const toScreen = (q) => {
        const [bx, by] = base(q);
        return [(bx - zoom.cx) * zoom.z + W / 2, (by - zoom.cy) * zoom.z + H / 2];
      };

      // Landscape state: which p_t is shown.
      let surfAb = 1;
      const project = (x, y, z) => toScreen(view(x, y, z));
      const onSurface = (x, y) => project(x, y, height(x, y, surfAb) + 0.02);

      function renderSurface(target, stride) {
        const s = target.getContext('2d');
        s.setTransform(1, 0, 0, 1, 0, 0);
        s.clearRect(0, 0, target.width, target.height);
        s.setTransform(ratio, 0, 0, ratio, 0, 0);
        const V = vertsFor(surfAb);
        const LINES = 6;
        const quads = [];
        for (let i = 0; i < GRID_X; i += stride) for (let j = 0; j < GRID_Y; j += stride) {
          const c = [V[i][j], V[i + stride][j], V[i + stride][j + stride], V[i][j + stride]];
          const cam = c.map((p) => view(p.x, p.y, p.z));
          const pts = cam.map(toScreen);
          // Skip quads outside the canvas (matters when zoomed in).
          if (pts.every((p) => p[0] < -2) || pts.every((p) => p[0] > W + 2) || pts.every((p) => p[1] < -2) || pts.every((p) => p[1] > H + 2)) continue;
          quads.push({ i, j, c, pts, depth: (cam[0][2] + cam[1][2] + cam[2][2] + cam[3][2]) / 4 });
        }
        quads.sort((a, b) => b.depth - a.depth);
        for (const q of quads) {
          const { pts } = q;
          const col = quadColor(...q.c);
          s.beginPath();
          s.moveTo(pts[0][0], pts[0][1]);
          for (let k = 1; k < 4; k++) s.lineTo(pts[k][0], pts[k][1]);
          s.closePath();
          s.fillStyle = col; s.strokeStyle = col; s.lineWidth = 0.8;
          s.fill(); s.stroke();
          // Sparse wireframe, drawn in painter's order so hidden lines stay hidden.
          s.strokeStyle = 'rgba(255,255,255,0.10)'; s.lineWidth = 0.7;
          s.beginPath();
          if (q.i % LINES === 0) { s.moveTo(pts[0][0], pts[0][1]); s.lineTo(pts[3][0], pts[3][1]); }
          if (q.j % LINES === 0) { s.moveTo(pts[0][0], pts[0][1]); s.lineTo(pts[1][0], pts[1][1]); }
          if (q.i + stride === GRID_X) { s.moveTo(pts[1][0], pts[1][1]); s.lineTo(pts[2][0], pts[2][1]); }
          if (q.j + stride === GRID_Y) { s.moveTo(pts[3][0], pts[3][1]); s.lineTo(pts[2][0], pts[2][1]); }
          s.stroke();
        }
      }

      // Cached surface for a still view; coarse mesh while something moves.
      const surface = document.createElement('canvas');
      surface.width = canvas.width; surface.height = canvas.height;
      let surfaceKey = null;
      function surfaceFor(still) {
        if (!still) { renderSurface(surface, 2); surfaceKey = null; return surface; }
        const key = `${u}|${surfAb}|${zoom.z}|${zoom.cx}|${zoom.cy}`;
        if (surfaceKey !== key) { renderSurface(surface, 1); surfaceKey = key; }
        return surface;
      }

      // ---- KaTeX labels over the canvas ----
      const tags = new Map();
      let usedTags = new Set();
      function tag(key, tex, x, y, color, plain) {
        usedTags.add(key);
        let t = tags.get(key);
        if (!t) {
          t = { div: document.createElement('div'), tex: null, katex: false };
          t.div.className = 'w15-tag';
          tagLayer.appendChild(t.div);
          tags.set(key, t);
        }
        const hasKatex = !!window.katex;
        if (t.tex !== tex || t.katex !== hasKatex) {
          if (hasKatex) window.katex.render(tex, t.div, { throwOnError: false });
          else t.div.textContent = plain || tex;
          t.tex = tex; t.katex = hasKatex;
        }
        t.div.style.left = `${x}px`;
        t.div.style.top = `${y}px`;
        t.div.style.color = color;
        t.div.style.display = '';
      }
      // Label next to a world point.
      const pointTag = (key, tex, p, color, plain, dx = 30, dy = -32) => {
        const [sx, sy] = onSurface(...p);
        tag(key, tex, sx + dx, sy + dy, color, plain);
      };
      // Label a world point on the side facing away from the other points in view, so it never
      // sits on an arrow between them.
      function tagAway(key, tex, p, others, color, plain, dist = 44) {
        const [sx, sy] = onSurface(...p);
        let dx = 0, dy = 0;
        for (const o of others) {
          const [ox, oy] = onSurface(...o);
          const l = Math.hypot(sx - ox, sy - oy) || 1;
          dx += (sx - ox) / l; dy += (sy - oy) / l;
        }
        const l = Math.hypot(dx, dy);
        if (l < 1e-3) { dx = 0.7; dy = -0.7; } else { dx /= l; dy /= l; }
        tag(key, tex, sx + dx * dist, sy + dy * dist, color, plain);
      }
      function hideUnusedTags() {
        for (const [k, t] of tags) if (!usedTags.has(k)) t.div.style.display = 'none';
        usedTags = new Set();
      }

      // ---- Markers ----
      const peaks = PEAKS.map((p, idx) => ({ ...p, idx, path: WALKS[idx], card: null }));
      const valley = { x: VALLEY.x, y: VALLEY.y, card: showValley ? makeCard(null, 77) : null };

      // Each image rides just above its dot, on a short stem, and wiggles along with it.
      function cardOnDot(card, sx, sy, ab, color, lin = false) {
        const cy = Math.max(CARD / 2 + 4, sy - LIFT - CARD / 2);
        ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, cy + CARD / 2); ctx.stroke();
        drawCard(card, sx, cy, ab, lin);
      }

      let s = 0; // forward step, fractional while the slider moves (diffuse mode)
      let ready = false;

      function drawCard(card, cx, cy, ab, lin = false) {
        card.render(ab, lin);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(card.cv, Math.round(cx - CARD / 2), Math.round(cy - CARD / 2), CARD, CARD);
      }

      function drawDot(sx, sy, color, r = 7) {
        ctx.beginPath(); ctx.arc(sx, sy, r + 2.5, 0, Math.PI * 2); ctx.fillStyle = '#0e0f13'; ctx.fill();
        ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      }

      // Point on a path at a fractional index.
      const at = (path, n) => {
        if (n <= 0) return path[0];
        if (n >= path.length - 1) return path[path.length - 1];
        const k = Math.floor(n);
        return lerp2(path[k], path[k + 1], n - k);
      };

      // Forward trail up to fractional step n, colored data -> noise.
      function drawTrail(path, n, alpha, width = 2) {
        ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        for (let k = 1; k <= Math.ceil(n) && k < path.length; k++) {
          const a = onSurface(...path[k - 1]), b = onSurface(...at(path, Math.min(n, k)));
          ctx.strokeStyle = rgb(mix(C_DATA, C_NOISE, k / T), alpha);
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
        ctx.lineCap = 'butt';
      }

      function drawPath(path, n, color, width, dash) {
        if (n <= 0) return;
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        const p0 = onSurface(...path[0]);
        ctx.moveTo(p0[0], p0[1]);
        for (let k = 1; k <= Math.ceil(n) && k < path.length; k++) {
          const p = onSurface(...at(path, Math.min(n, k)));
          ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Arrow between world points; optional KaTeX label beside its middle.
      function arrow(from, to, color, label, side = 1, key = 'arrow') {
        const [x1, y1] = onSurface(...from), [x2, y2] = onSurface(...to);
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
        if (len < 4) return;
        const ux = dx / len, uy = dy / len, head = Math.min(16, len * 0.4);
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - ux * head * 0.8, y2 - uy * head * 0.8); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - ux * head - uy * head * 0.5, y2 - uy * head + ux * head * 0.5);
        ctx.lineTo(x2 - ux * head + uy * head * 0.5, y2 - uy * head - ux * head * 0.5);
        ctx.closePath(); ctx.fill();
        ctx.lineCap = 'butt';
        if (label) tag(key, label.tex, (x1 + x2) / 2 - uy * 30 * side, (y1 + y2) / 2 + ux * 30 * side, color, label.plain);
      }

      // ---- Reverse mode: stages and their paths ----
      const focus = Math.min(PEAKS.length - 1, Math.max(0, +(el.dataset.focus ?? 3)));
      const fp = peaks[focus];
      const x0 = flowMode ? [FLOW_PK.mx, FLOW_PK.my] : [fp.mx, fp.my];
      const xT = flowMode ? FLOW_START : fp.path[T];
      const other = reverse ? secondPath(fp.path, fp, focus) : null;
      const sample = flowMode ? flowPath() : reverse ? ddpmSample(xT, fp, focus) : null;
      let stage = 0;       // highest data-w15-stage fragment shown
      let stageP = 1;      // animation progress of the current stage (0..1)

      // Where to zoom for each stage (top view): a box around the points being explained.
      function zoomFor(st) {
        if (!reverse || st === 0) return ZOOM0;
        let pts;
        if (st === 1) pts = [fp.path[T], fp.path[T - 1], fp.path[T - 2]];
        else if (st === 3) pts = [fp.path[T], fp.path[T - 1], other[T - 1]];
        else if (st <= 4) pts = [...fp.path, ...other, x0];
        else pts = [...fp.path, ...sample, x0];
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const [x, y] of pts) {
          const [bx, by] = base(view(x, y, 0));
          minX = Math.min(minX, bx); maxX = Math.max(maxX, bx);
          minY = Math.min(minY, by); maxY = Math.max(maxY, by);
        }
        const zmax = st === 1 || st === 3 ? 6 : 4;
        const z = Math.max(1, Math.min(zmax, (W - 300) / Math.max(1, maxX - minX), (H - 180) / Math.max(1, maxY - minY)));
        return { z, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
      }

      // The second path is drawn in white so the two zigzags are easy to tell apart.
      const C_OTHER = 'rgba(255,255,255,0.85)';
      const DRAW_PART = 0.8; // stage 2: share of the time spent drawing, the rest highlights x_t

      function drawReverse() {
        // During stage 5 the landscape grows back from N(0, I) into the peaks.
        const n5 = stage >= 5 ? stageP * T : 0;
        surfAb = stage >= 5 ? abAt(T - n5) : alphaBar[T];

        // Other walks fade back as the camera tilts; the focused one stays.
        for (const p of peaks) {
          if (p === fp) continue;
          drawTrail(p.path, T, 0.85 - 0.65 * u);
          const [sx, sy] = onSurface(...p.path[T]);
          drawDot(sx, sy, rgb(C_NOISE, 1 - 0.7 * u), 6);
        }
        drawTrail(fp.path, T, stage === 4 ? 0.45 : stage >= 5 ? 0.3 : 0.9, 2.2);

        // Cards from the previous slide fade out while the camera tilts.
        if (ready && u < 1) {
          ctx.globalAlpha = 1 - ease(u);
          for (const p of peaks) {
            const [sx, sy] = onSurface(...p.path[T]);
            cardOnDot(p.card, sx, sy, alphaBar[T], rgb(C_NOISE, 0.8));
          }
          ctx.globalAlpha = 1;
        }

        // Where the clean image is.
        if (u >= 1) {
          const [px, py] = onSurface(...x0);
          ctx.strokeStyle = rgb(C_DATA); ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.stroke();
          if (stage >= 2) tagAway('x0', 'x_0', x0, [stage >= 5 ? at(sample, Math.max(0, stageP * T - 3)) : xT], rgb(C_DATA), 'x0');
        }

        const prev = fp.path[T - 1];
        if (stage === 1 || stage === 3) {
          arrow(xT, prev, rgb(C_NOISE), null);
          drawDot(...onSurface(...prev), rgb(C_NOISE, 0.9), 5);
          tagAway('xprev', 'x_{t-1}', prev, stage === 3 ? [xT, other[T - 1]] : [xT], rgb(C_NOISE), 'x(t-1)');
        }

        if (stage === 2) {
          // Another zigzag from the same image, drawn from x_0 until it lands on our x_t,
          // then a ring around x_t: it is the same point.
          const n = Math.min(1, stageP / DRAW_PART) * T;
          drawPath(other, n, C_OTHER, 2.2);
          if (n < T) drawDot(...onSurface(...at(other, n)), C_OTHER, 5);
          const ring = Math.max(0, (stageP - DRAW_PART) / (1 - DRAW_PART));
          if (ring > 0) {
            const [sx, sy] = onSurface(...xT);
            ctx.strokeStyle = `rgba(255,255,255,${0.9 * ring})`; ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.arc(sx, sy, 12 + 10 * ease(ring), 0, Math.PI * 2); ctx.stroke();
          }
        }

        if (stage === 3) {
          // Same x_t, but the second path's last step came from somewhere else.
          drawPath(other, T, 'rgba(255,255,255,0.5)', 2);
          arrow(xT, other[T - 1], C_OTHER, null);
          drawDot(...onSurface(...other[T - 1]), C_OTHER, 5);
          tagAway('xprev2', "x'_{t-1}", other[T - 1], [xT, prev], 'rgba(255,255,255,0.95)', "x'(t-1)");
        }

        if (stage === 4) {
          // Both paths share the same way back: x_t and x_0 fix the entire noise.
          drawPath(other, T, 'rgba(255,255,255,0.45)', 2);
          arrow(xT, x0, 'rgba(255,255,255,0.95)', { tex: '\\text{total noise } \\varepsilon', plain: 'total noise eps' }, -1, 'a-eps');
        }

        if (stage >= 5) {
          const p = at(sample, n5);
          const tint = rgb(mix(C_NOISE, C_DATA, n5 / T));
          drawPath(sample, n5, rgb(mix(C_NOISE, C_DATA, n5 / T), 0.95), 2.2);
          if (n5 < T - 0.05) arrow(p, x0, 'rgba(255,255,255,0.85)', null);
          const [sx, sy] = onSurface(...p);
          drawDot(sx, sy, tint);
          if (n5 < T - 0.05) tagAway('xt', 'x_t', p, [x0], tint, 'x_t');
          if (ready) {
            const cx = Math.min(W - CARD / 2 - 6, Math.max(CARD / 2 + 6, sx - CARD * 0.9));
            const cy = Math.max(CARD / 2 + 6, Math.min(H - CARD / 2 - 6, sy - CARD * 0.8));
            drawCard(fp.card, cx, cy, abAt(T - n5));
          }
          return;
        }

        const [sx, sy] = onSurface(...xT);
        drawDot(sx, sy, rgb(C_NOISE));
        if (stage >= 1) {
          const away = stage === 3 ? [prev, other[T - 1]] : stage === 4 ? [x0] : stage === 2 ? [other[T - 1], prev] : [prev];
          tagAway('xt', 'x_t', xT, away, rgb(C_NOISE), 'x_t');
        }
      }

      // ---- SDE modes (single peak) ----
      const P = single ? sdeParticles() : null;
      const HL = [3];                   // the one particle we follow (thick, with its trail)
      const DECOMP_SCALE = 3;           // step arrows are drawn 3x so one step is visible

      // Thin arrow for vector fields and step decompositions (world coordinates on the surface).
      function thinArrow(from, to, color, width = 1.6, head = 7, outline = false) {
        const [x1, y1] = onSurface(...from), [x2, y2] = onSurface(...to);
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
        if (len < 2) return;
        const ux = dx / len, uy = dy / len, hd = Math.min(head, len * 0.45);
        const shaft = () => { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - ux * hd * 0.7, y2 - uy * hd * 0.7); };
        const tip = () => {
          ctx.beginPath();
          ctx.moveTo(x2, y2);
          ctx.lineTo(x2 - ux * hd - uy * hd * 0.5, y2 - uy * hd + ux * hd * 0.5);
          ctx.lineTo(x2 - ux * hd + uy * hd * 0.5, y2 - uy * hd - ux * hd * 0.5);
          ctx.closePath();
        };
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        if (outline) {
          // Dark rim so white arrows stay readable on the bright top of the hill.
          ctx.strokeStyle = 'rgba(14,15,19,0.85)'; ctx.lineWidth = width + 3;
          shaft(); ctx.stroke();
          tip(); ctx.stroke();
        }
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
        shaft(); ctx.stroke();
        tip(); ctx.fill();
        ctx.lineCap = 'butt';
      }

      // A vector field on a grid. Lengths are relative (scaled per frame), capped so arrows stay short.
      function drawField(fn, color, { minLen = 0, maxLen = 0.5, outline = false, width = 1.6, head = 7 } = {}) {
        const pts = [];
        for (let x = -3.9; x <= 3.95; x += 0.65) for (let y = -2.25; y <= 2.3; y += 0.65) {
          const [vx, vy] = fn(x, y);
          pts.push([x, y, vx, vy, Math.hypot(vx, vy)]);
        }
        const mags = pts.map((p) => p[4]).sort((a, b) => a - b);
        const k = 0.3 / (mags[Math.floor(mags.length / 2)] || 1);
        const mmax = mags[mags.length - 1] || 1;
        for (const [x, y, vx, vy, m] of pts) {
          if (m < 1e-9) continue;
          // Without minLen: relative lengths (median ~0.3), capped. With minLen: every arrow at least
          // minLen long, growing to maxLen for the strongest, so short ones still read as arrows.
          const len = minLen > 0 ? minLen + (maxLen - minLen) * (m / mmax) : Math.min(maxLen, m * k);
          thinArrow([x, y], [x + (vx / m) * len, y + (vy / m) * len], color, width, head, outline);
        }
      }

      // Trail of one particle up to fractional index n; colors follow time (data -> noise).
      function drawTimeTrail(path, n, timeAt, alpha) {
        ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        for (let i = 1; i <= Math.ceil(n) && i < path.length; i++) {
          const a = onSurface(...path[i - 1]), b = onSurface(...at(path, Math.min(n, i)));
          ctx.strokeStyle = rgb(mix(C_DATA, C_NOISE, timeAt(i) / T), alpha);
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
        ctx.lineCap = 'butt';
      }

      function drawSde() {
        const pk = P.peak;
        const paths = sde ? P.fwd : P.rev;
        const n = s;                              // index along the paths
        const tNow = sde ? s : T - s;             // diffusion time shown
        const timeAt = sde ? (i) => i : (i) => T - i;
        const ab = abAt(tNow), a = Math.sqrt(ab), v = ab * pk.s * pk.s + 1 - ab;
        const kStep = Math.min(T, Math.max(1, Math.round(tNow) + (sde ? 1 : 0)));
        const b = beta[kStep];

        // Stage 1: the deterministic part. Forward: the drift field -1/2 beta x pulls everything
        // to 0, and the peak's center slides along sqrt(alphaBar) mu. Reverse: the score field.
        // Stage 2 (forward) is about one particle's random kick: nothing else competes for attention.
        const focusKick = sde && stage >= 2;

        if (stage >= 1 && !focusKick) {
          if (sde) {
            drawField((x, y) => [-0.5 * b * x, -0.5 * b * y], 'rgba(255,255,255,0.55)');
            ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2.5; ctx.setLineDash([6, 5]);
            ctx.beginPath();
            for (let j = 0; j <= T; j += 2) {
              const q = Math.sqrt(alphaBar[j]);
              const [sx, sy] = onSurface(q * pk.mx, q * pk.my);
              if (j === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
            }
            ctx.stroke(); ctx.setLineDash([]);
            const [mx, my] = onSurface(a * pk.mx, a * pk.my);
            drawDot(mx, my, '#fff', 6);
          } else {
            drawField((x, y) => [-(x - a * pk.mx) / v, -(y - a * pk.my) / v], 'rgba(255,255,255,0.6)');
          }
        }

        if (!focusKick) for (const h of HL) drawTimeTrail(paths[h], n, timeAt, 0.9);

        const tint = rgb(mix(C_DATA, C_NOISE, tNow / T));
        const shown = HL;
        for (let i = 0; i < paths.length; i++) {
          if (shown.includes(i)) continue;
          const [sx, sy] = onSurface(...at(paths[i], n));
          ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = focusKick ? 'rgba(150,155,170,0.35)' : tint; ctx.fill();
        }

        // Stage 2 (forward): one particle's current step, split into drift (white) + random kick.
        if (focusKick) {
          const i0 = Math.min(T - 1, Math.floor(n));
          const x = paths[HL[0]][i0], x1 = paths[HL[0]][i0 + 1], bk = beta[i0 + 1];
          const d = [(Math.sqrt(1 - bk) - 1) * x[0], (Math.sqrt(1 - bk) - 1) * x[1]];
          const e = [x1[0] - x[0] - d[0], x1[1] - x[1] - d[1]];
          const tip = [x[0] + DECOMP_SCALE * d[0], x[1] + DECOMP_SCALE * d[1]];
          thinArrow(x, tip, '#fff', 3, 10);
          thinArrow(tip, [tip[0] + DECOMP_SCALE * e[0], tip[1] + DECOMP_SCALE * e[1]], rgb(C_NOISE), 3, 10);
        }

        for (const h of shown) {
          const [sx, sy] = onSurface(...at(paths[h], n));
          drawDot(sx, sy, tint, 7);
        }
      }

      // Score mode: arrows of grad log p on the surface. For N(mu, s^2 I) the score is -(x - mu) / s^2:
      // everywhere it points uphill, toward the top, and grows with the distance from it.
      function drawScore() {
        const k = SCORE.comps[0];
        drawField((x, y) => [-(x - k.mx) / (k.s * k.s), -(y - k.my) / (k.s * k.s)], '#fff',
          { minLen: 0.22, maxLen: 0.5, outline: true, width: 2.2, head: 9 });
      }

      // ---- Flow mode: the long reverse path, the deterministic one and a straight line, on one Gaussian ----
      // Stage 0: the reverse path draws itself, one dot per step = one network call.
      // Stage 1: without the noise (DDIM): one smooth but curved path, FLOW_DDIM steps.
      // Stage 2: flow matching goes straight from the noise to the image, in a few steps.
      // Stage 3: training: a point x_t = (1 - t) x_0 + t eps on the line; the network guesses x_0.
      const FEW = 4;
      function drawFlow() {
        // The straight path: straight on the map, lying on the surface (it climbs the hill).
        const linePt = (f) => onSurface(...lerp2(xT, x0, f));
        const lineTo = (f0, f1) => {
          const n = Math.max(2, Math.ceil(60 * Math.abs(f1 - f0)));
          ctx.beginPath();
          for (let i = 0; i <= n; i++) {
            const [lx, ly] = linePt(f0 + ((f1 - f0) * i) / n);
            if (i === 0) ctx.moveTo(lx, ly); else ctx.lineTo(lx, ly);
          }
        };

        // A path drawn up to fractional step n, one dot per step, with a live call counter.
        function stepPath(path, n, color, r, counter) {
          drawPath(path, n, color, 2);
          ctx.fillStyle = color;
          for (let i = 1; i <= Math.floor(n) && i < path.length; i++) {
            const [sx, sy] = onSurface(...path[i]);
            ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();
          }
          if (counter) tag('calls', `\\text{network calls: } ${Math.round(n)}`, W / 2, 26, '#fff', `network calls: ${Math.round(n)}`);
        }

        // The long random path and the deterministic one. Gone in the training stage, to keep it clean.
        if (stage <= 2) {
          const dim = stage >= 1;
          stepPath(sample, (stage === 0 ? stageP : 1) * T, rgb(C_NOISE, dim ? 0.3 : 0.95), 2.6, !dim);
        }
        if (stage === 1 || stage === 2) {
          const dim = stage >= 2;
          stepPath(FLOW_DDIM_PATH, (stage === 1 ? stageP : 1) * FLOW_DDIM, `rgba(255,255,255,${dim ? 0.35 : 0.95})`, 3.4, !dim);
        }

        // The straight line; in stage 2 with its few steps.
        if (stage >= 2) {
          const p1 = stage === 2 ? ease(stageP) : 1;
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          if (stage >= 3) ctx.setLineDash([8, 7]);
          lineTo(0, p1); ctx.stroke();
          ctx.setLineDash([]); ctx.lineCap = 'butt';
          if (stage === 2) {
            for (let i = 1; i < FEW && i / FEW <= p1; i++) drawDot(...linePt(i / FEW), '#fff', 5);
            if (stageP >= 1) tag('calls', `\\text{network calls: } ${FEW}`, W / 2, 26, '#fff', `network calls: ${FEW}`);
          }
        }

        // Endpoints: noise eps and the image x_0.
        const [px, py] = linePt(1);
        ctx.strokeStyle = rgb(C_DATA); ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.stroke();
        const [nx, ny] = linePt(0);
        drawDot(nx, ny, rgb(C_NOISE), 7);

        // Stage 3: a point x_t on the path, its image, and the direction the network learns
        // (orange): along the straight path toward x_0, i.e. -v.
        if (stage >= 3 && ready) {
          const t = 1 - 0.45 * ease(stage === 3 ? stageP : 1);
          const [sx, sy] = linePt(1 - t);
          // Arrow along the surface from x_t, stopping short of the ring at x_0.
          const fEnd = 1 - 16 / Math.max(16, Math.hypot(px - sx, py - sy)) * t;
          if (fEnd > 1 - t + 0.02) {
            // White with a dark rim, so it stays visible on the orange hill.
            const [ex, ey] = linePt(fEnd), [qx, qy] = linePt(fEnd - 0.03);
            const l = Math.hypot(ex - qx, ey - qy) || 1, ux = (ex - qx) / l, uy = (ey - qy) / l, hd = 16;
            const head = () => {
              ctx.beginPath(); ctx.moveTo(ex, ey);
              ctx.lineTo(ex - ux * hd - uy * hd * 0.55, ey - uy * hd + ux * hd * 0.55);
              ctx.lineTo(ex - ux * hd + uy * hd * 0.55, ey - uy * hd - ux * hd * 0.55);
              ctx.closePath();
            };
            ctx.lineCap = 'round'; ctx.lineJoin = 'round';
            ctx.strokeStyle = 'rgba(14,15,19,0.85)'; ctx.lineWidth = 8;
            lineTo(1 - t, fEnd - 0.02); ctx.stroke();
            head(); ctx.lineWidth = 4; ctx.stroke();
            ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 4;
            lineTo(1 - t, fEnd - 0.02); ctx.stroke();
            head(); ctx.fill();
            ctx.lineCap = 'butt';
          }
          cardOnDot(fp.card, nx, ny, 1, rgb(C_NOISE, 0.8), true);
          cardOnDot(fp.card2, px, py, 0, rgb(C_DATA, 0.8), true);
          cardOnDot(fp.card3, sx, sy, t, 'rgba(255,255,255,0.8)', true);
          drawDot(sx, sy, '#fff', 7);
          // Labels below the line, so they don't sit on it or on the images.
          tag('xt', 'x_t', sx, sy + 34, '#fff', 'x_t');
          tag('eps', '\\varepsilon', nx, ny + 34, rgb(C_NOISE), 'eps');
          tag('x0', 'x_0', px + 40, py + 8, rgb(C_DATA), 'x0');
        }
      }

      // ---- Manifold mode ----
      // Stage 1: the manifold as a dashed line along the ridge, and random points everywhere else:
      // almost all of the space is empty, i.e. noise.
      const offManifold = (() => {
        if (!manifoldMode) return [];
        const g = rng(321), pts = [];
        while (pts.length < 18) {
          const p = [g() * 0.45 * EXT_X, g() * 0.55 * EXT_Y];
          // Clearly away from the ridge, so no dot looks like it sits on it from the camera's view.
          let dmin = Infinity;
          for (let i = 0; i <= 60; i++) { const [cx, cy] = manifoldCurve(i / 60); dmin = Math.min(dmin, Math.hypot(p[0] - cx, p[1] - cy)); }
          // Dots just behind the ridge would look as if they sat on it (the camera looks along +y):
          // keep them in front of it, or far enough behind.
          const ridgeY = manifoldCurve(Math.max(0, Math.min(1, (p[0] + 3.7) / 7.4)))[1];
          const visible = p[1] < ridgeY - 0.9 || p[1] > ridgeY + 2.2;
          if (inside(...p) && dmin > 1.0 && visible) pts.push(p);
        }
        return pts;
      })();
      function drawManifold() {
        if (stage < 1) return;
        ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2.5; ctx.setLineDash([8, 6]); ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const [x, y] = manifoldCurve(i / 120);
          const [sx, sy] = project(x, y, height(x, y) + 0.03);
          if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }
        ctx.stroke(); ctx.setLineDash([]);
        for (const p of offManifold) drawDot(...onSurface(...p), rgb(C_NOISE, 0.9), 4);
        const [tx, ty] = project(...manifoldCurve(0.62), height(...manifoldCurve(0.62)) + 0.03);
        tag('mf', '\\text{images: a thin manifold}', tx, ty - 46, rgb(C_DATA), 'images: a thin manifold');
        const far = offManifold.reduce((a, b) => (b[1] < a[1] ? b : a));
        const [nx, ny] = onSurface(...far);
        tag('nz', '\\text{everywhere else: noise}', nx, ny + 30, rgb(C_NOISE), 'everywhere else: noise');
      }

      function drawForward() {
        const ab = abAt(s);
        const tint = rgb(mix(C_DATA, C_NOISE, s / T));

        if (diffuse && s > 0) for (const p of peaks) drawTrail(p.path, s, 0.9);

        if (valley.card) {
          const [sx, sy] = onSurface(valley.x, valley.y);
          drawDot(sx, sy, rgb(C_NOISE));
          cardOnDot(valley.card, sx, sy, 0, rgb(C_NOISE, 0.8));
        }

        // Farther dots (higher on screen) first, so nearer images overlap them.
        const items = peaks.map((p) => ({ p, pos: onSurface(...at(p.path, s)) })).sort((a, b) => a.pos[1] - b.pos[1]);
        for (const { p, pos: [sx, sy] } of items) {
          drawDot(sx, sy, tint);
          cardOnDot(p.card, sx, sy, ab, rgb(mix(C_DATA, C_NOISE, s / T), 0.8));
        }
      }

      function draw(still = true) {
        if (diffuse || sde) surfAb = abAt(s);
        if (sdeRev) surfAb = abAt(T - s);
        if (reverse) surfAb = stage >= 5 ? abAt(T - stageP * T) : alphaBar[T];
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(surfaceFor(still), 0, 0);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        if (reverse) drawReverse();
        else if (scoreMode) drawScore();
        else if (manifoldMode) drawManifold();
        else if (flowMode) drawFlow();
        else if (single) drawSde();
        else if (showMarkers && ready) drawForward();
        hideUnusedTags();
      }

      // ---- Animation: camera tilt, zoom, stage progress, slider snapping ----
      let raf = 0;
      let cam = null;    // { from, to, t0 }
      let zAnim = null;  // { from, to, t0 }
      let snap = null;   // { from, to, t0 } for the slider
      let stageT0 = 0;
      const CAM_MS = 1600, ZOOM_MS = 1100, SNAP_MS = 350, STAGE_MS = reverse ? { 2: 5000, 5: 7000 } : flowMode ? { 0: 4500, 1: 2500, 2: 1400, 3: 1800 } : {};
      const prog = (t0, ms, now) => Math.max(0, Math.min(1, (now - t0) / ms));
      function tick(now) {
        raf = 0;
        let moving = false;
        if (cam) {
          const p = prog(cam.t0, CAM_MS, now);
          setCamera(cam.from + (cam.to - cam.from) * p);
          if (p >= 1) cam = null; else moving = true;
        }
        if (zAnim) {
          const p = ease(prog(zAnim.t0, ZOOM_MS, now)), a = zAnim.from, b = zAnim.to;
          zoom = { z: Math.exp(Math.log(a.z) + (Math.log(b.z) - Math.log(a.z)) * p), cx: a.cx + (b.cx - a.cx) * p, cy: a.cy + (b.cy - a.cy) * p };
          if (p >= 1) zAnim = null; else moving = true;
        }
        if (snap) {
          const p = ease(prog(snap.t0, SNAP_MS, now));
          s = snap.from + (snap.to - snap.from) * p;
          if (p >= 1) snap = null; else moving = true;
        }
        if (stageP < 1) {
          stageP = prog(stageT0, STAGE_MS[stage] || 1, now);
          // Stage 5 morphs the landscape, so render it coarse while it plays.
          if (stage >= 5 && stageP < 1) moving = true;
        }
        draw(!moving);
        if (moving || stageP < 1) raf = requestAnimationFrame(tick);
        if (sliderSync) sliderSync();
      }
      function animate() { if (!raf) raf = requestAnimationFrame(tick); }
      function zoomTo(target) { zAnim = { from: { ...zoom }, to: target, t0: performance.now() }; animate(); }

      const slide = el.closest ? el.closest('section') : null;
      const stageFromDom = () => {
        let st = 0;
        if (slide) slide.querySelectorAll('.fragment.visible[data-w15-stage]').forEach((f) => { st = Math.max(st, +f.dataset.w15Stage); });
        return st;
      };
      function setStage(st, animated) {
        const prev = stage;
        stage = st;
        stageP = (reverse || flowMode) && animated && st === prev + 1 && STAGE_MS[st] ? 0 : 1;
        stageT0 = performance.now() + (reverse && (st === 2 || st === 5) ? 500 : 0);
        if (reverse) {
          if (st > 0 && (u < 1 || cam)) { cam = null; setCamera(1); }
          if (animated) zoomTo(zoomFor(st)); else zoom = zoomFor(st);
        }
        animate();
      }
      if ((reverse || single || flowMode || manifoldMode) && window.Reveal) {
        const onFragment = () => { if (el.dataset.active === '1') setStage(stageFromDom(), true); };
        Reveal.on('fragmentshown', onFragment);
        Reveal.on('fragmenthidden', onFragment);
      }

      // ---- Slider (diffuse mode) ----
      let sliderSync = null;
      if (diffuse || single) {
        const bar = document.createElement('div');
        bar.className = 'w15-controls';
        bar.setAttribute('data-prevent-swipe', '');
        // The reverse slide runs from noise to data; its slider counts reverse steps.
        const [left, right] = sdeRev ? ['noise', 'data'] : ['data', 'noise'];
        bar.innerHTML = `<span class="w15-end">${left}</span><input type="range" min="0" max="${T}" step="0.1" value="0" aria-label="diffusion step"><span class="w15-end">${right}</span><span class="w15-label"></span>`;
        const stepLabel = () => `t = ${sdeRev ? T - Math.round(s) : Math.round(s)} / ${T}`;
        el.appendChild(bar);
        const input = bar.querySelector('input');
        const label = bar.querySelector('.w15-label');
        sliderSync = () => {
          input.value = String(s);
          label.textContent = stepLabel();
        };
        let idle = 0;
        input.addEventListener('input', () => {
          snap = null;
          s = +input.value;
          label.textContent = stepLabel();
          // Coarse while dragging, full detail once the hand stops.
          draw(false);
          clearTimeout(idle);
          idle = setTimeout(() => draw(true), 150);
        });
        // On release, settle on the nearest whole step and hand the arrow keys back to reveal.
        input.addEventListener('change', () => {
          snap = { from: s, to: Math.round(s), t0: performance.now() };
          animate();
        });
        input.addEventListener('pointerup', () => input.blur());
        sliderSync();
      }

      // Images: data-images overrides the default photos.
      const custom = (el.dataset.images || '').split(',').map((x) => x.trim()).filter(Boolean);
      Promise.all(peaks.map(async (p, i) => {
        const cv = blankCanvas();
        const c = cv.getContext('2d');
        const im = await loadImage(custom[i] || IMAGES[i % IMAGES.length]);
        if (im) {
          const side = Math.min(im.naturalWidth, im.naturalHeight);
          c.drawImage(im, (im.naturalWidth - side) / 2, (im.naturalHeight - side) / 2, side, side, 0, 0, IMG, IMG);
        } else {
          c.fillStyle = '#555'; c.fillRect(0, 0, IMG, IMG);
        }
        const px = pixels(cv);
        p.card = makeCard(px, 500 + i);
        // Same noise seed: the flow slide shows eps, x_0 and their interpolation side by side.
        if (flowMode) { p.card2 = makeCard(px, 500 + i); p.card3 = makeCard(px, 500 + i); }
      })).then(() => { ready = true; draw(); });

      // The single-peak slides are seen from straight above.
      setCamera(single ? 1 : 0);
      draw();

      return {
        start() {
          if (flowMode) {
            // Entering from the previous slide: the long path draws itself step by step.
            stage = stageFromDom();
            if (stage === 0) { stageP = 0; stageT0 = performance.now() + 400; animate(); } else { stageP = 1; draw(); }
            return;
          }
          if (!reverse) { stage = stageFromDom(); draw(); return; }
          // Entering from the previous slide: tilt the camera down. Coming back from a later
          // slide (fragments already shown): jump straight to the top view at that stage.
          const st = stageFromDom();
          stage = st; stageP = 1; zAnim = null;
          if (st === 0) {
            zoom = ZOOM0; setCamera(0); draw();
            cam = { from: 0, to: 1, t0: performance.now() + 250 };
            animate();
          } else {
            cam = null; setCamera(1); zoom = zoomFor(st); draw();
          }
        },
        stop() {
          if (raf) cancelAnimationFrame(raf);
          raf = 0;
        },
      };
    },
  });
})();
