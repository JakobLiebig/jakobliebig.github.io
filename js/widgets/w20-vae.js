// W20: a variational autoencoder on MNIST, live in the browser, in three steps (fragments with
// data-w20-stage in the same slide):
//   0  input (pick one of ten MNIST test digits) -> encoder -> latent code -> decoder -> output
//   1  only latent code -> decoder -> output: a code alone is enough to make an image
//   2  the 2D latent space takes the code's place (the code becomes a note underneath); drag or
//      click in it to move z, and the decoded image follows
// The latent map shows 10,000 encoded test digits colored by class.
//
// Model: VAE Explainer (Bertucci & Endert 2024, MIT), TensorFlow.js graph models in
// data/latent/vae-explainer/. Encoder: 28x28x1 in [0, 1] -> [logvar_1, logvar_2, mu_1, mu_2].
// Decoder: z (2) -> 28x28x1. Runs on TensorFlow.js (vendor/tfjs) with the CPU backend.

(function () {
  const BASE = 'data/latent/vae-explainer/';
  const SAMPLES = 'data/latent/mnist-samples.json';
  const RANGE = 4;     // the latent map covers [-RANGE, RANGE]^2
  const FALLBACK_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];

  let modelsPromise = null;
  function loadAll() {
    if (!modelsPromise) {
      modelsPromise = (async () => {
        await tf.setBackend('cpu');
        const [enc, dec, latents, samples] = await Promise.all([
          tf.loadGraphModel(BASE + 'encoder/model.json'),
          tf.loadGraphModel(BASE + 'decoder/model.json'),
          fetch(BASE + 'latents.json').then((r) => r.json()),
          fetch(SAMPLES).then((r) => r.json()),
        ]);
        return { enc, dec, latents, samples };
      })();
    }
    return modelsPromise;
  }
  // Shared with the latent flow slide (W21), which decodes with the same model.
  window.W20Models = loadAll;

  // Draw 784 values in [0, 1] (or 0..255 with scale 1/255) into a 28x28 canvas.
  function drawDigit(cv, values, scale = 1) {
    const c = cv.getContext('2d');
    const img = c.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(255, values[i] * scale * 255));
      img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v;
      img.data[4 * i + 3] = 255;
    }
    c.putImageData(img, 0, 0);
  }

  function makeCanvas(cls, w = 28, h = 28) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h; cv.className = cls;
    return cv;
  }

  function trapezoid(kind) {
    // Encoder narrows (784 -> 2), decoder widens (2 -> 784).
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 220');
    svg.setAttribute('class', `w20-trap ${kind}`);
    const wide = [20, 200], narrow = [80, 140];
    const [l, r] = kind === 'enc' ? [wide, narrow] : [narrow, wide];
    const poly = document.createElementNS(ns, 'polygon');
    poly.setAttribute('points', `0,${l[0]} 120,${r[0]} 120,${r[1]} 0,${l[1]}`);
    svg.appendChild(poly);
    const t1 = document.createElementNS(ns, 'text');
    t1.setAttribute('x', 60); t1.setAttribute('y', 106); t1.setAttribute('class', 'w20-trap-title');
    t1.textContent = kind === 'enc' ? 'Encoder' : 'Decoder';
    const t2 = document.createElementNS(ns, 'text');
    t2.setAttribute('x', 60); t2.setAttribute('y', 128); t2.setAttribute('class', 'w20-trap-sub');
    t2.textContent = kind === 'enc' ? '784 → μ, σ' : 'z → 784';
    svg.appendChild(t1); svg.appendChild(t2);
    return svg;
  }

  Widgets.register('W20', {
    init(el) {
      el.classList.add('w20');
      el.innerHTML = '';

      // ---- Layout: one row; parts are shown or hidden per step ----
      const label = (txt) => { const d = document.createElement('div'); d.className = 'w20-label'; d.innerHTML = txt; return d; };
      const col = (cls, ...kids) => { const d = document.createElement('div'); d.className = `w20-col ${cls}`; d.append(...kids); return d; };

      const inCanvas = makeCanvas('w20-digit');
      const thumbs = document.createElement('div'); thumbs.className = 'w20-thumbs';
      const inputPart = col('w20-input', label('Input <span>784 pixels</span>'), inCanvas, thumbs);
      const encTrap = trapezoid('enc');

      const ratio = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
      const MAP = 230;
      const map = makeCanvas('w20-map', MAP * ratio, MAP * ratio);
      map.style.width = map.style.height = `${MAP}px`;
      const zNote = document.createElement('div'); zNote.className = 'w20-znote';
      const mapPart = col('w20-mappart', label('2D latent space <span>drag to pick z</span>'), map, zNote);

      const vec = document.createElement('div'); vec.className = 'w20-vec';
      const v1 = document.createElement('span'), v2 = document.createElement('span');
      vec.append(v1, v2);
      const sig = document.createElement('div'); sig.className = 'w20-sig';
      const codeLabel = label('Latent code z');
      const codePart = col('w20-codepart', codeLabel, vec, sig);

      const decTrap = trapezoid('dec');
      const outCanvas = makeCanvas('w20-digit');
      const outPart = col('w20-out', label('Output <span>784 pixels</span>'), outCanvas);

      const status = document.createElement('div'); status.className = 'w20-status'; status.textContent = 'loading model…';
      // Fixed slots, so nothing moves between steps: the left slot holds input + encoder (step 0 only);
      // the center slot holds the code vector (steps 0-1) or, in step 2, the 2D space with the code as
      // a small note underneath.
      const leftA = document.createElement('div'); leftA.className = 'w20-left-a'; leftA.append(inputPart, encTrap);
      const left = document.createElement('div'); left.className = 'w20-left'; left.append(leftA);
      const center = document.createElement('div'); center.className = 'w20-center'; center.append(codePart, mapPart);
      el.append(left, center, decTrap, outPart, status);
      el.setAttribute('data-prevent-swipe', '');

      // ---- State ----
      let M = null;            // models + data once loaded
      let z = [0, 0], mu = null, sigma = null;
      let stage = 0;
      let mapBase = null;      // cached scatter of all latents

      const toMap = (v) => ((v + RANGE) / (2 * RANGE)) * MAP;
      const fromMap = (px) => (px / MAP) * 2 * RANGE - RANGE;

      function applyStage() {
        el.dataset.stage = String(stage);
        sig.style.visibility = stage === 0 ? 'visible' : 'hidden';
        if (window.Reveal && Reveal.layout) Reveal.layout();
      }

      function drawMap() {
        const c = map.getContext('2d');
        c.setTransform(1, 0, 0, 1, 0, 0);
        c.clearRect(0, 0, map.width, map.height);
        if (mapBase) c.drawImage(mapBase, 0, 0);
        c.setTransform(ratio, 0, 0, ratio, 0, 0);
        const x = toMap(z[0]), y = MAP - toMap(z[1]);
        c.beginPath(); c.arc(x, y, 7, 0, Math.PI * 2); c.fillStyle = '#0e0f13'; c.fill();
        c.beginPath(); c.arc(x, y, 5.5, 0, Math.PI * 2); c.fillStyle = '#fff'; c.fill();
      }

      function buildMapBase() {
        mapBase = document.createElement('canvas');
        mapBase.width = map.width; mapBase.height = map.height;
        const c = mapBase.getContext('2d');
        c.setTransform(ratio, 0, 0, ratio, 0, 0);
        c.fillStyle = '#171920'; c.fillRect(0, 0, MAP, MAP);
        c.strokeStyle = 'rgba(255,255,255,0.08)'; c.lineWidth = 1;
        c.beginPath(); c.moveTo(MAP / 2, 0); c.lineTo(MAP / 2, MAP); c.moveTo(0, MAP / 2); c.lineTo(MAP, MAP / 2); c.stroke();
        const colors = (window.d3 && d3.schemeTableau10) || FALLBACK_COLORS;
        const { points, labels } = M.latents;
        c.globalAlpha = 0.35;
        for (let i = 0; i < points.length; i++) {
          const [a, b] = points[i];
          if (Math.abs(a) > RANGE || Math.abs(b) > RANGE) continue;
          c.fillStyle = colors[labels[i] % 10];
          c.fillRect(toMap(a) - 0.9, MAP - toMap(b) - 0.9, 1.8, 1.8);
        }
        c.globalAlpha = 1;
      }

      // The code vector, the decoded image and the map dot always show the current z.
      function decode() {
        if (!M) return;
        const out = tf.tidy(() => M.dec.predict(tf.tensor([z], [1, 2])).dataSync());
        drawDigit(outCanvas, out);
        v1.textContent = z[0].toFixed(2); v2.textContent = z[1].toFixed(2);
        zNote.textContent = `z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
        drawMap();
      }

      function encode(i) {
        const px = M.samples.digits[i].pixels;
        drawDigit(inCanvas, px, 1 / 255);
        [...thumbs.children].forEach((t, k) => t.classList.toggle('active', k === i));
        const code = tf.tidy(() => M.enc.predict(tf.tensor(Float32Array.from(px, (v) => v / 255), [1, 28, 28, 1])).dataSync());
        // Encoder output: [logvar_1, logvar_2, mu_1, mu_2]. The code shown is the mean mu.
        mu = [code[2], code[3]];
        sigma = [Math.exp(0.5 * code[0]), Math.exp(0.5 * code[1])];
        sig.textContent = `= μ, with σ = (${sigma[0].toFixed(2)}, ${sigma[1].toFixed(2)})`;
        z = mu.slice();
        decode();
      }

      // ---- Interaction: drag in the 2D space ----
      let dragging = false;
      const pick = (e) => {
        const r = map.getBoundingClientRect();
        const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
        z = [Math.max(-RANGE, Math.min(RANGE, fromMap(fx * MAP))), Math.max(-RANGE, Math.min(RANGE, -fromMap(fy * MAP)))];
        decode();
      };
      map.addEventListener('pointerdown', (e) => { dragging = true; map.setPointerCapture(e.pointerId); pick(e); });
      map.addEventListener('pointermove', (e) => { if (dragging) pick(e); });
      map.addEventListener('pointerup', () => { dragging = false; });

      // ---- Steps ----
      const slide = el.closest('section');
      const stageFromDom = () => {
        let st = 0;
        if (slide) slide.querySelectorAll('.fragment.visible[data-w20-stage]').forEach((f) => { st = Math.max(st, +f.dataset.w20Stage); });
        return st;
      };
      if (window.Reveal) {
        const onFragment = () => { if (el.dataset.active === '1') { stage = stageFromDom(); applyStage(); } };
        Reveal.on('fragmentshown', onFragment);
        Reveal.on('fragmenthidden', onFragment);
      }
      applyStage();

      loadAll().then((m) => {
        M = m;
        status.remove();
        M.samples.digits.forEach((d, i) => {
          const t = makeCanvas('w20-thumb');
          drawDigit(t, d.pixels, 1 / 255);
          t.title = `digit ${d.label}`;
          t.addEventListener('click', () => encode(i));
          thumbs.appendChild(t);
        });
        buildMapBase();
        encode(0);
        applyStage();
      }).catch((err) => {
        console.error('W20:', err);
        status.textContent = 'could not load the model (serve the deck over http)';
      });

      return {
        start() { stage = stageFromDom(); applyStage(); if (M) drawMap(); },
        stop() {},
      };
    },
  });
})();
