// W16: thumbnails for the DDPM Markov chain slide.
//
// Every <canvas data-ab="..."> inside the host shows the host's data-image noised to that level:
// sqrt(ab) x0 + sqrt(1 - ab) eps, with pixels in [-1, 1] as DDPM sees them. ab = 1 is the clean
// image, ab = 0 pure noise. One fixed eps is shared by all thumbnails, so they read as one chain.

(function () {
  const IMG = 96;

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

  function loadImage(src) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => { console.warn(`W16: could not load ${src}`); resolve(null); };
      im.src = src;
    });
  }

  Widgets.register('W16', {
    init(el) {
      const canvases = [...el.querySelectorAll('canvas[data-ab]')];
      const g = rng(+(el.dataset.seed || 503));
      const eps = new Float32Array(IMG * IMG * 3).map(() => g());

      loadImage(el.dataset.image).then((im) => {
        const src = document.createElement('canvas');
        src.width = src.height = IMG;
        const sc = src.getContext('2d');
        if (im) {
          const side = Math.min(im.naturalWidth, im.naturalHeight);
          sc.drawImage(im, (im.naturalWidth - side) / 2, (im.naturalHeight - side) / 2, side, side, 0, 0, IMG, IMG);
        } else {
          sc.fillStyle = '#555'; sc.fillRect(0, 0, IMG, IMG);
        }
        const d0 = sc.getImageData(0, 0, IMG, IMG).data;

        for (const cv of canvases) {
          const ab = Math.max(0, Math.min(1, +cv.dataset.ab));
          const a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
          cv.width = cv.height = IMG;
          const ctx = cv.getContext('2d');
          const out = ctx.createImageData(IMG, IMG);
          const d = out.data;
          for (let i = 0, j = 0; i < d.length; i += 4) {
            for (let c = 0; c < 3; c++, j++) {
              const v = a * (d0[i + c] / 127.5 - 1) + b * eps[j];
              d[i + c] = Math.max(0, Math.min(255, (v + 1) * 127.5));
            }
            d[i + 3] = 255;
          }
          ctx.putImageData(out, 0, 0);
        }
      });

      return { start() {}, stop() {} };
    },
  });
})();
