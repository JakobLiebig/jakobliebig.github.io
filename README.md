# Diffusion → Flow Matching

Seminar talk for "Open Generative AI Systems" (TUM, WS26) that doubles as a readable article.
Served by GitHub Pages from the repo root at https://jakobliebig.github.io. See `specs.md` for the full plan.

## Views

- Present: `index.html` (press `S` for speaker notes)
- Article: `index.html?view=scroll`
- PDF export: `index.html?print-pdf`, then print from Chrome

## Run locally

Everything is vendored, so no network is needed:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Layout

- `index.html`: all slides
- `css/theme.css`: theme and color code
- `js/main.js`: reveal init and widget lifecycle
- `js/widgets/`: one file per widget, each calls `Widgets.register(id, { init(el, data) })`
- `data/`: precomputed JSON and images
- `precompute/`: Python scripts (not used by the site)
- `vendor/`: reveal.js 5.2.1, KaTeX 0.16, D3 7, Inter
