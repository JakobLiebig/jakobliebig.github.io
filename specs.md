# Diffusion → Flow Matching: Talk & Website Build Spec

Oct 5, 2026 · @Jakob Liebig

## Overview

Build a reveal.js website that is both a 30-minute seminar talk (present mode) and a readable article (scroll view), hosted on GitHub Pages. The talk explains DDPM, score-based SDEs, the probability flow ODE and flow matching as one idea, then shows an original experiment on an open rectified flow model.

**Context:** TUM seminar "Open Generative AI Systems" (Prof. Lasser, WS26). Speaker: Jakob Liebig. Repo: `JakobLiebig/jakobliebig.github.io` (or a project repo; see open questions). Talk slot: 30 min.

**Thesis of the talk:** generation is reversing a physical diffusion process, like running a video of ink dissolving in water backwards. DDPM, score SDEs and flow matching are three ways of learning that reversal; flow matching learns the straightest one.

**Hard constraints for the builder:**

- Static site only. No backend, no in-browser training. All model outputs are precomputed in Python and loaded as JSON or images.
- One source, two modes: the same deck must work as keyboard slides and as `?view=scroll`.
- Every animated slide needs a static fallback frame (PDF export via `?print-pdf` must stay readable).
- Must run fully offline from a local clone (no CDN at talk time; vendor all JS).
- Relative asset paths only.
- Math in KaTeX. Equations: at most one key equation per slide in the talk; full derivations go in optional deep-dive sub-slides.

## Stack, repo layout, deployment

Plain HTML + reveal.js 5.x (vendored), KaTeX plugin, and vanilla JS + Canvas 2D for widgets (D3 only where axes are needed). No framework, no bundler, so GitHub Pages serves the repo root as is.

| Piece | Choice | Why |
| --- | --- | --- |
| Slides | reveal.js 5.x, vendored in `vendor/reveal/` | Present mode + built-in scroll view, PDF export |
| Math | reveal.js KaTeX plugin, vendored | Fast, offline |
| Widgets | Vanilla JS + Canvas 2D, D3 v7 for axes | Thousands of particles at 60 fps |
| Data | JSON (float32 rounded to 3 decimals), PNG/WebP frames | Precomputed offline |
| Precompute | Python 3.11, PyTorch, diffusers | Toy models + SD3.5/FLUX experiment |
| Hosting | GitHub Pages, deploy from `main` root, `.nojekyll` | No build step |

Repo layout:

```
/index.html            # the deck, all slides
/.nojekyll
/css/theme.css         # custom theme, CSS variables
/js/widgets/           # one file per widget, each exports init(el, data)
/js/main.js            # reveal init, widget lifecycle on slidechanged
/data/toy/             # precomputed 2D trajectories, fields
/data/experiment/      # similarity curves, prompt-switch frames
/media/                # hook videos (local copies, gitignored if large), ink frames
/vendor/               # reveal, katex, d3
/precompute/           # Python scripts, never served
/README.md
```

Widget lifecycle: each widget starts on `slidechanged` into its slide and stops when leaving (cancel `requestAnimationFrame`). In scroll view, start widgets with an IntersectionObserver instead. Widgets read `data-*` attributes for config so the same widget can appear on several slides.

reveal config: `hash: true`, `slideNumber: 'c/t'`, `transition: 'fade'`, `controls: false` in present mode, speaker notes via `<aside class="notes">`. Deep-dive material goes in vertical sub-slides that the speaker skips during the talk.

## Design system and the shared toy dataset

One 2D dataset and one fixed color code run through the whole talk, so the audience never has to re-learn a plot.

**Toy dataset:** two moons, 4,000 points, scaled to roughly \[-2, 2\]². Source distribution: standard Gaussian N(0, I). Same random seed for every method so trajectories are directly comparable.

**Color code (fixed meaning everywhere):**

| Element | Color | Meaning |
| --- | --- | --- |
| Data points | warm orange | samples from p\_data |
| Noise points | cool blue-grey | samples from N(0, I) |
| Particles in motion | gradient blue → orange by time t | where a sample is during generation |
| Score / velocity arrows | white at 60% opacity | the learned field |
| DDPM / SDE paths | jagged, thin | stochastic |
| ODE / flow matching paths | smooth, thicker | deterministic |

**Visual style:** dark background (#0e0f13), one sans-serif (Inter, vendored), monospace for code. Plots are square, no gridlines, axes hidden unless needed. Ink-in-water motif reused as the section divider animation.

**Time convention (state it once on screen):** t = 0 is data, t = 1 is noise, as in the SDE literature. Flow matching papers often use the opposite; the deck flips their notation to match and says so in a footnote.

## Slide-by-slide outline

32 main slides in 8 acts, about 29 minutes. Each slide lists: on-screen content, visual (widget IDs refer to the next section), and the speaker's point. Deep-dive sub-slides are marked ↓ and skipped live.

| Act | Slides | Time |
| --- | --- | --- |
| 0. Hook | 1–4 | 1.5 min |
| 1. The problem: sampling p(x) | 5–8 | 4.5 min |
| 2. Forward diffusion and the SDE | 9–12 | 4 min |
| 3. Learning the score | 13–15 | 3 min |
| 4. SDE → probability flow ODE | 16–17 | 2.5 min |
| 5. Flow matching | 18–22 | 4 min |
| 6. Scaling up | 23–25 | 3 min |
| 7. Experiment | 26–30 | 5.5 min |
| 8. Close | 31–32 | 1 min |

### Act 0: Hook (1.5 min)

1. **Title.** Talk title, name, seminar. Visual: W0 ink drop slowly diffusing behind the title. Point: none, just set the mood.
2. **"This is Will Smith eating spaghetti."** The viral 2023 ModelScope text-to-video clip, full screen, no other text. Present mode plays a local mp4; scroll view embeds YouTube.
3. **"…and this is also Will Smith eating spaghetti."** A recent clip (2025-era model). Same layout. Point: two years, same prompt, enormous jump.
4. **"…and this is a drop of ink in water."** Real ink-in-water footage, forward. Speaker: "By the end of this talk you'll see these are the same idea." Do not explain yet.

### Act 1: The problem (4.5 min)

5. **Images are points.** A 64×64 RGB image is one point in 12,288 dimensions. Visual: an image dissolving into a single dot on a plane. Point: from now on, the 2D plane stands in for image space; each orange dot is "an image".
6. **p(x).** Visual: W1 two-moons scatter, toggle to density heatmap. Point: realistic images sit where p(x) is high. Generating = drawing samples from p(x).
7. **Why not just build the map?** Visual: a grid over the plane, then a counter: 10 bins per dimension → 10^12288 cells. Point: we can never write down p(x), and we don't need to. We only need samples.
8. **Transport.** Visual: W2 blue Gaussian cloud morphing into the moons (uses the flow matching result as a preview). Point: start from noise we can sample trivially, learn a transformation into data. The rest of the talk is about how to learn that transformation.

### Act 2: Forward diffusion and the SDE (4 min)

9. **Destroying data is easy.** DDPM forward step: x\_t = √(1 − β\_t)·x\_{t−1} + √β\_t·ε. Visual: W3 slider over t, moons dissolve into the Gaussian. Point: adding noise needs no learning, and there is a closed form x\_t = √ᾱ\_t·x\_0 + √(1 − ᾱ\_t)·ε.
10. **Follow single particles.** Visual: W3 in particle mode, 50 highlighted jagged paths. Speaker callback: "This is the ink."
11. **Discrete → continuous.** Visual: W4 the same paths with 10 / 100 / 1000 steps, converging to a continuous random path. Point: as steps → 0, DDPM becomes an SDE: dx = f(x,t)dt + g(t)dW.
12. **The physics payoff.** The density obeys the Fokker–Planck equation, the same law that spreads ink in water. Credit Sohl-Dickstein et al. 2015 (inspired by non-equilibrium thermodynamics). Point: and physics tells us this process can be run backwards (Anderson 1982), if we know one thing: the score.
    - ↓ Deep dive: Fokker–Planck derivation sketch; VP vs. VE SDEs.

### Act 3: Learning the score (3 min)

13. **The reverse SDE.** dx = \[f − g²·∇ₓ log p\_t(x)\]dt + g·dW̄. Visual: W5 score field arrows over the plane, time slider; arrows point back toward the moons. Point: the score ∇ log p is "which way is more likely".
14. **But we don't know p\_t.** The trick: denoising score matching. Take a data point, add known noise, ask the network to predict that noise. The score of the noised distribution is −ε/σ\_t. Visual: one x\_0, its noisy x\_t, the arrow back.
15. **DDPM was secretly a score model.** Show the DDPM loss ‖ε − ε\_θ(x\_t, t)‖². Visual: W6 reverse sampling, particles flowing from noise to moons along jagged paths. Point: predict-the-noise = learn the score.

### Act 4: SDE → probability flow ODE (2.5 min)

16. **Same marginals, no randomness.** Probability flow ODE: dx/dt = f − ½g²·∇ log p\_t(x). Visual: W7 SDE paths vs. ODE paths side by side, with marginal histograms that match at every t. Point: we can drop the noise and still get the same distribution.
17. **Why care, and what's still wrong.** Deterministic, invertible, faster solvers. But paths are curved, so few-step sampling is blurry. Visual: W8 step-count slider (1–50) on the ODE; at 5 steps the moons smear. Point: curved paths are expensive. Can we make them straight?

### Act 5: Flow matching (4 min)

18. **Choose the path, then learn it.** Skip the SDE entirely. Pick a noise sample and a data sample, connect them with a straight line: x\_t = (1 − t)·x\_0 + t·ε. Visual: W9 lines between random pairs.
19. **Conditional vs. marginal.** Individual lines cross; the network learns their average, a single velocity field whose paths don't cross. Visual: W9 toggles from pair lines to learned trajectories.
20. **The loss.** ‖v\_θ(x\_t, t) − (ε − x\_0)‖². Put it next to the DDPM loss: nearly the same line of code. Point: same recipe, different regression target.
21. **The comparison.** Visual: W10 three panels (SDE, PF-ODE, flow matching) on the same seed, shared step slider. Flow matching holds up at few steps. Mention rectified flow and that SD3 and FLUX are trained this way.
22. **One idea, three parameterizations.** Given x\_t, noise prediction, score and velocity are linear transformations of each other. Ink callback: flow matching learns the straightest way to un-dissolve the ink.
    - ↓ Deep dive: the conversion formulas between ε, score and v; stochastic interpolants.

### Act 6: Scaling up (3 min)

23. **Latent space.** Real images lie near a low-dimensional manifold, so diffusing in pixel space wastes compute. Visual: W14, MNIST compressed by an autoencoder into a 2D latent plane, each dot a digit, colored by class; one particle flows from noise into a cluster while its decoded image sharpens beside it. Then a one-line diagram: image → encoder → latent (e.g. 8× downsampled) → flow → decoder. Point: this is latent diffusion (Rombach et al. 2022), used by Stable Diffusion and FLUX.
24. **The network.** Diagram: UNet (DDPM, SD 1.x) → Transformer (DiT; SD3's MMDiT; FLUX). Text enters via cross-attention in older models and via joint attention over text + image tokens in MMDiT. One slide, one sentence each.
    - ↓ Deep dive: UNet vs. DiT block diagrams.
25. **Classifier-free guidance.** v = v\_uncond + w·(v\_cond − v\_uncond). Visual: W14 in guidance mode: pick a digit, slider for w; particles get pulled ever harder into that digit's cluster. W11 (two-class moons) moves to a ↓ deep-dive sub-slide. Point: guidance is what makes the model actually follow the prompt.

### Act 7: Experiment (5.5 min)

26. **When does the model decide what it's drawing?** Hypothesis: layout and semantics are fixed early in sampling, texture and detail late. Cite prior work here: Wang & Vastola (2023), "Diffusion Models Generate Images Like Painters", analyze this outline-first behavior and find sampling trajectories are nearly 2D.
27. **Setup.** Model, fixed seeds, prompt 2×2 design, metric (details in the experiment section). Visual: the 2×2 prompt grid with thumbnails.
28. **Result 1: when do trajectories diverge?** Visual: W12 cosine-similarity-per-step curves, one line per prompt pair, plus a strip of predicted x̂\_0 at selected steps. Point: read off when each pair separates.
29. **Result 2: point of no return.** Visual: W13 prompt-switch slider: start with "a zebra", switch to "a horse" at step k; slider over k shows the final image. Point: after some step, the switch no longer changes what the object is, only its texture.
30. **What this means.** One or two takeaways, and honest limitations (one model, N seeds, prompt choice).

### Act 8: Close (1 min)

31. **Back to spaghetti.** Both clips side by side, small. "What changed: latent space, transformers, flow matching, scale." W0 ink drop now plays in reverse.
32. **Thanks.** QR code to the site, references list (Sohl-Dickstein 2015, Ho 2020, Song 2021, Lipman 2023, Liu 2023, Rombach 2022, Peebles & Xie 2023, Esser 2024, Ho & Salimans 2022, Wang & Vastola 2023, Helbling & Chau 2025).

## Interactive widgets

15 widgets, all reading precomputed data. Closest prior art for W1–W10 is Diffusion Explorer (Helbling & Chau 2025), an open-source browser tool for 2D diffusion models; review it for ideas and credit it. Note it uses the opposite time convention. Build them on one shared `ParticleCanvas` class (draws points, paths and arrow fields on a square canvas with the color code above), so most widgets are a thin config on top.

| ID | Slides | What it shows | Controls | Data file |
| --- | --- | --- | --- | --- |
| W0 | 1, 31 | Ink drop diffusing (or reversing) | autoplay; `data-reverse` | `media/ink.webm` or `data/toy/ink_particles.json` |
| W1 | 6 | Two-moons scatter ↔ density heatmap | toggle | `data/toy/moons.json`, `density.png` |
| W2 | 8 | Gaussian cloud morphing into moons | play/pause | `data/toy/fm_traj.json` |
| W3 | 9–10 | Forward noising, cloud or particle paths | t slider; mode toggle | `data/toy/forward.json` |
| W4 | 11 | Same paths at 10 / 100 / 1000 steps | 3-way switch | `data/toy/forward_steps.json` |
| W5 | 13 | Score arrow field over time | t slider | `data/toy/score_field.json` (20×20 grid × 21 t values) |
| W6 | 15 | Reverse SDE sampling, noise → moons | play/pause, scrub | `data/toy/sde_traj.json` |
| W7 | 16 | SDE vs. PF-ODE paths + marginal histograms | t slider | `sde_traj.json`, `ode_traj.json` |
| W8 | 17 | PF-ODE quality vs. step count | steps slider 1–50 | `data/toy/ode_steps/{n}.json` |
| W9 | 18–19 | Straight pair lines → learned FM paths | toggle | `data/toy/fm_pairs.json`, `fm_traj.json` |
| W10 | 21 | SDE / ODE / FM side by side | shared steps slider | `data/toy/compare/{method}_{n}.json` |
| W11 | 25 ↓ | Classifier-free guidance on 2-class moons (deep dive) | w slider 0–7 | `data/toy/cfg/{w}.json` |
| W12 | 28 | Similarity-per-step curves + x̂\_0 strip | hover step | `data/experiment/similarity.json`, `x0/*.webp` |
| W13 | 29 | Prompt-switch point of no return | k slider | `data/experiment/switch/{k}.webp` |
| W14 | 23, 25 | MNIST 2D latent: particles flow into digit clusters, decoded image beside the highlighted particle; guidance mode | play/scrub t; click particle; digit picker + w slider | `data/latent/scatter.json`, `traj_{class}_{w}.json`, `frames.webp` (sprite sheet) |

Rules for every widget:

- Particle count: 2,000 by default, 50 highlighted for path views.
- Must render a meaningful static frame without JS running (for PDF export and the scroll-view first paint): set a `data-poster` image.
- Sliders keyboard-accessible; arrow keys must not trigger reveal navigation while a slider has focus.
- Animations loop at about 4 s, ease-in-out over t.
- No widget may block the main thread for more than 50 ms on load; lazy-load data on first `slidechanged`.

## Precompute pipeline and experiment

Two independent Python stages: toy models (CPU, minutes) and the image-model experiment (GPU, hours). Both write only into `data/`.

### Stage A: 2D toy models (`precompute/toy/`)

1. `data.py` — two moons, 4,000 points, fixed seed; also a 2-class version (upper/lower moon) for CFG.
2. `train_score.py` — small MLP (3×128, SiLU, sinusoidal time embedding), VP-SDE, ε-prediction, 20k steps. Saves `score.pt`.
3. `train_fm.py` — same MLP, linear-path flow matching, velocity target. Saves `fm.pt`. Conditional variant with label dropout 10% for CFG.
4. `export.py` — writes every JSON listed in the widget table: forward paths, score grid, SDE/ODE/FM trajectories (Euler for ODE/FM, Euler–Maruyama for SDE), step-count variants n ∈ {1, 2, 4, 8, 16, 32, 50}, CFG at w ∈ {0, 1, 2, 3, 5, 7}.
5. `check.py` — renders each JSON to PNG for a visual sanity check, and saves these as the widgets' `data-poster` fallbacks.

JSON format: `{"t": [...], "frames": [[x0,y0,x1,y1,...], ...]}`, coordinates rounded to 3 decimals. Keep each file under 2 MB.

### Stage A2: latent MNIST (`precompute/latent/`, for W14)

1. `train_ae.py` — small conv autoencoder on MNIST, 2D bottleneck. Plain AE or a very small KL weight (≤ 0.01) so the latent keeps its class clusters instead of collapsing toward a Gaussian. Normalize latents to roughly unit scale. Minutes on a laptop.
2. `check_ae.py` — decode a 20×20 grid over the latent plane into one image. If digits are not readable, retry with a 3D latent (then W14 gets an orbit view in scroll mode).
3. `train_latent_fm.py` — the same MLP as the toy models, class-conditional with 10% label dropout, trained on the 2D latents.
4. `export_latent.py` — labeled latent scatter (2,000 points), 200 trajectories unconditional and per digit at w ∈ {0, 1, 3, 5}, and decoded 28×28 frames for 20 highlighted trajectories at 21 time steps, packed as one WebP sprite sheet.

Starting point: [Latent\_Flow\_Matching\_MNIST](https://github.com/Whalefishin/Latent_Flow_Matching_MNIST) (flow matching in a VAE latent for MNIST); swap its DiT for the MLP and its VAE for the 2D autoencoder.

### Stage B: experiment (`precompute/experiment/`)

**Question:** at which sampling step does a rectified flow model commit to what the image shows?

**Model:** Stable Diffusion 3.5 Medium via diffusers (open weights, flow matching, MMDiT). FLUX.1-schnell as an optional second model. 1024² resolution, 28 steps, guidance 4.5, fixed scheduler.

**Prompt design (2×2 + control), base prompt "a photo of a zebra standing in a field":**

| Pair | Second prompt | Shares with base |
| --- | --- | --- |
| Shape / semantics | a photo of a horse standing in a field | body shape, category, layout |
| Texture | a photo of a zebra-striped sofa standing in a field | stripes, colors |
| Control | a photo of a teapot standing in a field | scene only |
| Identity | base prompt, different seed | category, not noise |

**Run 1, divergence curves:** for each pair and each of 16 seeds, run both prompts from the same initial noise. At every step record the predicted velocity and the predicted clean latent x̂\_0 = x\_t − t·v. Compute per step: cosine similarity of velocities, and DINOv2 cosine similarity of decoded x̂\_0. Export mean ± std per step per pair to `similarity.json`. Save decoded x̂\_0 for one seed at steps {1, 3, 5, 8, 12, 20, 28} as WebP.

**Run 2, prompt switching:** start with the zebra prompt, switch to the horse prompt at step k for k ∈ {0, 2, 4, …, 28}; same for zebra → teapot. Save final images, 512² WebP. Report the first k at which a DINOv2 classifier (or CLIP zero-shot) still says "zebra" for at least 80% of seeds.

**Compute estimate:** about 16 seeds × 4 pairs × 2 prompts + 2 × 15 switch runs × 16 seeds ≈ 610 generations; on a single A100 at about 5 s each, under 1 hour. Use TUM or Lateration GPU.

**Reporting:** always show seed variance, state that results are for one model, and keep the prompt list visible on slide 27.

## Build order, acceptance criteria, open questions

Build the skeleton and toy pipeline first; the experiment runs in parallel and plugs in last.

1. Repo skeleton: vendored reveal.js + KaTeX, theme, all 32 slides as placeholders with titles and speaker notes, live on Pages.
2. Stage A precompute + `check.py` posters.
3. `ParticleCanvas` + W1, W2, W3 (validates the data format and lifecycle).
4. Remaining toy widgets W4–W11, then Stage A2 and W14, in slide order.
5. Hook and close media (W0, local video files, YouTube embeds for scroll view).
6. Stage B experiment, then W12–W13.
7. Polish: scroll view pass, PDF export pass, offline test, timing rehearsal.

Acceptance criteria:

- [ ] Site loads at the Pages URL and from `file://` / a local server with no network.
- [ ] Present mode: every slide reachable by keyboard and clicker; widgets start and stop with their slide.
- [ ] `?view=scroll` reads as a coherent article top to bottom; deep-dive sub-slides appear inline.
- [ ] `?print-pdf` export: every slide has a readable static frame.
- [ ] All widgets at 60 fps on a 2020-era laptop, total page weight under 30 MB excluding hook videos.
- [ ] Time convention and color code consistent on every plot.
- [ ] Full rehearsal fits in 29 minutes or less.

Open questions:

- [ ] Root site (`jakobliebig.github.io`) or seminar in a subfolder or project repo?
- [ ] Does the seminar require a PDF hand-in, and is Q&A inside the 30 minutes? If so, cut to 25 by trimming slides 11, 17 and 22.
- [ ] Which recent spaghetti clip to use, and its source link.
- [ ] GPU access for Stage B: TUM cluster or Lateration.
- [ ] Talk language: English or German.
- [ ] Latent MNIST: are 2D decodes readable enough, or switch to 3D? MNIST or Fashion-MNIST?
