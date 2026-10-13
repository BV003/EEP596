# EEP596

raw => md content => website

Deep Learning course notes (UW EEP 596, Fall 2026).

| Stage | Location | Notes |
|---|---|---|
| raw | `raw/` | Original lecture/lab PDFs (local only, git-ignored) |
| md content | `docs/lectures/`, `docs/labs/` | Concise English outlines, LaTeX math, code blocks |
| website | `docs/.vitepress/` | VitePress site, deployed to GitHub Pages |

## Usage

```bash
npm install
npm run dev      # local preview at http://localhost:5173/EEP596/
npm run build    # output: docs/.vitepress/dist
```

## Adding new material

1. Drop the PDF into `raw/`.
2. Write `docs/lectures/lecture-NN.md` or `docs/labs/lab-NN.md`.
3. Add a sidebar entry in `docs/.vitepress/config.mts`.
4. Push to `main` — GitHub Actions deploys automatically.
