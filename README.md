# justdoty.github.io

Source for my personal website, built with [Hugo](https://gohugo.io/) and a small custom theme (`themes/minimal`).

## Editing

| What | File |
|---|---|
| Name, title, buttons, interests, education, bio | `content/_index.md` |
| Papers, presentations, awards | `content/research.md` |
| Photo | `content/avatar.jpg` |
| PDFs (CV, papers) | `static/files/` |
| Contact links, menu, page description | `config/_default/config.toml` |
| Look and feel | `themes/minimal/assets/css/main.css` |

## Preview locally

```sh
hugo server
```

Then open http://localhost:1313/.

## Publishing

Push to `main`. The GitHub Actions workflow in `.github/workflows/deploy.yml` builds the site and deploys it to GitHub Pages.
