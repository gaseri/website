---
author: Vedran Miletić
authors:
  - vedranmiletic
date:
  created: 2026-04-14
tags:
  - git
  - markdown
  - mkdocs
---

# The future of MkDocs: ProperDocs and Zensical

---

![black and white penguin toy](https://unsplash.com/photos/wX2L8L-fGeA/download?w=1920)

Photo source: [Roman Synkevych (@synkevych) | Unsplash](https://unsplash.com/photos/black-and-white-penguin-toy-wX2L8L-fGeA)

---

[The slow collapse of MkDocs](https://fpgmaas.com/blog/collapse-of-mkdocs/) is, by now, well documented and worth reading if you are interested in the details of it.

In summary, MkDocs is unlikely to get future releases compatible with its current version, which makes [the instructions for setting up a GitHub Action for MkDocs deployment](2022-11-01-publishing-material-for-mkdocs-website-to-github-pages-using-custom-actions-workflow.md) in need of an update. Without further ado, here are two good options to move forward.

<!-- more -->

## ProperDocs

[ProperDocs](https://properdocs.org/) is MkDocs with [a dozen or so bug fixes](https://properdocs.org/about/release-notes/), some of which were long overdue. The existing plugins and themes for MkDocs continue to be supported, which makes the migration easy.

The `requirements.txt` file looks like:

``` text
properdocs[i18n]
mkdocs-material[recommended,imaging]
```

The `.github/workflows/properdocs-gh-pages.yml` file looks like:

``` yaml
# Sample workflow for building and deploying a ProperDocs site to GitHub Pages
name: Deploy ProperDocs with GitHub Pages dependencies preinstalled

on:
  # Runs on pushes targeting the default branch
  push:
    branches: ["main"]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v6
      - name: Setup Pages
        uses: actions/configure-pages@v6
      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.x'
      - name: Install yamllint
        run: pip install yamllint
      - name: Check ProperDocs YAML configuration
        run: yamllint ./properdocs.yml
        continue-on-error: true
      - name: Check Markdown files
        uses: DavidAnson/markdownlint-cli2-action@v23
        with:
          globs: '**/*.md'
        continue-on-error: true
      - name: Install required packages
        run: pip install -r requirements.txt
      - name: Setup caching
        uses: actions/cache@v5
        with:
          key: ${{ github.sha }}
          path: .cache
      - name: Build site
        run: properdocs build --config-file ./properdocs.yml --strict
        env:
          CI: true
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v5
        with:
          path: site

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v5
```

For example, [this website](../../../index.md) uses it.

## Zensical

[Zensical](https://zensical.org/) is a completely new project which aims to be compatible with MkDocs. However, at the moment, it might or might not [support all the features](https://zensical.org/compatibility/) needed for building your website. However, it aims to support most of the existing functionality in the future.

The `requirements.txt` file looks like:

``` text
zensical
```

The `.github/workflows/zensical-gh-pages.yml` file looks like:

``` yaml
# Sample workflow for building and deploying a Zensical site to GitHub Pages
name: Deploy Zensical with GitHub Pages dependencies preinstalled

on:
  # Runs on pushes targeting the default branch
  push:
    branches: ["main"]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v6
      - name: Setup Pages
        uses: actions/configure-pages@v6
      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.x'
      - name: Install yamllint
        run: pip install yamllint
      - name: Check Zensical YAML configuration
        run: yamllint ./mkdocs.yml
        continue-on-error: true
      - name: Check Markdown files
        uses: DavidAnson/markdownlint-cli2-action@v23
        with:
          globs: '**/*.md'
        continue-on-error: true
      - name: Install required packages
        run: pip install -r requirements.txt
      - name: Setup caching
        uses: actions/cache@v5
        with:
          key: ${{ github.sha }}
          path: .cache
      - name: Build site
        run: zensical build --config-file ./mkdocs.yml
        env:
          CI: true
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v5
        with:
          path: site

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v5
```

For example, the [core domain website](https://www.miletic.net/) uses it.
