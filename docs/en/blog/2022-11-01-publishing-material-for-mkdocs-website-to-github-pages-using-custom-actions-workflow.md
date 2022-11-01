---
author: Vedran Miletić
date: 2022-11-01
tags:
  - git
  - web server
  - letsencrypt
  - markdown
  - mkdocs
---

# Publishing (Material for) MkDocs website to GitHub Pages using custom Actions workflow

As you can probably see, this website is built using the [Material](https://squidfunk.github.io/mkdocs-material/) theme for [MkDocs](https://www.mkdocs.org/), which we are [happily using for over a year](2021-08-16-markdown-vs-restructuredtext-for-teaching-materials.md).

While we are manually building and deploying the website, there are [several](https://bluegenes.github.io/mkdocs-github-actions/) somewhat related [approaches](https://github.com/Tiryoh/actions-mkdocs) using [GitHub Actions](https://github.com/features/actions) for [deploying](https://github.com/marketplace/actions/deploy-mkdocs) MkDocs, usually with the Material theme, to [GitHub Pages](https://pages.github.com/). These guides are not only found on blogs written by enthusiasts; the official [Getting started section](https://squidfunk.github.io/mkdocs-material/getting-started/) of the Material for MkDocs documentation describes the usage of GitHub Actions for deployment and [provides a generic YAML file for that](https://squidfunk.github.io/mkdocs-material/publishing-your-site/#with-github-actions). Using that approach avoids the requirement to run the `build` and `gh-deploy` steps locally; GitHub Actions does both on GitHub's CI/CD servers. Additionally, the repository layout remains the same as it would be if the build and deployment steps were done locally; the `main` branch contains the site source in Markdown and the `gh-pages` branch contains the site files that get built for serving.

Since this summer, GitHub offers [publishing Pages using a custom Actions workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) as a [public beta](https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/), which was a unique feature of [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/) for years.

I thought that it would be interesting to see if we could use the GitHub Actions YAML configuration for Jekyll and replace Jekyll build step with MkDocs build step. This would streamline the usage of MkDocs with GitHub Pages, and, in particular, eliminate the requirement for publishing the site from a separate `gh-pages` branch, offering a Jekyll-like experience.

Let's see how far we can get. Here is the starting YAML file for Jekyll:

``` yaml hl_lines="30-36"
# Sample workflow for building and deploying a Jekyll site to GitHub Pages
name: Deploy Jekyll with GitHub Pages dependencies preinstalled

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

# Allow one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      - name: Setup Pages
        uses: actions/configure-pages@v2
      - name: Build with Jekyll
        uses: actions/jekyll-build-pages@v1
        with:
          source: ./
          destination: ./_site
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v1

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
        uses: actions/deploy-pages@v1
```

The highlighted lines are Jekyll-specific. We can easily replace these lines with the Python setup Action, the installation of MkDocs and Material for MkDocs (using pip), the installation of the optional dependencies required for the [generation of social cards](https://squidfunk.github.io/mkdocs-material/setup/setting-up-social-cards/) (that is, [Pillow](https://python-pillow.org/) and [CairoSVG](https://cairosvg.org/)), the optional caching setup for the downloaded fonts and the generated social cards, and, finally, the MkDocs site build command.

In this case, since we want a drop-in replacement for Jekyll so that the remaining commands work perfectly, we will perform the MkDocs build using the `mkdocs.yml` configuration file in the current directory and write the built site output files into the `_site` directory.

``` yaml hl_lines="30-44"
# Sample workflow for building and deploying a Jekyll site to GitHub Pages
name: Deploy Jekyll with GitHub Pages dependencies preinstalled

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

# Allow one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install MkDocs and Material for MkDocs
        run: pip install mkdocs[i18n] mkdocs-material
      - name: Install Pillow and CairoSVG (required for social card generation)
        run: pip install pillow cairosvg
      - name: Setup caching
        uses: actions/cache@v3
        with:
          key: ${{ github.ref }}
          path: .cache
      - name: Build site (_site directory name is used for Jekyll compatiblity)
        run: mkdocs build --config-file ./mkdocs.yml --site-dir ./_site
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v1

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
        uses: actions/deploy-pages@v1
```

And that's it! There is no more requirement for the `.nojekyll` file as Jekyll never gets ran in the build process. There is also no more separate `gh-pages` branch that the built files get pushed to, so there is also no more worry whether the site builds over time will add up to the [1 GB soft limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#repository-size-limits).

The next step in streamlining this approach further is probably patching [actions/configure-pages](https://github.com/actions/configure-pages) that will allow us to replace:

``` yaml
jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install MkDocs and Material for MkDocs
        run: pip install mkdocs[i18n] mkdocs-material
      - name: Install Pillow and CairoSVG (required for social card generation)
        run: pip install pillow cairosvg
      - name: Setup caching
        uses: actions/cache@v3
        with:
          key: ${{ github.ref }}
          path: .cache
```

with something along the lines of:

``` yaml
jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      - name: Setup Pages
        uses: actions/configure-pages@v2
        static_site_generator: mkdocs
```

We can even [dream bigger](https://youtu.be/GoJOVN2ycXQ) than that: specifying the `generator_config_file` should make some JavaScript-powered parsing magic/logic detect the requirement for the installation of optional dependencies and the caching setup, and enable them only if required.
