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

As you can probably see, this website is built using the [Material theme](https://squidfunk.github.io/mkdocs-material/) for [MkDocs](https://www.mkdocs.org/), which we have been [happily using for over one year](2021-08-16-markdown-vs-restructuredtext-for-teaching-materials.md) after [using Sphinx for many years prior](2017-07-29-why-we-use-restructuredtext-and-sphinx-static-site-generator-for-maintaining-teaching-materials.md). [GitHub Pages](https://pages.github.com/) offers [built-in support for Jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll), but not for MkDocs and therefore it requires the manual building and deployment of our website. However, it automates many other things, including [HTTPS certificate provisioning on our domain](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https) via [Let's Encrypt](https://letsencrypt.org/).

There are [several somewhat](https://bluegenes.github.io/mkdocs-github-actions/) [related approaches](https://github.com/Tiryoh/actions-mkdocs) using [GitHub Actions](https://github.com/features/actions) for [automating the deployment](https://github.com/marketplace/actions/deploy-mkdocs) of MkDocs-generated sites, usually with the Material theme, to GitHub Pages. These guides are not only found on blogs written by enthusiasts; the official [Getting started section](https://squidfunk.github.io/mkdocs-material/getting-started/) of the Material for MkDocs documentation describes the usage of GitHub Actions for deployment and [provides a generic YAML file for that purpose](https://squidfunk.github.io/mkdocs-material/publishing-your-site/#with-github-actions). Using this approach avoids the requirement to run the `build` and `gh-deploy` steps locally; GitHub Actions does both on GitHub's [CI/CD](https://resources.github.com/ci-cd/) servers, where the [free plan offers 2000 minutes of GitHub-hosted runners per month](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#included-storage-and-minutes). As many sites build in less than a minute, this amount allows from 50 to 100 builds and deployments *per day*, which is quite a bit more than most sites require. Additionally, the repository layout remains the same as it would be if the build and deployment steps were done locally; the `main` branch contains the site source in Markdown and the `gh-pages` branch contains the site files that get built for serving.

Since this summer, GitHub offers [publishing Pages using a custom Actions workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) as a [public beta](https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/), which was a unique feature of [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/) for years. I thought that it would be interesting to see if we could use the existing GitHub Actions workflow configuration for Jekyll and simply replace the Jekyll build step with the MkDocs build step. This would streamline the usage of MkDocs with GitHub Pages, and, in particular, eliminate the requirement for publishing the site from a separate `gh-pages` branch, offering a Jekyll-like experience.

Let's see how far we can get. Without going into details about the [syntax for GitHub Actions](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions), here is the starting workflow configuration file for Jekyll:

``` yaml hl_lines="1-2 32-36"
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

``` yaml hl_lines="1-2 32-44"
# Sample workflow for building and deploying a MkDocs site to GitHub Pages
name: Deploy MkDocs with GitHub Pages dependencies preinstalled

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
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install MkDocs and Material for MkDocs
        run: pip install mkdocs[i18n] mkdocs-material
      - name: Install Pillow and CairoSVG (required for social card generation)
        run: pip install pillow cairosvg
      - name: Setup caching
        uses: actions/cache@v3
        with:
          key: ${{ github.sha }}
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

Finally, if you want to use a custom domain, having the `CNAME` file in the repository root or the `docs` subfolder will no longer have the desired effect; the domain has to be [configured through the repository settings or using the API](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#creating-a-custom-github-actions-workflow-to-publish-your-site).

**Updated on 2022-11-25:** changed Python version from 3.10 to 3.11, resulting in faster docs builds (see [Faster CPython](https://docs.python.org/3.11/whatsnew/3.11.html#faster-cpython) for details).

**Updated on 2022-12-03:** changed caching to use `github.sha` instead of `github.ref`, enabling rebuilds of social cards when site contents change.
