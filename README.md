# 😎 Group for apps and services on exascale research infrastructure (GASERI) website

The group is interested in scientific software development, high-performance computers, cloud computing, and the use of free open-source software in the development of applications and services for supercomputers and cloud platforms, specifically in the application of exascale computing to solve problems in computational biochemistry and related fields. Visit [the group website](https://gaseri.org/en/) for more information.

## License

The contents of the website are licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), except teaching materials in Croatian under [/hr/nastava/materijali/](docs/hr/nastava/), teaching materials in English under [/en/teaching/materials/](docs/en/teaching/), and tutorials in English under [/en/tutorials/](docs/en/tutorials/gromacs/), which are licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## Prerequisites

Editing the website contents requires at least [the basic knowledge of writing Markdown in MkDocs](https://www.mkdocs.org/user-guide/writing-your-docs/).

## Obtaining the sources

Clone the repository using [Git](https://git-scm.com/):

``` shell
$ git clone https://github.com/gaseri/website.git
$ cd website
```

## Editing the contents

Make the changes you want using any text editor you like. Popular choices include [Visual Studio Code](https://code.visualstudio.com/), which supports [highlighting and previewing Markdown out of the box](https://code.visualstudio.com/docs/languages/markdown), [VSCodium](https://vscodium.com/), the community-driven and freely-licensed binary distribution of VS Code, [IntelliJ IDEA](https://www.jetbrains.com/idea/) with [the bundled Markdown plugin](https://www.jetbrains.com/help/idea/markdown.html), [Markdown Mode for GNU Emacs](https://www.emacswiki.org/emacs/MarkdownMode), and [Markdown Vim Mode](https://github.com/preservim/vim-markdown).

## Preparing the build environment

The website is built using [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Make sure that you have [Python](https://www.python.org/) and [pip](https://pip.pypa.io/) installed on your system; these two packages can usually be obtained via the operating system's package manager (e.g. [APT](https://wiki.debian.org/Apt), [DNF](https://dnf.readthedocs.io/), [Pacman](https://wiki.archlinux.org/title/Pacman), [Zypper](https://en.opensuse.org/SDB:Zypper_usage), [FreeBSD pkg](https://docs.freebsd.org/en/books/handbook/ports/#pkgng-intro), [Homebrew](https://brew.sh/), or [Windows Package Manager](https://docs.microsoft.com/en-us/windows/package-manager/)).

Once Python and pip are successfully set up, start by installing the required Python packages using the `pip` command:

``` shell
$ pip install -r requirements.txt
```

[By the way](https://iusearchbtw.lol/), if you are using [Arch Linux](https://archlinux.org/), you can alternatively use [mkdocs](https://aur.archlinux.org/packages/mkdocs) and [mkdocs-material](https://aur.archlinux.org/packages/mkdocs-material) from [Arch User Repository (AUR)](https://aur.archlinux.org/). Similarly, if you are using [FreeBSD](https://www.freebsd.org/), you can use [py-mkdocs](https://www.freshports.org/textproc/py-mkdocs/) and [py-mkdocs-material](https://www.freshports.org/textproc/py-mkdocs-material/) from [Ports](https://www.freebsd.org/ports/). These packages are mostly kept in sync with the upstream releases.

## Previewing the website

To open a local web server that will serve the website contents for previewing, use the [MkDocs's serve command](https://www.mkdocs.org/getting-started/):

``` shell
$ mkdocs serve
```

## Building the website

Build the website using [MkDocs's build command](https://www.mkdocs.org/getting-started/#building-the-site) (any remains of the previous build will be cleaned up automatically):

``` shell
$ mkdocs build
```

If the build was unsuccessful, fix the errors and repeat the building process.

## Saving the changes

Confirm that the website can be successfully built, add the changed files using Git, and commit the changes:

``` shell
$ git add docs
$ git commit
```

## Publishing the changes

To publish the changes, push them to GitHub:

``` shell
$ git push
```

Wait a few minutes until [GitHub Pages](https://pages.github.com/) finishes building the new site, including the changes you just pushed. Visit [gaseri.org](https://gaseri.org/) to make sure that your changes are visible.

**Note:** The official MkDocs approach, which we *don't* use, is to push the built contents to GitHub using the [MkDocs's GitHub deploy command](https://www.mkdocs.org/user-guide/deploying-your-docs/) (`mkdocs gh-deploy`).

*That's all, folks!*
