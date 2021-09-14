# Group for apps and services on exascale research infrastructure (GASERI) website

The group is interested in scientific software development, high-performance computers, cloud computing, and the use of free open-source software in the development of applications and services for supercomputers and cloud platforms, specifically in the application of exascale computing to solve problems in computational biochemistry and related fields. Visit [the group website](https://group.miletic.net/en/) for more information.

Editing the website contents requires at least [the basic knowledge of writing Markdown in MkDocs](https://www.mkdocs.org/user-guide/writing-your-docs/).

## Preparing the environment

The website is built using [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Make sure that you have [Python](https://www.python.org/) and [pip](https://pip.pypa.io/) installed on your system; these two packages can usually be obtained via the operating system's package manager (e.g. APT, DNF, Pacman, Zypper, FreeBSD pkg, Homebrew, or Windows Package Manager).

Once Python and pip are successfully set up, start by installing the required Python packages using the `pip` command:

``` shell
$ pip install mkdocs mkdocs[i18n] mkdocs-material
```

## Obtaining the sources

Clone the repository using [Git](https://git-scm.com/):

``` shell
$ git clone https://github.com/gaseri/website.git
```

## Editing the contents

Make the edits you want using any text editor you like (e.g. [Visual Studio Code](https://code.visualstudio.com/) supports [highlighting and previewing Markdown out of the box](https://code.visualstudio.com/docs/languages/markdown) and so does [Atom](https://atom.io/)).

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

## Saving the edits

Once the build is successful, add the edited files using Git and commit the changes:

``` shell
$ git add docs
$ git commit
```

## Publishing the edits

To publish your edits in source form, push them to GitHub:

``` shell
$ git push
```

## Publishing the website

Once you have confirmed that the build is successful, push the built contents to GitHub using the [MkDocs gh-deloy command](https://www.mkdocs.org/user-guide/deploying-your-docs/):

``` shell
$ mkdocs gh-deploy
```

Wait a few minutes until [GitHub Pages](https://pages.github.com/) finishes building the new site from the changes you just pushed and then visit [group.miletic.net](https://group.miletic.net/) to make sure that your changes are visible.

*That's all, folks!*
