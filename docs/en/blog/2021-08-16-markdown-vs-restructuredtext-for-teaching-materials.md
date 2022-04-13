---
author: Vedran Miletić
date: 2021-08-16
tags:
  - restructuredtext
  - sphinx
  - markdown
  - mkdocs
  - git
---

# Markdown vs reStructuredText for teaching materials

Back in summer 2017. I wrote an article explaining [why we used Sphinx and reStructuredText to produce teaching materials](2017-07-29-why-we-use-restructuredtext-and-sphinx-static-site-generator-for-maintaining-teaching-materials.md) and not a wiki. In addition to recommending Sphinx as the solution to use, it was general praise for generating static HTML files from Markdown or reStructuredText.

This summer I made the conversion of teaching materials from reStructuredText to Markdown. Unfortunately, the automated conversion using [Pandoc](https://pandoc.org/) didn't quite produce the result I wanted so I ended up cooking my own Python script that converted the specific dialect of reStructuredText that was used for writing the contents of [the group website](../../index.md) and fixing a myriad of inconsistencies in the writing style that accumulated over the years.

## reStructuredText as the obvious choice for software documentation

I personally prefer to write reStructuredText, which I find to be more powerful and better standardized than Markdown (I have heard the same is true about [AsciiDoc](https://asciidoc.org/), though I haven't personally used it). When we forked [rDock](https://rdock.gitlab.io/) to start [RxDock](https://rxdock.gitlab.io/), [reStructuredText](https://docutils.sourceforge.io/rst.html) and [Sphinx](https://www.sphinx-doc.org/) were the obvious choices for [its documentation](https://rxdock.gitlab.io/documentation/devel/html/). A good argument as to why a software developer prefers reStructuredText over Markdown for software documentation is given in [a very fine article](https://www.zverovich.net/2016/06/16/rst-vs-markdown.html) written by [Victor Zverovich](https://twitter.com/vzverovich). He mentions two main advantages, the first one being:

> reStructuredText provides standard extension mechanisms called [directives](https://docutils.sourceforge.io/docs/ref/rst/directives.html) and [roles](https://docutils.sourceforge.io/docs/ref/rst/roles.html) which make all the difference. For example, you can use the math role to write a mathematical equation (...) and it will be rendered nicely both in HTML using a Javascript library such as MathJax and in PDF via LaTeX or directly. With Markdown you’ll probably have to use MathJax and HTML to PDF conversion which is suboptimal or something like Pandoc to convert to another format first.

(For what it's worth, this has now been addressed by [PyMdown Extension](https://facelessuser.github.io/pymdown-extensions/) [Arithmatex](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/), which is [easy to enable](https://squidfunk.github.io/mkdocs-material/reference/mathjax/) when using [MkDocs](https://www.mkdocs.org/) with [Material theme](https://squidfunk.github.io/mkdocs-material/).)

The second advantage mentioned by Zverovich is very useful for software documentation and a feature that would be only nice to have elsewhere:

> In addition to this, Sphinx provides a set of roles and directives for different language constructs, for example, `:py:class:` for a Python class or `:cpp:enum:` for a C++ enum. This is very important because it adds semantics to otherwise purely presentational markup (...)

## Markdown as the obvious choice elsewhere

Despite recommending reStructuredText for software documentation, Victor opened his blog post with:

> In fact, I’m writing this blog post in Markdown.

This is the obvious option for blogs hosted on [GitHub Pages](https://pages.github.com/) since built-in Jekyll offers Markdown-to-HTML conversion. You don't have to worry about how the conversion is done so you can worry about writing the content. However, the same feature isn't available for reStructuredText and AsciiDoc as Jekyll doesn't support either of the two.

Is GitLab any different from GitHub? [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/) supports [almost anything you can imagine](https://gitlab.com/pages) [thanks to GitLab CI/CD](https://docs.gitlab.com/ee/user/project/pages/getting_started/pages_ci_cd_template.html), [including Sphinx](https://gitlab.com/pages/sphinx). However, the same isn't the case for project wikis where GitLab supports [Markdown](https://docs.gitlab.com/ee/user/markdown.html) and [AsciiDoc](https://docs.gitlab.com/ee/user/asciidoc.html), but not reStructuredText ([it was requested 5 years ago](https://gitlab.com/gitlab-org/gitlab/-/issues/15001)).

And it's a similar story elsewhere. Reddit? [Markdown](https://www.markdownguide.org/tools/reddit/). Slack, Mattermost? [Both](https://www.markdownguide.org/tools/slack/) [Markdown](https://www.markdownguide.org/tools/mattermost/). Visual Studio Code supports [Markdown](https://code.visualstudio.com/Docs/languages/markdown) without any extensions (but [there are 795 of them available](https://marketplace.visualstudio.com/search?term=markdown&target=VSCode&category=All%20categories&sortBy=Relevance) if you feel that something you require is not there, [compared to 21 for reStructuredText](https://marketplace.visualstudio.com/search?term=restructuredtext&target=VSCode&category=All%20categories&sortBy=Relevance)). Finally, it's a very popular choice among my colleagues and students, which is expected as there is nothing like [HackMD](https://hackmd.io/) for reStructuredText or AsciiDoc that I know of.

Obviously, many of these tools weren't around when we switched to Sphinx back in 2014. However, now that they are here to stay, Markdown is starting to look like a better choice among the two.

## Moving from reStructuredText to Markdown for teaching materials

In my particular case, the straw that broke the camel's back and made me decide to convert [the teaching materials](../../hr/nastava/index.md) from reStructuredText to Markdown was the student contribution of [ZeroMQ](https://zeromq.org/) exercises for the [Distributed systems](../../hr/nastava/kolegiji/DS.md) course (not included yet). I asked the student to write reStructuredText but got the materials in Markdown so I can understand why that is. Let's say that the student wanted to do things properly in reStructedText and Sphinx. The procedure is this:

1. Create a Git clone of the repository.
1. Open the folder in your preferred editor, say VS Code, notice it doesn't highlight rST out of the box. No problem, there should be an extension, right?
1. Install [the reStructuredText extension](https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext) ([homepage](https://www.restructuredtext.net/)), close all `NotImplemented` exception notes that appear when opening the project.
1. Open the file just to get a feel of how rST should look. Now try to preview it. Unknown directive type "sectionauthor". Oh, don't worry, it's just one command that is unsupported.
1. The source code blocks aren't highlighted either in the edit pane or in the preview pane. Oh well, it's not a show stopper.
1. Well, there are more errors in the preview. Don't worry, the compile is a real preview. Let's compile things every time something changes.
1. (...)
1. Send the changes by e-mail or git add, git commit, and git push.

Compare these steps with the Markdown workflow:

1. Create a Git clone of the repository.
1. Open the folder in VS Code and start writing.
1. Send the changes by e-mail or git add, git commit, and git push.

To be fair, VS Code Markdown preview is not rendering [Admonitions](https://python-markdown.github.io/extensions/admonition/), but that's how it goes with the language extensions. Still, it's much easier to get started with Markdown and MkDocs than with reStructuredText and Sphinx if you are new to documentation writing, which is the case with most of the students.

There are a number of other things I like:

- [Material theme for MkDocs](https://squidfunk.github.io/mkdocs-material/) is awesome. It's a set of extensions in addition to a good-looking theme.
- The Integrated Search feature is designed to "find-as-you-type" and provide a much better user experience.
- Much shorter time to build the website. It takes 11 seconds to build [the group website](../../index.md) with MkDocs, while it took 37 seconds to build the older version of the same website with Sphinx.
- Built-in GitHub Pages deployment functionality. [You can do the same with Sphinx](https://alkaline-ml.com/2018-12-23-automate-gh-builds/), but it's much nicer to have it built-in and maintained.
- Automatic building of the sitemap. (There's an extension for Sphinx that [does the same](https://github.com/jdillard/sphinx-sitemap).)

Overall, I am very satisfied with the results and I'm looking forward to using Markdown for writing teaching materials in the future. I'll continue to write RxDock documentation in reStructuredText since [fancy cross-references](https://rxdock.gitlab.io/documentation/devel/html/#references) and [numbered equation blocks](https://rxdock.gitlab.io/documentation/devel/html/reference-guide/scoring-functions.html) are very easy to do in reStructuredText. In addition, there is [an official way to produce PDF output via LaTeX](https://www.sphinx-doc.org/en/master/usage/builders/index.html), which is quite important to have for proper scientific software documentation. Also, the potential contributors, in this case, are somewhat experienced with documentation tools and can usually find their way around with reStructuredText and Sphinx so it's not that much of an issue.
