---
author: Vedran Miletić
date: 2017-07-29
tags:
  - restructuredtext
  - sphinx
  - markdown
  - mkdocs
  - git
  - mediawiki
  - php
---

# Why we use reStructuredText and Sphinx static site generator for maintaining teaching materials

Yesterday I was asked by [Edvin Močibob](https://edvin.me/), a friend and [a former student teaching assistant of mine](../people/index.md#former-students), the following question:

> You seem to be using [Sphinx](https://www.sphinx-doc.org/) for your [teaching materials](../../hr/nastava/index.md), right? As far as I can see, it doesn't have an online WYSIWYG editor. I would be interested in comparison of your solution with e.g. [MediaWiki](https://www.mediawiki.org/).

While the [advantages](https://www.sitepoint.com/7-reasons-use-static-site-generator/) and [disadvantages](https://www.sitepoint.com/7-reasons-not-use-static-site-generator/) of static site generators, when compared to content management systems, have been [written about](https://www.stevestreeting.com/2016/06/12/converting-this-blog-from-wordpress-to-hugo/) and [discussed](https://news.ycombinator.com/item?id=896634) already, I will outline our reasons for the choice of Sphinx below. Many of the points have probably already been presented elsewhere.

## Starting with MoinMoin

[Teaching materials](../../hr/nastava/index.md) for the courses some of my colleagues and I used to teach at [InfUniRi](https://www.inf.uniri.hr/) and [RiTeh](http://www.riteh.uniri.hr/), including laboratory exercises for [the Computer Networks 2 course](../../hr/nastava/kolegiji/RM2.md) [developed during early 2012](../../hr/istrazivanje-i-razvoj.md#razvoj-e-kolegija-racunalne-mreze-2-na-sveucilistu-u-rijeci-u-akademskoj-20112012-godini), were initially put online using [MoinMoin](https://moinmo.in/). I personally liked MoinMoin because it used flat text files and required no database and also because it was Python-based and I happen to know Python better than PHP.

During the summer of 2014, the decision was made to replace MoinMoin with *something better* because version 1.9 was lacking features compared to MediaWiki and also evolving slowly. Most of the development effort was put in [MoinMoin version 2.0](https://moin-20.readthedocs.io/), which, quite unfortunately, still isn't released as of 2017. My colleagues and I especially cared about mobile device support (we wanted [responsive design](https://en.wikipedia.org/wiki/Responsive_web_design)), as it was requested by students quite often and, by that time, every other relevant actor on the internet had it.

## The search for alternatives begins

[DokuWiki](https://www.dokuwiki.org/) was a nice alternative and it offered responsive design, but I wasn't particularly impressed by it and was also slightly worried it might go the way of MoinMoin (as of 2017, this does not seem to be the case). It also used [a custom markup/syntax](https://www.dokuwiki.org/wiki:syntax), while I would have much preferred something [Markdown](https://daringfireball.net/projects/markdown/)/[reStructuredText](https://docutils.sourceforge.io/rst.html)-based.

We really wanted to go open with the teaching materials and release them under [a Creative Commons license](https://creativecommons.org/choose/). Legally, that can be done with any wiki or similar software. Ideally, however, a user should not be tied to your running instance of the materials to contribute improvements and should not be required to invest a lot of effort to set up a personal instance where changes can be made.

[MediaWiki](https://www.mediawiki.org/) was another option. Thanks to [Wikipedia](https://www.wikipedia.org/), MediaWiki's markup is widely understood, and [WYSIWYG editor](https://www.mediawiki.org/wiki/WYSIWYG_editor) was at the time being created.

In an unrelated sequence of events I have set up [a MediaWiki instance, later replaced by a static site generator](https://svedruziclab.github.io/software.html) in [BioSFGroup](https://svedruziclab.github.io/group.html) (where I also [participated in research projects for almost two years](2015-07-28-joys-and-pains-of-interdisciplinary-research.md)) and can say that setting up such an instance presents a number of challenges:

- You need a database, unlike when using MoinMoin and DokuWiki; these days most common choice is [MariaDB](https://mariadb.org/). I'll give kudos to MediaWiki devs for also supporting [PostgreSQL](https://www.postgresql.org/), which I prefer to MariaDB/MySQL.
- You need to be running PHP, which is fairly easy these days. There's [some extra work to do if SELinux is enabled](https://www.mediawiki.org/wiki/SELinux) (on my systems, it [is](https://stopdisablingselinux.com/)).
- You need to frequently patch all components of the stack, especially MediaWiki itself, which does get tedious over time. In addition, while PHP devs [have a reasonably long security and bugfix support time and also provide a detailed migration path after EOL](https://secure.php.net/eol.php) and while newer MediaWiki will likely support more recent PHP versions that include security fixes, your distribution [might](https://security-tracker.debian.org/tracker/source-package/php5) [require](https://security-tracker.debian.org/tracker/source-package/php7.0) you to do a distribution upgrade to a newer version to provide you with a newer (and supported) version of PHP containing the security fixes.

When migrating a MediaWiki instance from a server to another server, you have to dump/restore the database and adjust the config files (if you're lucky it won't be required to convert [Apache](https://httpd.apache.org/) configuration directives to [Nginx](https://nginx.org/) ones or vice versa). None of this is especially complicated, but it's extra work compared to flat file wikis and static websites.

Finally, my favorite MediaWiki theme (*skin* in its terminology) is [Vector](https://www.mediawiki.org/wiki/Skin:Vector), so my potential wiki with teaching materials would look exactly like Wikipedia. While nice and trendy, it is not very original to look like Wikipedia.

## Going static, going reStructured

Therefore, we opted to use [Sphinx](https://www.sphinx-doc.org/) and reStructuredText, as it was and still is a more powerful format than Markdown. We specifically cared about the [built-in admonitions](https://www.sphinx-doc.org/en/stable/rest.html#directives), which made it easier for us to convert the existing contents ([Python socket module lecture](../../hr/nastava/materijali/python-modul-socket.md) is a decent example). The advantages of Sphinx were and still are the following:

- [Git](https://git-scm.com/)-based diff,
- reduced attack surface: no web application, PHP, no database,
- very low amount of server maintenance required, can even be outsourced by dumping your content on e.g. [GitLab pages](https://docs.gitlab.com/ee/user/project/pages/index.html),
- one can still have social share links by including e.g. [AddToAny](https://www.addtoany.com/) in a post-processing script after the compilation,
- one can still have comments (our teaching materials will not) by including e.g. [Disqus](https://disqus.com/),
- super easy deployment: choose whether you want your URLs to end in `.html` or `/`, compile accordingly, and dump the generated HTML files,
- plently of good looking themes, some of which are responsive ([reStructuredHgWiki](../../hr/povijest.md#restructuredtext-sphinx-i-bootstrap) uses [sphinx-bootstrap-theme](https://ryan-roemer.github.io/sphinx-bootstrap-theme/), [CNPSLab](../../hr/povijest.md#laboratorij-za-racunalne-mreze-paralelizaciju-i-simulaciju) uses [sphinx-rtd-theme](https://sphinx-rtd-theme.readthedocs.io/), and there are [many more to choose from](https://github.com/search?q=sphinx+theme)),
- Sphinx can produce (LaTeX) PDF from the reStructuredText source (and [a nice-looking one](https://www.sphinx-doc.org/_/downloads/en/master/pdf/), I must say), but [Pandoc can produce almost anything you can imagine](https://pandoc.org/).

There is a number of issues which affected us:

- the time to deployment after the change: varies depending on the change, but it's in the order of tens of seconds in the worst case,
- the need to automate the deployment upon `git push` (note that this does *not* increase attack surface since git uses SSH or HTTPS for authentication and transfer).
- learning curve to add content: MediaWiki's WYSIWYG editor beats using git and reStructuredText in terms of simplicity.

## Conclusion

A rule of thumb here would be:

- if many people inside of an organization are going to edit content a lot and the content is more like notes than proper documentation, then MediaWiki (or DokuWiki) is the choice,
- if the content has an obvious hierarchy of parts, chapters, sections, etc. and/or it is evolving like a piece documentation changes with software it documents, then Sphinx (or any of Markdown-based generators, e.g. [HotDoc](https://hotdoc.github.io/) or [MkDocs](https://www.mkdocs.org/)) will do a better job.
