---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2024-05-28
tags:
  - git
  - web server
  - web standards
  - javascript
  - mkdocs
---

# Serverless URL redirects using JavaScript on GitHub Pages

As many readers of this blog are already aware, we make great use of [GitHub Pages](2023-06-23-what-hardware-software-and-cloud-services-do-we-use.md#cloud-services) for [hosting](2022-11-01-publishing-material-for-mkdocs-website-to-github-pages-using-custom-actions-workflow.md) this website and [several](https://www.miletic.net/) [others](https://fidit-rijeka.github.io/elarsportal/). In particular, after [FIDIT's inf2 server](2017-07-22-enabling-http2-https-and-going-https-only-on-inf2.md) was finally decomissioned, Pages was the obvious choice for replacing the remaining services it offered.

Since the number and variety of applications and services hosted on inf2 server grew and diminished organically over time, what remained afterward was a collection of complex, but unrelated link hierarchies that had to be redirected to new locations (remember that [Cool URIs don't change](https://www.w3.org/Provider/Style/URI)).

<!-- more -->

In particular, redirecting just the index is very simple, e.g. for ELARS Portal accessing <https://inf2.uniri.hr/elarsportal/> will redirect to <https://fidit-rijeka.github.io/elarsportal/> thanks to the `index.html` file in the `elarsportal` directory with the following contents:

``` html
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="refresh" content="0; url='https://fidit-rijeka.github.io/elarsportal/'" />
  </head>
</html>
```

This is too tedious to do for every URI, but luckily there are better approaches. We briefly looked into building a "redirect site" using [mkdocs-redirects](https://github.com/mkdocs/mkdocs-redirects) plugin for [MkDocs](https://www.mkdocs.org/), but ultimately decided even this approach was going to be too much hassle to maintain.

GitHub Pages itself doesn't offer a proper HTTP redirection service, but there is a [neat trick using 404 page](https://gist.github.com/domenic/1f286d415559b56d725bee51a62c24a7?permalink_comment_id=3945701#gistcomment-3945701). After some tinkering coupled together with reading [MDN's JavaScript basics](https://developer.mozilla.org/en-US/docs/Learn/Getting_started_with_the_web/JavaScript_basics), we got to the `404.html` file with the following contents:

``` html linenums="1"
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <title>inf2 is now retired</title>

    <script>
        window.onload = function () {
            const currentLocation = window.location.href;
            var redirectLocation = 'https://noc.miletic.net/';
            if (currentLocation.startsWith('https://inf2.uniri.hr/elarsportal')) {
                redirectLocation = currentLocation.replace(
                    'https://inf2.uniri.hr/elarsportal',
                    'https://fidit-rijeka.github.io/elarsportal'
                );
            } else if (currentLocation.startsWith('https://inf2.uniri.hr/reStructuredHgWiki')) {
                redirectLocation = currentLocation.replace(
                    'https://inf2.uniri.hr/reStructuredHgWiki',
                    'https://group.miletic.net/hr/nastava'
                );
            } else if (currentLocation.startsWith('https://inf2.uniri.hr/hgwiki') ||
                currentLocation.startsWith('https://inf2.uniri.hr/heterogeneouswiki') ||
                currentLocation.startsWith('https://inf2.uniri.hr/redwiki')) {
                redirectLocation = currentLocation.replace(
                    'https://inf2.uniri.hr/hgwiki',
                    'https://group.miletic.net/hr/nastava/kolegiji'
                ).replace(
                    'https://inf2.uniri.hr/heterogeneouswiki',
                    'https://group.miletic.net/hr/nastava/kolegiji'
                ).replace(
                    'https://inf2.uniri.hr/redwiki',
                    'https://group.miletic.net/hr/nastava/kolegiji'
                );
            } else if (currentLocation.startsWith('https://inf2.uniri.hr/bluewiki')) {
                redirectLocation = 'https://www.inf.uniri.hr/~amestrovic/';
            } else if (currentLocation.startsWith('https://inf2.uniri.hr/request') ||
                currentLocation.startsWith('https://inf2.uniri.hr/server')) {
                redirectLocation = currentLocation.replace(
                    'https://inf2.uniri.hr',
                    'https://apps.group.miletic.net'
                );
            } else if (currentLocation.startsWith('https://inf2.uniri.hr/metod')) {
                redirectLocation = 'https://moodle.srce.hr/';
            }
            document.body.innerHTML = '<p>Please follow <a href="' + redirectLocation + '">this link</a>.</p>';
            window.location.replace(redirectLocation);
        }
    </script>
</head>

<body>
    <p>Please follow <a href="https://noc.miletic.net/">this link</a>.</p>
</body>

</html>
```

Let's go briefly over the contents and redirect options:

- lines 12--16 handle [ELARS Portal](https://fidit-rijeka.github.io/elarsportal/) link hierarchy,
- lines 17--21 handle [reStructuredText-powered CNSPSLab website](../../../hr/povijest.md#restructuredtext-sphinx-i-bootstrap) link hierarchy,
- lines 22--36 handle [MoinMoin-powered wiki](../../../hr/povijest.md#moinmoin) link hierarchy, with all historical name changes taken into account,
- lines 37--42 handle two applications used in teaching [Computer Networks](../../../hr/nastava/kolegiji/RM.md), [Computer Networks (RiTeh)](../../../hr/nastava/kolegiji/RM-RiTeh.md), and [Computer Networks 2](../../../hr/nastava/kolegiji/RM2.md) (obviously, JavaScript will not be executed when accesed via [cURL command-line interface](../../../hr/nastava/materijali/curl-protokoli-aplikacijske-razine.md), but the hint about the correct URI in the output should be fairly obvious),
- lines 43--44 handle our internal instance of [Moodle](https://moodle.org/) that got replaced by [Srce](https://www.srce.unizg.hr/)'s, and
- lines 11 and 47 ensure that all other requests, i.e. not matched by any of the rules above, get redirected to the [network operations center](https://noc.miletic.net/).

And that's it! Sure, we can extend it further to also handle renamed pages and moved files by adding nested `if`s, but this is working well enough for the most of the URIs that we care about. It is also interesting to note that [Google recognizes the redirects](https://www.google.com/search?q=site%3Ainf2.uniri.hr), despite the HTTP 404 status code returned by the `GitHub.com` server and the need to execute JavaScript while crawling to figure out the destination URI.
