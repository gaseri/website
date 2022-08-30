---
author: Vedran Miletić
date: 2017-07-22
tags:
  - web server
  - sphinx
  - letsencrypt
---

# Enabling HTTP/2, HTTPS, and going HTTPS-only on inf2

Inf2 is a web server at [University of Rijeka](https://uniri.hr/) [Department of Informatics](https://www.inf.uniri.hr/), hosting Sphinx-produced static HTML course materials ([mirrored](../../hr/nastava/index.md) [elsewhere](../../hr/index.md)), some big files, a Wordpress instance ([archived](https://fidit-rijeka.github.io/elarsportal/) [elsewhere](https://github.com/fidit-rijeka)), and an internal instance of [Moodle](https://moodle.org/).

HTTPS was enabled on inf2 for a long time, albeit using a self-signed certificate. However, with [Let's Encrpyt](https://letsencrypt.org/) coming into [public beta](https://letsencrypt.org/2015/11/12/public-beta-timing.html), we decided to [join the movement to HTTPS](https://www.facebook.com/inf.uniri/posts/972284382811042).

HTTPS was optional. Almost a year and a half later, we also enabled HTTP/2 for the users who access the site using HTTPS. This was [very straightforward](https://twitter.com/VedranMiletic/status/855700609323986944).

Mozilla has a long-term plan to [deprecate non-secure HTTP](https://blog.mozilla.org/security/2015/04/30/deprecating-non-secure-http/). The likes of [NCBI](https://www.ncbi.nlm.nih.gov/home/develop/https-guidance/) (and [the rest of the US Federal Government](https://https.cio.gov/)), [Wired](https://www.wired.com/2016/09/wired-completely-encrypted/), and [StackOverflow](https://stackoverflow.blog/2017/05/22/stack-overflow-flipped-switch-https/) have already moved to HTTPS-only. We decided to do the same.

Configuring [nginx to redirect to HTTPS](https://serverfault.com/questions/250476/how-to-force-or-redirect-to-ssl-in-nginx) is very easy, but configuring particular web applications at the same time can be tricky. Let's go through them one by one.

Sphinx-produced static content does not hardcode local URLs, and the resources loaded from CDNs in [Sphinx Bootstrap Theme](https://ryan-roemer.github.io/sphinx-bootstrap-theme/) are already loaded via HTTPS. No changes were needed.

WordPress requires you to set the HTTPS URL in Admin, Settings/General. If you forget to do so before you go HTTPS only, you can [still use the config file to adjust the URL](https://codex.wordpress.org/Changing_The_Site_URL).

Moodle requires you to [set $CFG->wwwroot in the config file](https://docs.moodle.org/33/en/Configuration_file) to the HTTPS URL of your website.

And that's it! Since there is a dedicated IP address used just for the inf2 domain, we can afford to not require [Server Name Indication](https://en.wikipedia.org/wiki/Server_Name_Indication) support from the clients (I'm sure that both of our Android 2.3 users are happy for it).
