---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2017-09-24
tags:
  - web server
  - free and open-source software
  - academia
  - decentralization
  - linux
  - debian
  - red hat
---

# Mirroring free and open-source software matters

---

![gold and silver steel wall decor](https://unsplash.com/photos/tszceVXBPos/download?w=1920)

Photo source: [Tuva Mathilde Løland (@tuvaloland) | Unsplash](https://unsplash.com/photos/gold-and-silver-steel-wall-decor-tszceVXBPos)

---

Post theme song: [Mirror mirror](https://youtu.be/SVg8eP7KPNQ) by [Blind Guardian](https://www.blind-guardian.com/)

A mirror is a local copy of a website that's used to speed up access for the users residing in the area geographically close to it and reduce the load on the original website. [Content distribution networks (CDNs)](https://en.wikipedia.org/wiki/Content_delivery_network), which are a newer concept and perhaps more familiar to younger readers, serve the same purpose, but do it in a way that's transparent to the user; when using a mirror, the user will see explicitly which mirror is being used because the domain will be different from the original website, while, in case of CDNs, the domain will remain the same, and the DNS resolution (which is invisible to the user) will select a different server.

Free and open-source software was distributed via (FTP) mirrors, usually residing in the universities, basically since its inception. [The story of Linux](https://en.wikipedia.org/wiki/Revolution_OS) mentions [a directory](ftp://ftp.funet.fi/pub/Linux/00Directory_info.txt) on `ftp.funet.fi` ([FUNET](https://en.wikipedia.org/wiki/FUNET) is the Finnish University and Research Network) where Linus Torvalds uploaded the sources, which was soon after [mirrored by Ted Ts'o on MIT's FTP server](https://linuxdevices.org/ted-tso-to-boost-the-linux-standards-base/). [The GNU Project](https://www.gnu.org/)'s history contains an analogous process of making local copies of the software for faster downloading, which was especially important in the times of pre-broadband Internet, and it [continues today](https://www.gnu.org/prep/ftp.html).

<!-- more -->

Many Linux distributions, including this author's favorite [Debian](https://www.debian.org/) and [Fedora](https://getfedora.org/) use mirroring (see [here](https://www.debian.org/mirror/list) and [here](https://mirrors.fedoraproject.org/)) to be more easily available to the users in various parts of the world. If you look carefully at those lists, you can observe that the universities and institutes host a significant number of mirrors, which is both a historical legacy and an important role of these research institutions today: the researchers and the students in many areas depend on free and open-source software for their work, and it's much easier (and faster!) if that software is downloadable locally.

Furthermore, my personal experience leads me to believe that hosting a mirror as a university is a great way to reach potential students in computer science. For example, I heard of TU Vienna thanks to `ftp.tuwien.ac.at` and, if I was willing to do a Ph.D. outside of Croatia at the time, would certainly look into the programs they offered. As another example, Stanford has some very interesting courses/programs at the [Center for Computer Research in Music and Acoustics (CCRMA)](https://ccrma.stanford.edu/). How do I know that? They went even a bit further than mirroring, they offered software packages for Fedora at [Planet CCRMA](http://ccrma.stanford.edu/planetccrma/software/). I bet I wasn't the only Fedora user who played/worked with their software packages and in the process got interested to check out what else they are doing aside from packaging those RPMs.

That being said, we wanted to do both at the University of Rijeka: serve the software to the local community and reach the potential students/collaborators. Back in late 2013, we started with setting up a mirror for [Eclipse](https://eclipse.org/); it first appeared at inf2 server under mirrors directory and later moved to a dedicated mirrors server, where it still resides. [LibreOffice](https://download.documentfoundation.org/mirmon/allmirrors.html) was also added early in the process, and [Cygwin](https://www.cygwin.com/mirrors.html) quite a bit later. Finally, we [started mirroring](https://www.facebook.com/inf.uniri/posts/1519708868068588) [CentOS](https://www.centos.org/)'s [official](https://www.centos.org/download/mirrors/) and [alternative architectures](https://www.centos.org/download/mirrors-altarch/) as a second mirror in Croatia (but the first one in Rijeka!), the first Croatian one being hosted by [Plus Hosting](https://www.plus.hr/) in Zagreb.

University's mirrors server already syncs a number of other projects on a regular basis, and we will make sure we are added to their mirror lists in the coming months. As it has been mentioned, this is both an important historical legacy role of a university and a way to serve the local community, and a university should be glad to do it. In our case, it certainly is.
