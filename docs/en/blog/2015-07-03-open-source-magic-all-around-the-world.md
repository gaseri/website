---
author: Vedran Miletić
date: 2015-07-03
tags:
  - free and open-source software
  - popular science
  - information revolution
  - red hat
  - systemd
---

# Open-source magic all around the world

Last week brought us two interesting events related to open-source movement: [2015 Red Hat Summit](https://www.redhat.com/summit/2015/resources/) (June 23-26, [Boston, MA](https://www.openstreetmap.org/way/29739137)) and [Skeptics in the pub](https://www.facebook.com/events/105693939772055/) (June 26, [Rijeka, Croatia](https://www.openstreetmap.org/way/358439113)).

## 2015 Red Hat Summit

[Red Hat](https://www.redhat.com/) provided live streaming of keynotes (kudos to them); [Domagoj](https://domargan.net/), [Luka](https://luka.vretenar.pro/) and I watched [the one from Craig Muzilla where they announced a partnership with Samsung](https://youtu.be/wWNVpFibayA). We made stupid jokes by substituting [FICO](https://en.wikipedia.org/wiki/FICO) (a company name) for [fićo](https://en.wikipedia.org/wiki/Zastava_750) (a car that is legendary in the Balkans due to its price and popularity). [Jim Whitehurst](https://twitter.com/JWhitehurst) was *so inspired he almost did not want to speak*, but luckily [spoke nonetheless](https://youtu.be/n6WBrYbkPD0). The interesting part was where he spoke about how the predicted [economics of information revolution](https://youtu.be/6ag8DiOWG1I) is already coming true.

Paul Cormier continued on Jim Whitehurst in terms of showing how predictions come true; [his keynote](https://youtu.be/tekg8OjrfDM) starts with the story of how Microsoft and VMware changed their attitude towards Linux and virtualization. He also presents [a study (starting at 1:45) showing that only Linux and Windows remain operating in the datacenter](https://youtu.be/tekg8OjrfDM?t=1m45s), and also that Windows is falling in market share, while Linux is rising. This is great news; let's hope it inspires Microsoft to learn from Red Hat how to be more open. Finally, [Marco Bill-Peter](https://twitter.com/marcobillpeter) is presenting [advances in customer support](https://youtu.be/x2TuacPvPNw) (in a heavy German accent).

## Skeptics in the pub

Watching streaming events is cool, but having them in your city is even cooler. I was invited to speak at the local [Skeptics in the pub](https://en.wikipedia.org/wiki/Skeptics_in_the_Pub) on whether the proprietary technologies are dying. Aside from being happy to speak about open source in public, I was also happy to speak about computer science more broadly. Too often people make the mistake of thinking that computer science researchers look for better ways to repair iPhones, clean up printers, and reinstall Windows. Well, in some way we do, but we don't care about those particular products and aren't focused on how to root the latest generation of Samsung phones, or clean up some nasty virus that is spreading.

That isn't to say that any of these things should be undervalued, as most of them aren't trivial. It's just that our research work approaches technology in a much broader way, and (somewhat counter-intuitively) solves very specific problems. For example, one such overview would be to look at the [historical development of Internet protocols at Berkeley in the '80s](https://youtu.be/ds77e3aO9nA) and later; one such problem would be the implementation of [Multipath TCP](https://en.wikipedia.org/wiki/Multipath_TCP) on Linux or Android.

As usual, the presentation was recorded, so the videos will appear^W^W are [on PZKM's YouTube channel](https://www.youtube.com/user/dzpzikm): [leture](https://youtu.be/aG_O88vaH60) and [discussion](https://youtu.be/GV5nM-EQDZk).

The short version of the presentation is: [computers are not just PCs anymore, but a range of devices](https://blogs.windows.com/windows-insider/2015/01/21/the-next-generation-of-windows-windows-10/). Non-proprietary software vendors recognized this expansion first, so open technologies are leading the way in many new areas (e.g. mobile phones and tablets), and have also taken the lead in some of the more traditional ones (e.g. infrastructure, web browsers). The "moving to the open-source technologies" trend is very obvious, at least in software. However, even in software, the proprietary vendors are far from being dead yet.

Regardless, there are two aspects of the move towards open-source technologies that make me particularly happy. The first aspect is that, with open technologies taking the lead, we can finally provide everyone with highly sophisticated tools for education. Imagine a kid downloading [Eclipse](https://www.eclipse.org/) (for free) to learn to program in Java, C, or C++; one does get to experience the real-world development environment that developers all around the world use daily. Imagine if, instead of the fully functional Eclipse integrated development environment, the kid got some kind of stripped-down toy software or a functionally limited demo version of the software. The kid's potential for learning would be severely restricted by the software limitations. (This is not a new idea, most people who cheer open source have been using this in open-source advocacy for many years. I was very happy to hear [Thomas Cameron](https://twitter.com/thomasdcameron) use the same argument at 2015 Red Hat Summit welcome speech.)

The second aspect is that LEGO blocks of the software world are starting to emerge in a way, especially considering container technologies. Again, imagine a kid wanting to run a web, mail, or database server at his home. Want to set it up all by yourself from scratch? There are hundreds of guides available. Want to see how it works when it works? Use a container, or a pre-built virtual appliance image, and spin a server in minutes. Then play with it until it breaks to see how to break it and how to fix it, preferably without starting from scratch. But even when you have to start from scratch, rebuilding a working operating system takes minutes (if not seconds), not tens of minutes or hours.

When I was learning my way around Linux in the '00s, virtualization was not yet widespread. So, whenever you broke something beyond repair, you had to reinstall your Linux distribution or restore it from a disk/filesystem image of some kind. If you dual-booted Linux and Windows, you could destroy the Windows installation if you did not know what you were doing. Finally, before the [Ubuntu](https://ubuntu.com/) and subsequent [Fedora](https://getfedora.org/) Live media, installation used to take longer than it does today. And if you think further and consider the geeks who grew up in the '90s, it's easy to see that they had even less ease of use in open-source software available to them. Yet, both 00's and 90's generations of geeks are creating awesome things in the world of open-source software today.

## OK, enough talk, where is the code?

I was so inspired by all this, so I got down to coding. I made [a small contribution](https://github.com/systemd/systemd/pull/466) to [systemd](https://freedesktop.org/wiki/Software/systemd/) (which made my OCD hurt less by wrapping the output text in the terminal better) and [a bit larger one](https://github.com/cp2k/cp2k/commit/7e11faa4da61f07e88f8fbb2d206f01f8f74655c) to [CP2K](https://www.cp2k.org/) (which [crossed off an item from dev:todo list](https://cp2k.org/dev:todo?do=diff&rev2%5B0%5D=1434371588&rev2%5B1%5D=1435853840&difftype=sidebyside)). On the [ns-3](https://www.nsnam.org/) front, we have just finished the [Google Summer of Code 2015](https://www.nsnam.org/wiki/GSOC2015AcceptedProjects) midterm reviews, with all our students passing. Good times ahead.
