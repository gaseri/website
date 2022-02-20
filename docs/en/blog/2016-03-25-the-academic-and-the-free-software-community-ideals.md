---
author: Vedran Miletić
date: 2016-03-25
tags:
  - academia
  - free and open-source software
  - software licensing
  - software patents
  - unicode
---

# The academic and the free software community ideals

Today I vaguely remembered there was one occasion in 2006 or 2007 when some guy from the academia doing something with Java and Unicode posted on some mailing list related to the free and open-source software about a tool he was developing. What made it interesting was that **the tool was open source, and he filed a patent on the algorithm**.

## Few searches after, boom, there it is

Google is a powerful tool. [The original thread](https://www.mail-archive.com/linux-utf8@nl.linux.org/msg05444.html) from March 2007 on (now defunct) linux-utf8 mailing list can be found on [The Mail Archive](https://www.mail-archive.com/). [The software website](http://u8u16.costar.sfu.ca/) is still up. [The patent](https://www.google.com/patents/US7783862) is out there as well.

Back in 2007 I was in my 3rd year of undergraduate study of mathematics (major) and computer science (minor), used to do Linux workshops in my spare time, and was aiming to do a Ph.D. in mathematics. I disliked the usage and development of proprietary research software which was quite common in much of the computer science research I saw back then. Unlike these researchers, I believed that that **academia and free software community agreed that knowledge should be free as in freedom**, and I wanted to be a part of such a community.

## Academic ideals

As a student, you are told continuously that **academia is for idealists**. People who put freedom before money. People who care about knowledge in and of itself and not how to sell it. And along with these ideas about academia, you are passed one more very important idea: **the authority of academia**. Whatever the issue, academia (not science, bear in mind) will provide a solution. Teaching? Academia knows how to do it best. Research? You bet. Sure, some professors here and other professors there might disagree on whatever topic, and one of them might be wrong. Regardless, academia will resolve whatever conflict that arises and produce the right answer. Nothing else but academia.

The idea, in essence, is that people outside of academia are just outsiders and their work is not relevant because it is not sanctioned by academics. They do not get the right to decide on relevant research. Their criticism of the work of someone from academia does not matter.

## Free software community ideals

Unlike academia, the free software community is based on **decentralization**, a lack of imposed hierarchy, individual creativity, and **strong opposition to this idea of requiring some sanction from some arbitrary central authority**. If you disagree, you are free to create software your way and invite others to do the same. There is no "officially right" and "officially wrong" way.

## Patent pending open-source code

"Some guy from the academia" in the case I mentioned above was Robert D. Cameron from [Simon Fraser University](https://www.sfu.ca/), asking the free software community to look at his code:

> u8u16-0.9 is available as open source software under an OSL 3.0 license at <http://u8u16.costar.sfu.ca/>

Rich Felker [was enthusiastic at first](https://www.mail-archive.com/linux-utf8@nl.linux.org/msg05445.html), but quickly [saw the software in question was patent pending](https://www.mail-archive.com/linux-utf8@nl.linux.org/msg05446.html):

> On second thought, I will not offer any further advice on this. The website refers to "patent-pending technology". Software patents are fundamentally wrong and unless you withdraw this nonsense you are an enemy of Free Software, of programmers, and users in general, and deserve to be ostracized by the community. Even if you intend to license the patents obtained freely for use in Free Software, it's still wrong to obtain them because it furthers a precedent that software patents are valid, particularly stupid patents like "applying vectorization in the obvious way to existing problem X".

There were also doubts presented regarding the relevance of this research at all, along with suggestions for better methods. While interesting, they are outside the scope of this blog post.

A patent is a [state-granted monopoly](https://mises.org/blog/are-patents-%E2%80%9Cmonopolies%E2%80%9D) designed to stimulate research, yet frequently used to [stop competition and delay access to new knowledge](https://www.eff.org/patent). Both [Mises Institute](https://mises.org/) and [Electronic Frontier Foundation](https://www.eff.org/) have written many articles on patents which I highly recommend for more information. In addition, as an excellent overview of the issues regarding the patent system, I can recommend the [Patent Absurdity: How software patents broke the system](https://youtu.be/_RPKtMTjXHg) movie.

So, there was a guy from the *idealistic* academia, who from my perspective seemed to take the wrong stance. And there was a guy outside of the *idealistic* academia and was seemingly taking the right stance. It made absolutely no sense at first that academia was working against freedom and an outsider was standing for freedom. Then it finally hit me: **the academia and the free software community do not hold the same ideals and do not pursue the same goals**. And this was also the moment I chose my side: the free software community first and the academia second.

However, academics tend to be very creative in proving they care about freedom of knowledge. Section 9 of [the paper](http://u8u16.costar.sfu.ca/attachment/wiki/WikiStart/ppopp074-cameron.pdf) (the only part of the paper I read) goes:

> A Simon Fraser University spin-off company, International Characters, has been formed to commercialize the results of the ongoing parallel bit stream research using an open-source model. Several patent applications have been filed on various aspects of parallel bit stream technology and inductive doubling architecture.

Whoa, open-source and patents. What's going on here?

> However, any issued patents are being dedicated to free use in research, teaching, experimentation and open-source software. This includes commercial use of open-source software provided that the software is actually publically available. However, commercial licenses are required for proprietary software applications as well as combinations of hardware and software.

Were it not for the patents, but for the licenses, I would completely agree with this approach. "If you are open sourcing your stuff, you are free to use my open-source stuff. If you are not open source, you are required to get a different license from me." That is how copyleft licenses work.

The problem is, as Rich says above, **every filling of a patent enforces the validity of the patent system itself**. The patent in question is just a normal patent and this is precisely the problem. Furthermore:

> From an industry perspective, the growth of software patents and open-source software are both undeniable phenomena. However, these industry trends are often seen to be in conflict, even though both are based in principle on disclosure and publication of technology details.

Unlike patents, free and open-source software is based on the principle of free unrestricted usage, modification, and distribution. These industry trends are seen in conflict, and that is the right way to see them.

> It is hoped that the patentleft model advanced by this commercialization effort will be seen as at least one constructive approach to resolving the conflict. A fine result would ultimately be legislative reform that publication of open source software is a form of knowledge dissemination that is considered fair use of any patented technology.

While it would certainly be nice if open source was protected from patent lawsuits, this tries to shift the focus from the **real issue, which is the patent itself and restrictions it imposes**.

## Opening the patents

The first possible solution is not to patent at all.

The second possible solution is to license the patent differently. Instead of being picky about the applications of the patent to decide whether royalties ought to be paid, which is the classical academic approach and also used above, one can simply license it royalty-free to everyone. This way, one **prevents innovations from being patented and licensed in a classical way**. This is what [Tesla Motors does](https://www.tesla.com/blog/all-our-patent-are-belong-you).

The third possible solution is to use the [copyleft-style patent license](https://en.wikipedia.org/wiki/Patentleft), which allows royalty-free use of knowledge given that you license your developments under the same terms. The approach uses the existing patent system in a reverse way, just like the copyleft licenses use the copyright system in a reverse way. This can be seen as an evolution of what [Open Invention Network](https://openinventionnetwork.com/) and [Biological Open Source (BiOS)](https://cambia.org/bios-landing/) already do.

This approach still relies on giving validity to the patent system, but unlike the classical academic approach, it also forces anyone to either go copyleft with their derivative patents or not use your technology. Effectively, this approach uses the patent system to **expand the technology commons accessible to everyone**, which is an interesting reverse of its originally intended usage.
