---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2023-06-10
tags:
  - phd
  - academia
  - free and open-source software
  - scientific software
  - cp2k
  - gromacs
  - amd
  - intel
---

# Should I do a Ph.D.?

---

![a bike is parked in front of a building](https://unsplash.com/photos/mLcCS4HU4yg/download?w=1920)

Photo source: [Santeri Liukkonen (@iamsanteri) | Unsplash](https://unsplash.com/photos/a-bike-is-parked-in-front-of-a-building-mLcCS4HU4yg)

---

Tough question, and the one that has been asked and [answered](https://www.princetonreview.com/grad-school-advice/why-you-shouldnt-pursue-phd) [over](https://www.theguardian.com/commentisfree/2018/aug/15/should-do-phd-you-asked-autocomplete-questions) [and](https://www.elsevier.com/connect/9-things-you-should-consider-before-embarking-on-a-phd) [over](https://awis.org/to-phd-or-not-phd/). The simplest answer is, of course, it depends on many factors.

As I [started blogging](2015-05-01-browser-wars.md) at [the end](2015-06-18-the-follow-up.md#phd-done) of my journey as a doctoral student, the topic of how I selected the field and ultimately decided to enroll in the postgraduate studies never really came up. In the following paragraphs, I will give a personal perspective on my Ph.D. endeavor. Just like other perspectives from doctors of *not that kind*, it is specific to the person in the situation, but parts of it might apply more broadly.

<!-- more -->

## Intrinsic motivation

Note that my motivation was largely intrinsic and there were few considerations in terms of monetary compensation. Therefore, I will not discuss whether getting a Ph.D. has made me richer than I would be if I hadn't embarked on it (it probably hasn't).

People tend to differ in intrinsic motivation quite a bit. To illustrate my point, my colleague [Arian](http://www.riteh.uniri.hr/osoba/arian-skoki) [Skoki](https://scholar.google.com/citations?user=n-k-YmgAAAAJ) usually mentions how he enjoys transforming scientific results into papers and choosing the right words to describe the observed phenomena. While [my track record](../../people/principal-investigator.md#publications-and-presentations) demonstrates I don't hate writing papers, that isn't my primary driving force either.

So, what exactly was my intrinsic motivation?

## Free and open-source software revolution

During [the early days of Mozilla](2015-05-01-browser-wars.md#browser-wars), I believed that free and open-source software is morally right and great in theory, but that it could never work in practice. I understood hobbyism and how open source can work in that domain, but it didn't extend to a belief that a sufficient number of people is ever going to volunteer their time and effort to create open-source software competitive with state-of-the-art proprietary software.

However, by [2000 and late milestone releases of Mozilla Application Suite](https://en.wikipedia.org/wiki/History_of_Mozilla_Application_Suite#Release_history), I started seriously questioning my assumption that open source could never work in practice. The product was far from perfect, but it was evolving at a rapid pace. By 2001 and Mozilla's 0.x releases, it became usable for day-to-day browsing and e-mail. Watching all these events unfold in real-time, I soon became a believer; I was certain that not only will we develop an open-source web browser and e-mail client that is better than the proprietary competition, but we will also develop [an office suite](https://en.wikipedia.org/wiki/OpenOffice.org), [an operating system](https://en.wikipedia.org/wiki/Fedora_Project), and everything else. Free and open-source software will win and, to paraphrase [Marc Andreessen](https://a16z.com/2011/08/20/why-software-is-eating-the-world/), it will eat the world.

To put it another way, I saw the revolution that was happening, I understood why it was inevitable, and wanted to help by being an active participant in it. It was soon obvious that installing Linux distributions on people's personal computers, while useful, isn't the only thing we should do if we want free and open-source software to succeed. I went to study mathematics and informatics to learn how software is created and then apply this knowledge in furthering our cause.

## Power of participation in the community

By the time I was close to finishing my diploma studies and deciding whether to do a Ph.D., I decided to start trying to contribute code to free and open-source software projects. Until then, I have only contributed translations, bug reports, website fixes, tutorials, and similar. As I was working with audio as a hobby quite a bit at that time, I [sent a few patches](../../people/principal-investigator.md#linux) to [Advanced Linux Sound Architecture (ALSA)](https://en.wikipedia.org/wiki/Advanced_Linux_Sound_Architecture). Previous involvement with [kX Project](https://github.com/kxproject) surely helped.

This was interesting and I learned a lot in the process, but it was not sustainable. There was simply no way to get a job in Croatia at the time that would require even the basic knowledge of sound card driver development, let alone include submitting patches to the audio subsystem in the kernel.

It also did not use any of my knowledge in the field of mathematics, which I just invested four years into acquiring. There had to be some other approach to put my skills to better use.

## Industry vs. academia

I figured very soon that the Croatian private sector was seldom involved in free and open-source software development, at least at the time. At best, you could get some web development work using open-source frameworks or system administration on Linux and [KVM](https://en.wikipedia.org/wiki/Kernel-based_Virtual_Machine)/[QEMU](https://en.wikipedia.org/wiki/QEMU). It sounded good enough for plan B, but I wanted to see if there was something even better than that.

I got the opportunity to start my academic career as a research and teaching assistant at [University of Rijeka](https://uniri.hr/)'s [Department of Informatics](https://www.inf.uniri.hr/) while obtaining a Ph.D. in computer science from [FER](https://www.fer.unizg.hr/) at the University of Zagreb.

Compared to working in the private sector, I hoped there would be more freedom to do things however you like as long as you get them done properly. In the end, I would say that I got what I wanted. Quoting from the *Thanks* section of [my Ph.D. thesis](../../people/principal-investigator.md#books-and-theses):

> I am particularly grateful that \[my supervisor Professor Branko Mikac\] enabled me to solve scientific and engineering problems by using and further developing free and open-source software, which is a life passion of mine.

The [further-developed](../../people/principal-investigator.md#ns-3) software in question was, of course, [the ns-3 network simulator](https://www.nsnam.org/). The members of the project that formed around the development of ns-3 are rightfully mentioned later in the same section:

> Network simulator ns-3 has built an awesome community over the years. I want to thank everyone from the community who helped me in some way, but the list would be far too long, so here is a short one: Mathieu Lacage for describing trampoline objects (among others) and encouraging me to code what I need, Peter D. Barnes for all the design alternatives that were thrown away and never implemented (but taught me a lot about the software design), Tommaso Pecorella for that *energetic* midnight discussion in front of the hotel, Alina Quereilhac and Alex Afanasyev for fixing Waf with me, and last, but certainly not least, Tom Henderson for being the leader of an open organization and herding cats more often than not.

## Open-source software already ate the world?

As years went by, I started to feel the shift in attitudes toward free and open-source software among my peers and my students alike.

This shift was reflected in the world more broadly. In 2013, [Jim Whitehurst](https://en.wikipedia.org/wiki/Jim_Whitehurst), then president and CEO of [Red Hat](https://www.redhat.com/), said that open source is ["a viable alternative to proprietary software in a whole bunch of categories"](https://youtu.be/abTSM8hvkb8?t=30m57s) and that "the next twenty years is no longer about being a viable alternative and it's about being the default choice for the next generation of IT infrastructure". However, only a year after that, we already had [Ansible](https://en.wikipedia.org/wiki/Ansible_(software)), [Terraform](https://en.wikipedia.org/wiki/Terraform_(software)), [Docker](https://en.wikipedia.org/wiki/Docker_(software)), and [Kubernetes](https://en.wikipedia.org/wiki/Kubernetes) and these tools were also popular enough to be considered the default choice.

By the time I finished my thesis in 2015, I felt that something major has changed and it would be great if such a transition can happen with other software, especially scientific software. While ns-3 was pretty advanced in terms of using best practices in software engineering, scientific software, in general, can be considered as lagging behind IT infrastructure software in that regard. Therefore, I figured it could use some help in fixing that.

I soon ended up working on [computational chemistry software](2015-07-28-joys-and-pains-of-interdisciplinary-research.md) due to the University of Rijeka's [Department of Biotechnology](https://www.biotech.uniri.hr/) using it in [drug design and related research](https://svedruziclab.github.io/research.html) and found myself enjoying being productive and finding many opportunities to apply what I learned previously. This especially came to fruition later, during my time in [Prof. Dr. Frauke Gräter](https://www.h-its.org/people/prof-dr-frauke-grater/)'s [Molecular Biomechanics](https://www.h-its.org/research/mbm/) group at [Heidelberg Institute for Theoretical Studies](https://www.h-its.org/). I published [three journal papers](../../people/principal-investigator.md#research-papers-in-journals), while making numerous contributions to [Mesa](../../people/principal-investigator.md#mesa), [LLVM](../../people/principal-investigator.md#llvm), [CP2K](../../people/principal-investigator.md#cp2k), and, of course, [GROMACS](../../people/principal-investigator.md#gromacs).

I also started regularly writing about what I was developing for popularization and activism purposes. Specifically, the best blog post I ever wrote, [What is the price of open-source fear, uncertainty, and doubt?](2015-09-14-what-is-the-price-of-open-source-fear-uncertainty-and-doubt.md), is from the early months of my postdoc, inspired by the discussions at the [4th](https://www.cp2k.org/events:2015_cecam_tutorial:index) [CPK2](https://www.cp2k.org/exercises:2015_cecam_tutorial:index) [Tutorial](https://www.cecam.org/workshop-details/480). Another great blog post, [AMD and the open-source community are writing history](2016-01-17-amd-and-the-open-source-community-are-writing-history.md), came out some months later.

## Independence and final words

In 2021, I finally became an assistant professor, started [my group](../../index.md), and got my first Ph.D. student, [Matea](../../people/phd-students.md#matea-turalija). Good times are ahead!

Looking back now, would I do it again? I surely would. There were certainly hard times and the way forward wasn't always clear, but I am very glad that things turned out the way they did. Along the way, I took many opportunities to work with great people on the topics that mattered while applying my skills in free and open-source software development.

Would I pursue the topic in the same field? Possibly, as there is still much work to be done in computational chemistry software and the exascale supercomputer architectures we use are still [evolving and expanding](../../projects.md#dpu-offload-of-force-reduction-calculations-in-molecular-dynamics-simulations).

Finally, what would I be doing if I wasn't doing what I am now? I love that question since it forces you to be creative and think about areas you are not that familiar with, which is a useful thought experiment. I would look into research topics around cryptography processors (such as [Trusted Platform Module](https://en.wikipedia.org/wiki/Trusted_Platform_Module), [Intel Management Engine](https://en.wikipedia.org/wiki/Intel_Management_Engine), and [AMD Platform Security Processor](https://en.wikipedia.org/wiki/AMD_Platform_Security_Processor)), possibly using [coreboot](https://en.wikipedia.org/wiki/Coreboot). To give an example of the actual topic that comes to mind, perhaps I would aim for something similar to the [Christian Werling](https://www.user.tu-berlin.de/cwerling/)'s and [Robert Buhren](https://www.linkedin.com/in/robert-buhren/)'s CCC talk from several years ago, titled [Dissecting the AMD Platform Security Processor](https://media.ccc.de/v/thms-38-dissecting-the-amd-platform-security-processor). I found it very intriguing how many interesting things can be performed using common hardware and some specialized open-source software.
