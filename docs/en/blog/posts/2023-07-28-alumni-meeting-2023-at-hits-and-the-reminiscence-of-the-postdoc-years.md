---
author: Vedran Miletić
authors:
  - vedranmiletic
date:
  created: 2023-07-28
categories:
  - Academia
  - Computing
tags:
  - academic careers
  - GROMACS
  - GPU computing
  - interdisciplinary research
---

# Alumni Meeting 2023 at HITS and the reminiscence of the postdoc years

---

![a fountain in the middle of a town square](https://unsplash.com/photos/xaWRyEnLk10/download?w=1920)

Photo source: [Jahanzeb Ahsan (@jahan_photobox) | Unsplash](https://unsplash.com/photos/a-fountain-in-the-middle-of-a-town-square-xaWRyEnLk10)

---

This month we had [Alumni Meeting 2023](https://alumni2023.h-its.org/) at the [Heidelberg Institute for Theoretical Studies](https://www.h-its.org/), or HITS for short. I was very glad to attend this whole-day event and reconnect with my former colleagues as well as [researchers](https://www.h-its.org/research/mbm/) [currently](https://www.h-its.org/research/mcm/) [working](https://www.h-its.org/research/ccc/) in the area of computational biochemistry at HITS. After all, this is the place and the institution where I worked for more than half of my time as a postdoc, where I started regularly [contributing code](../../people/principal-investigator.md#gromacs) to [GROMACS](https://www.gromacs.org/) molecular dynamics simulator, and published some of [my best papers](../../people/principal-investigator.md#research-papers-in-journals).

<!-- more -->

The day started with math colloquium talks by [Steve Trettel](https://stevejtrettel.site/) and [Richard Schwarz](https://www.math.brown.edu/reschwar/) in [Studio Villa Bosch](https://www.studio-villa-bosch.de/). I was particularly impressed by [3-Dimensional Space](https://www.3-dimensional.space/) visualizations of eight different geometries, some of which were shown during the talk.

## Dr. Johannes Wagner's career talk

After the lunch brake, career talks followed. I was especially happy that one of the speakers at the event was [Johannes Wagner](https://alumni2023.h-its.org/blog/hits-people/johannes-wagner/), who was at HITS as a PhD student under [Prof. Dr. Frauke Gräter](https://www.h-its.org/people/prof-dr-frauke-grater/) at the same time I was a postdoc. These days, Johannes is working as a managing director of [Eraneos Analytics](https://www.eraneos.com/service/data-ai/) and in his talk he compared working in the industry with consulting jobs and [academic careers](2023-06-10-should-i-do-a-phd.md#industry-vs-academia), emphasizing pros and cons of all three options. It is great to hear first-hand experiences from someone who has tried all three career paths at various stages in life. I am certian that current and former HITSters will benefit from having this information at their disposal in the future.

For what it's worth, I have actually kept contact with Johannes over the last few years. He visited Rijeka in the late summer of last year and we spent some time exploring the city and its surroundings:

<div data-embed-consent="instagram">
  <p data-embed-placeholder>
    Enable Instagram posts in cookie settings to view this post here.
    <button type="button" class="md-button embed-cookie-settings">Cookie settings</button> ·
    <a href="https://www.instagram.com/p/CiSynJeq_Cp/">View on Instagram</a>
  </p>
  <template>
    <blockquote class="instagram-media" data-instgrm-captioned
      data-instgrm-permalink="https://www.instagram.com/p/CiSynJeq_Cp/" data-instgrm-version="14">
      <a href="https://www.instagram.com/p/CiSynJeq_Cp/">A post shared by Dr. Vedran Miletić (@drvedranmiletic)</a>
    </blockquote>
  </template>
</div>

He also got a chance to meet other members of [my group](../../index.md), in particular our PhD student [Matea](../../people/phd-students.md#matea-turalija), and share with them his stories and lessons learned from working in our field for many years.

## Meeting the current members of the MBM group

Since I arrived at HITS the day before the alumni meeting, I had an afternoon free and went to meet current members of Frauke's [Molecular Biomechanics (MBM) group](https://www.h-its.org/research/mbm/). The group has grown quite a bit in the last five years and it now has [twenty members](https://www.h-its.org/research/mbm/people/) working on several computational [projects](https://www.h-its.org/research/mbm/projects/), such as simulation of [reactive molecular dynamics](https://www.h-its.org/projects/emulation-in-simulation/) and [collagen failure](https://www.h-its.org/projects/mechanoradicals-in-collagen/). Additionally, the group now also has an experimental lab, where [studies of dihydroxyphenylalanine](https://www.h-its.org/projects/mechanoradicals-and-dopa-in-collagen-tissue/) as a marker of mechanical stress and radical scavenger in collagen are performed.

## The reminiscence of the postdoc years

I worked with Frauke Gräter's group from November 2015 until March 2018. During these two and almost-half years, I had no teaching obligations and was almost fully focused on research and development.

### GROMACS developments

I developed [two features](https://github.com/HITS-MBM/gromacs-developments) we needed for then-ongoing research projects in the MBM group: conditional stop and sliced pull. The first feature allows stopping of simulation under certain conditions such as atoms being close to each other, which enabled us to more efficiently study [the stability of biological membranes](https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.8b01861) upon mechanical indentation. The second feature adds the option to introduce a directed force into the solvent around biomolecules, which we used to study [the peptides in uniform flow](https://www.cell.com/biophysj/fulltext/S0006-3495(19)30109-2).

Both of these features are now being reworked and reimplemented to be more generally useful. When they are finished, the plan is to submit them for inclusion in the official version of GROMACS. In particular, the conditional stop feature is much easier to implement in GROMACS from 2019 onwards thanks to the work on stop signal setting and handling by [Pascal Merz](https://www.linkedin.com/in/pascal-t-merz/) and [M. Eric Irrgang](https://www.linkedin.com/in/m-eric-irrgang-6a9566108/).

During my time at HITS, I also started participating in GROMACS development by adding documentation, fixing smaller bugs, participating in the meetings, and reviewing other people's code. I continue to do so today, albeit there was a pause for several years while [Patrik Nikolić](https://nikoli.ch/) and I worked under [RxTx Research](https://rxtxresearch.github.io/) to adapt [rDock](https://rdock.gitlab.io/) for modern computer systems under the name [RxDock](https://rxdock.gitlab.io/).

### AMD Radeon open-source compute

I [got excited about AMD's efforts under the GPUOpen umbrella](2016-01-17-amd-and-the-open-source-community-are-writing-history.md) and helped make GROMACS run via OpenCL on AMD Radeon GPUs using open-source drive stack. Aside from GROMACS, there were changes to [Mesa](../../people/principal-investigator.md#mesa) and [various LLVM sub-projects](../../people/principal-investigator.md#llvm). While contributions to open-source software matter on its own, it is worth noting that AMD's [ROCm](https://www.amd.com/en/products/software/rocm.html) uses the same Clang and LLVM that Mesa does.

This work was presented in [two European LLVM Developers' Meetings](../../people/principal-investigator.md#abstracts-in-conference-proceedings). The second presentation in particular expanded the set of supported software much wider than just GROMACS by including other molecular dynamics simulators such as [LAMMPS](https://lammps.org/) and [OpenMM](https://openmm.org/), as well as other libraries such as [ASL](https://asl.org.il/).

### Miscellaneous activities

I was also a system administrator for our group supercomputer, which included teaching system administration during the last few weeks before I left. This was, in a nutshell, a slightly customized and time-intensive version of [the Computer System Administration course](../../../hr/nastava/kolegiji/URS.md), covering [Red Hat Enterprise Linux](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux)/[CentOS](https://www.centos.org/), [FreeIPA](https://www.freeipa.org/), [systemd](https://systemd.io/), [firewalld](https://firewalld.org/), and [Slurm](https://slurm.schedmd.com/). I enjoy teaching this material very much and I taught it in many different settings and various forms over the years.

Finally, we also had some time for fun activities which you could barely count as work. The one that comes to mind is when our group participated in recording of this Floppy Keys video, narrated by [group](https://www.biophysics.org/profiles/frauke-graeter) [leaders](https://lemkelab.com/the-team/about-edward-lemke/) and [Davide Mercadante](https://lab.mercadante.net/):

<div data-embed-consent="youtube">
  <p data-embed-placeholder>
    Enable YouTube videos in cookie settings to play this video here.
    <button type="button" class="md-button embed-cookie-settings">Cookie settings</button> ·
    <a href="https://www.youtube.com/watch?v=XyiN1BpXimk">Watch on YouTube</a>
  </p>
  <iframe hidden data-src="https://www.youtube-nocookie.com/embed/XyiN1BpXimk"
    title="Floppy Keys" style="border: 1px solid black; width: 100%; height: 25rem"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>
</div>

## Long-term impact of postdoctoral choices

Many researchers from Croatia visit institutions abroad during their careers. Some of them choose to stay for years before returning, like I did. In such cases, returning to alumni meetings afterwards is a great chance to catch up with former colleagues and see what they are working on at present. These discussions also provide many ideas for the potential directions of your future research. This is particularly true at HITS where people from different fields of computational science come together and collaborate on research projects (Davide agrees with me in [his alumni interview](https://youtu.be/XbJrth8iKQI)).

I will not deny that doing my postdoc in such an interdisciplinary setting had some level of risk in it due to the formal rules of Croatian academic system not recognizing and rewarding [interdisciplinary research](2015-07-28-joys-and-pains-of-interdisciplinary-research.md) particularly well. Regardless, the experience gained while working in an intersection of several fields of science at HITS was unmatched by any other environment and well worth the risks involved.

In the end, with a bit of luck, it also worked out great in formal terms. In 2021 I became an assistant professor of computer science at [Faculty of Informatics and Digital Technologies, University of Rijeka](https://www.inf.uniri.hr/), got the opportunity to form [my own group](../../index.md), and started mentoring [my first PhD student](../../people/phd-students.md#matea-turalija). Looking forward to what's coming!
