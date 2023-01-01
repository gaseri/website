---
author: Vedran Miletić
date: 2015-07-28
tags:
  - academia
  - researchers night
  - cp2k
  - nwchem
  - gromacs
  - nvidia
  - gpu computing
---

# Joys and pains of interdisciplinary research

In 2012 University of Rijeka became [NVIDIA](https://www.nvidia.com/) [GPU Education Center](https://developer.nvidia.com/education_centers) (back then it was called CUDA Teaching Center). For non-techies: NVIDIA is a company producing graphical processors (GPUs), the computer chips that draw 3D graphics in games and the effects in modern movies. In the last couple of years, NVIDIA and other manufacturers allowed the usage of GPUs for general computations, so one can use them to do really fast multiplication of large matrices, finding paths in graphs, and other mathematical operations.

## Partnership with NVIDIA

To become a GPU Education Center, NVIDIA required us to have at least one recurring course in the curriculum and also hold regular workshops. In return, we got the GPUs to work with. Aside from allowing us to teach, having this hardware gave us an opportunity to initiate research projects using GPU computing. If we are successful in research, we can take the next step and become a [GPU Research Center](https://developer.nvidia.com/research_centers), and hopefully end up being [GPU Center of Excellence](https://developer.nvidia.com/centers_of_excellence) at some point. Either of these would give us access to special events, pre-release hardware, special pricing, etc. (Access to [NVIDIA HQ](https://commons.wikimedia.org/wiki/File:Nvidiaheadquarters.jpg) not included.)

Roughly a year later, in September 2013, we had the [Researchers night](https://hrcak.srce.hr/243373) in Rijeka. The goal was to get researchers from various disciplines to showcase their work, and potentially find collaborators or options for joint projects. I came there to find scientist interested in applying computation in their research, ideally using GPU computing. I was inspired by Assistant Professor [Željko Svedružić](https://svedruziclab.github.io/principal-investigator.html)'s [enthusiasm](https://youtu.be/JYiQ-cEw0b8?t=2m7s), and saw the potential for collaboration. A bit later I joined [BioSFGroup](https://svedruziclab.github.io/group.html) to do research work in computational chemistry in my spare time. At that point I had a PhD thesis to finish and there was little time to do other things.

## Dipping toes in computational chemistry

However, computational chemistry seemed worth gambling my spare time, due to a number of resons. First, I had the hardware that would eventually be obsolete, used or unused; second, there were open-source software computational chemistry packages which I could contribute to; third, I wanted to move the GPU Education Center closer to becoming the GPU Research Center. Very soon [Patrik Nikolić](https://nikoli.ch/) and I were in the lab AM to PM, five to six days a week. [GROMACS](https://www.gromacs.org/) was running day and night, and we were juggling visualizations in [VMD](https://www.ks.uiuc.edu/Research/vmd/), [Chimera](https://www.cgl.ucsf.edu/chimera/), [Avogadro](https://avogadro.cc/), and [Marvin](https://chemaxon.com/products/marvin) (occasionally we hated each of these packages). At some point, we also figured out how to do "simple" quantum mechanics calculations in [NWChem](https://www.nwchem-sw.org/) and [CP2K](https://www.cp2k.org/) (for the latter, it goes something like [this](https://www.cp2k.org/exercises:2015_ethz_mmm:mo_ethene)).

Ṛegardless of the extra work, the experience was very rewarding due to a number of things. First, both GROMACS and CP2K are meant to run on Linux. A biochemist might or might not have experience with compiling Linux software and linking it with GPU compute libraries such as NVIDIA CUDA; however, a biochemist does not want to be blocked by taking time to do these things. A computer scientist, on the other hand, is used to working with different operating systems and software. Software, and specifically scientific software, is what you do as a computer scientist. In my particular case, this experties includes both Linux and CUDA. Suddenly, the research group I was a part of started to iterate very fast since all of us did not have to learn the others domain to move forward.

## The exchange of knowledge

Second, the knowledge is flowing both ways. After a couple of months, Patrik was using Linux as his primary OS, and I had no problem reading through Professor Svedružić's copy of [Lehninger Principles of Biochemistry](https://www.amazon.com/Lehninger-Principles-Biochemistry-David-Nelson/dp/1429234148). With each new method (e.g. molecular dynamics or nudged elastic band) we exchanged more knowledge. "Let's try to plot these results using [Gnuplot](http://gnuplot.info/)" from my side was met with "why don't we try [Diels-Alder reaction](https://en.wikipedia.org/wiki/Diels-Alder_reaction)" from Patrik's. Eventually, I could assess approximations of forces resulting from different force fields as good or bad, and Patrik benchmarked GROMACS on one or more GPUs to decide how to run it. (By the way, GROMACS benefits from using two GPUs for calculation instead of one, kudos to developers on making that possible. We would test with three GPUs, but our most powerful systems have "only" two.)

There is a number of downsides as well. Instead of taking time to expand my horizons, I could have just followed the (un)written rules and take my time to work on projects that will result in papers strictly in field of computer science, because these count. I could have explored opportunities to squeeze more papers by re-exploiting my previous research work. This would enable me to avoid learning to use new software or to postpone developing new features in existing software packages. I could have done either, but I did not because I believed and still believe there are many more productive ways to use my time. (Just to be clear: we did create a publication resulting from this work. Namely, a book chapter written by our group will appear in a book by [Elsevier](https://www.elsevier.com/) in 2016.)

## Formal critera for professorship in support of individual passion and creativity

Present [classification of areas of sciences, engineering, biomedicine, biotechnology, social sciences, humanities, and arts in Croatia](https://narodne-novine.nn.hr/clanci/sluzbeni/2009_09_118_2929.html) recognizes interdisciplinary fields of science, but only a handful of them. However, [the minimal criteria for professorship in Croatia](https://narodne-novine.nn.hr/clanci/sluzbeni/289156.html) recognizes interdisciplinary papers only in sciences, biotechnology, and humanities. I am well aware it is hard to write precise criteria about a myriad of possible interdisciplinary combinations of different fields. But I am also aware that having such criteria would expand the amount of possibilities one has to get professorship, and in turn motivate more researchers to look into their options.

I might be an idealist, writing all this. I don't expect to motivate anyone to do the same; people have very different motivations for doing the work they do. Regardless, I have sort of an addiction to epic quotes, so here is one from Ralph Waldo Emerson:

> Every revolution was first a thought in one man's mind; and when the same thought occurs to another man, it is the key to that era.

That is, we are not the only group in Croatia combining life sciences and computer science. I'm very happy to say that [Mile Šikić](https://www.fer.unizg.hr/mile.sikic) from [University of Zagreb](https://www.unizg.hr/) [Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/), working in area of computer science, has [a number of papers in field of bionformatics](https://www.bib.irb.hr/pregled/profil/27663) (look for papers published in [Nuclelic Acids Research](https://academic.oup.com/nar)). Do these papers count for professorship? I have no idea, I guess we will find out eventually, but I doubt that getting counts up was the primary motivation for writing those papers.
