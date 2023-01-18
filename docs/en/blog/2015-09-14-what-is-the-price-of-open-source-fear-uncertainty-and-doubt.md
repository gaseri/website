---
author: Vedran Miletić
date: 2015-09-14
tags:
  - american chemical society
  - cp2k
  - nwchem
  - q-chem
  - free and open-source software
  - scientific software
  - software licensing
---

# What is the price of open-source fear, uncertainty, and doubt?

[The Journal of Physical Chemistry Letters (JPCL)](https://pubs.acs.org/journal/jpclcd), published by [American Chemical Society](https://www.acs.org/), recently put out two Viewpoints discussing open-source software:

1. [Open Source and Open Data Should Be Standard Practices](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00285) by [J. Daniel Gezelter](https://chemistry.nd.edu/people/j-daniel-gezelter/), and
2. [What Is the Price of Open-Source Software?](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b01258) by [Anna I. Krylov](https://iopenshell.usc.edu/), [John M. Herbert](https://chemistry.osu.edu/people/herbert.44), Filipp Furche, Martin Head-Gordon, Peter J. Knowles, Roland Lindh, Frederick R. Manby, Peter Pulay, Chris-Kriton Skylaris, and Hans-Joachim Werner.

Viewpoints are not detailed reviews of the topic, but instead, present the author's view on the state-of-the-art of a particular field.

The first of two articles stands for open source and open data. The article describes Quantum Chemical Program Exchange (QCPE), which was used in the 1980s and 1990s for the exchange of quantum chemistry codes between researchers and is roughly equivalent to the modern-day [GitHub](https://github.com/). The second of two articles questions the open-source software development practice, advocating the usage and development of proprietary software. I will dissect and counter some of the key points from the second article below.

Just to be clear: I will not discuss the issues of Open Data and Open Access; they are very important and they deserve a separate post. I will focus solely on the use of [free and open-source software (FOSS)](https://en.wikipedia.org/wiki/Free_and_open-source_software) and proprietary software in computational chemistry research.

## Reactions and replies by others

There are reactions to both articles already posted on the Internet. [Christoph Jacob](https://www.tu-braunschweig.de/pci/agjacob/mitarbeitende/jacob) replied with a blog post titled [How Open are Commercial Scientific Software Packages?](https://christophjacob.wordpress.com/2015/07/18/how-open-are-commercial-scientific-software-packages/) Among the rest, he says:

> To develop, test, and finally use a new idea, it needs to be implemented in software. Usually, this requires using a lot of well-established tools, such as integral codes, basic methods developed many decades ago, and advanced numerical algorithms. All of these are a prerequisite for new developments, but not “interesting” by itself anymore today. Even though all these tools are well-documented in the scientific literature, recreating them would be a major effort that cannot be repeated every time and by every research group – because both time and funding are limited resources, especially for young researchers with rather small groups such as myself.
>
> Therefore, method developers in quantum chemistry need some existing program package as a “development platform”. Both open-source and commercial codes can offer such a platform. Open-source codes have the advantage that there is no barrier to access. Anyone can download the source code and start working on a new method.

I fully agree with this idea and the rest of his post, so I will not repeat it here. What is interesting to note, however, is that Christoph is a contributor to [ADF](https://www.scm.com/), a proprietary quantum chemistry software.

!!! success
    Christoph Jacob later published a Viewpoint in JPCL titled [How Open Is Commercial Scientific Software?](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.5b02609) based on his blog post.

[Another well-put reply](https://plus.google.com/u/0/+MaximilianKubillus/posts/M8HS3VCMgjT) is posted by [Maximilian Kubillus](https://www.ipc.kit.edu/tcb/english/Staff_427.php). This part is particularly well put:

> A letter about scientific open-source software, written by TEN authors that own, work for or are founding members of closed-source software companies, saying that \[open-source software\] could never reach the quality of good closed-source software. It also states that good code review won't happen in \[open-source\] environments and efficient algorithms can only be developed by professional scientific programmers, using words like cyberinfrastructure without any reference to what they mean here and calling promoters of \[open-source software\] naive without giving a real foundation on why their presented open software models don't work (in their eyes).

I agree with this post as well and I will not repeat it here.

## Dissecting the Viewpoint arguments

All quotations in the following text are copied from the [Full-Text HTML version of the "What Is the Price of Open-Source Software?" Viewpoint](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.5b01258).

### Is open source mostly mandated?

> The notion that all scientific software should be open-source and free has been actively promoted in recent years, mostly from the top down via mandates from funding agencies but occasionally from the bottom up, as exemplified by a recent Viewpoint in this journal.

It is true that both funding agencies and individuals promote FOSS. However, the authors did not cite any data from which they could conclude that promotion happened **mostly** in a top-down way, and only **occasionally** in a bottom-up way. In fact, I would argue that the opposite is true.

As someone who is involved in the open-source community since the early 2000s, I am well aware of the efforts that were put by open-source supporters to get governments and funding agencies to understand the importance of open-source software and mandate it in regulations. As one example of a monumental effort, it took to move the public sector to open-source software and [OpenDocument](https://opendocumentformat.org/), see [Open source lobbying](https://youtu.be/WgHIunOA9Us), a story about writing a national policy of open source in the Netherlands, presented at 24th CCC in Berlin, 2007.

To summarize: for any top-down mandate of open source to happen, a lot of bottom-up efforts are required. These efforts are not trivial and usually take a long time and a lot of (mostly volunteer) effort.

### Does it really matter who has the software development skills?

> To bring new ideas to the production level, with software that is accessible to (and useful for) the broader scientific community, contributions from expert programmers are required. These technical tasks usually cannot—and generally should not—be conducted by graduate students or postdocs, who should instead be focused on science and innovation. To this end, Q-Chem employs four scientific programmers.

The notion that a certain percentage of scientists (graduate students, postdocs) do not possess technical skills (i.e. software engineering) required for the development of complex codes makes sense; whether this still applies to most of the scientists studying quantum chemistry today can be discussed. Still, the argument does not imply anywhere that the software these "four scientific programmers" are developing should be proprietary or closed source.

### Is selling licenses a sustainable model of software development, let alone the only sustainable model?

> Sales revenue cannot support the entire development cost of an academic code, but it contributes critically to its sustainability. The cost that the customer pays for a code like Q-Chem reflects this funding model: it is vastly lower than the development cost, particularly for academic customers but also for industry. It primarily reflects the sustainability cost.

A software like Q-Chem earns money by selling licenses and uses this money to fund programmers who develop it, which is the traditional proprietary software business model. This model is very simple but has many flaws.

Imagine an academic lab using Q-Chem in its protocols. Suddenly, Q-Chem changes a feature the lab cares about, or the lab's university buys a particular HPC based on the architecture that is not very common on the market and happens to be unsupported by Q-Chem. Or even worse, imagine the company behind Q-Chem disappearing.

If any of these scenarios do occur, the lab in our story is left with a binary that runs on their current computer system. Since the lab has no access to the source code, they are unable to port the code to the new systems. There is only one provider of the service they need (i.e. the improvement of the software they use): the company behind Q-Chem, which **owns the intellectual property rights** to Q-Chem code. If the lab then decides to switch to another software, they are met with the unexpected additional license costs from buying another proprietary software. That is not all, however, as the update of lab protocols and retraining of lab scientists takes time and effort.

The lab from our story is in a [vendor lock-in situation](https://en.wikipedia.org/wiki/Vendor_lock-in): they can not easily change the vendor providing support for software they use, because **the software is closed source and the property of one and only one vendor**. Imagine instead the software used by the lab was FOSS. Suddenly, the vendor providing support and maintenance changed conditions so they no longer suit the requirements of the lab, or the vendor goes bankrupt altogether. Another vendor can easily start working on the source code (since it is available to everyone), and this vendor can start providing support to the lab under any mutually agreed contract.

### Is scientific software equivalent to a sophisticated machine in the physical world?

> Nevertheless, the software itself is a product, not a scientific finding, more akin to, say, an NMR spectrometer—a sophisticated instrument—than to the spectra produced by that instrument.

This is true. This is why projects such as [Open Source Ecology](https://www.opensourceecology.org/) provide blueprints (source-code equivalent) for industrial machines. The reason why the open-source movement succeed first in the domain of software is the low cost of making copies of the software (both the source code and the binaries). The cost of the data transmission and storage became so low with the technology advancements that today the only significant cost in producing new software is the development itself. Basically, once developed, the software can be distributed to any number of users at a very low cost.

### Is free as in *free beer* the same as free as in *free speech* when it comes to software?

> There Is No Free Software.

Nice try. There is [free software](https://www.gnu.org/philosophy/free-sw.html), there is even the [Free Software Foundation](https://www.fsf.org/). The authors should **clearly separate the issues of software cost and software freedom**, which they did not. The following text demonstrates this clearly.

### Are free and open-source software companies and customers naïve?

> Gezelter acknowledges the cost of maintaining scientific software and suggests alternative models to defray these costs including selling support, consulting, or an interface, all the while making the source code available for free. These suggestions strike us as naïve, something akin to giving away automobiles but charging for the mechanic who services them. Such a model creates a financial incentive to release a less-than-stellar product into the public domain, then charge to make it useful and usable. It is better to release a top-of-the-line product for a nominal fee.

A free and open-source quantum chemistry tool can have a graphical user interface (GUI) which would specifically target common lab protocols in, say, material science. If the vendor makes the GUI proprietary, no functionality of the original software is lost. The **GUI just makes the same functionality available in a different way**, potentially simpler to use. If you want to use the GUI and save time, you have to pay the license fee. If you want to use the quantum chemistry software without the GUI, you are free to do so, and you can even write your own GUI, and even give it out under a free software license. Your **freedom to use the original FOSS tool is preserved**.

It is in the interest of the software vendor to make both the software and its GUI as high quality and as easy to maintain as possible, to attract code contributions from the outside. In terms of proprietary software, **these contributions are equivalent to getting the development work done for free** (i.e. without paying the programmer doing it).

As for this business model being naïve, consider the open-source leader, [Red Hat](https://www.redhat.com/), which has 1.5 billion dollars in revenues per year. As for [Fortune Global 500](https://fortune.com/global500/) companies, [100% of airlines, telcos, healthcare companies, commercial banks, and U.S. Executive Departments rely on Red Hat](https://www.redhat.com/en/resources/who-relies-on-red-hat). If you look up the names of the companies, you will find out they are anything but naïve.

Finally, the car analogy the authors use is completely flawed. Red Hat did not score the business from [NASA](https://www.nasa.gov/), [NYSE](https://www.nyse.com/), or any other organization by giving bad software for free and then charging for service fees; had they, they would have been easily overtaken by a competitor and out of business by now. Since **all of the Red Hat supported software is free and open-source**, the potential competitor would just take the source code, improve it, and build support around it. Red Hat would find itself in a situation where they have to understand the changes their competitor is making, and the competitor would find that supporting better code would be easier and much cheaper.

To summarize: Red Hat has to be high-quality as it **does not have the luxury of owning the source code**, which would automatically exclude the competitors from the market.

### Can a researcher decide which free software to support without paying the license?

> Is “free” software genuinely free of charge to individual researchers? Consider software developed in the U.S. national laboratories. These ventures are supported by full-time scientific programmers employed specifically for the task, and the cost to support and develop these products is subtracted from the pool of research funding available to the rest of the community. The individual researcher pays for these codes, in a sense, with his rejected grant proposals in times of lean funding. In contrast to using one’s own performance metrics to guide software purchases, within this system, one has no choice in what one pays for. In other words, “free software” is not free for you; the only sense in which it is “free” is that you are freed from making a choice about how to spend your research money.

Research funding comes from public money, and the **public should be granted full access to the research results** it, in a sense, bought. In particular, this access includes access to the source code of the software developed using public funding. By transferring the ownership of the source code to any company we are basically funding private ventures with public money. Furthermore, we are letting the company that gets the ownership of the source code **dictate the terms under which the public will access the results it has already paid for**.

The interested companies are free to sell support for the software, additional functionality (such as a GUI) designed for the software, or even their development services (say, implementation or integration of a particular feature in an open-source way), but the part of the **software developed using the public money must remain available to everyone under a FOSS license**.

Claiming that the individual researcher who did not receive funding for research paid anything is a flawed argument in any sense. However, **an individual researcher has a choice what FOSS he will support**: as a group leader, he can assign the implementation of his research requirements in a particular software of his choosing to his students and postdocs, he can do the implementation himself, or he can pay an external contractor to do it for him. Furthermore, he can choose an external contractor among many, which is impossible in the case of proprietary software since – again – the **company behind the software has exclusive access to the code**.

### Is saving time worth losing freedom?

> Computational chemistry software must balance the needs of two audiences: users, who gauge their productivity based on the speed, functionality, and user-friendliness of a given program; and developers, who may be more concerned with whether the structure “under the hood” provides an environment that fosters innovation and ease of implementation. As a quantitative example, consider that the cost of supporting a postdoctoral associate (salary plus benefits) is perhaps $4,800/month. If the use of well-supported commercial software can save 2 weeks of a postdoc’s time, then this would justify an expense of ≳$2,000 to purchase a software license. This amount exceeds the cost of an academic license for many computational chemistry programs. Given the choice between a free product and a commercial one, a scientist should make a decision based on her own needs and her own criteria for doing innovative research.

This is a sensible argument. However, it does not address the freedom issue already discussed above. **The lab that buys the license to use the software depends on a single vendor to maintain the software for them.** Furthermore, the lab is not granted the right to modify the code to fit their needs and to redistribute their modifications to their colleagues. The issue here is not the price of the license but **the freedom that is taken away from the paying user**.

### What Is “Open Source”?

> The term “open source” is ubiquitous but its meaning is ambiguous. Some codes are “free” but are not open, whereas others make the source code available, albeit without binary executables so that responsibility for compilation and installation is left to the user. Insofar as the use of commercial quantum chemistry software is a mainstay of modern chemical research and teaching, there exists a broad consensus that the commercial model offers the stability and user support that the community desires.

[Wikipedia provides a definition for open source](https://en.wikipedia.org/wiki/Open_source) that says: "open source as a development model promotes a universal access via a free license to a product's design or blueprint, and universal redistribution of that design or blueprint, including subsequent improvements to it by anyone". A simple, clear-cut definition.

The **authors again confuse freeware with FOSS** and then talk about the requirement to compile and install FOSS from source as an issue, which it simply is not. GNU/Linux distributions such as [Debian Science](https://wiki.debian.org/DebianScience) (including [Chemistry](https://wiki.debian.org/DebianScience/Chemistry)) and [Fedora Scientific](https://labs.fedoraproject.org/en/scientific/) provide ready to use binaries for end users that prefer to avoid compiling software.

Finally, the data supporting "broad consensus that the commercial model offers the stability and user support that the community desires" is lacking (and the term itself is ambiguous); I would honestly like to see relevant market share data of quantum chemistry tools presented and discussed. For illustrative purposes, let's assume a number of citations as a rough metric for the number of users and therefore a consensus on the usage of proprietary vs the usage of FOSS codes. Searching for citations in Web of Science gives [542 articles](https://www.webofscience.com/wos/woscc/full-record/WOS:000165132900011) for [Q-Chem 2.0 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/1096-987X%28200012%2921%3A16%3C1532%3A%3AAID-JCC10%3E3.0.CO%3B2-W) and [1581 articles](https://www.webofscience.com/wos/woscc/full-record/WOS:000239251500002) for [Advances in methods paper](https://pubs.rsc.org/en/content/articlelanding/2006/cp/b517914a); on the other hand, [CP2K QUICKSTEP paper](https://www.sciencedirect.com/science/article/pii/S0010465505000615) has [938 citations](https://www.webofscience.com/wos/woscc/full-record/WOS:000228421500005). Despite CP2K having a lower total number of articles citing the relevant paper then Q-Chem, **the number of citations is comparable in the order of magnitude**. Since there are other FOSS codes (for example, [NWChem](https://www.nwchem-sw.org/) and [Quantum ESPRESSO](https://www.quantum-espresso.org/)) as well as other proprietary codes, this result does not prove much. However, this result questions the "broad consensus" claimed by the authors.

### Does being open source imply anyone can modify (and break) the main source code repository?

> Strict coding guidelines can be enforced within a model where source code access is limited to qualified developers, and this kind of stability offers one counterbalance to the “reproducibility crisis”. To the extent that such a crisis exists, it has occurred in spite of the existence of open-source electronic structure codes such as GAMESS, NWChem, and CP2K.

**Strict coding guidelines can be enforced in any project**, be it FOSS or proprietary software. The [ns-3 network simulator](https://www.nsnam.org/) and [Linux kernel](https://www.kernel.org/) are both good examples of FOSS projects with strict rules on coding style, API usage, and on not breaking the existing functionality.

The "reproducibility crisis" is two separate issues: **being able to run the code someone else had run previously** and having the code produce the same result within tolerance despite changes over time. The first issue is actually **better solved by open-source software since anyone can access the code**, and the second one is unrelated to code being open or proprietary, as described above.

### Does a good description of the algorithm make the implementation code unnecessary?

> Occasionally the open-source model is touted on the grounds that one can use the source code to learn about the underlying algorithms, but this hardly seems relevant if the methods and algorithms are published in the scientific literature. Source code itself rarely constitutes enjoyable reading, and using source code to learn about an algorithm is a last resort forced by poorly written scientific papers. Better peer review is a more desirable solution.

This is true, and we should also note that having both the source code and its detailed description is an ideal situation: you can study both, you can learn implementation tricks that could easily have been omitted from the description, and you can modify the algorithm without having to reimplement it first.

### Is freely available to academics free enough?

> A more practical use of openly available source code is to reuse parts of it in other programs, provided that the terms of the software license allow this. Often, they do not. Some ostensibly “open” chemistry codes forbid reuse, or even redistribution.

Here the authors cite [ORCA](https://www.kofo.mpg.de/en/research/services/orca) and [GAMESS](https://www.msg.chem.iastate.edu/gamess/index.html). **ORCA and GAMESS are not free and open-source software.** (They are available to academics free of charge – it might be that this fact was the source of confusion.)

### Is viral license the problem for adoption?

> Others, such as CP2K, use the restrictive General Public License that requires any derivative built on the original code to be open-source itself. Variation in design structure from one program to the next also severely hampers transferability, even if the license terms are amenable.

GNU General Public License (GPL) is a viral license, meaning that any code which reuses GPL code must also be licensed under GPL. This way, **more and more code becomes FOSS over time**. The authors are trying to imply that FOSS quantum chemistry tools are the problem for the quantum chemistry software ecosystem due to the GPL. Such implication is a **misunderstanding of how FOSS works**. The analogous misunderstanding was presented by [Steve Ballmer, back in 2001, who said Linux was "cancer that attaches itself in an intellectual property sense to everything it touches" due to it being licensed under the GPL](https://en.wikiquote.org/wiki/Steve_Ballmer). [Microsoft's lost decade](https://www.vanityfair.com/news/business/2012/08/microsoft-lost-mojo-steve-ballmer) followed, and one can argue there is a causation since the companies like Apple, Google, and Facebook gladly reused FOSS when they could, and subsequently contributed their improvements back to the FOSS community.

### Is it open enough if the software project is open by invite?

> To facilitate innovation by developers, source code needs only to be available to people who intend to build upon it. This is commonly accomplished in the framework of “closed-source” software projects by granting academic groups access to the source code for development purposes.

This is too easy to counter. **Contributions to a FOSS project can come from anywhere.** A student studying a particular variant of an algorithm wants to implement it and contribute it back. A professor trying different algorithms and contributing the best one. With so much FOSS out there, the potential contributor is not going to bother with software source code that is behind an NDA, an email form, or an invitation of some kind. **Such a potential contributor is unlikely to give his freedom away and sign his code away for the benefits of a particular private venture.**

### Will open source destroy proprietary software?

> What would the impact be on computational chemistry of destroying other teamware projects such as Molpro, Turbomole, Jaguar, Molcas, PQS, or ONETEP, in the interest of satisfying some “open-source” mandate?

I fail to see how the open-source mandate per se destroys any proprietary software. Namely, all these proprietary software projects have the option to open source their code and **change their business model to compete based on quality, not code ownership**. Alternatively, if they desire to continue to maintain the present development practices, they are still free to find other sources of income.

### Is proprietary software more optimized?

> \[Open-source mandate\] would, in our view, detract from the merit-based review process. When evaluating grant proposals that involve software development, the questions to be asked should be: 1. What will be the quality of the software in terms of the new science that it enables, either on the applications side or on the development side? 2. How will the software foster productivity? For example, how computationally efficient is it for a given task? How usable will the software be, and how quickly will other scientists be able to learn to use it for their own research? A rigid, mindless focus on an open-source mantra is a distraction from these more important criteria. It can even be an excuse to ignore them, and creates an uneven playing field in which developers who prefer to work with a commercial platform are put at a disadvantage and potentially forced to adopt less efficient practices.

The quality argument presented in the first point has already been addressed above. The second point does not say much without the accompanying benchmarks that would support the idea that, when both are implementing the same method, the proprietary software is computationally more efficient than FOSS. However, there is a [report on CP2K performance from Bethune, Reid, and Lazzaro](https://www.research.ed.ac.uk/en/publications/cp2k-performance-from-cray-xt3-to-xc30) showing the improvement of computational performance over time. These **measurements only prove that there is a FOSS project that specifically cares about computational efficiency**, but it does not say anything in absolute terms.

### Does open source force a scientist to open everything straight away?

> Open-source requirements potentially force a scientist to choose between pursuing a funding opportunity versus implementing an idea in the quickest, most efficient, and highest-impact way. A strictly open-source environment may furthermore disincentivize young researchers to make new code available right away, lest their ability to publish papers be short-circuited by a more senior researcher with an army of postdocs poised to take advantage of any new code.

Using open-source software under GPL version 2 or later allows a researcher to make private changes and never release them to the public. Namely, **GPL version 2 only mandates that a release of the software in binary form be accompanied by the release of the matching source code**. Therefore, if anything, **the scientist has more options how (and if) to release the code**, not less.

As for the "army of postdocs jumping on any new code", I see many more advantages than disadvantages of this particular situation. Namely, since **nobody can claim the authorship of anyone else's code**, one can use this heightened interest in the new code to explain to other scientists the research he is doing, and open opportunities for collaboration.

### Is orphaned code more common in open source than in proprietary software?

> This would contribute directly to the scenario that Gezelter wishes to avoid, namely, one where students leave behind “orphaned” code that will never be incorporated into mainstream, production-level software. Viewed in these terms, an open-source mandate degrades, rather than enhances, cyberinfrastructure.

If the students were developing proprietary instead of open-source software, their "orphaned" code would automatically not be available to other researchers for further development. Whether or not any code will be incorporated in production-level software depends on the code quality, its usefulness, and community interest.

### Are software freedom and software quality competing features?

> How should the impact of software be measured? Scientific publications are a more sound metric than either the price of a product or whether its source code is available in the public domain. Software is meant to serve scientific research, in the same way that any other scientific instrument is intended. As such, the question should not be whether software is free or open-source, but rather, what new science can be accomplished with it?

True, scientific publications are one possible way to measure the impact of software. However, **open-source software is certainly not released in the public domain**. As for the question, why not **require both the quality and the freedom from software**? Are really these two requirements competing against each other?

### Is software freedom a political rhetoric?

> Let us not allow political rhetoric to dictate how we are to do science. Let different ideas and different models (including open source!) compete freely and flourish, and let the community focus instead on the most important metric of all: what is good for scientific discovery.

The **issue of software freedom is both an ethical issue and a practical one**, as described above, so it is hardly a "political rhetoric". I would propose instead that we let the community choose the software to use, based on both freedom and quality. While at it, **we should stand firm on the requirement that the publicly-funded scientific software development results in free and open-source software**. Whether the proprietary vendors will be willing to adapt their business models is ultimately their choice.

## Conflict of interest statement

I am a contributor to [CP2K](https://www.cp2k.org/) and [GROMACS](https://www.gromacs.org/) open-source molecular dynamics software packages. So far, I have attended two CP2K developer meetings, one remotely and one being physically present in Zürich. For my contributions in general and for these attendances in particular, I have not received any monetary compensation from [ETH Zürich](https://www.ethz.ch/), [University of Zürich](https://www.uzh.ch/), or any other party involved in CP2K development.
