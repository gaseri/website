---
author: Vedran Miletić
date: 2016-01-17
tags:
  - amd
  - radeon
  - gpuopen
  - free and open-source software
  - firmware
  - gpu drivers
  - intel
  - nvidia
  - gpu computing
---

# AMD and the open-source community are writing history

Over the last few years, [AMD](https://www.amd.com/) has slowly been walking the path towards having fully [open](https://www.phoronix.com/scan.php?page=news_item&px=ODg5Nw) [source](https://www.phoronix.com/scan.php?page=news_item&px=OTQzNQ) [drivers](https://www.phoronix.com/scan.php?page=news_item&px=AMD-Two-More-Open-Linux-Devs) on Linux. AMD did not walk alone, they got help from [Red](https://www.phoronix.com/scan.php?page=news_item&px=MTc2NTY) [Hat](https://www.phoronix.com/scan.php?page=news_item&px=Red-Hat-Hiring-Fedora-Work), [SUSE](https://www.phoronix.com/scan.php?page=news_item&px=SUSE-Hiring-Another-Gfx-Dev), and probably others. Phoronix also mentions [PathScale](https://www.phoronix.com/scan.php?page=news_item&px=MTcwNDI), but I have been told on Freenode channel #radeon this is not the case and found no trace of their involvement.

AMD finally [publically unveiled](https://youtu.be/eXCXJoRsgJc) the GPUOpen initiative on the 15th of December 2015. The story was covered on [AnandTech](https://www.anandtech.com/show/9853/amd-gpuopen-linux-open-source), [Maximum PC](https://www.maximumpc.com/amd-rtg-summit-gpuopen-and-software/), [Ars Technica](https://arstechnica.com/information-technology/2015/12/amd-embraces-open-source-to-take-on-nvidias-gameworks/), [Softpedia](https://news.softpedia.com/news/amd-going-open-source-with-amdgpu-linux-driver-and-gpuopen-tools-497663.shtml), and others. For the open-source community that follows the development of Linux graphics and computing stack, this announcement comes as hardly surprising: Alex Deucher and Jammy Zhou [presented plans regarding amdgpu on XDC2015](https://youtu.be/lXi0ByVTFyY) in September 2015. Regardless, public announcement in mainstream media proves that AMD is serious about GPUOpen.

I believe GPUOpen is **the best chance we will get in this decade** to open up the driver and software stacks in the graphics and computing industry. I will outline the reasons for my optimism below. As for the history behind open-source drivers for ATi/AMD GPUs, I suggest [the well-written reminiscence on Phoronix](https://www.phoronix.com/scan.php?page=news_item&px=Reminiscing-OSS-AMD-2016).

## Intel and NVIDIA

AMD's only competitors are Intel and NVIDIA. More than a decade ago, these three had other companies competing with them. However, all the companies that used to produce graphics processors either ceased to exist due to bankruptcy/acquisition or changed their focus to other markets.

Intel has **very good open-source drivers** and this has been the case for almost a decade now. However, they only produce integrated GPU which are not very interesting for gaming and heterogeneous computing. Sadly, their open-source support *does not* include [Xeon Phi](https://www.intel.com/content/www/us/en/processors/xeon/xeon-phi-detail.html), which is a sort of interesting device for heterogeneous computing.

NVIDIA, on the other hand, has **very good proprietary drivers**, and this has been true for more than a decade. Aside from Linux, these drivers also support FreeBSD and Solaris (however, CUDA, the compute stack, is Linux-only).

To put it simply, **if using a proprietary driver for graphics and computing is acceptable, NVIDIA simply does a better job with proprietary drivers than AMD**. You buy the hardware, you install the proprietary driver on Linux, and you play your games or run the computations. From a consumer's perspective, this is how hardware should work: stable and on the release day. From the perspective of an activist fighting for software freedom, this is unacceptable.

Yes, if AMD tries to compete with proprietary drivers against NVIDIA's proprietary drivers, NVIDA wins. When both companies do not care about free and open-source software, I (and probably others) will just pick the one that works better at this moment, and not think much about it.

To give a real-world example, back in 2012 we started a new course on GPU computing at [University of Rijeka](https://uniri.hr/) [Department of Informatics](https://www.inf.uniri.hr/). If AMD had the **open-source heterogeneous computing stack** ready, we would gladly pick their technology, even if hardware had slightly lower performance (you do not care for teaching anyway). However, since it came down to proprietary vs. proprietary, NVIDIA offered a more stable and mature solution and we went with them.

Even with the arguments that [NVIDIA is anti-competitive because G-Sync works only on their hardware](https://youtu.be/OnbMjhB8xQk), that AMD's hardware is not so bad and you can still play games on it, and that if AMD crashes NVIDIA will have a monopoly, I could not care less. It is completely useless to buy AMD's hardware just so that they don't crash as a company; **AMD is not a charity** and I require value in return when I give money to them.

To summarize, AMD with (usually more buggy and less stable) proprietary drivers just did not have an attractive value proposition.

## GPUOpen changing the game

However, AMD having the open-source driver as their main one gives a reason to ignore their slight disadvantage in terms of the performance per watt and the performance per dollar. Now that AMD is developing a part of the open-source graphics ecosystem and improving it for them as well as the rest of the community, they are a very valuable graphics hardware vendor.

This change empowers the community to disagree with AMD about what should be developed first and take the lead. As a user, you can **fix the bug** that annoys you **when you decide** and do not need to wait for AMD to fix it when they care to do it. Even if you don't have sufficient knowledge to do it yourself, you can pay someone to fix it for you. And this freedom is what is very valuable with open-source drivers.

Critics might say, this is easy to promise, AMD has said many things many times. And this is true; however, the **commits by AMD developers** in the [Kernel](https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/), [LLVM](https://github.com/llvm-mirror/llvm), and [Mesa](https://cgit.freedesktop.org/mesa/mesa) repositories shows that AMD is walking the walk. Doing a quick grep for e-mail addresses that contain amd.com shows a nice and steady increase in both the number of developers and the number of commits since 2011.

Critics might also say that AMD is just getting free work from the community and giving 'nothing' in return. Well, I wish more companies sourced free work from the community in this way and gave their code as free and open-source software (the 'nothing'). Specifically, I wish NVIDIA would follow AMD's lead. Anyhow, this is precisely the way [Netscape](https://en.wikipedia.org/wiki/Netscape) started what we know today as [Firefox](https://www.mozilla.org/firefox/), and [Sun Microsystems](https://en.wikipedia.org/wiki/Sun_Microsystems) started what we know today as [LibreOffice](https://www.libreoffice.org/).

To summarize, AMD with open-source drivers as the main lineup is very attractive. Free and open-source software enthusiasts do not (nor they should) care if AMD is 'slightly less evil', 'more pro-free market', 'cares more about the teddy bear from your childhood' than NVIDIA (other types of activists might or might not care about some of these issues). For the open-source community, including Linux users, **AMD either has the open-source drivers and works to improve the open-source graphics ecosystem as a whole or they do are not doing anything that matters**. If AMD wants Linux users on their side, they **have to remain committed to developing open-source drivers**. It's that simple, it's the choice between open and irrelevant.

## Non-free Radeon firmware

Free Software Foundation calls for [reverse engineering of the Radeon firmware](https://www.fsf.org/campaigns/priority-projects/reverse-engineering). While I do believe we should aim for the free firmware **and hardware**, I have two problems with this. First, I disagree with a part of [Stallman's position](https://stallman.org/to-4chan.html) (which is mirrored by FSF):

> I wish ATI would free this microcode, or put it in ROM, so that we could endorse its products and stop preferring the products of a company that is no friend of ours.

I can not agree with the idea that the non-free firmware, when included in a ROM on the card, is somehow better than the **same** non-free firmware uploaded by the driver. The reasoning behind this argument makes exactly zero sense to me. Finally, the same reasoning has been applied elsewhere: in 2011 [LWN covered the story of GTA04](https://lwn.net/Articles/460654/), which used the 'include firmware in hardware' trick to be compliant with FSF's goals.

Second, AMD, for whatever reason, does not want to release firmware as free and open-source, but their firmware is freely redistributable. (They have the freedom not to open it and even disagree with us that they should, of course.) While obviously not ideal, for me this is a reasonable compromise that works in practice. I can install the latest [Fedora](https://getfedora.org/) or [Debian](https://www.debian.org/) and a small firmware blob is packaged with the distro, despite being non-free. It doesn't depend on the kernel version, it doesn't even depend on whether I run Linux or FreeBSD kernel.

To summarize, I would like to see AMD release free firmware as much as anyone supporting FSF's ideas. And I do not hide that from AMD nor do I think anyone else should. However, I do not consider this issue of non-free firmware to be anywhere as important as having a supported free and open-source driver, which they finally have. Since **NVIDIA is no better regarding free firmware**, I do not believe that right now we have the leverage required to convince AMD to change their position.

## AMD and the open-source community

Just like Netscape and Sun Microsystems before them, AMD right now needs the community as much as the community needs them. I sincerely hope AMD is aware of this, and I know that the community is. Together, **we have the chance of the decade** to free another part of the **industry that has been locked down with proprietary software** dominating it for so long. Together, we have the chance to start a new chapter in graphics and computing history.
