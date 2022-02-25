---
author: Vedran Miletić
date: 2022-02-24
tags:
  - archiving and compression
  - free and open-source software
  - software licensing
  - web standards
---

# Don't use RAR

I sometimes joke with my TA [Milan Petrović](https://milanxpetrovic.github.io/) that his usage of RAR does not imply that he will be driving a [rari](https://www.urbandictionary.com/define.php?term=rari). After all, he is not [Devito rapping^Wsinging Uh 😤](https://youtu.be/_uOoV0mtX3E). Jokes aside, if you search for "should I use RAR" or a similar phrase on your favorite search engine, you'll see articles like 2007 [Don't Use ZIP, Use RAR](https://blog.codinghorror.com/dont-use-zip-use-rar/) and 2011 [Why RAR Is Better Than ZIP & The Best RAR Software Available](https://www.makeuseof.com/tag/rar-zip-rar-software/).

So, why shouldn't we use RAR? The non-free license and Windows-centric aspects have already been addressed by [Kyle Cordes](https://kylecordes.com/) in the 2007 blog post [Why I do not use RAR](https://kylecordes.com/2007/no-rar). These reasons are still relevant; the [official versions](https://www.rarlab.com/rar_add.htm) of [rar](https://tracker.debian.org/pkg/rar) and [unrar](https://tracker.debian.org/pkg/unrar-nonfree) are still non-free. There is an [unofficial unrar](https://gitlab.com/bgermann/unrar-free) that is free and open-source software, but there is [no free and open-source rar](https://peazip.github.io/free-rar-create.html) as creating one is prohibited by the [RAR license](https://www.win-rar.com/winrarlicense.html).

Let's address what else is new in the last 15 years and why the arguments for using RAR over open archiving and compression formats are long obsolete.

First off, [7-Zip](https://www.7-zip.org/) achieves a better compression ratio, but it is much slower to compress than RAR. However, since 2013 [Google's Brotli](https://www.brotli.org/) and since 2015 [Facebook's Zstandard (Zstd)](https://facebook.github.io/zstd/) are two good options for file compression. Aside from the file compression, [Brotli is well-supported in HTTP compression](https://caniuse.com/brotli) and [Zstd is used in OpenZFS](https://papers.freebsd.org/2018/bsdcan/jude-zfs_zstd/). They are [quite competitive against RAR in terms of speed and size](https://peazip.github.io/fast-compression-benchmark-brotli-zstandard.html) too: where RAR compresses a file to 318 MB in 100 seconds, Brotli gets 322 MB in 52.3 s, while Zstd gets 321 MB in 63 s (when 128 MB window size is used).

Furthermore, the small differences in the resulting file size matter less over time as bandwidth is [increasing and increasing fast](https://www.speedtest.net/global-index).

Finally, for those who dislike CLI, [PeaZip](https://peazip.github.io/) is a good-looking [cross-platform](https://peazip.github.io/peazip-linux.html) [GUI-based](https://peazip.github.io/peazip-macos.html) [archiver](https://peazip.github.io/peazip-64bit.html) [licensed under GNU LGPLv3](https://peazip.github.io/peazip-sources.html). Think 7-Zip, but with nicer icons and support for Brotli and Zstd.

All things considered, there are no reasons left to use RAR.
