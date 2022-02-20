---
author: Vedran Miletić
date: 2016-01-30
tags:
  - microsoft
  - opendocument
  - free and open-source software
  - secure boot
  - web standards
---

# I am still not buying the new-open-source-friendly-Microsoft narrative

This week Microsoft [released Computational Network Toolkit (CNTK) on GitHub](https://blogs.microsoft.com/ai/microsoft-releases-cntk-its-open-source-deep-learning-toolkit-on-github/), after [open sourcing Edge's JavaScript engine last month](https://blogs.windows.com/msedgedev/2015/12/05/open-source-chakra-core/) and a whole bunch of projects before that.

Even though the open sourcing of a bunch of their software is a very nice move from Microsoft, I am still not convinced that they have changed to the core. I am sure there are parts of the company who believe that free and open source is the way to go, but it still looks like a change just on the periphery.

All the projects they have open-sourced so far are not the core of their business. Their latest version of Windows is no more friendly to alternative operating systems than any version of Windows before it, and one could argue it is even *less* friendly due to more Secure Boot restrictions. Using Office still basically requires you to use Microsoft's formats and, in turn, accept their vendor lock-in.

Put simply, I think all the projects Microsoft has opened up so far are a nice start, but they still have a long way to go to gain respect from the open-source community. What follows are three steps Microsoft could take in that direction.

## Step 1: fully support OpenDocument and make it the default format in Office applications

Microsoft has accepted the web standards defined by [W3C](https://www.w3.org/). Making [OpenDocument](https://opendocumentformat.org/) the default format in Office would be the equivalent of accepting independently standardized HTML and CSS. Even after accepting the format, Microsoft could still compete with free and open-source office suites. They could offer more features, a more beautiful user interface, better performance, or better quality support. They would, however, lose the vendor lock-in ability.

## Step 2: Open source large parts of Windows and the tools required to build custom versions

Apple has been [open sourcing large parts of OS X](https://opensource.apple.com/) (but not all of it, one should say) since version 10.0. With significant effort, it is possible to build something like [PureDarwin](https://www.puredarwin.org/), an open-source operating system based on the source released by Apple. Note that, for example, PureDarwin does not use OS X GUI, since Apple has not open-sourced it.

Microsoft could do the same with Windows as Apple did with OS X: open source large parts of the code, and allow people to combine it with other software to build custom versions. Even if some parts of the code remain proprietary, it is still a big improvement over what Microsoft is doing now.

## Step 3: Spin-off the department for Secure Boot bootloader signing into an independent non-profit entity

Since 2012, machines with UEFI Secure Boot have started to appear on the market. To get your laptop or desktop PC certified for Windows 8, a manufacturer had to support Secure Boot, including Microsoft keys, turn Secure Boot on by default, **and allow the user to turn it off**. Microsoft agreed to [sign binaries](https://blog.hansenpartnership.com/adventures-in-microsoft-uefi-signing/) for vendors of other operating systems, and vendors like [Fedora](https://arstechnica.com/information-technology/2012/06/fedora-could-seek-microsoft-code-signing-to-contend-with-secure-boot/) and [Canonical](https://www.pcworld.com/article/456863/two-ubuntu-linux-versions-can-now-work-with-secure-boot.html) got the signatures.

With Windows 10, [the requirement to allow the user to turn Secure Boot off vanished](https://arstechnica.com/information-technology/2015/03/windows-10-to-make-the-secure-boot-alt-os-lock-out-a-reality/), which prevents booting of unsigned operating systems. Furthermore, Microsoft can at any time revoke the key used for signing operating systems other than Windows and render all of them unbootable. Finally, since the key used to sign other operating systems is a separate key from the one used to sign Windows, the revoking would not affect Windows in any way.

The situation gives Microsoft an enormous amount of power and control over desktops and laptops. It would be much better if the signing process and management of keys were done by an independent non-profit entity, governed by a consortium of companies.

## Summary

I am sure there are people, even among those who work for Microsoft right now, who would agree with these ideas. However, the support for these ideas itself does not matter much unless and until Microsoft starts taking action in that direction.

And, unless and until that happens, I am not buying the new-open-source-friendly-Microsoft narrative.
