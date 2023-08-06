---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2023-06-23
tags:
  - academia
  - free and open-source software
  - scientific software
  - mozilla
  - web server
  - amd
---

# What hardware, software, and cloud services do we use?

Our everyday [scientific](../../projects.md) and [educational](../../teaching/index.md) work relies heavily on hardware, software, and, in modern times, cloud services. The equipment that we will mention below is specific to our group; common services used by university and/or faculty employees will not be specifically mentioned here.

<!-- more -->

## Laptops

- [Lenovo V15 G2-ALC](https://pcsupport.lenovo.com/us/en/products/laptops-and-netbooks/lenovo-v-series-laptops/v15-g2-alc) running [Garuda Linux](https://garudalinux.org/)
- [HP 255 G7](https://support.hp.com/us-en/product/hp-255-g7-notebook-pc/24381324) running [Manjaro](https://manjaro.org/)

### Userland software

We use [Mozilla](https://www.mozilla.org/) [Firefox](https://www.mozilla.org/firefox/), [FireDragon](https://forum.garudalinux.org/t/firedragon-librewolf-fork/5018), and [Brave](https://brave.com/) for web browing and development.

We use [Visual Studio Code](https://code.visualstudio.com/) for writing and editing [Markdown](https://code.visualstudio.com/docs/languages/markdown) (to be processed by [MkDocs](https://www.mkdocs.org/) or [Pandoc](https://pandoc.org/)) and [LaTeX](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop), [C++ development](https://code.visualstudio.com/docs/languages/cpp), and [Python development](https://code.visualstudio.com/docs/languages/python).

## Servers

- [SuperMicro A+ Server 1013S-MTR](https://www.supermicro.com/en/Aplus/system/1U/1013/AS-1013S-MTR.cfm) running [Proxmox VE](https://www.proxmox.com/proxmox-ve) with several virtual machines, used for compute
- Custom server built with [ASRock FM2A88M-HD+](https://www.asrock.com/mb/AMD/FM2A88M-HD+/index.asp) and eight [Seagate IronWolf 8TB](https://www.seagate.com/products/nas-drives/ironwolf-hard-drive/) running [TrueNAS CORE](https://www.truenas.com/truenas-core/), used for storage

### Future changes

We plan to switch the storage server to [TrueNAS SCALE](https://www.truenas.com/truenas-scale/) in the near future. We will consider it for the compute server as well to reduce the number of different appliance solutions we use.

## Cloud services

- [Hetzner Cloud x86 VPS](https://www.hetzner.com/cloud) running [Arch Linux](https://archlinux.org/) with [Apache HTTP Server](https://httpd.apache.org/), [OpenSSL](https://www.openssl.org/), [PHP](https://www.php.net/), and [Tor](https://www.torproject.org/), used for hosting [miletic.net](https://www.miletic.net/) (web applications and services)
- [GitHub Actions](https://github.com/features/actions), used for building, and [GitHub Pages](https://pages.github.com/), used for hosting [gaseri.org](../../../index.md) (web site)
- [HackMD](https://hackmd.io/), used for collaborative drafting of Markdown documents
- [Overleaf](https://www.overleaf.com/), used for collaborative drafting of LaTeX documents
