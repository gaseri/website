---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2024-06-14
tags:
  - freebsd
  - linux
  - red hat
  - free and open-source software
---

# How to run Visual Studio (VS) Code Remote over SSH on FreeBSD 13 and 14

---

![white car parked in front of store during night time](https://unsplash.com/photos/HA7e2KX3tqg/download?w=1920)

Photo source: [Lemon Pepper Pictures (@lemonpepperpictures) | Unsplash](https://unsplash.com/photos/white-car-parked-in-front-of-store-during-night-time-HA7e2KX3tqg)

---

[FreeBSD](https://www.freebsd.org/) [Ports](https://www.freebsd.org/ports/) provide [editors/vscode](https://ports.freebsd.org/cgi/ports.cgi?query=vscode&stype=name&sektion=editors) with the latest stable version of [Visual Studio Code](https://code.visualstudio.com/) and the FreeBSD Foundation provides [an excellent guide](https://freebsdfoundation.org/resource/how-to-use-vs-code-on-freebsd/) how to install and use it. Unfortunately, the latest stable version of [Visual Studio Code Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) still [does not officially support FreeBSD](https://github.com/microsoft/vscode-remote-release/issues/727), but [only Linux, Windows, and macOS](https://code.visualstudio.com/docs/remote/ssh#_system-requirements).

<!-- more -->

Until recently, it was possible to use [Linuxulator](https://wiki.freebsd.org/Linuxulator) and CentOS 7 ([emulators/linux_base-c7](https://ports.freebsd.org/cgi/ports.cgi?query=linux_base-c7&stype=name&sektion=emulators) in Ports) to [run](https://gist.github.com/mateuszkwiatkowski/ce486d692b4cb18afc2c8c68dcfe8602) it [successfully](https://www.gaelanlloyd.com/blog/how-to-connect-visual-studio-code-to-freebsd-servers/), but this is not possible anymore since early February, when [version 1.86](https://code.visualstudio.com/updates/v1_86#_linux-minimum-requirements-update) updated minimum glibc version requirement to 2.28 and glibcxx to 3.42.5, which implies CentOS 8 or newer.

Recent [effort by Gleb Popov and Dima Panov](https://cgit.freebsd.org/ports/commit/?id=5aa75e1ca0fca26372479bd36773428e2c24f1e4) (sponsored by Serenity Cybersecurity, LLC) brought [Rocky Linux](https://rockylinux.org/) [9](https://rockylinux.org/news/rocky-linux-9-0-ga-release) to Linuxulator. It is now again possible (and quite simple) to run Visual Studio Code Remote - SSH on FreeBSD's currently [supported releases](https://www.freebsd.org/security/#sup) [13.2-RELEASE](https://www.freebsd.org/releases/13.2R/), [13.3-RELEASE](https://www.freebsd.org/releases/13.3R/), [14.0-RELEASE](https://www.freebsd.org/releases/14.0R/), and [14.1-RELEASE](https://www.freebsd.org/releases/14.1R/).

Let's start with enabling Linuxulator and installing the required packages:

``` tcsh
doas sysrc linux_enable="YES"
doas service linux start
doas pkg install linux-rl9-libsigsegv
```

Note that installing [devel/linux-rl9-libsigsegv](https://www.freshports.org/devel/linux-rl9-libsigsegv/) will automatically pull [emulators/linux_base-rl9](https://www.freshports.org/emulators/linux_base-rl9/) as a dependancy. After installing it, we should make sure that Linuxulator works correctly (using `uname`):

``` tcsh
/compat/linux/usr/bin/uname -a
```

``` tcshcon
Linux gaser 5.15.0 FreeBSD 14.1-RELEASE releng/14.1-n267679-10e31f0946d8 GENERIC x86_64 x86_64 x86_64 GNU/Linux
```

We should also check that `bash` got installed successfully:

``` tcsh
/compat/linux/usr/bin/bash --version
```

``` tcshcon
GNU bash, version 5.1.8(1)-release (x86_64-redhat-linux-gnu)
Copyright (C) 2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>

This is free software; you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
```

On the client side, you should adjust OpenSSH's config file to use Linuxulator's `bash`, for example:

``` shell
cat ~/.ssh/config
```

``` apacheconf
Host example.group.miletic.net
    RemoteCommand /compat/linux/usr/bin/bash
```

And that's it! Upon the first connection, the VS Code Remote server will be installed. Afterward, you can add your favorite extensions, as you would do it on Linux, Windows, or macOS.
