---
author: Vedran Miletić
authors:
 - vedranmiletic
date: 2026-01-31
tags:
 - free and open-source software
 - freebsd
 - linux
 - openssh
 - wireguard
---

# OpenSSH connections with post-quantum key exchange through WireGuard tunnel

![a close up of a piece of paper with writing on it](https://unsplash.com/photos/dvMJR9-Drbs/download?w=1920)

Photo source: [Bozhin Karaivanov (@murrayc) | Unsplash](https://unsplash.com/photos/a-close-up-of-a-piece-of-paper-with-writing-on-it-dvMJR9-Drbs)

The other day, I set up a machine using a fairly standard [Fedora](https://fedoraproject.org/) [43](https://fedoramagazine.org/announcing-fedora-linux-43/) [Server](https://fedoraproject.org/server/) installation with a [WireGuard](https://www.wireguard.com/) VPN tunnel to another machine running a fairly standard [Arch Linux](https://archlinux.org/) installation ([of course](https://knowyourmeme.com/memes/btw-i-use-arch)).

When I tried to SSH from the Arch machine to the Fedora machine, the [OpenSSH](https://www.openssh.org/) client would hang.

<!-- more -->

## Verbose post-quantum OpenSSH key exchange

It was weird, but I thought, well, we used to teach [that stuff](../../../hr/nastava/materijali/openssh-sigurna-ljuska.md) [several years ago](2023-08-28-my-perspective-after-two-years-as-a-research-and-teaching-assistant-at-fidit.md) as part of [the Security of Information and Communication Systems course](../../../hr/nastava/kolegiji/SIKS.md), it should be possible to figure out what is going on.

I started debugging the issue by enabling verbose mode:

``` shell
ssh -v my-fedora-machine
```

``` shell-session
(...)
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: mlkem768x25519-sha256
debug1: kex: host key algorithm: ssh-ed25519
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
```

Key exchange hangs waiting for the reply. Could it be some unreadable file on the server? I went down the list of the usual suspects and disabled SELinux (hello, `setenforce 0`, my old friend) before even looking at the audit log. (Sorry, [Major Hayden](https://major.io/), I know [I shouldn't have done that](https://stopdisablingselinux.com/).) As you might guess, given that the length of this blog post goes beyond this paragraph, disabling SELinux did not help.

## Legacy pre-quantum key exchange comes to the rescue

Despite [the OpenSSH major version](https://www.openssh.org/releasenotes.html) being 10 on [both](https://packages.fedoraproject.org/pkgs/openssh/openssh/) [sides](https://archlinux.org/packages/core/x86_64/openssh/), I wondered if there could be a case of some subtle cipher incompatibility. After all, [post-quantum](https://www.openssh.org/pq.html) [ML-KEM algorithm](https://en.wikipedia.org/wiki/Kyber) (`mlkem768x25519-sha256`) is somewhat new. Just in case, I tried using a non-post-quantum cipher instead:

``` shell
ssh -v -oKexAlgorithms=curve25519-sha256 my-fedora-machine
```

``` shell-session
(...)
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ssh-ed25519
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: SSH2_MSG_KEX_ECDH_REPLY received
debug1: Server host key: ssh-ed25519 SHA256:0ru7bD+izhNW+qTNFkxqHtDoiyDRNLUHHvvuF0O0I84
(...)
```

That works, hm, very interesting!

I would have almost left it at that and tried it again after OpenSSH version updates (or whenever I remembered). However, some hours later, I randomly noticed the SSH session hanging on large command outputs, such as running `dmesg`. Wait, I know how to solve that, but could it be that post-quantum OpenSSH key exchange also fails for the same underlying reason?

## WireGuard interface MTU setting

Of course, I forgot to [reduce](https://schroederdennis.de/vpn/wireguard-mtu-size-1420-1412-best-practices-ipv4-ipv6-mtu-berechnen/) [the maximum transmission unit](https://en.wikipedia.org/wiki/Maximum_transmission_unit), that is, add the `MTU = 1280` setting in [the WireGuard interface config](https://www.wireguard.com/quickstart/). (Whether the MTU value of 1280 is optimal is [debatable at best](https://gist.github.com/nitred/f16850ca48c48c79bf422e90ee5b9d95), but it has worked for me thus far.) Because of the missing setting, the large packets aren't properly fragmented and cannot pass through.

I edited the config to have:

``` ini
[Interface]
MTU = 1280
```

After restarting the `wg-quick@.service`, the setting is applied:

``` shell
ip link
```

``` shell-session
(...)
3: wgbtw: <POINTOPOINT,NOARP,UP,LOWER_UP> mtu 1280 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/none
(...)
```

Does the original post-quantum key exchange algorithm also work after reducing the interface MTU? Sure enough, it does. OpenSSH with post-quantum cryptography now goes past the `expecting SSH2_MSG_KEX_ECDH_REPLY` message:

``` shell
ssh -v -oKexAlgorithms=mlkem768x25519-sha256 my-fedora-machine
```

``` shell-session
(...)
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: mlkem768x25519-sha256
debug1: kex: host key algorithm: ssh-ed25519
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: SSH2_MSG_KEX_ECDH_REPLY received
debug1: Server host key: ssh-ed25519 SHA256:0ru7bD+izhNW+qTNFkxqHtDoiyDRNLUHHvvuF0O0I84
(...)
```

Reducing the MTU is a good practice in general, but, as you can see, it is possible to forget it and still have a VPN tunnel that works reasonably well with pre-quantum key exchange and small amounts of text in command outputs. (In fact, the [recommendation to do so](https://wiki.archlinux.org/title/WireGuard#Adjusting_the_MTU_value) appears in [the ArchWiki page about WireGuard](https://wiki.archlinux.org/title/WireGuard), where pages lean toward being concise rather than exhaustive.) To put it plainly, the issue in this particular case is that the missing WireGuard interface MTU setting is hardly the first suspect when OpenSSH key exchange fails to complete.

## Related issues elsewhere

Is this behavior of post-quantum cryptographic algorithms well known, both in OpenSSH and [elsewhere](https://openssl.foundation/news/the-features-of-3-5-post-quantum-cryptography)?

Well, yeah, sort of: both [ControlPlane](https://control-plane.io/posts/the-quantum-leap-navigating-pqc-adoption-in-todays-digital-infrastructure/) and [Sophos](https://support.sophos.com/support/s/article/KBA-000009276) wrote about a large `ClientHello` message in ML-KEM having issues with firewalls and requiring a reduced MTU setting. The former also mentions OpenSSH explicitly.

As explained above, the issue is not limited to firewalls, but also relevant for WireGuard (and possibly other VPNs), in which case the interface MTU setting isn't the obvious suspect that comes to mind. Finally, this behavior of OpenSSH through WireGuard is unlikely to be limited to Linux too; [the FreeBSD 15.0-RELEASE announcement](https://www.freebsd.org/releases/15.0R/announce/) mentions:

> OpenSSH has been upgraded to 10.0p2 which includes support for quantum-resistant key agreement by default.

It should be reasonable to expect the same behavior on [FreeBSD 15.0-RELEASE](https://www.freebsd.org/releases/15.0R/relnotes/) and newer as well.
