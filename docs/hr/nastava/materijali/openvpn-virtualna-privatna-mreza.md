---
author: Manuel Maraš, Vedran Miletić
---

# Konfiguracija virtualne privatne mreže alatom OpenVPN

[Virtualna privatna mreža](https://en.wikipedia.org/wiki/Virtual_private_network) (engl. *virtual private netvork*, kraće VPN) pruža usluge privatne mreže korištenjem neke javne mreže kao što je internet. Pomoću VPN-a možemo slati i primati podatke preko javne mreže, a istovremeno koristiti konfiguraciju privatne lokalne mreže. VPN koristi različite sigurnosne mehanizme koji osiguravaju tajnost i autentičnost podaka koje šaljemo. Virtualnu privatnu mrežu moguće je stvoriti brojnim protokolima, npr. [L2TP](https://en.wikipedia.org/wiki/Layer_2_Tunneling_Protocol) na veznom sloju (koji implementira [xl2tpd](https://github.com/xelerance/xl2tpd)), [IPsec](https://en.wikipedia.org/wiki/IPsec) na mrežnom sloju (koji implementira [strongSwan](https://www.strongswan.org/)) i [SSTP](https://en.wikipedia.org/wiki/Secure_Socket_Tunneling_Protocol) na transportnom sloju (koji implementira [SoftEther VPN](https://www.softether.org/)).

[OpenVPN](https://openvpn.net/community/) je vrlo popularan VPN poslužitelj i klijent otvorenog koda korištenjem kojeg je moguće izgraditi virtualnu privatnu mrežu. Koristi [vlastiti protokol](https://openvpn.net/community-resources/openvpn-protocol/) koji nije standardiziran od strane IETF-a (specijalno, informativni [RFC 2764: A Framework for IP Based Virtual Private Networks](https://datatracker.ietf.org/doc/html/rfc2764) samo općenito opisuje način rada virtualnih privatnih mreža), ali je otvoren i može ga bilo tko implementirati (npr. postoji [roburova](https://robur.io/Our%20Work/Projects) [implementacija u programskom jeziku OCaml](https://github.com/roburio/openvpn)).

OpenVPN podržava Linux, FreeBSD, macOS, Windows i [druge operacijske sustave](https://openvpn.net/faq/openvpn-compatibility/); u praksi to znači da se, primjerice, na OpenVPN poslužitelj koji se pokreće na Linuxu mogu povezati OpenVPN klijenti koji se pokreću na macOS-u i Windowsima. Osim sučelja naredbenog retka koje je dostupno svugdje i koje koristimo u nastavku, na Windowsima ima [vlastito grafičko korisničko sučelje](https://community.openvpn.net/openvpn/wiki/OpenVPN-GUI), na Linuxu je [podržan u okviru NetworkManagera](https://wiki.gnome.org/Projects/NetworkManager/VPN), a na macOS-u postoji [Tunnelblick](https://tunnelblick.net/) ([tunel](https://en.wikipedia.org/wiki/Tunneling_protocol) (engl. *tunnel*) je naziv za put koji VPN uspostavlja između računala kroz internet).

OpenVPN je podržan od strane brojnih komercijalnih pružatelja usluge VPN-a poput [Private Internet Accessa](https://www.privateinternetaccess.com/vpn-features/open-source-vpn) i [NordVPN-a](https://nordvpn.com/ovpn/) koji korisnicima omogućuju skrivanje vlastite IP adrese i time anonimnost na internetu. Također ga koriste brojni proizvođači hardvera kao što su [Teltonika Networks](https://wiki.teltonika-networks.com/view/OpenVPN_configuration_examples), [Linksys](https://www.linksys.com/us/support-article?articleNum=156241), [TP-Link](https://www.tp-link.com/en/support/faq/1544/), [Cisco](https://www.cisco.com/c/en/us/support/docs/smb/routers/cisco-rv-series-small-business-routers/smb5879-openvpn-on-rv160-rv260.html) i [Turris](https://doc.turris.cz/doc/en/howto/theoretical_article_on_openvpn) OpenVPN koriste brojne korporacije, sveučilišta, instituti i druge organizacije za organizaciju udaljenog rada na resursima unutar organizacije i, ako postoji potreba, za međusobno povezivanje više različitih fizičkih lokacija na kojima organizacija ima svoje podružnice.

Osim OpenVPN-a, nešto novije rješenje koje vrijedi spomenuti je [WireGuard](https://www.wireguard.com/), koji razvija kompanija [Edge Security](https://www.edgesecurity.com/) od 2015. godine. WireGuard je [2020. godine prošao sigurnosni audit](https://lore.kernel.org/netdev/20200319003047.113501-1-Jason@zx2c4.com/) i postao [dio jezgre Linuxa od verzije 5.6](https://arstechnica.com/gadgets/2020/03/wireguard-vpn-makes-it-to-1-0-0-and-into-the-next-linux-kernel/) pa se očekuje da će u budućnosti biti ozbiljna konkurencija OpenVPN-u.

Nakon povezivanja OpenVPN klijenta s OpenVPN poslužiteljem (koje opisujemo u nastavku), stvara se tunel kroz internet kroz koji se onda šalju podaci. Podaci koji prolaze tunelom su šifirirani kako ih treće strane na internetu ne bi mogle čitati, a dodatno se mogu koristiti i razne vrste kompresije kako bi podataka bilo manje. Zbog toga korištenje OpenVPN-a na brzim komunikacijskim mrežama zahtijeva značajnu količinu procesorskog vremena i s time treba računati kod planiranja hardverske konfiguracije OpenVPN poslužitelja.

## Statički ključ

!!! note
    Ovaj dio je složen prema članku [Static Key Mini-HOWTO](https://openvpn.net/community-resources/static-key-mini-howto/) koji je dio [službene dokumentacije OpenVPN-a namijenjene za zajednicu](https://openvpn.net/community-resources/).

OpenVPN podržava dva načina rada: statički ključ kod kojeg klijent i poslužitelj koriste dijeljeni statički tajni ključ te sigurnosni mehanizam javnog ključa kod kojeg poslužitelj ima vlastiti autoritet certifikata koji koristi za provjeru valjanosti certifikata klijenata koji se žele povezati. Prvi način rada je jednostavniji, ali podržava samo rad točka-do-točke s jednim poslužiteljem i jednim klijentom. Drugi način rada je složeniji, ali podržava povezivanje više klijenata na jedan poslužitelj (i međusobno, ako poslužitelj tako konfigurira). Statičkim ključem se bavimo sada, a sigurnosnim mehanizmom javnog ključa u idućem dijelu.

Uvjerimo se prvo da imamo instaliran OpenVPN i saznajmo o kojoj se verziji radi pokretanjem naredbe `openvpn` s parametrom `--version`:

``` shell
$ openvpn --version
OpenVPN 2.4.9 [git:makepkg/9b0dafca6c50b8bb+] x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Apr 20 2020
library versions: OpenSSL 1.1.1g  21 Apr 2020, LZO 2.10
Originally developed by James Yonan
Copyright (C) 2002-2018 OpenVPN Inc <sales@openvpn.net>
Compile time defines: enable_async_push=no enable_comp_stub=no enable_crypto=yes enable_crypto_ofb_cfb=yes enable_debug=yes enable_def_auth=yes enable_dlopen=unknown enable_dlopen_self=unknown enable_dlopen_self_static=unknown enable_fast_install=yes enable_fragment=yes enable_iproute2=yes enable_libtool_lock=yes enable_lz4=yes enable_lzo=yes enable_management=yes enable_multihome=yes enable_pam_dlopen=no enable_pedantic=no enable_pf=yes enable_pkcs11=yes enable_plugin_auth_pam=yes enable_plugin_down_root=yes enable_plugins=yes enable_port_share=yes enable_selinux=no enable_server=yes enable_shared=yes enable_shared_with_static_runtimes=no enable_small=no enable_static=yes enable_strict=no enable_strict_options=no enable_systemd=yes enable_werror=no enable_win32_dll=yes enable_x509_alt_username=yes with_aix_soname=aix with_crypto_library=openssl with_gnu_ld=yes with_mem_check=no with_sysroot=no
```

Naredba ima brojne parametre čiji se popis može dobiti korištenjem parametra `--help`:

``` shell
$ openvpn --help
OpenVPN 2.4.9 [git:makepkg/9b0dafca6c50b8bb+] x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Apr 20 2020

General Options:
--config file   : Read configuration options from file.
--help          : Show options.
--version       : Show copyright and version information.

Tunnel Options:
--local host    : Local host name or ip address. Implies --bind.
--remote host [port] : Remote host name or ip address.
(...)
```

Detaljniji opis svakog od pojedinih parametara dan je u man stranici `openvpn(8)` (naredba `man 8 openvpn`). Za početak generirajmo statički ključ:

``` shell
$ openvpn --genkey --secret static.key
```

Uvjerimo se da je ključ ispravno generiran:

``` shell
$ cat static.key
# 2048 bit OpenVPN static key
#
-----BEGIN OpenVPN Static key V1-----
8a806dab5e61999f2c1b5deae9f3ed31
73c3b8f4a611ddcc2d097947d42652c3
43124fbae8ee35fc121d9cd4b5ba6b8c
a5a173c4db0c74657c74da37c503cb74
d5133b8cf6b5d4cce41009c5754df2b7
bf45455dcd3a1dd6fb48a322f2a32b93
ba3f04f0fb0a410d472550fa9f47e0f1
5213a8f289c78f7dd01a0652c49926cd
c95ee36faa8912147c620e1e864633f8
57e30b1dcb85199a81751bb2b44cb5a8
7a3cc17ccf283989129dcb4e9f55307f
785cb103e64c74d0e4f1bec469c16cc0
934f6022ee7d75779f5f93bd73e031f8
7ffffa5836b0a003cd4054e2734353b9
ed89fa055a72dc20e9df748844030008
91da5920e1e572fd05f75c2fac49d12e
-----END OpenVPN Static key V1-----
```

Ovaj će ključ biti korišten na klijentskoj i na poslužiteljskoj strani za generiranje dva ključa: ključ za računanje koda autentifikacije poruke korištenjem hashiranja i ključ za šifriranje i dešifriranje. Konfigurirajmo poslužitelj, stvorimo datoteku imena `server.ovpn` sadržaja:

```
dev tun
ifconfig 172.20.0.1 172.20.0.2
secret static.key
```

[TUN i TAP](https://en.wikipedia.org/wiki/TUN/TAP) su virtualna mrežna sučelja koja postoje na različitim operacijskim sustavima sličnim Unixu. Kako su u cijelosti implementirana softverski, njihov upravljački program nije vezan niti za jedan hardverski mrežni adapter, a danas se održava [u sklopu Linuxa](https://www.kernel.org/doc/html/latest/networking/tuntap.html) i FreeBSD-a ([tun](https://www.freebsd.org/cgi/man.cgi?query=tun&sektion=4), [tap](https://www.freebsd.org/cgi/man.cgi?query=tap&sektion=4)). TUN (skraćeno od *network TUNnel*) emulira uređaj mrežnog sloja i prenosi IP pakete. TAP (skraćeno od *Terminal Access Point*) emulira uređaj veznog sloja i prenosi Ethernet okvire. U našem slučaju će OpenVPN raditi na mrežnom sloju (`dev tun`).

Zatim konfiguriramo IP adresu (`ifconfig`) poslužitelja (`172.20.0.1`) i klijenta koji će se tek povezati (`172.20.0.2`).

Naposlijetku navodimo ime datoteke u kojoj se nalazi tajni ključ koji će se koristiti (`secret static.key`). Pokrenimo poslužitelj:

``` shell
$ sudo openvpn --config server.ovpn
Tue May 19 19:51:32 2020 disabling NCP mode (--ncp-disable) because not in P2MP client or server mode
Tue May 19 19:51:32 2020 OpenVPN 2.4.9 [git:makepkg/9b0dafca6c50b8bb+] x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Apr 20 2020
Tue May 19 19:51:32 2020 library versions: OpenSSL 1.1.1g  21 Apr 2020, LZO 2.10
Tue May 19 19:51:32 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:51:32 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:51:32 2020 TUN/TAP device tun0 opened
Tue May 19 19:51:32 2020 /usr/bin/ip link set dev tun0 up mtu 1500
Tue May 19 19:51:32 2020 /usr/bin/ip addr add dev tun0 local 172.20.0.1 peer 172.20.0.2
Tue May 19 19:51:32 2020 Could not determine IPv4/IPv6 protocol. Using AF_INET
Tue May 19 19:51:32 2020 UDPv4 link local (bound): [AF_INET][undef]:1194
Tue May 19 19:51:32 2020 UDPv4 link remote: [AF_UNSPEC]
```

Primijetimo upozorenje u vezi korištenja nesigurnog šifrarnika `WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.` i prijedlog rješenja `Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).` koji ćemo iskoristiti kad se budemo bavili dodatnim opcijama.

Provjerimo popis mrežnih sučelja na računalu naredbom `ip address show` ili `ifconfig` pa uočimo da je stvoreno jedno novo sučelje `tun0`:

``` shell
$ ip address show
(...)
10: tun0: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 100
    link/none
    inet 172.20.0.1 peer 172.20.0.2/32 scope global tun0
       valid_lft forever preferred_lft forever
    inet6 fe80::6e2c:b3c5:f1ca:1cf5/64 scope link stable-privacy
       valid_lft forever preferred_lft forever
(...)
```

Poslužitelj će slušati za povezivanje klijenata sve do prekida kombinacijom tipki Control + C (^C), nakon čega će ispisati:

```
Tue May 19 19:54:19 2020 event_wait : Interrupted system call (code=4)
Tue May 19 19:54:19 2020 /usr/bin/ip addr del dev tun0 local 172.20.0.1 peer 172.20.0.1
Tue May 19 19:54:19 2020 SIGINT[hard,] received, process exiting
```

Ostavimo ga pokrenutog ili ga pokrenimo opet ako smo ga zaustavili. U drugom terminalu pokrenut ćemo klijent. Konfigurirajmo ga, stvorimo datoteku imena `client.ovpn` sadržaja:

``` squid hl_lines="1 3"
remote localhost
dev tun
ifconfig 172.20.0.2 172.20.0.1
secret static.key
```

Prva linija postavlja da se radi o klijentu koji se povezuje na poslužitelj (`remote`) na danoj adresi, ovdje `localhost`. U stvarnosti neće poslužitelj i klijent raditi na istom računalu pa će umjesto `localhost` adresa biti oblika, primjerice, `vpn-service.uniri.hr` ili `secret-vpn-entrance.rm.miletic.net`. Nama na ovim vježbama nisu zanimljivi način rada usmjeravanja kroz VPN, uspješnost kompresije, mjerenje performansi u ovisnosti o opterećenju osnovnog procesora, već samo način konfiguracije sigurnosnih mehanizama te ćemo u nastavku koristiti poslužitelj i klijent na istom računalu.

Uočimo da treća linija s konfiguracijskom naredbom `ifconfig` ima obrnute adrese od iste konfiguracijske naredbe na poslužitelju, odnosno prvo je sada klijentska adresa (`172.20.0.2`), a onda poslužiteljska (`172.20.0.1`). Pokrenimo klijent:

``` shell
$ sudo openvpn --config client.ovpn
[sudo] lozinka za vedranm:
Tue May 19 19:51:49 2020 disabling NCP mode (--ncp-disable) because not in P2MP client or server mode
Tue May 19 19:51:49 2020 OpenVPN 2.4.9 [git:makepkg/9b0dafca6c50b8bb+] x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Apr 20 2020
Tue May 19 19:51:49 2020 library versions: OpenSSL 1.1.1g  21 Apr 2020, LZO 2.10
Tue May 19 19:51:49 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:51:49 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:51:49 2020 TUN/TAP device tun1 opened
Tue May 19 19:51:49 2020 /usr/bin/ip link set dev tun1 up mtu 1500
Tue May 19 19:51:49 2020 /usr/bin/ip addr add dev tun1 local 172.20.0.2 peer 172.20.0.1
Tue May 19 19:51:49 2020 TCP/UDP: Preserving recently used remote address: [AF_INET6]::1:1194
Tue May 19 19:51:49 2020 setsockopt(IPV6_V6ONLY=0)
Tue May 19 19:51:49 2020 TCP/UDP: Socket bind failed on local address [AF_INET][undef]:1194: Address already in use (errno=98)
Tue May 19 19:51:49 2020 Exiting due to fatal error
Tue May 19 19:51:49 2020 /usr/bin/ip addr del dev tun1 local 172.20.0.2 peer 172.20.0.1
```

Klijent se nije uspio pokrenuti jer su mu adresa i vrata već zauzeti od strane ranije pokrenutog poslužitelja. OpenVPN sa zadanim postavkama u tom smislu radi atipično jer klijent uz operaciju `connect()` na klijentskoj utičnici dodatno radi i operaciju `bind()` na još jednoj utičnici, odnosno ponaša se kao da je poslužitelj. Potrebno je eksplicitno reći da ne želimo da se klijent ponaša na taj način konfiguracijskom naredbom `nobind`:

``` squid hl_lines="2"
remote localhost
nobind
dev tun
ifconfig 172.20.0.2 172.20.0.1
secret static.key
```

Pokretanje klijenta je sada uspješno:

``` shell
$ sudo openvpn --config client.ovpn
Tue May 19 19:52:48 2020 disabling NCP mode (--ncp-disable) because not in P2MP client or server mode
Tue May 19 19:52:48 2020 OpenVPN 2.4.9 [git:makepkg/9b0dafca6c50b8bb+] x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Apr 20 2020
Tue May 19 19:52:48 2020 library versions: OpenSSL 1.1.1g  21 Apr 2020, LZO 2.10
Tue May 19 19:52:48 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:52:48 2020 WARNING: INSECURE cipher with block size less than 128 bit (64 bit).  This allows attacks like SWEET32.  Mitigate by using a --cipher with a larger block size (e.g. AES-256-CBC).
Tue May 19 19:52:48 2020 TUN/TAP device tun1 opened
Tue May 19 19:52:48 2020 /usr/bin/ip link set dev tun1 up mtu 1500
Tue May 19 19:52:48 2020 /usr/bin/ip addr add dev tun1 local 172.20.0.2 peer 172.20.0.1
Tue May 19 19:52:48 2020 TCP/UDP: Preserving recently used remote address: [AF_INET6]::1:1194
Tue May 19 19:52:48 2020 UDP link local: (not bound)
Tue May 19 19:52:48 2020 UDP link remote: [AF_INET6]::1:1194
```

Pokretanje klijenta stvorilo je još jedno mrežno sučelje naziva `tun1`:

``` shell
$ ip address show
(...)
11: tun1: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 100
    link/none
    inet 172.20.0.2 peer 172.20.0.1/32 scope global tun1
       valid_lft forever preferred_lft forever
    inet6 fe80::365d:7be7:d60b:369c/64 scope link stable-privacy
       valid_lft forever preferred_lft foreve
(...)
```

Klijent će raditi sve do prekida kombinacijom tipki Control + C (^C), nakon čega će ispisati:

```
Tue May 19 19:54:15 2020 event_wait : Interrupted system call (code=4)
Tue May 19 19:54:15 2020 /usr/bin/ip addr del dev tun1 local 172.20.0.2 peer 172.20.0.1
Tue May 19 19:54:15 2020 SIGINT[hard,] received, process exiting
```

Ostvarili smo vezu klijenta i poslužitelja osnovnom konfiguracijom. Moguće je stvoriti konfiguraciju kod koje se koriste dva statička ključa na klijentskoj i poslužiteljskoj strani, svaki u jednom smjeru. U tom slučaju kod generiranja navodimo vrijednost smjera (`direction`) 0 ili 1.

``` shell
$ openvpn --genkey --secret static-direction0.key 0
```

``` shell
$ openvpn --genkey --secret static-direction1.key 1
```

Tada bismo u konfiguraciji klijenta i poslužitelja imali dvije naredbe:

```
secret static-direction0.key 0
secret static-direction1.key 1
```

Sada će i generirani ključevi za računanje koda autentifikacije poruke korištenjem hashiranja biti različiti u svakom od smjerova. Pored toga, bit će korištena i dva ključa za šifriranje i dešifriranje, po jedan u svakom od smjerova.

## Konfiguracijske naredbe i parametri

Kao što smo vidjeli na početku, OpenVPN nudi brojne parametre koje možemo navesti kod pokretanja, a isti postoje i kao konfiguracijske naredbe. Osim u man stranici, ista dokumentacija koja sadrži opis parametara (i konfiguracijskih naredbi) može se naći i u dokumentu [Reference manual for OpenVPN 2.4](https://openvpn.net/community-resources/reference-manual-for-openvpn-2-4/). Brojne [upute za konfiguraciju](https://dnaeon.github.io/openvpn-freebsd/) [koje se mogu](https://www.c0ffee.net/blog/openvpn-guide/) [pronaći na internetu](https://wiki.gentoo.org/wiki/OpenVPN) koriste te još neke konfiguracijske naredbe uz nužne naredbi koje smo vidjeli iznad. Pokazat ćemo sad kako rade neke od njih:

- `cipher alg` postavlja šifrarnik koji se koristi na `alg`; zadana postavka je nesigurni `BF-CBC` zbog kojeg dobivamo upozorenje da koristimo nesigurni šifrarnik i preporuku da iskoristimo `AES-256-CBC` (u načinu rada statičkog ključa možemo koristiti samo CBC šifrarnike, inače se preporuča `AES-256-GCM`); popis svih dostupnih šifrarnika dostupan je naredbom `openvpn --show-ciphers`
- `local host` ograničava vezivanje poslužitelja samo na adresu `host` koja može biti dana kao ime domaćina ili kao IP adresa; u zadanim postavkama OpenVPN poslužitelj se veže na sve adrese
- `port number` postavlja vezivanje poslužitelja na vrata `number` umjesto na zadana vrata 1194, korisno kad na vatrozidu nemamo otvorena vrata 1194
- `proto p` postavlja protokol na UDP (`udp`) ili TCP (`tcp-client` na klijentskoj strani, `tcp-server` na poslužiteljskoj strani); zadan i preporučen je UDP
- `resolv-retry n` pokušavaj pronaći adresu na koju se klijent povezuje `n` sekundi prije odustajanja; zadana vrijendost za `n` je `infinite`
- `keepalive interval timeout` šalje ping svakih `interval` sekundi ako nema primljenih paketa i vrši ponovno spajanje (šalje signal `SIGUSR1`) ako `timeout` sekundi nema primljenih paketa
- `compress algorithm` uključuje komprimiranje, `algorithm` može biti `lzo` (starije verzije OpenVPN-a imaju opciju `comp-lzo` koja je ekvivalentna `compress lzo` u novijim verzijama) ili `lz4` (preporučen ako ga podržavaju i poslužitelj i klijenti)
- `status file` zapisuje status programa u datoteku `file`
- `persist-tun` isključuje ponovno stvaranje TUN/TAP uređaja kod restarta korištenjem signala `SIGUSR1`
- `persist-key` isključuje ponovno čitanje ključeva kod restarta korištenjem signala `SIGUSR1`
- `persist-local-ip` ostavlja prvotno postavljenu lokalnu adresu kod restarta korištenjem signala `SIGUSR1`
- `persist-remote-ip` ostavlja prvotno postavljenu udaljenu adresu kod restarta korištenjem signala `SIGUSR1`
- `verb n` postavlja rječitost izlaza programa na `n`, zadano 1; 0 -- ništa osim fatalnih grešaka; 1 do 4 -- normalna količina izlaza; 5 -- ispis čitanja i pisanja svakog paketa; 6 do 11 -- poruke za debugging
- `mute n` navedi u izlazu najviše `n` poruka iste vrste
- `user u` mijenja korisnika kojem pripada OpenVPN-ov proces nakon pokretanja u `u`, često se postavlja `nobody` kako bi se smanjila količina privilegija koje proces ima i povećala sigurnost
- `group g` mijenja grupu kojoj pripada OpenVPN-ov u `g`, često se postavlja `nogroup` analogno `nobody` kod korisnika

Detaljniji opis ovih i brojnih drugih parametara dan je u man stranici `openvpn(8)` (naredba `man 8 openvpn`).
