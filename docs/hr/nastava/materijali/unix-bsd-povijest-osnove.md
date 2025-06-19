---
author: Vedran Miletić
---

# Povijesni pregled razvoja Unixa na Kalifornijskom sveučilištu u Berkeleyu

## Berkeley Software Distribution (BSD)

- BSD Unix nastaje [krajem 1970-tih i tijekom 1980-tih godina kao derivat AT&T-evog Unixa](https://klarasystems.com/articles/history-of-freebsd-unix-and-bsd/), začetnik je [Bill Joy](https://en.wikipedia.org/wiki/Bill_Joy) sa [University of California, Berkeley](https://www.berkeley.edu/)
- razvija ga [Computer Systems Research Group (CSRG)](https://en.wikipedia.org/wiki/Computer_Systems_Research_Group): [Keith Bostic](https://en.wikipedia.org/wiki/Keith_Bostic), [Bill Joy](https://en.wikipedia.org/wiki/Bill_Joy), [Marshall Kirk McKusick](https://en.wikipedia.org/wiki/Marshall_Kirk_McKusick), [Sam Leffler](https://en.wikipedia.org/wiki/Samuel_J._Leffler), [Mike Karels](https://en.wikipedia.org/wiki/Michael_J._Karels) i drugi
- brzo postaje popularan na sveučilištima i institutima, zaslužan za popularizaciju Unixa na sveučilištima u SAD-u
- verzije 1BSD, 2BSD, 3BSD, 4BSD, 4.1BSD, 4.2BSD, 4.3BSD, ...; dobar povijesni izvor je [Twenty Years of Berkeley Unix: From AT&T-Owned to Freely Redistributable (Open Sources: Voices from the Open Source Revolution)](https://www.oreilly.com/openbook/opensources/book/kirkmck.html)
- vrlo je važan za razvoj tehnologija na kojima se zasniva [internet](https://en.wikipedia.org/wiki/Internet); ranih 1980-ih godina [DARPA](https://www.darpa.mil/) financira razvoj tehnologija interneta u okviru BSD Unixa

    - [BBN Technologies (Bolt, Beranek and Newman)](https://en.wikipedia.org/wiki/BBN_Technologies) natječe se s entuzijastima na Sveučilištu u Berkeleyu među kojima glavnu ulogu imaju Bill Joy i Mike Karels
    - [A Narrative History of BSD](https://youtu.be/ds77e3aO9nA), video u kojem Marshall Kirk McKusick priča o razvoju Unixa i interneta na Berkeleyu (u [povijesti interneta](https://en.wikipedia.org/wiki/History_of_the_Internet) to je doba gdje se [više dotad odvojenih i međusobno različitih mreža spaja u jednu](https://en.wikipedia.org/wiki/History_of_the_Internet#Merging_the_networks_and_creating_the_Internet_(1973%E2%80%9395)) i nastaje međumrežje, odnosno internet)
    - kako je BSD već bio raširen po sveučilištima diljem SAD-a, tako su se kasnijim verzijama proširile tehnologije koje omogućuju rad interneta razvijene na Berkeleyu i drugdje

## FreeBSD

- 1989\. godine Keith Bostic predlaže zamjenu čitavog preostalog AT&T-evog koda otvorenim kodom i do 1991. ostaje šest datoteka za zamjenu
- 1992\. [William](https://en.wikipedia.org/wiki/William_Jolitz) i [Lynne Jolitz](https://en.wikipedia.org/wiki/Lynne_Jolitz) pišu zamjene za šest datoteka, a na temelju njihovog rada od 1993. nadalje razvija se [FreeBSD](https://en.wikipedia.org/wiki/FreeBSD)
- 1992\. također ide tužba [UNIX System Laboratories, Inc. (USL) v. Berkeley Software Design, Inc. (BSDi)](https://en.wikipedia.org/wiki/UNIX_System_Laboratories,_Inc._v._Berkeley_Software_Design,_Inc.) ([dokumentacija](https://www.bell-labs.com/usr/dmr/www/bsdi/bsdisuit.html)) koja nakon protutužbe od strane Sveučilišta u Berkeleyu i preuzimanja USL-a od strane Novella 1993. godine završava dogovorom koji paše obje strane (specijalno, Sveučilište u Berkeleyu ima autorsko pravo nad kodom na kojem se temelji FreeBSD)
- početnu motivaciju za pretvorbu akademskog projekta Berkeleyevog Unixa u projekt slobodnog softvera otvorenog koda dobro opisuje danski programski inženjer [Poul-Henning Kamp](https://en.wikipedia.org/wiki/Poul-Henning_Kamp), jedan od utjecajnijih razvijatelja FreeBSD-a, u članku [Free and Open Source Software–and Other Market Failures](https://cacm.acm.org/practice/free-and-open-source-software-and-other-market-failures/):

    > When the Unix revolution arrived in the mid- to late-1980s, everybody would try to sell you their “open Unix computer,” but everything from their product catalog to the sales force’s behavior screamed “vendor lock-in.” And that was on top of their products being pretty bad and overpriced in the first place, and everybody actively and deliberately trying to be incompatible with everybody else.
    >
    > (...)
    >
    > Out of that utter market failure came Minix, (Net/Free/Open)BSD, and Linux, at a median year of approximately 1991. I can absolutely guarantee that if we had been able to buy a reasonably priced and solid Unix for our 32-bit PCs—no strings attached—nobody would be running FreeBSD or Linux today, except possibly as an obscure hobby.

- danas je osobito značajan zbog vrlo malo licenčnih ograničenja kod korištenja unutar vlasničkih hardvera i softvera, na njemu se temelje

    - [Netflix Open Connect](https://openconnect.netflix.com/en/appliances/#software)
    - [Panasonic Viera TV](https://www.panasonic.com/caribbean/consumer/tv/viera-tv.html)
    - [PlayStation 3](https://en.wikipedia.org/wiki/PlayStation_3), [PlayStation 4](https://www.playstation.com/ps4/) i [PlayStation Vita](https://en.wikipedia.org/wiki/PlayStation_Vita)
    - [Nintendo Switch](https://www.nintendo.com/switch/)
    - [Darwin](https://en.wikipedia.org/wiki/Darwin_(operating_system)), koji je srž otvorenog koda [na kojoj je izgrađen](https://thenewstack.io/apples-open-source-roots-the-bsd-heritage-behind-macos-and-ios/) vlasnički operacijski sustav [Apple macOS](https://www.apple.com/macos/)
    - [Whatsapp](https://blog.whatsapp.com/1-million-is-so-2011) ([prije](https://blog.whatsapp.com/on-e-millio-n) [kupnje od strane Facebooka](https://investor.atmeta.com/investor-news/press-release-details/2014/Facebook-to-Acquire-WhatsApp/default.aspx))

- ukupno je manje popularan od Linuxa i Apple macOS-a, ali je vrlo korišten u pojedinim nišama

    - vrlo popularan kod pojedinih pružatelja internetskih usluga (vrlo rano dobio podršku za IPv6 zahvaljujući projektu [KAME](https://www.kame.net/)) i uređajima za pohranu podataka (uglavnom koriste derivat FreeBSD-a [TrueNAS](https://www.truenas.com/))
    - korišten u specijaliziranim akademskim i industrijskim istraživanjima, npr. projekt [CHERI](https://freebsdfoundation.org/blog/freebsd-for-research-cheri-morello/) s [Odjela za računarstvo i tehnologiju](https://www.cst.cam.ac.uk/about) [Sveučilišta u Cambridgeu](https://www.cam.ac.uk/) u suradnji s tvrtkom [Arm](https://www.arm.com/architecture/cpu/morello), koji razvija sigurnosne tehnike u hardveru i sustavskom softveru
    - marginalan kod superračunala ([Brooks Davis - FreeBSD, Building a Computing Cluster](https://youtu.be/BpsRb9fJ4Ds) i [Brooks Davis: Porting HPC Tools to FreeBSD](https://youtu.be/BpsRb9fJ4Ds))
    - šaljiva predavanja na temu relativne (ne)popularnosti FreeBSD-a: [BSD is Dying, Jason Dixon, NYCBSDCon 2007](https://youtu.be/g7tvI6JCXD0) i [Jason Dixon Closing Remarks of DCBSDCon - BSD is Still Dying](https://youtu.be/YzXLwhkBe80), parodija na ["Netcraft now confirms: \*BSD is dying](https://everything2.com/title/BSD+is+dying), ima priču o povijesti Unixa od početka do doba FreeBSD-a

## Ostali derivati BSD-a

- Apple [NextSTEP](https://en.wikipedia.org/wiki/NeXTSTEP)/[OPENSTEP](https://en.wikipedia.org/wiki/OpenStep), kasnije [Mac OS X i OS X](https://en.wikipedia.org/wiki/MacOS_version_history), danas [macOS](https://www.apple.com/macos/)
- [DragonFlyBSD](https://www.dragonflybsd.org/)
- [NetBSD](https://www.netbsd.org/) i [OpenBSD](https://www.openbsd.org/)

!!! tip
    Detaljniji pregled povijesti Unixa i operacijskih sustava sličnih Unixu moguće je pronaći na Wikipedijinoj stranici [History of Unix](https://en.wikipedia.org/wiki/History_of_Unix).

!!! question "Ispitna pitanja"
    1. Opišite početak BSD-a.
    1. Objasnite zašto kažemo da je BSD značajan za razvoj interneta.
    1. Opišite nastanak FreeBSD-a.
    1. Navedite dva proizvoda koji se temelje na FreeBSD-u.
    1. Objasnite zašto Apple macOS specifično vežemo uz BSD i općenito uz Unix.
