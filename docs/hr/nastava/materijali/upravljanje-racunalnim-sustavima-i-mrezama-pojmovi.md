---
author: Vedran Miletić
---

# Osnovni pojmovi upravljanja računalnim sustavima i mrežama

Dosad smo se u studiju uglavnom sretali s kolegijima iz razvoja softvera u raznim područjima računarske znanosti, pri čemu bi se sadržaj kolegija dao apstrahirati kao:

1. pronađi problem unutar područja računarstva ili nekog drugog područja ljudske djelatnosti (lingvistika, poslovanje, kemija, grafika, genetika, kartografija, sociologija, ...)
1. razvij algoritam koji rješava taj problem (kod egzaktnih problema) ili nalazi rješenje zadovoljavajuće točnosti (kod problema koji imaju više rješenja čija se kvaliteta može mjeriti)
1. implementiraj algoritam u prikladnom programskom jeziku i time dokaži da razvijeni algoritam radi što se očekuje
1. (opcionalno) složi skup algoritama u upotrebljiv softverski alat

Područje razvoj i održavanje računalnih sustava i mreža, popularno zvano *sistemašenje*, kreće s idejom da su svi potrebni softveri već velikim dijelom razvijeni, a zanima nas njihovo međusobno povezivanje u stogove softvera koji onda nude određene usluge i udovoljavaju zahtjevima korisnika. Drugim riječima, kod proučavanja razvoja softvera smatrali smo implementaciju gotovog softvera zajedno s potrebnim elementima stoga u nekom okruženju postupkom koji nas ne treba brinuti, dok ćemo sada smatrati proces razvoja softvera postupkom koji je riješen i baviti se upravo implementacijom raznih gotovih softvera koji udovoljavaju zahtjevima korisnika.

## Problemi u domeni sistemašenja

Upravitelj računalnih sustava i mreža, popularno zvan sistem admistrator, od milja *sistemac* ili *sistemaš*, je osoba koja se bavi problemima kao što su:

- Janko ne može ispisati dokument već 10 minuta; dijeljeni pisač Canon iR pokazuje da ima tonera i papira, ali dokument koji je on poslao na ispis nije izišao iz pisača, zapravo, pisač se ponaša kao da uopće nije dobio naredbu da išta ispiše.
- U laboratoriju za teorijsku astrofiziku Ivana želi napraviti upload obrađenih podataka na poslužitelj [kolaboracije ARIANNA](https://arianna.ps.uci.edu/) kako bi ih njezini suradnici mogli pregledati, ali veza na internet je višestruko sporija nego ona u njenom uredu.
- Sanjin želi udaljeni pristup poslužitelju s datotekama unutar tvrtke. Poslužitelj je inače dostupan jedino iz ureda zaposlenika, ali Sanjin ide na službeni put u Australiju na 3 tjedna i želi moći surađivati na kvartalnom izvještaju koji treba biti spreman u manje od mjesec dana.
- Andreji treba instalirati novu verziju alata za upravljanje referencama [Zotero](https://www.zotero.org/) u računalnoj učionici za nastavu iz Uvoda u istraživački rad u biotehnologiji. Nova verzija 5.0 jako poboljšava sinkronizaciju što će omogućiti studentima da budu produktivniji na satovima vježbi.
- Tvrtko želi dual-boot Windowsa 10 i Ubuntua 18.04 na novom laptopu [HP Spectre](https://www.hp.com/us-en/shop/slp/spectre-family/hp-spectre-x-360), koji mu je predan na korištenje kao zamjena za upravo rashodovani, 4 godine stari [Lenovo ThinkPad](https://www.lenovo.com/hr/hr/thinkpad/).
- Martina bi htjela imati udaljeni pristup desktopu svog uredskog računala. Naime, neki od softvera koje ona i suradnici razvijaju mogu se izvoditi samo unutar organizacije obzirom da su jedino tamo dostupne licencirane verzije vlasničkih softvera koje ti lokalno razvijeni softveri zahtijevaju za svoj rad. Obzirom da izgradnja i izvođenje testova tih softvera traje gotovo 2 sata, ponekad Martina želi pokrenuti taj proces dok je na terenu ili na sastanku u drugoj tvrtci.
- Franka želi rezultate simulacija [molekularne dinamike](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/molecular-dynamics) veličine približno 10 TB učiniti dostupnima za udaljeni pristup suradnicima sa [Sveučilišta Rurh u Bochumu](https://www.ruhr-uni-bochum.de/). Zbog veličine datoteka, oni ih nemaju namjeru preuzimati u cijelosti, ali bi ih htjeli obrađivati na udaljenom računalu i eventualno preuzeti dio njih za daljnju analizu vlastitim alatima.
- Marin je pronašao bug u softveru VoIP telefona Cisco koji onemogućuje korištenje funkcije imenika kad je jezik postavljen na hrvatski i kad je aktivirana funkcija automatskog prelaska u low power način rada nakon 15 minuta. Taj je bug popravljen u verziji softvera koja je od pred dva tjedna dostupna na Ciscovim stranicama.
- [MIT](https://web.mit.edu/) želi uspostaviti [udrugu diplomiranih bivših studenata](https://alum.mit.edu/about) koja će, među ostalim, svakom članu osigurati da može zauvijek koristiti e-mail alias oblika `<ime>@alum.mit.edu` i URL alias oblika `http(s)://alum.mit.edu/~<ime>/`.
- Nakon zadnjeg updatea Windowsa 10 u travnju 2018. godine, Ivona ima probleme sa "zamrzavanjem" računala nakon pokretanja memorijski zahtjevnih aplikacija kao što su Adobe Dreamweaver i SideFX Houdini na svom uredskom desktop računalu [Dell OptiPlex](https://www.dell.com/en-us/work/shop/desktop-and-all-in-one-pcs/sf/optiplex-desktops). Postoji mogućnost da je problem u UEFI firmwareu na tom računalu, obzirom da njeno računalo koristi verziju iz veljače 2016. godine, a od tada je Dell izdao 8 verzija s uključenim zakrpama za različite probleme koji su im prijavljeni od strane korisnika ili koje su sami pronašli.
- Moreno želi da se njegove statičke mrežne stranice pisane u [Markdownu](https://daringfireball.net/projects/markdown/) automatski izgrađuju [MkDocsom](https://www.mkdocs.org/) kad napraviti `git push` u određeni repozitorij i da se, ako izgradnja prođe bez greške, datoteke kopiraju u `/var/www` umjesto postojećih.
- Podravka želi nadograditi svoj službeni automobil [Tesla Model S](https://www.tesla.com/models) [novim softverom](https://www.tesla.com/support/software-updates) koji donosi poboljšanja mnogih značajki, uključujući i [samovožnju](https://www.tesla.com/autopilot).

## Mjesta na webu koja vrijedi pratiti

### Sistemašenje

- [2.5 Admins](https://2.5admins.com/)
- [CARNET sys.portal](https://sysportal.carnet.hr/)
- [It's A Digital Disease! (/r/DataHoarder/)](https://www.reddit.com/r/DataHoarder/)
- [Everything DevOps (/r/devops/)](https://www.reddit.com/r/devops/)
- [(☞ﾟ∀ﾟ)☞ (/r/homelab/)](https://www.reddit.com/r/homelab/)
- [linuxadmin: Expanding Linux SysAdmin knowledge (/r/linuxadmin/)](https://www.reddit.com/r/linuxadmin/)
- [The Lone Sysadmin](https://lonesysadmin.net/)
- [nixCraft](https://www.cyberciti.biz/)
- [Red Hat Enable Sysadmin](https://www.redhat.com/sysadmin/)
- [Self Hosted (Jupiter Broadcasting)](https://www.jupiterbroadcasting.com/show/self-hosted/)
- [Server Fault](https://serverfault.com/)
- [ServeTheHome](https://www.servethehome.com/)
- [Sysadmin on Reddit (/r/sysadmin/)](https://www.reddit.com/r/sysadmin/)
- [Sysadmin Humor - lulz from crazy IT peoplez (/r/Sysadminhumor)](https://www.reddit.com/r/Sysadminhumor/)
- [Unix & Linux Stack Exchange](https://unix.stackexchange.com/)
- [Wandering Thoughts (Chris Siebenmann)](https://utcc.utoronto.ca/~cks/space/blog/)

### Slobodni softver otvorenog koda

- [BSD Now](https://www.bsdnow.tv/)
- [Cat-v.org Random Contrarian Insurgent Organization](https://cat-v.org/)
- [installgentoo (the often imitated but never duplicated)](https://installgentoo.com/)
- ["Linux, GNU/Linux, free software... on Reddit (r/linux/)](https://www.reddit.com/r/linux/)
- [May your htop stats be low and your beard grow long on Reddit (r/linuxmasterrace/)](https://www.reddit.com/r/linuxmasterrace/)
- [Linux @ OneAndOneIs2](https://linux.oneandoneis2.org/)
- [LWN.net](https://lwn.net/)
- [OMG! Ubuntu](https://www.omgubuntu.co.uk/)
- [OSTechNix](https://www.ostechnix.com/)
- [Phoronix](https://www.phoronix.com/)
- [Planet CentOS](https://planet.centos.org/)
- [Planet Debian](https://planet.debian.org/)
- [Planet Fedora](https://planet.fedoraproject.org/)
- [Planet FreeBSD](https://planet.xbsd.net/)
- [Planet openSUSE](https://planet.opensuse.org/)
- [Planet Ubuntu](https://planet.ubuntu.com/)

### Sigurnost

- [Common Vulnerabilities and Exposures (CVE)](https://cve.mitre.org/)
- [Debian -- Security Information](https://www.debian.org/security/)
- [The Hacker News](https://thehackernews.com/)
- [Naked Security (Sophos)](https://nakedsecurity.sophos.com/)

### Superračunala i podatkovni centri

- [Data Center Knowledge](https://www.datacenterknowledge.com/)
- [HPC News from Livermore Computing Center](https://hpc.llnl.gov/news)
- [HPCwire](https://www.hpcwire.com/)
- [insideHPC](https://insidehpc.com/)
- [The Next Platform](https://www.nextplatform.com/)

### Internet i umrežavanje

- [APNIC Blog](https://blog.apnic.net/)
- [ICANN Announcements](https://www.icann.org/en/announcements)
- [IETF Blog](https://ietf.org/blog/)
- [RIPE Labs](https://labs.ripe.net/)

### Softver i tehnologija općenito

- [AdoredTV](https://adoredtv.com/)
- [AnandTech](https://www.anandtech.com/)
- [Coreteks](https://coreteks.tech/)
- [ExtremeTech](https://www.extremetech.com/)
- [/g/ on 4chan](https://boards.4chan.org/g/)
- [Hacker News (Y Combinator)](https://news.ycombinator.com/)
- [HotHardware](https://hothardware.com/)
- [InfoWorld](https://www.infoworld.com/)
- [PCWorld](https://www.pcworld.com/)
- [The Register](https://www.theregister.com/)
- [Semiconductor Engineering](https://semiengineering.com/)
- [Softpedia News](https://news.softpedia.com/)
- [Techdirt](https://www.techdirt.com/)
- [TechPowerUp](https://www.techpowerup.com/)
- [Tom's Hardware](https://www.tomshardware.com/)
- [/v/technology](https://voat.co/v/technology)
- [Wccftech](https://wccftech.com/)
- [ZDNet](https://www.zdnet.com/)

### Privatnost i ostala društvena pitanja

- [Privacy News Online](https://www.privateinternetaccess.com/blog/)
- [EFF Deeplinks Blog](https://www.eff.org/deeplinks)
- [La Quadrature du Net](https://www.laquadrature.net/)

### Računalne igre

- [Boiling Steam](https://boilingsteam.com/)
- [IGN](https://www.ign.com/)
- [Niche Gamer](https://nichegamer.com/)
- [GamingOnLinux](https://www.gamingonlinux.com/)
- [Gematsu](https://www.gematsu.com/)
- [GNU/Linux Gaming on Reddit (r/linux_gaming/)](https://www.reddit.com/r/linux_gaming/)
- [PCGamer](https://www.pcgamer.com/)
- [Rock, Paper, Shotgun](https://www.rockpapershotgun.com/)
- [TechRaptor](https://techraptor.net/)
- [/v/ on 4chan](https://boards.4chan.org/v/)
- [/v/gaming](https://voat.co/v/gaming)

### Dokumentacija

- [A-Z Index of the Linux command line (SS64.com)](https://ss64.com/bash/)
- [ArchWiki](https://wiki.archlinux.org/)
- [Gentoo Wiki](https://wiki.gentoo.org/)
- [The Linux Documentation Project](https://tldp.org/)
- [OSDev Wiki](https://wiki.osdev.org/)
- [Unix Toolbox (Colin Barschel)](https://www.linuxsc.net/pdf/unixtoolbox.pdf)
