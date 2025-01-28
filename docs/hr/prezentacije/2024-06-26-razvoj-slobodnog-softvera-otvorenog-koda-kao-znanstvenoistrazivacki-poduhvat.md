---
marp: true
theme: default
class: default
paginate: true
author: Vedran Miletić
title: Razvoj slobodnog softvera otvorenog koda kao znanstvenoistraživački poduhvat -- motivacija, izvedba i utjecaj
description: Open Science Café
keywords: free software, open source, scientific software, gromacs, github
abstract: |
  Slobodni softver otvorenog koda je u posljednja četiri destljeća narastao od poduhvata malobrojnih entuzijasta okupljenih oko projekta GNU i neprofitne organizacije FSF do globalno prepoznatljivog fenomena koji pogoni 500 najvećih svjetskih superračunala, o kojem ovise javne uprave većine razvijenih država, na kojem poslovanje temelje brojne međunarodne tvrtke i o kojem pišu mediji, a nastavlja rasti i dalje. Danas već davne 2018. godine Microsoft je dao 7,5 milijardi američkih dolara kako bi preuzeo GitHub, najveću svjetsku platformu za razvoj i dijeljenje otvorenog koda, a godinu kasnije je IBM za 34 milijarde dolara kupio Red Hat, tada vodeću kompaniju za razvoj slobodnog operacijskog sustava za poslovne korisnike temeljenog na jezgri Linux. Osim u poslovanju, slobodni softver otvorenog koda je u posljednjem desetljeću postao prepoznat kao istovremeno tržišno relevantan i dugoročno održiv način razvoja softvera i u brojnim drugim područjima, među kojima je i razvoj znanstvenoistraživačkog softvera. Suvremena znanost koristi softver primarno kao alat u procesu istraživanja, slično kao fizičke uređaje u pojedinim područjima, ali i za automatizaciju repetitivnih radnji, izvođenje velikog broja virtualnih eksperimenata, formalnu provjeru teorijskih modela te brojne druge primjene. Kako se napredak u znanosti temelji na "stajanju na ramenima divova", otvoreni kod koji omogućuje nadogradnju i prilagodbu u skladu s vlastitim potrebama i interesima je prirodni način razvoja znanstvenog softvera.

  Predavanje će govoriti o razvoju slobodnog softvera otvorenog koda kao znanstvenog alata iz tri osnovne perspektive: znanstvenoistraživačke, gdje će biti govora o motivaciji za razvoj znanstvenog softvera koji služi kao neposredna podrška vlastitim istraživačkim aktivnostima, tehničke, gdje će biti dan pregled alata koji se koriste prilikom izvedbe razvoja i davanja doprinosa, te organizacijske, gdje će biti razmotren utjecaj doprinosa slobodnim softverima na povećanje vidljivosti i potencijala za suradnjama.
curriculum-vitae: |
  [Vedran Miletić](https://vedran.miletic.net/) je zaposlen kao docent u polju računarstva na Fakultetu informatike i digitalnih tehnologija, gdje vodi Grupu za aplikacije i usluge na eksaskalarnoj istraživačkoj infrastrukturi. Član je Centra za popularizaciju i promociju znanosti te Povjerenstva za superračunalne resurse Sveučilišta u Rijeci. Trenutno je na dopustu zbog znanstvenog usavršavanja na Računarskom i podatkovnom postrojenju Max Plancka, gdje se bavi poboljšanjem metoda simulacije molekulske dinamike u slobodnim softverima otvorenog koda namijenjenim za akademsku i industrijsku primjenu na modernim superračunalima. Posljednjih dva destljeća aktivno radi na promociji korištenja slobodnosoftverskih rješenja kroz javna predavanja, radionice i organizaciju proslava Dana slobode dokumenata te rado ističe [What is the price of open-source fear, uncertainty, and doubt?](https://group.miletic.net/en/blog/2015-09-14-what-is-the-price-of-open-source-fear-uncertainty-and-doubt/) kao svoj najbolji neznanstveni rad.
---

# Razvoj slobodnog softvera otvorenog koda kao znanstvenoistraživački poduhvat – motivacija, izvedba i utjecaj

## [Vedran](https://vedran.miletic.net/) [Miletic](https://www.miletic.net/), HPC application expert, MPCDF, MPG; docent i voditelj grupe (na dopustu), FIDIT, UniRi

![MPG MPCDF logos](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc2a89605e5cb6effc55ad732f103d71afb8c7060ecaa95c5fb93987e4c8acbd.svg)

### [Centar za otvorenu znanost](https://otvorenaznanostuniri.svkri.hr/centar-za-otvorenu-znanost/), [Open Science Café](https://otvorenaznanostuniri.svkri.hr/open-science-cafe/), [26. lipnja 2024.](https://otvorenaznanostuniri.svkri.hr/najava-open-science-cafe-razvoj-slobodnog-softvera-otvorenog-koda-kao-znanstvenoistrazivacki-poduhvat-motivacija-izvedba-i-utjecaj/)

---

## Dr. Vedran Miletic

- doktorat, FER; razvoj simulatora telekomunikacijske mreže [ns-3](https://www.nsnam.org/)
- postdok, Heidelberški institut za teorijske studije (*Heidelberg Institute for Theoretical Studies*, HITS); razvoj simulatora molekulske dinamike [GROMACS](https://www.gromacs.org/)
    - pomoć u pripremi za intervju: [pok.](https://svedruziclab.github.io/news/2023/04/21/obituary-dr-%C5%BEeljko-svedru%C5%BEi%C4%87.html) [izv. prof. dr. sc. Željko Svedružić](https://svedruziclab.github.io/principal-investigator.html)
- viši predavač, UniRi; razvoj softvera za predviđanje pristajanja malenih molekula na proteine [RxDock](https://rxdock.gitlab.io/), [RxTx Research](https://rxtxresearch.github.io/)
- docent i voditelj grupe, UniRi / HPC application expert, Računarsko i podatkovno postrojenje Max Plancka (*Max Planck Computing and Data Facility*, MPCDF); razvoj simulatora molekulske dinamike [GROMACS](https://www.gromacs.org/)
- van fakulteta: [Udruga Penkala](https://udruga-penkala.hr/) ([Radionica: Izradite svoj web u 4 sata!](https://udruga-penkala.hr/radionica-izradite-svoj-web-u-4-sata/2024/)), [Riječka podružnica Hrvatske udruge Linux korisnika](https://www.ri.linux.hr/) i dr.

---

## "Je li razvoj softvera za mene ili moju grupu? To je više za informatičare."

- [izv. prof. dr. sc. Željko Svedružić](https://svedruziclab.github.io/principal-investigator.html) (BioTech): [razvoj alata za kvantnu kemiju CP2K](https://www.cp2k.org/)
- [prof. dr. sc. Nataša Hoić-Božić](https://www.inf.uniri.hr/~natasah/), izv. prof. dr. sc. Martina Holenko Dlab (FIDIT); doc. dr. sc. Jasminka Mezak (UFRi): [razvoj sustava preporučivanja za računalom podržano učenje ELARS](https://fidit-rijeka.github.io/elarsportal/)
- [Petra Mrša](https://apuri.uniri.hr/nastavnik/petra-mrsa/), [Dijana Protić](https://apuri.uniri.hr/dijana-protic/) (APURi): [razvoj platforme za agregaciju digitalnih audiovizualnih umjetničkih djela CubeCommons](https://cubecommons.ca/)
- brojne druge suradnje manjeg opsega: [izv. prof. dr. sc. Kristijan Lenac](https://klenac.weebly.com/) (RiTeh), doc. dr. sc. Benedikt Perak (FFRi)

![bg right:25%](https://unsplash.com/photos/1LLh8k2_YFk/download?w=1920) <!-- https://unsplash.com/photos/a-computer-screen-with-a-bunch-of-text-on-it-1LLh8k2_YFk -->

---

## Sadržaj predavanja

- Vlasnički i slobodni softver
- Znanstveni softver
- Licence znanstvenog softvera: komercijalne, akademske, slobodne/otvorene
- Motivacija, izvedba i utjecaj
- Zaključak i idući koraci

![bg left](https://unsplash.com/photos/4kCGEB7Kt4k/download?w=1920) <!-- https://unsplash.com/photos/open-led-signage-4kCGEB7Kt4k -->

---

## Pogled u povijest

- u počecima softvera: otvoren kao i formule u fizici ili matematici
- komercijalizacija: prodaja hardvera i softvera u paketu
    - kao Apple macOS danas
    - nastanak i grananje Unixa
    - npr. grafika samo SGI
    - Berkeley Software Distribution (BSD)
- Microsoftova vizija: Windowsi kao univerzalno sučelje između različitog hardvera i različitog softvera

![bg right:40%](https://unsplash.com/photos/klWUhr-wPJ8/download?w=1920) <!-- https://unsplash.com/photos/img-ix-mining-rig-inside-white-and-gray-room-klWUhr-wPJ8 -->

---

## Slobodni softver i otvoreni kod

- softver koji možete prilagođavati svojim potrebama i dijeliti prilagođene verzije (!= besplatan softver)
    - slično kao znanstvene rezultate
- [Richard M. Stallman](https://stallman.org/)
    - radi u MIT AI Labu; [priča o printeru](https://en.wikipedia.org/wiki/Richard_Stallman#Events_leading_to_GNU)
    - 1983\. osniva projekt [GNU](https://www.gnu.org/), kratica za *GNU's Not Unix* (GNU zaista nije Unix jer je neovisno razvijen kao slobodni softver, a Unix je neslobodni softver)
    - 1985\. osniva Free Software Foundation (FSF) kako bi financirao razvoj slobodnog softvera
- Netscape/Mozilla 1998., preteča Firefoxa
    - otvoreni kod (engl. *open source*)

![bg left:25%](https://unsplash.com/photos/8bghKxNU1j0/download?w=1920) <!-- https://unsplash.com/photos/purple-and-blue-light-digital-wallpaper-8bghKxNU1j0 -->

---

## Linus Torvalds, autor Linuxa

> I often compare open source to science. To where science took this whole notion of developing ideas in the open and improving on other peoples' ideas and making it into what science is today and the incredible advances that we have had. And I compare that to witchcraft and alchemy, where openness was something you didn't do.

![bg right:40%](https://unsplash.com/photos/iS1NV9yN0Lg/download?w=1920) <!-- https://unsplash.com/photos/penguins-on-brown-rock-formation-iS1NV9yN0Lg -->

## Video: [Truth Happens Remix](https://youtu.be/5EkkMfjetEY)

---

## Koliko vrijedi slobodni softver otvorenog koda?

- 2018\. godine Microsoft je dao 7,5 milijardi američkih dolara kako bi preuzeo [GitHub](https://github.com/), najveću svjetsku platformu za razvoj i dijeljenje otvorenog koda
    - ~= Facebook/Instagram/TikTok za programere
    - zbog toga danas može ponuditi [umjetnu inteligenciju Copilot](https://github.com/features/copilot) koja pomaže programerima u razvoju softvera
    - kroz [Archive Program](https://archiveprogram.github.com/) radi na očuvanju otvorenog koda za buduće generacije
- godinu kasnije IBM je za 34 milijarde dolara kupio [Red Hat](https://www.redhat.com/), tada vodeću kompaniju za razvoj slobodnog operacijskog sustava za poslovne korisnike temeljenog na jezgri Linux
    - financijske institucije, zdravstvo, [superračunala](https://cnrm.uniri.hr/bura/) i dr.

![bg right:20%](https://unsplash.com/photos/BV49ACKmMtY/download?w=1920) <!-- https://unsplash.com/photos/purple-and-white-love-neon-light-signage-BV49ACKmMtY -->

---

## Mentimeter: Što je za vas znanstveni softver?

## `menti.com` i kod `1522 0077`

---

## Znanstveni softver

- vrsta izvršnog softvera (kao web preglednik ili kasa u Konzumu)
- razvija se kako bi riješio neki problem u nekoj grani znanosti
    - automatizacija repetitivnih postupaka
        - roboti u stvarnom svijetu
        - obrada podataka u računalu
    - modeliranje i simulacija stvarnih sustava u računalu
- može se dijeliti među znanstvenicima kao znanstveni rezultati ili komercijalizirati

![bg left:40%](https://unsplash.com/photos/pwcKF7L4-no/download?w=1920) <!-- https://unsplash.com/photos/refill-of-liquid-on-tubes-pwcKF7L4-no -->

---

## Licence znanstvenog softvera

- komercijalne licence
    - deseci različitih modela uvjeta i obračuna troškova (proporcionalno veličini superračunala, broju ljudi u grupi, broju eksperimenata/simulacija na godinu, ...)
- akademske licence
    - besplatno ili simbolično niska cijena za nekomercijalno korištenje u akademske istraživačke i nastavne svrhe
    - kompetitivna cijena za komercijalno korištenje u industriji
- slobodne, odnosno otvorene licence

![bg right:35%](https://unsplash.com/photos/OQMZwNd3ThU/download?w=1920) <!-- https://unsplash.com/photos/man-writing-on-paper-OQMZwNd3ThU -->

---

## Slobodne, odnosno otvorene licence znanstvenog softvera

- copyleft (~= zahtijevaju dijeljenje pod istim uvjetima)
    - [GNU Affero General Public License (AGPL)](https://en.wikipedia.org/wiki/GNU_Affero_General_Public_License)
    - [GNU General Public License (GPL)](https://en.wikipedia.org/wiki/GNU_General_Public_License)
    - [GNU Lesser General Public License (LGPL)](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License)
    - [Mozilla Public License (MPL)](https://en.wikipedia.org/wiki/Mozilla_Public_License)
- ne-copyleft
    - [Apache License](https://en.wikipedia.org/wiki/Apache_License)
    - [MIT License](https://en.wikipedia.org/wiki/MIT_License)
    - [BSD Licenses](https://en.wikipedia.org/wiki/BSD_licenses): najpopularnija Simplified BSD (FreeBSD) License

![bg left:35%](https://unsplash.com/photos/ZYBl6VnUd_0/download?w=1920) <!-- https://unsplash.com/photos/green-and-white-open-signage-ZYBl6VnUd_0 -->

---

## Motivacija: Hrvatska je periferija zapadnog prostora znanosti

- realnost je da su Kina, Rusija, Iran i s njima povezane manje države van zapadnog prostora znanosti
    - nažalost je sve izraženija tendencija da objavljuju kod sebe na svom jeziku
- kaskamo u usporedbi sa zapadnijim državama: Nature metrike, ARWU, QS, THE, ...
    - razlozi su dobro poznati: financiranje, organizacija, povijesni razlozi (prof. Ivan Đikić, prof. Gordan Lauc i brojni drugi)
- kroz razvoj znanstvenog softvera moguće je jače se povezati sa zapadnim kolegama i manje ovisiti o financiranju od strane MZO RH (HrZZ)

![bg right:20%](https://unsplash.com/photos/6gSyEKq4Pvg/download?w=1920) <!-- https://unsplash.com/photos/north-east-west-and-south-wall-decor-6gSyEKq4Pvg -->

---

## Motivacija: softver kao digitalni robot

- automatizacija repetitivnih radnji u stvarnom svijetu -> **robot**
- automatizacija repetitivne obrade podataka -> **softver**
    - naravno, softver može i više od toga (!)

![bg left:55%](https://unsplash.com/photos/F49x0Vct5Lo/download?w=1920) <!-- https://unsplash.com/photos/assorted-hand-tools-on-wall-F49x0Vct5Lo -->

---

## Motivacija: *software is eating the world*

- softver oko nas: pametni telefoni, televizije, auti, kućanski aparati, ...
- modeliranje i simulacije umjesto teško dostupnih, sporih ili skupih eksperimenata u procesu istraživanja
    - većina objavljenih radova kombinira eksperimentalne i simulacijske metode, prilagođava ih po potrebi

![bg right](https://unsplash.com/photos/PBtfsP3eEZ4/download?w=1920) <!-- https://unsplash.com/photos/person-holding-black-iphone-4-PBtfsP3eEZ4 -->

---

## Motivacija: stajanje na ramenima divova

> if I have seen further \[than others\], it is by standing on the shoulders of giants.

-- Isaac Newton

- ~= daljnji razvoj postojećih (slobodnih) znanstvenih softvera
- slobodni softver je zbog toga posebno prikladan za razvoj znanstvenog softvera

![bg left:50%](https://unsplash.com/photos/sqWiX2-BxLg/download?w=1920) <!-- https://unsplash.com/photos/a-woman-standing-next-to-a-statue-of-a-woman-sqWiX2-BxLg -->

---

## Izvedba: osnovne potrepštine

- problem -> postupak rješavanja
- programski jezik: [Python](https://www.python.org/), [Julia](https://julialang.org/), [R](https://www.r-project.org/), [Perl](https://www.perl.org/); [C++](https://isocpp.org/), [C](https://en.wikipedia.org/wiki/C_(programming_language)), [Fortran](https://fortran-lang.org/)
    - Python je danas *lingua franca*
    - gotove biblioteke treće strane koje se mogu iskoristiti su lakše za pronaći u popularnijim programskim jezicima
- razvojno okruženje: Microsoft Visual Studio (VS) Code, JetBrains PyCharm ili sl.

![bg right:40%](https://unsplash.com/photos/fZP5-34c91Q/download?w=1920) <!-- https://unsplash.com/photos/a-black-and-white-photo-of-the-top-of-a-tower-fZP5-34c91Q -->

---

## Izvedba: odnos programera i korisnika softvera

- korisnik ima domensko znanje => problem, potreba
- programer ima tehničko znanje => razvoj, rješenje
- ista osoba može biti i korisnik i programer: jezici kao Python drastično su smanjili razinu potrebnih znanja za započeti razvoj softvera
- zajedno ulažu vrijeme i trud => softver usklađen s potrebama

![bg left](https://unsplash.com/photos/5fNmWej4tAA/download?w=1920) <!-- https://unsplash.com/photos/person-holding-pencil-near-laptop-computer-5fNmWej4tAA -->

---

## Izvedba: alati za suradnički razvoj softvera

- platforme [GitHub](https://github.com/) i [GitLab](https://gitlab.com/) omogućuju čuvanje promjena i dijeljenje razvijenog softvera:
    - softver može steći zvjezdice ~= lajkovi na Facebooku, Instagramu i TikToku
    - automatsko testiranje kod dodavanja promjena
    - mogućnost praćenja prijavljenih problema, planova razvoja, ...

![bg right](https://unsplash.com/photos/842ofHC6MaI/download?w=1920) <!-- https://unsplash.com/photos/a-close-up-of-a-text-description-on-a-computer-screen-842ofHC6MaI -->

---

## Izvedba: imamo li dovoljno računalnog hardvera?

- za početak je svako moderno računalo (do ~10 godina starosti) dovoljno dobro
- idući korak: [Bura](https://cnrm.uniri.hr/bura/) i [Supek](https://www.srce.unizg.hr/napredno-racunanje)
- kad domaća supperračunala postanu premala: [The European High Performance Computing Joint Undertaking (EuroHPC JU)](https://eurohpc-ju.europa.eu/)
- alternativa: računala u oblaku ([Amazon Web Services](https://aws.amazon.com/), [Google Cloud Platform](https://cloud.google.com/), [Microsoft Azure](https://azure.microsoft.com/))

![bg left](https://unsplash.com/photos/rvfm_b1C6lc/download?w=1920) <!-- https://unsplash.com/photos/an-old-computer-sitting-on-the-floor-next-to-a-wall-rvfm_b1C6lc -->

---

## Treba li razvijati softver u skladu sa svojim znanstveno-istraživačkim potrebama?

![bg](https://unsplash.com/photos/tYVkjjMYFBo/download?w=1920) <!-- https://unsplash.com/photos/person-writing-on-dry-erase-board-tYVkjjMYFBo -->

---

## Ako već postoji gotov, vjerojatno ne

- iz znanstvene perspektive, upitan doprinos
- iz praktične perspektive, gubitak vremena i pogrešno uloženi trud

![bg left:60%](https://unsplash.com/photos/wD1LRb9OeEo/download?w=1920) <!-- https://unsplash.com/photos/three-men-sitting-while-using-laptops-and-watching-man-beside-whiteboard-wD1LRb9OeEo -->

---

## Ako postoji sličan ("skoro gotov"), bolje je doraditi

- uključivanje u razvoj postaje "penjanje na ramena divova"
- odlična prilika za povezivanje i suradnju sa znanstvenicima koji razvijaju i koriste taj softver

![bg right:60%](https://unsplash.com/photos/wMRIcT86SWU/download?w=1920) <!-- https://unsplash.com/photos/people-sitting-on-chair-in-front-of-table-wMRIcT86SWU -->

---

## Ako ne postoji, da

- (anegdotalno) znanstvenici u HR općenito koriste bitno manje automatizacije repetitivnih radnji u procesu istraživanja u odnosu na znanstvenike u zapadnijim državama
- odabrati jezik i licencu, iskoristiti što više gotovih biblioteka treće strane
- moguće je uključiti suradnike i podijeliti posao (i zasluge)

![bg left](https://unsplash.com/photos/2jTu7H9l6JA/download?w=1920) <!-- https://unsplash.com/photos/green-and-black-audio-mixer-2jTu7H9l6JA -->

---

## Utjecaj: javno dostupne metrike

- softveri se distribuiraju putem različitih repozitorija, npr. [conda-forge](https://conda-forge.org/), [PyPI](https://pypi.org/)
    - imaju javne metrike broja preuzimanja
- GitHub/GitLab zvjezdice -> kao lajkovi na društvenim mrežama, pomažu rangiranju u pretragama
- arhiva verzija softvera na Zenodu pomaže nalažljivosti softvera povezanim s radovima
    - ima javne metrike broja pogleda
    - može se citirati pomoću DOI

![bg right:40%](https://unsplash.com/photos/3XisDwg6jAE/download?w=1920) <!-- https://unsplash.com/photos/white-and-brown-elephant-figurine-3XisDwg6jAE -->

---

## Utjecaj: vidljivost na tražilicama

- dokumentacija (ili web sjedište) znanstvenog softvera navodi autore, grupe i sveučilišta te može imati poveznice na njihova web sjedišta
    - rast posjećenosti dokumentacije i rast broja poveznica na nju utječe na rast važnosti web sjedišta autora, grupa i sveučilišta
    - spontani nalaznici mogu doprinijeti softveru i postaju potencijalni suradnici u znanstvenim projektima

![bg left:40%](https://unsplash.com/photos/yeB9jDmHm6M/download?w=1920) <!-- https://unsplash.com/photos/smartphone-showing-google-site-yeB9jDmHm6M -->

---

## Utjecaj: objavljeni radovi i citati

- u mnogim područjima moguće je objaviti znanstveni rad u časopisu koji opisuje razvijeni znanstveni softver
    - ponekad bude dio rada koji opisuje istraživanje u kojem je taj softver prvi put iskorišten
- korisnici rado citiraju softver koji koriste iz zahvalnosti što im je besplatno dostupan i što ga mogu prilagoditi svojim potrebama
    - rad koji opisuje softver može lako imati 10x ili 100x više citata od prosječnog rada neke grupe

![bg right:40%](https://unsplash.com/photos/hy116XBXS6g/download?w=1920) <!-- https://unsplash.com/photos/a-close-up-of-a-book-with-writing-on-it-hy116XBXS6g -->

---

## Uspješni primjeri

- [GROMACS](https://www.gromacs.org/), [OpenMM](https://openmm.org/)
    - samo naš tutorial za GROMACS ima više od 10 000 pogleda mjesečno
- [PyMOL](https://pymol.org/), [VIAMD](https://github.com/scanberg/viamd)
- [PLUMED](https://www.plumed.org/)
- [Open](https://www.openfoam.com/)[FOAM](https://openfoam.org/)
- [NWChem](https://nwchem-sw.org/)
- [Gephi](https://gephi.org/)
- [Orange](https://orangedatamining.com/)
- [PyTorch](https://pytorch.org/)

![bg left](https://unsplash.com/photos/DYHx6h3lMdY/download?w=1920) <!-- https://unsplash.com/photos/a-ferris-wheel-lit-up-at-night-with-colorful-lights-DYHx6h3lMdY -->

---

## Promjene u pravnom okviru akademske zajednice

- u akademskoj zajednici u UK, uz mjesta doktoranda, predavača, profesora i dr., već više od 10 godina postoji radno mjesto inženjer znanstvenog softvera (engl. *research software engineer*, RSE)
    - u DE/FR se ljude zapošljava na radna mjesta za poslijedoktorande, istraživače i znanstvenike koji su članovi osoblja, ali nisu voditelji grupa
    - kako RH slijedi zapadne trendove, realno je očekivati da će se pravni okvir na neki način kretati u tom smjeru

![bg right:35%](https://unsplash.com/photos/j06gLuKK0GM/download?w=1920) <!-- https://unsplash.com/photos/gold-and-silver-round-frame-magnifying-glass-j06gLuKK0GM -->

---

## Zaključak i idući koraci

- grupe koje razvijaju znanstveni znanstveni softver često surađuju i s industrijom
- grupe koje razvijaju znanstveni slobodni softver otvorenog koda često budu jake i međunarodno poznate, iako nisu nužno uvijek dio najjačih sveučilišta
    - mogućnost za pozicioniranje grupa s hrvatskih sveučilišta (pa i UniRi)
- danas je razvoj znanstvenog softvera istovremeno kompetitivan i kolaborativan
    - razvojni alati sve bolji => kvaliteta razvijenog softvera je sve bolja
    - teško je biti bolji od nekog postojećeg rješenja na kojem više grupa radi desetljećima i koje ima velik broj korisnika
    - bolje je ući u nišu koja nije bila pokrivena ili se otvorila napretkom u znanosti (npr. šira primjena umjetne inteligencije)

![bg](https://unsplash.com/photos/OxHPDs4WV8Y/download?w=1920) <!-- https://unsplash.com/photos/polyhedral-wall-OxHPDs4WV8Y -->

---

## [Open Playground](https://youtu.be/oDzecz8F4yI)

![bg right](https://unsplash.com/photos/GlQawc9OMyI/download?w=1920) <!-- https://unsplash.com/photos/orange-and-black-playground-slide-GlQawc9OMyI -->

---

## Hvala na pažnji!

## Prezentacija: `miletic.net/otvorenaznanost2024`

## Osobna stranica: <https://vedran.miletic.net/>

## Kontakt e-mail: <vedran@miletic.net>
