---
author: Vedran Miletić
---

# O laboratoriju

(Ova je stranica također dostupna [na engleskom jeziku](../en/index.md).)

**Laboratorij za računalne mreže, paralelizaciju i simulaciju** (CNPSLab, international name: *Computer Networks, Parallelization, and Simulation Laboratory*) je istraživačka, razvojna i nastavna jedinica na [Odjelu za informatiku](https://www.inf.uniri.hr/) [Sveučilišta u Rijeci](https://www.uniri.hr/). Osnovan je u travnju 2008. godine kao Laboratorij za računalne mreže (international name: *Computer Networks Laboratory*), a ime je promijenjeno zajedno s proširenjem znanstvenih i edukacijskih interesa.

Specifično, zanima nas računarska znanost i računarstvo visokih performansi korištenjem slobodnog softvera otvorenog koda i otvorenog hardvera koliko god je to moguće.

Općenito, naš cilj je raditi bolju znanost na način otvorenog izvora; znanost je u suštini otvorena i takvom treba i ostati. To uključuje otvorene formate datoteka, softver otvorenog koda, otvoreni hardver, obrambeno licenciranje patenata i otvoreni pristup publikacijama.

**Ova stranica je u izradi (stanje sredinom 2017. godine).** Više detalja će biti dodano kasnije, uključujući i sadržaj "spašen" s prethodne stranice.

## Misija laboratorija

Laboratorij za računalne mreže, paralelizaciju i simulaciju (Computer Networks, Parallelization, and Simulation Laboratory, CNPSLab) je ustrojbena jedinica Odjela za informatiku osnovana s ciljem organizacije i izvođenja laboratorijskih vježbi iz kolegija: Računalne mreže 1, Računalne mreže 2, Upravljanje mrežnim sustavima, te sudjelovanja u izvođenju laboratorijskih vježbi iz kolegija Distribuirani sustavi i Paralelno programiranje na heterogenim sustavima. Laboratorij:

- brine o održavanju i nabavi laboratorijske opreme,
- organizira i provodi direktni rad sa studentima,
- predlaže inovacije u sadržaju povjerenih predmeta,
- organizira i predlaže znanstveno-istraživačke i razvojne projekte,

te time vodi brigu o unaprjeđenju praktične nastave i znanstveno-istraživačke djelatnosti. Uz navedeno, Laboratorij obavlja i druge poslove po nalogu pročelnika Odjela za informatiku, predstojnika Zavoda za komunikacijske sustave ili voditelja Katedre za mrežne sustave.

## Vizija laboratorija

Laboratorij za računalne mreže, paralelizaciju i simulaciju želi se unutar Odjela za informatiku Sveučilišta u Rijeci pozicionirati kao priznata ustrojbena jedinica po vlastitom nastavnom, znanstvenom i stručnom radu, koja laboratorijsku nastavu povjerenih predmeta izvodi u skladu s Europskim i svjetskim trendovima te sudjeluje u međunarodno priznatim znanstveno-istraživačkim i razvojnim projektima.

Nastavni, znanstveno-istraživački i stručni rad laboratorija izvodi se u tri međusobno izrazito povezana područja.

### Modeliranje i simulacija računalnih mreža

U području modeliranja i simulacije računalnih mreža CNPSLab koristi svoju računalnu i mrežnu infrastrukturu kako bi omogućio studentima rad s alatima za emulaciju, simulaciju i analizu svojstava računalnih mreža u okviru kolegija Računalne mreže 1 i Računalne mreže 2.

Znanstveno-istraživački i razvojni rad fokusiran je na:

- proširenje funkcionalnosti mrežnog simulatora ns-3 u smjeru modeliranja i simulacije kvarova i popravaka u mreži, s posebnim fokusom na modeliranje zavisnosti kvarove,
- implementaciju modela optičke telekomunikacijske mreže u mrežnom simulatoru ns-3, uključujući sustav za upravljanje mrežom,
- implementaciju algoritama za usmjeravanje i dodjelu valnih duljina u optičkoj telekomunikacijskoj mreži, te
- implementaciju različitih metoda zaštite i obnavljanja mreže u slučaju kvara.

### Distribuirani heterogeni sustavi

U području distribuiranih heterogenih sustava sustava CNPSLab omogućuju studentima korištenje računalne i mrežne infrastrukture laboratorija kako bi mogli raditi s alatima koji pomažu razvoj aplikacija temeljenih na standardu MPI i simulaciju distribuiranih sustava u okviru kolegija Distribuirani sustavi. U području heterogenih sustava omogućuje studentima rad s alatima koji pomažu razvoj aplikacija temeljenih na tehnologijama NVIDIA CUDA i OpenCL u okviru kolegija Paralelno programiranje na heterogenim sustavima, a povezivanje ta dva područja izvodi se kroz završne i diplomske radove.

Znanstveno-istraživački i razvojni rad fokusiran je na:

- optimizacija mrežnog simulatora ns-3 korištenjem distribuiranih (MPI) i heterogenih (CUDA C/C++, OpenCL) tehnologija paralelizacije, te
- istraživanje mogućnosti kombinirane primjene distribuiranog i heterogenog programskog modela za optimizaciju algoritama u domeni:

    - primijenjene matematike i računarstva, specifično optimizaciju algoritama na grafovima,
    - biotehnologije i istraživanja lijekova, specifično optimizaciju alata za molekularnu dinamiku.

### Softverska infrastruktura računala visokih performansi

CNPSLab održava vlastitu softversku infrastrukturu koja se trenutno zasniva na operacijskom sustavu Debian GNU/Linux. Pored navedene distribucije, za određene projekte i izvođenje nastave koriste se po potrebi i distribucije Debian GNU/kFreeBSD, [Fedora](https://fedoramagazine.org/fedora-computer-lab-university/) i FreeBSD. U okviru kolegija Upravljanje mrežnim sustavima studentima se nudi uvid u proces pripreme i održavanja softverske infrastrukture umreženog računala.

Znanstveno-istraživački i razvojni rad fokusiran je na:

- poboljšanje performansi mrežnog simulatora ns-3 na platformama GNU/Linux, GNU/kFreeBSD i FreeBSD,
- izučavanje sustava za izgradnju softvera i održavanje sustava waf za izgradnju mrežnog simulatora ns-3, te
- izučavanje razvojnih okruženja za paralelno programiranje NVIDIA Nsight i Eclipse PTP i drugih razvojnih alata zasnovanih na platformi Eclipse.

## Kako se uključiti

U nastavku je kratka poruka studentima zainteresiranim za sudjelovanje u znanstveno-istraživačkom radu laboratorija.

Naš znanstveno-istraživački program zasniva se uvelike na razvoju softvera, fokusirajući se uglavnom na području modeliranja i simulacije računalnih mreža, ali i primjeni paralenih, distribuiranih i heterogenih metoda programiranja u tom području. U širem kontekstu zainteresirani smo za primjenu distribuiranih i heterogenih metoda programiranja u područjima primijenjene matematike, računarstva te biotehnologije i istraživanja lijekova. Ako želite biti dio našeg programa laboratoriju istraživanja, dobrodošli ste, ali molimo vas da pored intenzivnog rada na kolegijima u području kojim se laboratorij bavi razmislite o sljedećim točkama.

- Potrebno je imati vještinu programiranja u C/C++-u i Pythonu, i želju za radom sa softverom kao sastavnim dijelom znanstvenog istraživanja. To uključuje iskustvo s objektno-orijentiranim dizajnom softvera, pisanjem testova, korištenjem i pisanjem API dokumentacije, radom sa sustavima za upravljanje verzijama, debuggerima i profilerima. Iskustvo s pojedinim razvojnim okruženjima, sustavima za praćenje grešaka, sustavima za kontinuiranu integraciju i ostalim pomoćnim alatima je korisno.
- Stvaranje paralelnih programa korištenjem tehnologija kao što su OpenMP, MPI i OpenCL na Linux/Unix distribuiranim sustavima (clusterima i gridovima) je vrlo korisno, ali nije preduvjet. Pored toga, iskustvo sa sustavima za upravljanje oblacima kao što su OpenStack i Eucalyptus je plus.
- Potrebno je imati iskustvo u radu u komandnolinjskom sučelju na Linuxu, FreeBSD-u, Mac OS X-u ili bilo kojem drugom operacijskom sustavu sličnom Unixu (iako međusobno nisu isti, dovojno su slični da se većina znanja s jednog može primijeniti na preostalima). Ako imate iskustva samo u radu s Windowsima, iako je ono općenito vrlo korisno, ono vam nažalost neće ovdje biti od velike pomoći.

Ako posjedujete ove vještine i entuzijazam za znanstveno-istraživačkim radom, i želite se pridružiti našem istraživačkom programu, molimo da voditelju laboratorija pošaljete e-mail koji uključuje:

- Opis vašeg istraživačkog interesa unutar našeg programa u 300 do 500 riječi.
- Primjer Linux/Unix softvera koji ste napisali; tar arhiva je dovoljna, ali poveznica na kod na Bitbucketu ili GitHubu je preferirana. Priložite detaljni opis softvera: što radi i kako. Ukoliko vaš kod koristi POSIX threads, OpenMP, MPI, OpenCL ili NVIDIA CUDA-u, to je plus.
