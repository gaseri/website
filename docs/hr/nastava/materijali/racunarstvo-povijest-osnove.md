---
author: Vedran Miletić
---

# Povijesni pregled razvoja računala i temelji računarske znanosti

## Računarstvo i računarska znanost

- Uobičajena percepcija: "nešto što ima veze s računalima", "znanost o računalima"

    - FER-ovi [preddiplomski](https://www.fer.unizg.hr/studiji/fer3/racunarstvo) i [diplomski studiji](https://www.fer.unizg.hr/studiji/dipl/rac/rz)

- izučava procese koji djeluju na podatke i koji se sami mogu prikazati kao podaci u obliku programa
- [brojne discipline](https://en.wikipedia.org/wiki/Computer_science#Fields): neke su teoretske (npr. računska teorija složenosti), neke praktične (npr. računalna grafika), neke se bave dizajnom sustava (npr. računalne mreže, paralelno i distribuirano računarstvo)
- nije isključivo znanost o računalima, mnogo je starija je od samih računala

    - "*Computer science is no more about computers than astronomy is about telescopes."* ([Michael Fellows](https://www.mrfellows.net/))
    - nije ni znanost u užem smislu, spada pod inženjerstvo

### Primjeri problema kojima se računarstvo bavi

- [smanjenje veličine konzolaških igara](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/bloat2.pdf) (spada u disciplinu koja se naziva [dizajn program prevoditelja](https://holub.com/compiler/))
- [procjena koliko je moguće poboljšati performanse zbrajanja i množenja u tabličnom kalkulatoru na novoj generaciji AMD-ovih procesora](https://www.techenablement.com/libreoffice-opencl-acceleration-for-the-masses-intel-vs-amd-gpu-performance/) (optimizacija programskog koda)
- [procjena koliko bi koštao razvoj softvera temeljenog na poznatim algoritmima](http://softwarecost.org/tools/COCOMO/) u domeni računalne dinamike fluida koji nudi zadovoljavajuću točnost za primjenu u industriji automobila (softversko inženjerstvo)
- [grafički prikaz ljudske kože](https://techxplore.com/news/2020-05-gaming-characters-realistic-skin.html) (računalna grafika)
- na koji način poboljšati performanse procesora unutar iPhonea za barem 20% kako bi aplikacije nove generacije radile prihvatljivo brzo (arhitektura računala)
- [prilagodba računalne igre za povećanje zaronjenja igrača](https://publications.lib.chalmers.se/records/fulltext/111921.pdf) (engl. *player immersion*) (interakcija čovjeka i računala)
- mogu li Ante i Frane komunicirati preko veze koja se prisluškuje, ali tako da njihov razgovor nije moguće dešifrirati (kriptografija)
- može li se izračunati koji od prvih n prirodnih brojeva su prosti brojevi i s kojim računalnim resursima (teorija izračunljivosti)
- ako se može izračunati koji od prvih n prirodnih brojeva su prosti brojevi u jednom danu, može li se korištenjem 24 umrežena računala isto izračunati u jednom satu (paralelno i distribuirano računarstvo)

## Informatika

- znanost o obradi i prijenosu podataka i informacija (informacija je podatak stavljen u kontekst koji time dobiva značenje)

    - u Hrvatskoj je klasificirana u području društvenih znanosti, polju informacijskih i komunikacijskih znanosti (prema [Pravilniku o znanstvenim i umjetničkim područjima, poljima i granama](https://narodne-novine.nn.hr/clanci/sluzbeni/2008_07_78_2563.html))

- pojam iz njemačkog govornog područja (1957. godina, Karl Steinbuch), informacija + automatika

## Sklopovska oprema (hardver)

[Sklopovska oprema (hardver)](https://en.wikipedia.org/wiki/Computer_hardware) (engl. *hardware*) uključuje fizičke, opipljve dijelove računala kao što su kućište, osnovni procesor, grafički procesor, monitor, tipkovnica, zvučna kartica itd.

### Matična ploča

[Matična ploča](https://en.wikipedia.org/wiki/Motherboard) (engl. *motherboard*) je osnovna ploča u računalu koja omogućuje komunikaciju između komponenata računalnog sustava.

### Osnovni procesor

[Osnovni procesor](https://en.wikipedia.org/wiki/Central_processing_unit) (engl. *central processing unit*, CPU) je elektronički krug u računalu koji izvodi instrukcije programa kao što su aritmetika, logika, upravljačke i ulazno-izlazne operacije.

Pojam procesora se mijenjao tijekom godina, ali uvijek ima [aritmetičko-logičku jedinicu](https://en.wikipedia.org/wiki/Arithmetic_logic_unit) (engl. *arithmetic logic unit*) i registre. Vremenom je dobio i:

- [jedinicu za operacije s pomičnim zarezom](https://en.wikipedia.org/wiki/Floating-point_unit) (engl. *floating point unit*),
- nekoliko nivoa priručne memorije (engl. *cache*),
- memorijski kontroler za upravljanje glavnom memorijom i dio ulazno-izlaznih kontrolera,
- u novije vrijeme i grafički procesor (npr. [AMD Accelerated Processing Unit](https://en.wikipedia.org/wiki/AMD_Accelerated_Processing_Unit) i [Intel Core i3, i5, i7](https://en.wikipedia.org/wiki/Intel_Core)).

### Grafički procesor

[Grafički procesor](https://en.wikipedia.org/wiki/Graphics_processing_unit) (engl. *graphics processing unit*, GPU) je elektronički krug koji stvara slike u [međuspremniku okvira (engl.](https://en.wikipedia.org/wiki/Framebuffer) (engl. *framebuffer*) namijenjene za prikaz na zaslonu.

### Memorija

[Glavna memorija](https://en.wikipedia.org/wiki/Computer_data_storage#Primary_storage), često samo memorija, je jedina izravno memorija dostupna osnovnom procesoru. U njoj se nalaze programi koje procesor dohvaća i izvršava te podaci na kojima se operacije vrše.

### Pohrana podataka

[Pohrana podataka](https://en.wikipedia.org/wiki/Computer_data_storage#Secondary_storage) je memorija koja nije izravno dostupna osnovnom procesoru, ali se može doseći putem ulazno-izlaznih kanala. Programi i podaci iz pohrane podataka prvo idu u glavnu memoriju, a tek se onda koriste.

### Mrežni adapter

[Mrežni adapter](https://en.wikipedia.org/wiki/Network_interface_controller) je uređaj u računalu koji služi za povezivanje računala na računalnu mrežu. Mrežni adapter omogućuje komunikaciju računala putem računalne mreže, koja može biti povezana ili kabelima ili bežično.

### Napajanje

[Napajanje](https://en.wikipedia.org/wiki/Power_supply_unit_(computer)) pretvara izmjeničnu struju (uglavnom iz utičnice) u istosmjernu struju potrebnu računalnim komponentama.

## Programska oprema (softver)

[Programska oprema (softver)](https://en.wikipedia.org/wiki/Computer_hardware) (engl. *software*) je skup računalnih instrukcija i podataka koji kaže računalu kako treba raditi.

### Izvršni softver

[Izvršni softver](https://en.wikipedia.org/wiki/Application_software) (engl *application software*) je softver napravljen da izvede posebne funkcije koje nisu dio osnovnog rada računala. To su, primjerice, obrada teksta, tablična kalkulacija, [3D modeliranje i animacija](https://youtu.be/p2qgIlpcZ1w), računovodstvo, pregledavanje weba, komunikacija putem računalne pošte i izvođenje snimljenog audiovizualnog sadržaja. U izvršni softver spadaju i [računalne](https://youtu.be/dS-ti6XjT4A) [igre](https://youtu.be/GY8QmXnIZEw). Nama interesantni znanstveni softveri su također vrsta izvršnih softvera.

### Sustavni softver

[Sustavni softver](https://en.wikipedia.org/wiki/System_software) (engl *system software*) je softver koji služi kao platforma za drugi softver.

- Operacijski sustav: npr. [Manjaro](https://manjaro.org/), [Ubuntu](https://ubuntu.com/), [FreeBSD](https://www.freebsd.org/), [macOS](https://www.apple.com/macos/), [Microsoft Windows](https://www.microsoft.com/windows/)
- Upravljački programi uređaja: npr. [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx), [AMD Radeon Software](https://www.amd.com/en/technologies/radeon-software), [Mellanox EN Driver](https://www.mellanox.com/products/ethernet-drivers/linux/mlnx_en)
- Pomoćni alati: npr. [systemd](https://www.freedesktop.org/wiki/Software/systemd/), [GCC](https://gcc.gnu.org/)

### Zlonamjerni softver

[Zlonamjerni softver](https://en.wikipedia.org/wiki/Malware) je softver namjerno dizajniran da napravi štetu računalu ili računalnoj mreži.
