---
author: Vedran Miletić, Maja Grubišić
---

# Instalacija operacijskih sustava Fedora i CentOS korištenjem instalacijskog alata Anaconda

!!! hint
    Za više informacija proučite [Red Hat Enterprise Linux 7 Installation Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/installation_guide/index).

!!! todo
    Ovaj dio je vrlo sirov i treba ga temeljito pročistiti i dopuniti.

[Anaconda](https://fedoraproject.org/wiki/Anaconda) ([Wikipedia](https://en.wikipedia.org/wiki/Anaconda_(installer)), [službena dokumentacija](https://anaconda-installer.readthedocs.io/)) je instalacijski program koji koriste Fedora, Red Hat Enterprise Linux i neke druge distribucije, i napisana je u programskom jeziku Python. Ona dopušta korisniku instalaciju operativnog sustava na željeno računalo i takoder daje mogućnost nadogradnje postojeće instalacije starije verzije neke od distribucija.

Podržava instalaciju s lokalnog (npr. CD-a, DVD-a, USB flash diska i slike operacijskog sustava pohranjene na tvrdom disku) ili udaljenog izvora (npr. NFS-a, HTTP-a i FTP-a). Takoder podržava i [kickstart](https://pykickstart.readthedocs.io/) instalaciju kojom administrator može stvoriti jedan dokument koji sadrži odgovore na sva pitanja koja su postavljena prilikom instalacije, i ta metoda značajno olakšava i ubrzava instalaciju distribucije na više strojeva.

Izlaskom [Fedore 18](https://fedoraproject.org/wiki/Releases/18/FeatureList) Anaconda je [doživjela značajne promjene](https://fedoraproject.org/wiki/Anaconda/NewInstaller), a [verzije koje su uslijedile su nastavile s manjim promjenama i dodatnim značajkama](https://rhinstaller.wordpress.com/2018/03/28/anaconda-a-look-back-at-10-fedora-releases-with-new-ui/) (detaljnije o promjenama u Fedori [29](https://fedoramagazine.org/whats-coming-fedora-29-anaconda/), [28](https://fedoramagazine.org/whats-new-in-the-anaconda-installer-for-fedora-28/), [26](https://fedoramagazine.org/anaconda-fedora-26-new-features/)). Najveća promjena je što [grafičko sučelje sada koristi model središta i zubaca](https://rhinstaller.github.io/anaconda-addon-development-guide/sect-anaconda-hub-and-spoke.html) (engl. *hub and spoke*).

## Fedora, Red Hat Enterprise Linux i CentOS

[Fedora](https://fedoraproject.org/) je operacijski sustav koji kompanija [Red Hat](https://www.redhat.com/) razvija u suradnji sa zajednicom. Fedora cilja na entuzijaste i nudi podršku od strane zajednice.

Periodički, približno svakih 3 do 5 godina, Red Hat uzima tehnologiju razvijenu u suradnji sa zajednicom i od nje gradi operacijski sustav [Red Hat Enterprise Linux (RHEL)](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux) koji će moći [biti podržan zakrpama softvera približno 10 godina nakon izlaska](https://access.redhat.com/support/policy/updates/errata). Red Hat naplaćuje podršku za Enterprise Linux i zbog toga polaže pravo na svoj zaštitni znak, no sam softver je besplatan (i slobodan i otvorenog koda).

[CentOS](https://centos.org/) je RHEL koji ima drugačiju robnu marku (engl. *branding*), što omogućuje neograničenu distribuciju bez kršenja Red Hatovog autorskog prava (engl. *copyright*) i zaštitnog znaka (engl. *trademark*). CentOS cilja na poslovne i akademske korisnike koji nemaju potrebu ili želju za plaćanjem Red Hatove podrške, a sam nudi podršku bez naknade (i bez garancija) od strane zajednice okupljene oko projekta.

Dokumentacija koju vrijedi konzultirati je:

- [Red Hat Enterprise Linux dokumentacija](https://access.redhat.com/products/red-hat-enterprise-linux/)

    - CentOS je 100% kompatibilan s istom verzijom RHEL-a pa sva dokumentacija RHEL-a vrijedi, a nadopunjuje je [CentOS Wiki](https://wiki.centos.org/)

- [Fedora dokumentacija](https://docs.fedoraproject.org/)

    - velikim dijelom, ali ne sasvim, odnosi se i na RHEL/CentOS
    - dio dokumentacije je neovisan o verziji, a kad je ovisan o verziji vrijede sljedeća pravila o [odnosu](https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux#Relationship_with_Fedora) [Fedore](https://docs.fedoraproject.org/en-US/quick-docs/fedora-and-red-hat-enterprise-linux/) i [RHEL-a](https://access.redhat.com/articles/3078):

        - RHEL/CentOS 5 ~= [Fedora Core 6](https://docs.fedoraproject.org/en-US/releases/f6/)
        - RHEL/CentOS 6 ~= [Fedora 12](https://docs.fedoraproject.org/en-US/releases/f12/) i [Fedora 13](https://docs.fedoraproject.org/en-US/releases/f13/)
        - RHEL/CentOS 7 ~= [Fedora 19](https://docs.fedoraproject.org/en-US/releases/f19/) i [Fedora 20](https://docs.fedoraproject.org/en-US/releases/f20/)
        - RHEL/CentOS 8 ~= [Fedora 28](https://docs.fedoraproject.org/en-US/releases/f28/)

## Instalacije Fedore korištenjem Anaconde

Da bi mogli koristiti operacijski sustav, moramo ga instalirati. To ćemo na vježbama učiniti *dva puta*, i to redom koristeći instalacijske medije:

1. [Fedora Server](https://getfedora.org/en/server/), uobičajena instalacija u grafičkom okruženju.
1. [Fedora Workstation](https://getfedora.org/en/workstation/), živa instalacija u grafičkom okruženju.

Instalacije ćemo vršiti u KVM/QEMU virtualnoj mašini koja koristi klasični BIOS ili UEFI.

Namjera je da steknete što je više moguće različit pogled na istu stvar, obzirom da su programi koji vrše instalaciju u ova dva slučaja na površini dosta različiti, ali obavljaju istu funkciju. Kada jednom steknete uvid u način funkcioniranja programa za instalaciju sustava, vrlo ćete se lako koristiti i drugim, uglavnom vrlo sličnim alatima.

Razlikujemo dvije osnovne vrste instalacije:

1. uobičajena instalacija (engl. *default install*), češće korištena kod instalacije poslužitelja ili kad god je potrebna sloboda u izboru paketa (dijelova sustava) koji će se instalirati, kod koje se pokreće minimalan sustava i on vrši instalaciju paketa iz svog repozitorija na disk računala, te
1. živa instalacija (engl. *live install*), češće korištena od strane običnih korisnika, kod koje se sustav prvo pokrene sa eksternog medija (npr. CD-a, DVD-a, USB diska), a zatim se slika pokrenutog sustava kopira na disk računala.

Kod uobičajene instalacije moguće je napraviti odabir paketa (dijelova sustava) koji će se instalirati, ali zato instalacija sustava najčešće traje duže jer se vrši instalacija svakog paketa pojedinačno. To je jedna od osnovnih prednosti žive instalacije; naime, instalacija u većini slučajeva traje svega nekoliko minuta. Međutim, kako se na disk kopira slika pokrenutog sustava, nije moguće odabrati koji će se dijelovi sustava instalirati, a koji ne.

## Živa instalacija (Fedora Workstation)

Kod [žive instalacije](https://fedoraproject.org/wiki/LiveOS_image) se prvo pokreće boot učitavač [ISOLINUX](https://wiki.syslinux.org/wiki/index.php?title=ISOLINUX). U slučaju da korisnik ne pritisne nijednu tipku, on provjerava medij i pokreće Fedoru nakon 60 sekundi. U slučaju da korisnik želi nešto drugo, ima izbor:

- pokretanje sustava (`Start Fedora-Workstation-Live`),
- provjera medij i pokretanje sustava (`Test this media & start Fedora-Workstation-Live`),
- rješavanje problema (`Troubleshooting`), npr. pokrene sustav bez uključivanja postavljanja rezolucije zaslona od strane jezgre (parametar komandne linije jezgre `nomodeset`).

Kod pokretanja sustava, jezgra pokreće [Plymouth](https://fedoramagazine.org/howto-change-the-plymouth-theme/) (parametra komandne linije jezgre `rhgb`, kratica od Red Hat Graphical Boot), koji vizualno neatraktivan skrolajući tekst zamjenjuje vizualno atraktivnim logotipom Fedore koji se polako pojavljuje ili, u slučaju da grafički procesor ne podržava visoku rezoluciju ili se ne koristi postavljanje rezolucije zaslona od strane jezgre, jednostavnim ekranom koji pomoću tri točke označava učitavanje operacijskog sustava.

U slučaju da je odabran provjera medija, ona će biti izvršena prije nastavka pokeranja sustava. Iznimno, ako provjera pronađe grešku na instalacijskom mediju, pokretanje sustava, a time i instalacija, će biti prekinuti.

U pozadini Plymouthovih efekata pojavljuju se sistemske poruke, a u slučaju da se dogodi nekakva greška, ona će se pokazati na ekranu (Plymouth zbog toga stalno "sluša" jezgru i očekuje vijesti o greškama).

Zatim se pokreće poslužitelj zaslona Wayland ili X, a na njemu aplikacija [GNOME Display Manager (GDM)](https://wiki.gnome.org/Projects/GDM), koja služi za prijavu korisnika u grafičkom sučelju. Mi GDM ne vidimo, jer je postavljena automatska prijava zadanog korisnika bez čekanja na korisničku intervenciju.

GDM pokreće standardni [GNOME Desktop Environment](https://www.gnome.org/) i moguće je pokrenuti sve njegove programe ([tekst editor](https://wiki.gnome.org/Apps/Gedit), [terminal](https://wiki.gnome.org/Apps/Terminal), [video player](https://wiki.gnome.org/Apps/Videos) itd.), kao da je sustav instaliran. Ovo služi da korisnici mogu isprobati radi li sustav na njihovom računalu kako treba prije nego se odluče na instalaciju. Naravno, zbog manje brzine čitanja sa CD-a, DVD-a i USB flash diskova realno je za očekivati da sustav radi nešto sporije nego kad je instaliran.

Sustav koji se pokrene bit će na engleskom jeziku. Korisniku se nudi isprobavanje Fedore i instalacija na čvrsti disk. U slučaju da korisnik odabere isprobavanje, instalaciju je uvijek moguće pokrenuti iz Dasha (niza ikona koje se pojave na lijevoj strani radne površine nakon klika na `Activities`).

U slučaju da korisnik odabere instalaciju Fedore, pokrenuti će se instalacijski program Anaconda. Sama instalacija ima nekoliko koraka, koje redom opisujemo u nastavku.

- `Ekran dobrodošli i odabir jezika`: ako smo povezani na internet i sustav uspije prepoznati našu lokaciju na temelju [baze GeoIP](https://www.maxmind.com/en/geoip2-databases), Anaconda će nam kao prvi izbor ponuditi hrvatski jezik.
- `Kratak pregled instalacije`: ovo je središte u modelu središta i zubaca. Omogućuje se prijelaz na odabir tipkovnice, podešavanje vremena i datuma, instalacijskog odredišta te mreže i imena domaćina.

    - `Raspored tipkovnice`: hrvatski će već biti odabran ako je odabran jezik instalacije hrvatski, eventualno se može dodati engleski.
    - `Vrijeme i datum`: vremenska zona Europa/Zagreb će već biti odabrana ako je odabran jezik instalacije hrvatski. Mrežno vrijeme je uključeno u zadanim postavkama, što znači da će sat na računalu biti točan ako se računalo nakon instalacije bar jednom poveže na internet i izvrši sinkronizaciju s [nekim od poslužitelja vremena putem Network Time Protocola (NTP)](https://www.ntppool.org/).
    - `Odredište instalacije`: odabiru se lokalni standardni diskovi koji će se koristiti u instalaciji, a nakon toga, osim u slučaju kad je odabrano automatska konfiguracija pohrane podataka, otvara dodatno sučelje koje omogućuje finiju konfiguraciju particija koje će biti stvorene.

        - Ručno particioniranje omogućuje automatsko stvaranje particija korištenjem standardnih particija, [Btrfs-a](https://btrfs.wiki.kernel.org/), [LVM-a](https://www.sourceware.org/lvm2/) ili LVM-ovog odmjerenog dodjeljivanja. Osim toga, moguće je ručno napraviti particiju po particiju.
        - Napredno ručno particioniranje koristi [Blivet-GUI](https://fedoraproject.org/wiki/Blivet-gui) ([Blivet](https://fedoraproject.org/wiki/Blivet) je biblioteka za upravljanje uređajima za pohranu podataka), alat [dostupan u Fedori od verzije 21](https://fedoramagazine.org/manage-your-partitions-like-in-anaconda-with-blivet-gui/).

            Ovo je vjerojatno najteži korak u instalaciji jer zahtjeva poznavanje datotečnih sustava i načina na koji su strukturirani diskovi na računalu na koje želimo instalirati Fedoru. Ako nismo pažljivi i ne znamo što radimo, postoji mogućnost brisanja particije koja sadrži druge operacijske sustave, u praksi specijalno Windowse.

            U slučaju da disk nema particijsku tablicu (što je slučaj kod prvog korištenja novih fizičkih ili virtualnih diskova), ona se stvara pritiskom na gumb `Nova particijska tablica`.

            Prisjetimo se da disk može imati ili 4 primarne particije ili 3 primarne i 1 sekundarnu (koja tada u sebi može imati 64 logičke).

            Zatim je omogućeno dodavanje i brisanje particija. Napraviti ćemo uobičajeni raspored, koji se za naš virtualni disk od 20 sastoji od:

    - particije za `/boot` direktorij od 500 MiB, koja sadrži datoteke za pokretanje sustava (modernija preporuka, i za sada nerealno velika; nekad se preporučalo od nekoliko desetaka do najviše 100),
    - particije za `/` direktorij od svog preostalog prostora (u našem slučaju 18.5 GiB),
    - `swap` particije, koja je prostor za virtualnu memoriju, dvostruke veličine radne memorije računala (u našem slučaju 2 GiB, obzirom da smo virtualnom stroju dodijelili 1 GiB memorije).

    Još jedan vrlo često korišteni raspored je s direktorijem `/home` izdvojenim na posebnoj particiji, pa imamo:

    - `/boot` -- 500 MiB,
    - `/` -- 8 GiB (čak i to je u većini slučajeva nerealno veliko; Fedora nakon instalacije zauzima oko 3 GiB),
    - `/home` -- sve ostalo (u našem slučaju 10.5 GiB),
    - `swap` -- 2 GiB.

    Za `/boot` particiju ćemo koristiti datotečni sustav `ext4` da bude u skladu s ostalima, ali mogli bi koristiti i njegovu stariju inačicu `ext2` jer ona ne treba dnevničenje (kako je relativno male veličine, provjera cijele particije traje vrlo kratko). Za ostale particije (`/` i `/home`) koristit ćemo datotečni sustav `ext4`. Mogu se koristiti i drugi datotečni sustavi koji podržavaju dnevničenje, ali treba biti svjestan da nisu svi podjednako dobar odabir. Jedan od boljih konkurenata datotečnom sustavu `ext4` je `xfs`, koji je inače i zadani datotečni sustav u poslužiteljskoj inačici Fedore, CentOS-u i RHEL-u.

    Pritiskom na gumb `Instaliraj odmah` instalacija započinje i više nema povratka. Promjene se zapisuju na disk, i što je učinjeno, učinjeno je. Zatim se, ukoliko sve prođe dobro, događa kopiranje slike pokrenutog sustava na disk. Za to vrijeme mi nastavljamo konfigurirati sustav.

    - `Mreža i ime domaćina`: postavlja se ime domaćina po želji, zadana konfiguracija mreže (dohvaćanje adrese pomoću DHCP-a) je u redu.

## Uobičajena instalacija (Fedora Server)

!!! todo
    Ovaj dio treba napisati.
