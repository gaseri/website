---
author: Luka Vretenar, Vedran Miletić
---

# Upravljanje uslugama operacijskog sustava

!!! hint
    Za dodatne primjere naredbi proučite [stranicu systemd na ArchWikiju](https://wiki.archlinux.org/title/Systemd) i [Chapter 10. Managing Services with systemd](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/chap-managing_services_with_systemd) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/index).

- moderni operacijski sustavi bazirani na jezgri Linux koriste [systemd](https://systemd.io/) za upravljane procesima, specifično pokretanje usluga kod podizanje sustava
    - relativno novi alat, razvoj započeo Lennart Poettering 2010. godine tekstom [Rethinking PID 1](https://0pointer.de/blog/projects/systemd.html)
    - koristi značajke specifične za Linux jezgru kao što su [kontrolne grupe](https://en.wikipedia.org/wiki/Cgroups) (engl. *control groups*, cgroups), [signalfd](https://en.wikipedia.org/wiki/Event_loop), [epoll](https://en.wikipedia.org/wiki/Epoll) i druge
    - radi puno više od običnog sustava za baratanje uslugama: praćenje događaja i stvaranje zapisnika, konfiguracija lokalnih i regionalnih postavki, konfiguracija datuma i vremena, konfiguracija mreže, konfiguracija DNS-a, praćenje zauzueća memorije i drugo
        - prema autoru, cilj je spriječiti fragmentaciju (koju autor politički nekorektno naziva "balkanizacijom") Linuxa sučelja, unificirati način rada između različitih distribucija i precizno definirati što Linux jest
    - predmet [brojnih](https://www.zdnet.com/article/linus-torvalds-and-others-on-linuxs-systemd/) [debata](https://wiki.debian.org/Debate/initsystem/systemd), [kritika](https://ewontfix.com/14/) i [inženjerskih](https://web.archive.org/web/20150101045521/https://plus.google.com/+LennartPoetteringTheOneAndOnly/posts/VUzeRLf5g5m) [šala](https://desuarchive.org/g/thread/44805517)
- najpopularniji prethodno korišteni sustav te vrste `sysvinit` izlazi iz prakse i vrijedan je samo spomena; postoji [šalabahter](https://fedoraproject.org/wiki/SysVinit_to_Systemd_Cheatsheet) za prebacivanje između ta dva sustava

## Systemd ekosustav

Skup programa koji se naziva systemd zapravo sadrži veći broj međusobno povezanih i donekle neovisnih alata, od kojih su najznačajniji:

- `systemd` init proces, to je prvi proces koji pokreće jezgra Linux nakon učitavanja i on je zadužen za pokretanje i upravljanje životnim vijekom drugih procesa
- systemd usluga, koje su datoteke za definiranje zahtjeva, redoslijeda i instrukcija za pokretanje ostalih procesa potrebnih za rad sustava
- alata `systemctl`, za pregled i upravljanje stanjima definiranih usluga
- alata `journalctl`, za pregled sustavskih događaja i događaja vezanih za pojedinu uslugu

!!! admonition "Zadatak"
    Proučite na [systemdovim stranicama na freedesktop.org-u](https://www.freedesktop.org/wiki/Software/systemd/) dio *Spelling* koji govori o imenovanju alata, a zatim pokušajte objasniti zašto se osnovni alat za upravljanje `systemd`-om naziva `systemctl`.

## Pokretanje sustava

Pokretanje sustava razlikuje se u situaciji kada se koristi SysVinit od situacije kada se koristi Systemd.

### Pokretanje sustava korištenjem SysVinita

- [init](https://en.wikipedia.org/wiki/Init) (naredba `init`) pokreće usluge na operacijskom sustavu
    - BSD-style vs. SysV-style
    - razina pokretanja (engl. *runlevel*) opisuje stanje računala
        - sastoji se od usluga (engl. *services*)
        - razine 0, 1, 2, 3, 4, 5, 6, S/s
        - naredba `telinit` postavlja razinu pokretanja na zadanu
    - rezervirane razine
        - 0 -> isključivanje računala (engl. *halt*)
        - 1 -> jednokorisnički način rada (engl. *single-user mode*) u kojem se može prijaviti samo `root` korisnik
        - 6 -> ponovno pokretanje računala (engl. *reboot*)
    - najčešće se koriste razine pokretanja 3 i 5, razina pokretanja 4 se ne koristi

Kada se za init koristi SysVinit vrijedi:

- razina je definirana `:initdefault:` u `/etc/inittab`
- usluge su u `/etc/init.d/`
- ostale naredbe su u `/etc/rc.local`
- poveznice na usluge po razinama pokretanja su u `/etc/rc[0123456].d/`

!!! admonition "Zadatak"
    - Pokreće li se usluga `apache2` u razinama pokretanja 1, 2 i 3?
    - Pokreće li se usluga `postgresql` prije ili nakon procesa `exim4` u razinama pokretanja 3 i 5?

### Systemd jedinka cilja pokretanja

Systemd jedinka (engl. *unit*) je konfiguracijska datoteka koja se koristi za implementaciju usluga i ciljeva pokretanja:

- usluge koje pokreću procese demone definirane su `*.service` datotekama,
- ciljevi pokretanja definirani su `*.target` datotekama.

Alat `systemd` podiže sustav u predefinirani cilj koji se sastoji od određenih pokrenutih usluga. Predefinirani ciljevi pokretanja su:

- `poweroff.target` (pandan SysV `runlevel0'`)
- `reboot.target` (pandan SysV `runlevel6`)
- `multi-user.target` (pandan SysV `runlevel[234]`)
- `graphical.target` (pandan SysV `runlevel5`)
- `rescure.target` (pandan SysV `runlevel1`)
- `emergency.target`

U općem slučaju `systemd` podiže sustav u stanje `default.target` koji tipično pokazuje na `graphical.target`. Sustav će se pokrenuti do grafičkog sučelja i omogućiti prijavu preko istog samo ako je na njemu prisutan neki od [upravitelja prikaza](https://en.wikipedia.org/wiki/X_display_manager) kao što su [Simple Desktop Display Manager (SDDM)](https://en.wikipedia.org/wiki/Simple_Desktop_Display_Manager) i [GNOME Display Manager (GDM)](https://en.wikipedia.org/wiki/GNOME_Display_Manager).

Maredbom `systemctl isolate TARGET` moguće je prebaciti sustav u željeni cilj pokretanja.

!!! admonition "Zadatak"
    Provjerite koji je cilj pokretanja trenutno aktivan na sustavu. Prebacite se u cilj pokretanja `multi-user.target`, a zatim usporedite popis procesa. Prebacite se u cilj pokretanja `reboot.target` da provjerite radi li ono što očekujete.

- sve konfiguracijske datoteke i datoteke koje opisuju usluge se nalaze u direktorijima `/lib/systemd` i `/etc/systemd`
    - systemd prvo traži datoteku u `/lib/systemd`, a onda u `/etc/systemd`, što omogućuje nadjačavanje sustavske verzije datoteke u `/lib/systemd` s vlastitom, prilagođenom verzijom u `/etc/systemd`
    - specijalno, usluge se nalaze u direktorijima `/lib/systemd/system` i `/etc/systemd/system`

!!! hint
    Podsjetimo se da već desetak godina većina distribucija Linuxa [spaja prvi i drugi nivo hijerarhije datotečnog sustava](https://www.freedesktop.org/wiki/Software/systemd/TheCaseForTheUsrMerge/), odnosno da su direktoriji `/bin`, `/lib` (`/lib64`) i `/sbin` samo simboličke poveznice na `/usr/bin`, `/usr/lib` (`/usr/lib64`) i `/usr/sbin`. Posljedično, putanje `/lib/systemd/system` i `/usr/lib/systemd/system` su ekvivalentne.

!!! admonition "Zadatak"
    Instalirajte Apache ga nemate već instaliranog. Apachejeva usluga naziva se `http.service`; prvo pronađite tu datoteke na datotečnom sustavu, a zatim je kopirajte na mjesto na kojem ćete je moći uređivati tako da systemd koristi uređenu verziju, a izvorna verzija ostane sačuvana drugdje na datotečnom sustavu.

## Upravljanje pojedinom uslugom

Baratanje pojedinom uslugom razlikuje se u situaciji kada se koristi SysVinit od situacije kada se koristi Systemd.

### Upravljanje uslugom korištenjem SysVinita

- `/etc/init.d/usluga start` pokreće uslugu, ekvivalentno `service usluga start`
- `/etc/init.d/usluga stop` zaustavlja uslugu, ekvivalentno `service usluga stop`
- `/etc/init.d/usluga status` ispisuje stanje usluge, ekvivalentno `service usluga status`
- naredba `service --status-all` prikazuje status svih usluga

!!! admonition "Zadatak"
    - Provjerite status usluga `ntpd` i `ntpdate`, a zatim ih zaustavite i ponovno pokrenite.
    - Naredba `netstat -a` daje, između ostalog, popis otvorenih lokalnih portova (konekcije u stanju `LISTEN`). Otkrijte koje su to usluge u našem slučaju. Ima li `ntpd` u popisu nakon zaustavljanja daemona?

!!! admonition "Zadatak"
    Iskoristite `yum` da bi instalirali `vsftpd`, a zatim ga pokrenite. Iskoristite `ftp` dostupan na računalu na kojem radite da bi se povezali na poslužitelj (sama prijava vam neće raditi; to zasad zanemarite).

### Systemd naredba `systemctl`

- pregled stanja pojedine usluge `USLUGA` možemo izvršiti naredbom `systemctl status USLUGA`
- upravljati stanjem usluge `USLUGA` možemo slijedećim naredbama:
    - `systemctl start USLUGA`
    - `systemctl stop USLUGA`
    - `systemctl restart USLUGA`
    - `systemctl reload USLUGA`

Pregled svih praćenih usluga i njihovih stanja se vrši pokretanjem alata `systemctl`.

!!! admonition "Zadatak"
    Pronađite dva poslužiteljska programa (izuzev Apacheja) po želji iz [Category:Servers na ArchWikiju](https://wiki.archlinux.org/title/Category:Servers). Instalirajte ih i isprobajte mogu li se pokrenuti i zaustaviti.

## Upravljanje uslugama pojedinog cilja pokretanja

Baratanje uslugama cilja ili razine pokretanja razlikuje se u situaciji kada se koristi SysVinit od situacije kada se koristi Systemd.

### Upravljanje uslugama citalja pokretanja korištenjem SysVinita

- naredba `chkconfig` omogućuje da se uključuje ili isključuje pojedine usluge u pojedinim razinama pokretanja
- `chkconfig --list usluga` ispisuje jesu li usluge uključene ili isključene u pojedinim razinama pokretanja
- `chkconfig usluga on --level 2345` uključuje uslugu za pokretanje u razinama 2, 3, 4 i 5
- `chkconfig usluga off --level 01` isključuje uslugu za pokretanje u razinama 0 i 1

!!! admonition "Zadatak"
    Uključite uslugu `vsftpd` u runlevelima 3, 4 i 5.

### Upravljanje uslugama citalja pokretanja korištenjem Sytemda

- usluge koji žele biti pokrenute u određenom cilju pokretanja moraju o istome ovisiti
- `systemctl enable` uključuje pokretanje usluge u nivoima navedenim u dijelu `[Install]` pod `WantedBy=`
- `systemctl disable` isključuje pokretanje usluge

## Systemd jedinka usluge

Usluge su implementirane kao *service units*, npr. `acpid.service`, `sshd.service`. Detaljnu specifikaciju moguće je pronaći u man stranici `systemd.service(5)`.

!!! admonition "Zadatak"
    Uredite Apachejevu systemd uslugu tako da opis umjesto `Apache Web Server` bude `Apache HTTP Server` i da se pokreće tek nakon što se računalo poveže na internet (**uputa:** provjerite popis ciljeva i pronađite cilj koji to osigurava).
