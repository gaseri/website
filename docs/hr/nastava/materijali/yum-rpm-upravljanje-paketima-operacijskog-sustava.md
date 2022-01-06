---
author: Vedran Miletić
---

# Upravljanje paketima

!!! hint
    Za više informacija proučite [Part III. Installing and Managing Software](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/part-installing_and_managing_software) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/index).

- softverski paket (engl. *software package*)

    - kraće *paket*; arhiva koja sadrži datoteke aplikacije, biblioteke ili dokumentacije (npr. `firefox`, `firefox-branding`, `firefox-gnome-support`)
    - najpoznatiji `.rpm` i `.deb`

- međuovisnost

    - međuovisnost kod prevođenja (engl. *compile-time dependancy*), programi i/ili bibliotečne datoteke koje program zahtjeva za uspješno prevođenje (npr. `flex`, `bison` ili `iostream`)
    - međuovisnost kod izvršavanja (engl. *run-time dependancy*), programi i/ili bibliotečne datoteke koje program zahtjeva dok radi

## Osnove rada s upraviteljem paketa niže razine

- upravitelj paketima niže razine

    - koristi se za stvaranje, instalaciju, deinstalaciju i konfiguriranje paketa
    - *"dependency hell"* (pakao međuovisnosti?)
    - najpoznatiji su [RPM Package Manager](https://rpm.org/), naredba `rpm`, i [dpkg](https://wiki.debian.org/Teams/Dpkg), naredba `dpkg`

!!! admonition "Zadatak"
    Proučite `rpm(8)` (notacija za `man 8 rpm`), specifično dio o naredbama za pretraživanje (engl. query, `rpm -q`).

    - Provjerite koji paketi koji u imenu sadrže niz znakova `python` su instalirani na sustavu. Ne zaboravite da ljuska Bash smatra `*` specijalnim znakom te ga je potrebno escapeati prefiksom `\`.
    - Pronađite koje datoteke sadrži paket `openssh`.
    - Pronađite koji paket sadrži datoteku `/bin/ls`, a koji datoteku `/usr/bin/scp`. Objasnite što se dogodi ako izostavite putanju.

## Osnove rada s upraviteljem paketa više razine

- upravitelj paketima više razine

    - koristi se za pretraživanje, dohvaćanje i nadogradnju paketa, nalaženje paketa koji zadovoljavaju potrebne međuovisnosti

        - `rpm` i `dpkg` ne znaju kako doći do paketa koji im trebaju da zadovolje međuovisnosti pa nad njima rade upravitelji paketa više razine koji provjeravaju postoje li u repozitoriju paketi koji su potrebni za instalaciju programa

    - najpoznatiji su [Yellowdog Updater Modified](http://yum.baseurl.org/), kraće yum, nareba `yum`, njegov nasljednik [Dandified YUM](https://github.com/rpm-software-management/dnf), kraće DNF, naredba `dnf`, i [Advanced Packaging Tool](https://wiki.debian.org/Apt), kraće APT, naredbe `apt-get`, `apt-cache`, `apt-mark`, `apt-key`, `apt-config`, ..., a od verzije 1.0 i `apt`
    - komandnolinijsko i ncurses sučelje za APT i dpkg [aptitude](https://wiki.debian.org/Aptitude) podržava većinu APT naredbi, npr. `install`, `remove`, `purge`, `update`, `upgrade`, `show`, ali donosi i neke nove značajke, npr. `changelog`

- [repozitorij softvera](https://en.wikipedia.org/wiki/Software_repository) (engl. *software repository*)

    - skup paketa i metapodataka o njima, najčešće udaljen i dostupan putem Interneta
    - sadrži metapodatke o paketima (naziv paketa, opis sadržaja, popis međuovisnosti, ...) i same pakete
    - repozitoriji softvera uključeni na sustavu navedeni su u `/etc/yum.repos.d`

- popis paketa

    - doslovno popis paketa, dio repozitorija na poslužitelju, mijenja se sukladno promjenama u skupu paketa koji se s poslužitelja mogu preuzeti
    - upravitelj paketima više razine ima kopije jednog ili više popisa u lokalnom međuspremniku
    - naredbe `yum list` i `dnf list`

!!! admonition "Zadatak"
    - Pronađite u `/etc` koju datoteku yum koristi za konfiguraciju i pročitajte u njoj koji direktorij yum koristi za međuspremnik. Provjerite njegov trenutni sadržaj.
    - Ispišite popis paketa iz lokalnog međuspremnika i prebrojite ih. Radi li se o instaliranim ili dostupnim paketima?
    - Provjerite postoji li u popisu paketa koji yum ispisuje paket `python3`.
    - Pronađite informacije o paketu `dosfstools`, specifično kolika mu je veličina i iz kojeg je repozitorija, a zatim popis paketa o kojima ovisi. (*Napomena:* Možda će vam trebati više od jedne naredbe.)

!!! admonition "Zadatak"
    - Pronađite koji direktorij APT koristi za međuspremnik. Provjerite njegov trenutni sadržaj.
    - Pročitajte u statistikama koliko ukupno ima paketa u popisu u lokalnom međuspremniku.
    - Provjerite ima li u popisu paketa paket `python3.6`.
    - Pronađite informacije o paketu `python2.7`, specifično kolika mu je veličina i koji mu je SHA1 hash, a zatim popis paketa o kojima ovisi. (*Napomena:* Možda će vam trebati više od jedne naredbe.)

- `yum check-update` / `apt update` -- dohvaća nove popise paketa s poslužitelja i sprema ih u lokalni međuspremnik
- `yum update` / `apt upgrade` -- dohvaća nove popise paketa s poslužitelja i sprema ih u lokalni međuspremnik, a zatim radi nadogradnju svih instaliranih paketa
- `yum install paket` / `apt install paket` -- instalira paket ukoliko on postoji u popisima
- `yum remove/erase paket` / `apt remove paket` -- briše paket sa sustava zajedno sa svim paketima koji o njemu ovise, a konfiguracijske datoteke paketa sprema u `.rpmsave` (`.dpkg-old`) u slučaju da su bile promijenjene od strane korisnika
- `yum search paket` / `apt search paket` -- traži paket u popisima

!!! admonition "Zadatak"
    - Instalirajte paket `nano`.
    - Pronađite i instalirajte paket koji sadrži [Liberation fonts](https://www.redhat.com/en/blog/liberation-fonts).
    - Izbrišite paket `wireshark`.
    - Izbrišite paket koji sadrži bazu PostgreSQL tako da ostanu sačuvane i baze i konfiguracijske datoteke.

        - Ako je datoteka `pg_hba.conf` promijenjena, a datoteka `postgresql.conf` nije; što će yum napraviti?

    - Nadogradite sve pakete na sustavu. Hoće li `yum` automatski obnoviti popise? Objasnite što bi se dogodilo u slučaju da prethodno ne obnovi popise paketa. (*Uputa:* Za ilustraciju možete razmišljati u situaciji s popisima paketa starim nekoliko mjeseci.)

- [vrste](https://rpm-software-management.github.io/rpm/manual/dependencies.html) [međuovisnosti](https://rpm-software-management.github.io/rpm/manual/more_dependencies.html) (RPM): `Provides`, `PreReq`, `Requires`, `Conflicts`, `Obsoletes`; `Requires(pre)`, `Requires(post)`, `BuildRequires`, `BuildConflicts`, `BuildPreReq`
- [vrste međuovisnosti](https://www.debian.org/doc/debian-policy/ch-relationships.html) (dpkg): `Depends`, `Recommends`, `Suggests`, `Enhances`, `Pre-Depends`, `Breaks`, `Conflicts`, `Replaces`, `Provides`

- [PackageKit](https://www.freedesktop.org/software/PackageKit/)

    - sučelje visoke razine za više upravitelja paketima, uključujući APT i yum
    - koriste ga aplikacije koje nude funkcionalnost sličnu Appleovom App Storeu i Microsoftovom Windows Apps, npr. [GNOME Software](https://wiki.gnome.org/Apps/Software) i [KDE Discover](https://userbase.kde.org/Discover) (
    - koriste ga i jednostavniji grafički upravitelji paketima kao [gnome-packagekit](https://help.gnome.org/users/gnome-packagekit/stable/) [KPackageKit](https://userbase.kde.org/KPackageKit)

## Dodatni upravitelji paketima koji nadopunjuju sustavske

- [Snap](https://snapcraft.io/)
- [Flatpak](https://flatpak.org/)
