---
author: Irena Hartmann, Vedran Miletić
---

# Pakiranje deb paketa

Paket je oblik arhive koja sadrži dodatne informacije i metapodatke potrebne za ispravnu instalaciju softvera na određenom sustavu.

## Postupak pakiranja

Postupak pakiranja, iako nije trivijalan, može se sažeti na tri koraka:

1. Dobavljanje programa koji se pakira, instalacija, testiranje i preimenovanje
1. Generiranje i modifikacija konfiguracijskih datoteka i pakiranje
1. Testiranje

Za svaki od ovih koraka postoje određene konvencije koje se moraju poštovati -- ako se ispune nužni minimalni uvjeti, paket će biti tehnički ispravan, ali to ne garantira prihvaćanje u Debian repozitorij paketa.

Za ovo pakiranje korišten je [VirtualBox](https://www.virtualbox.org/) 4.3.2 i na njemu virtualni [Debian](https://www.debian.org/) 7, Wheezy. Od alata koji se koriste za pakiranje, postoji [niz potrebnih i korisnih alata](https://www.debian.org/doc/manuals/maint-guide/start.en.html#needprogs) koji olakšavaju i više ili manje automatiziraju proces pakiranja, među kojima bih istaknula `debhelper`, `dh-make` i `build-essentials`.

### Dobavljanje programa

Prvi korak je nužan kako bi se osoba pri pakiranju barem donekle upoznala s alatom, iako nije presudno razumijevanje svih detalja funkcionalnosti. U ovom konkretnom slučaju, instalacija Avogadra 2 dala je i vrijedan uvid u ovisnosti programa o drugim alatima koji moraju biti u pakiranju riješeni kako bi se osiguralo ispravno funkcioniranje paketa.

Također, potrebno je ponekad modificirati u određenoj mjeri oblik izvornog programa -- jedna od stvari koje bih istaknula u ovom koraku je poštovanje norme imenovanja verzije programa i paketa.

### Pakiranje paketa za Debian

Kako je Debian iznimno popularna Linux distribucija, dokumentacija o pakiranju je raznovrsna i varira po pitanju čitljivosti i kompleksnosti. U pakiranju Avogadra 2 korišten je najviše [Debian New Maintainers' Guide](https://www.debian.org/doc/manuals/maint-guide/index.en.html), koji ažurno i pregledno vodi korisnika kroz postupak pakiranja. U ovom tekstu preskačem uvod o društvenoj dinamici Debiana, potrebnim programima i dokumentaciji, kao i modificiranje alata koji se koriste. Ipak, valja istaknuti da je i to iznimno važan dio pakiranja i održavanja paketa koji može znatno olakšati traženje i popravljanje kasnijih grešaka.

Ipak, srž problematike pakiranja je najjasnije vidljiv u trenutku kada se od arhiva originalnog programa započne konfiguriranje datoteka potrebnih za pakiranje. Moguće je koristiti, kao primjerice u ovom slučaju, naredbu `dh_make` koja napravi kopiju arhiva, preimenuje ga i generira poddirektorij `debian/` i u njemu datoteke potrebne za daljnju konfiguraciju paketa. Od niza datoteka koje su generirane i korisne za upravljanje Debian paketom, datoteke `control`, `copyright`, `changelog` i `rules` su najbitnije i najviše se koriste. Ostale (primjerice, datoteke `TODO`, `watch`, `source/format`, `README.Debian`, `compat` i slično) su isto važne, kako za funkcionalnost tako i za norme Debian paketa ali ih se nešto manje mijenja u samom procesu izrade paketa i manje je vjerojatno da će se nepažljivim mijenjanjem tih podataka uzrokovati trajniji problemi.

#### Datoteka `control`

Datoteka `control` sadrži sve važne informacije koje `dpkg`, `dselect`, `apt-get`, `apt-cache`, `aptitude` i ostali alati koriste pri upravljanju paketom. Tu se nalaze sve važne informacije i izvoru i binarnom obliku paketa, podaci o autorima, osobi koja je pakirala, međuovisnostima.

#### Datoteka `copyright`

Datoteka `copyright` bi morala sadržavati sve informacije o tome gdje je program nabavljen, o autorima, samom copyrightu i legalnosti. Naredba `dh_make` može izraditi generičku formu, ali bi je valjalo popuniti po [Debian standardima](https://www.debian.org/doc/debian-policy/ch-docs.html#s-copyrightfile).

#### Datoteka `changelog`

Datoteku `changelog` koriste alati kao što je `dpkg` za dohvaćanje broja verzije, revizije, distribucije i važnosti instalacije paketa, što je koristan skup informacija kako za osobu koja ga održava tako i za krajnje korisnike.

#### Datoteka `rules`

Datoteka `rules` je datoteka u kojoj su zapisana pravila koje će `dpkg-buildpackage` koristiti u samom stvaranju paketa. Datoteka `rules` funkcionira kao Makefile -- niz ključnih riječi (primjerice, `clean`, `build`, `build-arch`, `install`, `binary`) i skup instrukcija instrukcija kako ih izvesti. Ipak, nije ih uvijek potrebno pisati sve ručno, naredba `dh_make` i u ovom slučaju izgenerira uobičajen, zadan oblik koji pri izradi paketa poziva sve potrebno u dvije izrazito efektne linije koda (`%:` i `dh $@`), što implicitno omogućuje pozivanje bilo koje ključne riječi (`target`) programom `dh`. Primjerice, naredba `debian/rules` build pokreće `dh_build` što pak pokrene `dh_testdir`, `dh_auto_configure`, `dh_auto_build` i `dh_auto_test`.

Sama datoteka `rules` se može dalje prilagođavati i mijenjati, ovisno o potrebama. Često se problemi pri pakiranju rješavaju upravo tim putem -- primjerice, da se zaobiđe zadano mjesto instalacije koje nije po Debian standardima ili nešto slično. To čini rules ako ne najvažnijom, onda barem datotekom koja omogućuje najviše fleksibilnosti pri izradi paketa.

### Testiranje

Nakon što se paket izradi sa `dpkg-buildpackage`, izgenerirane su datoteke `.deb` (koje služi za instalaciju) i datoteka `.changes` s imenom programa koji se pakira.

Važno je provjeriti može li se paket instalirati, deinstalirati, ažurirati sa stare verzije (ako postoji). Alat `lintian` je koristan za pronalaženje daljnjih grešaka i upozorenja, kako bi se paket uskladilo sa traženim standardima Debian repozitorija.

## Pakiranje Avogadra

Kako bi se olakšala instalacija i korištenje [Avogadra 2](https://www.openchemistry.org/projects/avogadro2/) na određenim Linux distribucijama, kao što je primjerice Debian, potrebno ga je zapakirati u deb paket.

Kao konkretan primjer, Avogadro 2 nije bio posebno problematičan za pakiranje. Nakon početnih problema i neuspjeha koji su nastali kao rezultat neiskustva, proces je tekao relativno glatko. Bilo je potrebno prvo instalirati a onda i zapakirati i pripadajuće biblioteke i program [MoleQueue](https://www.openchemistry.org/projects/molequeue/), što je zbog međuovisnosti koje se pojavljuju znalo zahtijevati vremena i strpljenja. Ipak, pakiranje je izrazito olakšano kvalitetno napisanim Make datotekama autora programa, automatizirani postupak je bio relativno jednostavan i bez većih grešaka.

Ono što je nepotpuno su još uvijek finese koje treba doraditi kako bi se podiglo paket na standarde potrebne za pokretanje procesa uključivanja Debian direktorij.

Osim toga, jednom naučen, proces pakiranja je korisna vještina koja se može dalje primjenjivati i razvijati. Daljni napredak bi se mogao kretati u smjeru proučavanja pakiranja za ostale distribucije, bilo za Avogadro 2, bilo za neki drugi softver -- Debian i postojeći alati su dobro dokumentirani, ali to ne mora biti slučaj za manje popularne Linux distribucije.
