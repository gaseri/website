---
author: Krešimir Marnika, Vedran Miletić
---

# Konfiguracija sustava za upravljanje bazom podataka PostgreSQL

Prvi sustavi upravljanja bazom podataka razvijeni su 1960-ih. Začetnik u tom polju bio je Charles Bachman. Njegov je cilj bio stvaranje djelotvornije uporabe novih uređaja s izravnim pristupom pohrane koji su postali dostupni -- do tada se obrada podataka temeljila na bušenim karticama i magnetskoj vrpci te je serijska obrada bila dominantna.

SUBP je softverski sustav koji osigurava izvođenje osnovnih funkcija velikih količina podataka kao što su jednostavno pretraživanje, pamćenje i ažuriranje podataka, višestruko paralelno korištenje skupa podataka, sigurnost i pouzdanost.

Osnovne zadaće SUBP su:

- opis i manipulacija podacima pomoću posebnog jezika (SQL),
- zaštititi bazu podataka od neovlaštenog korištenja (data security)
- spriječiti narušavanje pravila integriteta (data integrity)
- osigurati obnovu podataka u slučaju uništenja (recovery),
- visok nivo sučelja prema korisniku, bez obzira na strukturu podataka
- omogućavanje konkurentnosti tj. pristupa istim podacima od strane više različitih korisnika istovremeno i
- postojanje skupa programskih pomagala za jednostavno razumijevanje i korištenje podataka spremljenih u bazi.

Administrator baze podataka (DBA -- Database Administrator) je osoba zadužena za izvedbu i održavanje baze podataka. Prilikom izrade baze podataka, osoba koja stvara bazu postaje automatski njezin administrator.

Administrator ima najveću razinu korisničkih prava, što se tiče pristupa bazi i manipuliranja podacima. On dodaje ostale korisnike, u njegovoj je nadležnosti da određenim korisnicima dozvoli ili zabrani pristup pojedinim podacima itd. Zadužen je i za održavanje baze (backup).

Glavne su zadaće administatora sljedeće:

- suradništvo s ostalim kompjuterskim profesionalcima (analitičari podataka, programeri...),
- identifikacija potreba klijenata,
- skupljanje podataka, organizacija i upravljanje podacima,
- razvijanje kompjuterskih baza podataka i njihovo postavljanje,
- osigravanje pravilnog rada sustava i baza podataka,
- testiranje kompjutorskih sustava baza podataka s ciljem osiguravanja njihova pravilnog rada,
- osigurati podršku za podatke u slučaju da se podaci izgube ili budu uništeni,
- oporavak izgubljenih kompjutorskih podataka ili podataka koji su bili izmjenjeni,
- osiguravanje integriteta i sigurnosti svih skupljenih podataka,
- nadgledanje i kontrola izvora podataka u organizaciji i
- osiguravanje dostupnosti informacije svima kojima su potrebne.

PostgreSQL je sustav za upravljanje relacijskim bazama podataka otvorenog koda. PostgreSQL je objektno relacijska baza podataka, tj. to je baza podataka koja podatke sprema u odvojene tablice, što daje brzinu i fleksibilnost bazama podataka. Tablice su povezane definiranim relacijama što omogućuje kombiniranje podataka iz nekoliko tablica. Pored uobičajenih svojstava sustava za upravljanje bazom podataka, PostgreSQL nudi mnogo mogućnosti za sistem administratore: backup, replikaciju, balansiranje opterećenja i nadgledanje rada.

## Povijest

Razvoj Ingres projekta započeo je krajem 70-ih godina na Sveučilištu Berkley u Kaliforniji. Profesor informacijskih znanosti, Michael Stonebraker, njegov je začetnik. Nakon što je Ingres, koji je razvijao od 1977. -- 1985., preuzela tvrtka Relational Technologies/Ingres Corporation, Stonebraker je 1986. godine počeo raditi na boljem sustavu za upravljanje bazama podataka. Nadograđuje Ingres objektno orijentiranim svojstvima i daje mu novo ime: Postgres. Njegova je popularnost u nadolazećim godinama značajno rasla.

90-ih godina Postgres je koristio specifični upitini jezik Postquel. Andrew Yu i Jolly Chen dodali su Postgres-u podršku za SQL upitni jezik sredinom 90-ih godina te on tada mijenja i ime: Postgres u PostgreSQL. Danas PostgreSQL razvija međunarodna grupa stručnjaka PostgreSQL Global Development Group.

## Značajke

PostgreSQL sustav je utemeljen na modelu klijent-poslužitelj. Poslužiteljske (postgres) i klijentske aplikacije međusobno su povezane. Zadaće su poslužiteljske aplikacije: rad s datotekama baze, prihvaćanje konekcija od klijentskih aplikacija prema bazi podataka i njihova obrada, dok je zadaća klijentskih aplikacija obavljanje pojedinih radnji s bazom podataka. Komunikacija se obavlja preko TCP/IP sučelja.

PostgreSQL akcije komuniciranja klijentske aplikacije s glavnim poslužiteljskim procesom te kasnije s novonastalim procesom (stvorenim baš za tu klijentsku aplikaciju) obavlja interno i krajnji korisnik ne mora biti upoznat s njima.

## Alati

Komunikacija s PostgreSQL bazom podataka može se ostvariti na više načina: postoje ugrađeni alati koji su dostupni instalacijom PostgreSQL-a na računalo, a postoje i alati koji se mogu dodatno instalirati. Budući da PostgreSQL sustav za upravljanje bazama podataka ima otvoreni kod, ne začuđuje postojanje velikog broja ekstenzija i programa za administraciju. Na stranici [pgFoundry](https://wiki.postgresql.org/wiki/Pgfoundry) postojala je jedna cijela zajednica odvojena od samih PostgreSQL programera. Navedena zajednica je nastojala na jednom mjestu okupiti sve nadogradnje PostgreSQL sustava za upravljanje bazama podataka.

Alat naredbenog retka `psql` omogućuje potpunu slobodu pri komunikaciji s bazom podataka. Ovaj alat s bazom podataka komunicira preko komandne linije. Omogućava kreiranje i brisanje baze podataka i tablica, unošenje, ažuriranje i brisanje podataka.

pgAdmin III jedan je od poznatijih programa za administriranje i održavanje PostgreSQL-a. Navedena aplikacija koristi grafičko sučelje za administraciju sustava za upravljanje bazama podataka. Podržava PostgreSQL, EnterpriseDB, Mammoth PostgreSQL, Bizgres i Greenplum sustave. Otvorenog je koda i besplatna je za korištenje. pgAdmin III pruža mnoge mogućnosti:

- kreiranje i brisanje baza podataka, tablica i shema,
- izvršavanje SQL upita u prozoru za upite,
- izvoz rezultata upita u razne datoteke,
- izrada sigurnosnih kopija i obnavljanje baza podataka ili tablica,
- podešavanje korisnika, grupa korisnika i privilegija,
- pregledavanje, uređivanje i unos podataka u tablice.

pgAdmin III je dizajniran na način da može udovoljiti različitim potrebama korisnika -- omogućuje pisanje jednostavnih SQL upita pa sve do razvoja kompleksnih baza podataka. Aplikaciju razvija zajednica PostgreSQL programera diljem svijeta. Dostupna je u više jezika.

phpPgAdmin pokrećemo u web pregledniku upisujući URL adresu http://ime_servera/phppgadmin/. To je aplikacija napisana u PHP-u koja pruža grafičko sučelje za upravljanje bazom podataka. Kako bismo pokrenuli phpPgAdmin potrebno je instalirati i web server, primjerice Apache.

Osim ranije navedenih, postoje još neke aplikacije koje se koriste za upravljanje bazama podataka. Primjer je aplikacija TODO. Također, PostgreSQL može komunicirati i s nizom drugih aplikacija kao što su
Microsoft Access, Microsoft Excel, kao i s većinom open-source aplikacija (npr. Mapserver, Geoserver itd.).

## Instalacija

PosgreSQL se može instalirati na gotov svim platformama te na većini novijih Unix kompatibilnih platformi. Na CentOS-u to činimo naredbom

```
yum install postgresql-server
```

Zatim pokrećemo uslugu naredbom

```
service postgresql start

```

Nakon instalacije, incijalno podešavanje vršimo naredbom

``` shell
$ sudo -i -u postgresql
$ /usr/lib/postgresql/10/bin/initdb -D /var/lib/postgresql
$ /usr/lib/postgresql/10/bin/postmaster -D /var/lib/postgresql >logfile 2>&1 &
```

Naredbama `createdb` i `dropdb` možemo kreirati i brisati baze. Ukoliko je sve prošlo bez greške, moguće je stvoriti probnu bazu i ući u nju naredbama

``` shell
$ createdb testnabaza
$ psql testnabaza
```

Ukoliko je logiranje na bazu bilo uspješno, trebalo bi se prikazati:

``` shell
# testnabaza=#
```

Bazu je moguće testirati jednostavnim upitom:

``` shell
# testnabaza=# SELECT current_date;
# date
------------
2018-01-20
(1 row)
```

Naredbe koje počinju s ključnim znakom "\" su ugrađene od strane PostgreSQL-a. Dvije najčešće naredbe koje se koriste su:

``` shell
# testnabaza=> \h -> kao help za naredbe jezika SQL
# testnabaza=> \q -> za izlaz iz PostgreSQL-a
```

Mnoge naredbe koje rade u ostalim SQL okruženjima kao funkcije i procedure rade i na PostgreSQL-u, primjerice: `SELECT FROM`, `INSERT INTO`, `DELETE FROM`.

## Upravljanje

Jednako kao i kod drugih servera koji su dostupni s weba, vrijedi preporuka da se PostgreSQL pokreće kao drugi, a ne kao ne administratorski korisnik. Samo administrator mora moći upravljati podacima, dok ostali ostali korisnici to nemaju tu ovlast. U nastavku pretpostavljamo da administraciju baze vršimo kao korisnik `postgres`.

Prilikom kreiranja grupa (clustera) potrebno je definirati prostor na disku predodređen
za bazu. Grupa baze podataka je skup baza kojima se upravlja jednim pokretanjem date baze.
Nakon inicijalizacije grupa, baze podataka će pokretati bazu pod imenom postgres, koja je
default baza. Sam server baze podataka ne zahtjeva postojanje spomenute baze, ali drugi
pomoćni programi pretpostavljaju da ona postoji. Grupa baze podataka će biti jedan direktorij
u kojem će biti spremljeni svi podatci. Može se koristiti bilo koji direktorij u tu svrhu, no
najčešće se koristi /usr/local/pgsql/data ili /var/lib/pgsql/data. Kako bi se inicirala grupa baze
podataka koristi se naredba initdb. Odabir lokacije se u sustavu određuje naredbom:

```
postgres$ initdb -D /var/lib/postgresql
```

Naredba initdb neće biti prihvaćena ako direktorij već postoji.
Budući da su u direktoriju smješteni svi podaci vezani za PostgreSQL neophodno je da
on bude zaštićen od neautoriziranog pristupa. Iz tog će razloga initdb odbijati pristup svakome
osim PostgreSQL korisniku.
Dok je sadržaj direktorija siguran, po osnovnim postavkama autorizacije klijenta
dozvoljeno je spojiti se na bazu podataka i čak postati administrator. Zato se preporučuje
korištenje jedne od ovih opcija:

```
initdb's -W, --pwprompt ili --pwfile za dodjeljivanje lozinke administratoru baze podataka.
```

## Pokretanje poslužitelja

Kako bismo pristupili bazi podataka moramo prvenstveno pokrenuti server.
PostgreSQLov program koji služi pokretanju servera naziva se postmaster. On mora znati
gdje se nalaze potrebni podaci. Naredba za pokretanje je:

``` shell
$ postmaster -D /usr/local/pgsql/data
```

Ukoloko želimo da radi u pozadini koristimo naredbu:

``` shell
$ postmaster -D /usr/local/pgsql/data >logfile 2>&1 &
```

Postoji mogućnosti javljanja grešaka prilikom pokretanja servera te prilikom spajanja klijenata.

## Pojavljivanje grešaka

Prilikom podizanja servera moguće je javljanje sljedeće pogreške:

```
LOG: could not bind IPv4 socket: Address already in use
HINT: Is another postmaster already running on port 5555? If not, wait a few seconds and retry.
FATAL: could not create TCP/IP listen socket
```

Takva greška označava problem zauzetosti porta, tj. taj je port već zauzet drugim postmasterom. To je najčešća greška.

Moguće su još neke greške: primjerice zauzetost adrese ili ograničenje na dodijeljenu memoriju od strane krenela.

Prilikom pristupa na server moguće je javljanje greške:

```
psql: could not connect to server: Connection refused
Is the server running on host "server.proba.com" and accepting TCP/IP connections on port 5432?
```

Najčešće je uzrok ovakve greške neomogućavanje TCP/IP-a prilikom konfiguracije servera.

Ukoliko na traženom mjestu nije pokrenut server, javlja se jedna od dviju mogućih poruka: ili `Connection refused` ili `No such file or directory`. Poruke tipa `Connection timed out` najvjerovatnije znače da je mrežna veza prekinuta.

## Zaustavljanje poslužitelja

Postoji nekoliko načina za gašenje servera. Način gašenja se može kontrolirati slanjem različitih signala postmastera procesima:

- `SIGTERM`: Nakon primanja ovog signala, server ne dozvoljava nove konekcije ali dozvoljava postojećim sesijama (prijavama) da završe svoj posao bez ometanja. Ovo je takozvano pametno gašenje (Smart Shutdown).
- `SIGINT`: Server ne dozvoljava nove konekcije i šalje svim postojećim procesima SIGTERM, server izvršava prekid svih transakcija. Tada se čeka završavanje procesa samog servera i nakon toga se gasi. Ovo je brzo gašenje (Fast Shutdown).
- `SIGQUIT`: Poslužitelj provodi instant gašenje (Immediate Shutdown). Preporučuje se samo u najhitnijim slučajevima. Server se ne gasi pravilno što uzrokuje oporavak prilikom sljedećeg pokretanja.

## Povezivanje klijenta SSH tunelima

Prvi je korak provjera je li podignut SSH server na istoj mašini na kojoj je i PostgreSQL server te je li moguće logirati se koristeći SSH kao neki od postojećih korisnika (npr. root).

Po završetku provjere, uspostavlja se spajanje sa klijentske strane naredbom:

``` shell
# ssh -L 3333:proba_server.com:5432 netko@ proba_server.com
```

Za localhost:

``` shell
# psql -h localhost -p 3333 postgres
```

## Uloge i prava

PostgreSQL upravlja dozvolama za pristup bazi podataka koristeći koncept uloga (engl. *roles*). Uloga može biti ili korisnik baze podataka ili grupa korisnika. To mogu biti vlasnici objekata baze podataka (tablica) i mogu dodjeljivati prava za objekte u drugim ulogama. Koncept uloga obuhvaća koncept korisnika i grupa. Za kreiranje uloga koristi se SQL naredba `CREATE ROLE`:

``` shell
# CREATE ROLE ime;
```

Za brisanje uloge koristi se SQL naredba `DROP ROLE`:

``` shell
# DROP ROLE ime;
```

Za kreiranje i brisanje korisnika koristi se shell naredbe `createuser` i `dropuser` (korisnik je uloga koja ima pravo logina):

```
postgres$ createuser ime
postgres$ dropuser ime
```

Kako bi se pregledale postojeće uloge potrebno je ispisati tablicu `pg_roles`, to možemo učiniti idućim SQL upitom:

``` shell
# SELECT rolname FROM pg_roles;
```

Uloga može imati više osobina koje će definirati njene privilegije i komunicirati sa sustavom za autorizaciju klijenta.

Samo uloge koje imaju LOGIN osobinu mogu biti korištene kao inicijalne uloge za konekciju na bazu podataka. Takve uloge su korisnici baze podataka. Koristi se jedna od dvije međusobno ekvivalentne naredbe:

``` shell
# CREATE ROLE ime LOGIN;
# CREATE USER ime;
```

Superkorisnik baze podataka zaobilazi sve provjere dozvola. Uvijek se preporučuje da se najveći dio posla obavlja sa ulogom koja nije superkorisnik. Za kreiranje novog superkorisnika, koristi se naredba:

``` shell
# CREATE ROLE ime SUPERUSER;
```

Uloga bi morala imati dozvolu za kreiranje baze (osim za superkorisnika). Koristi se naredba:

``` shell
# CREATE ROLE ime CREATEDB;
```

Uloga bi morala imati dozvolu za kreiranje drugih uloga (osim za superkorisnika). Naredba:

``` shell
# CREATE ROLE ime PASSWORD 'string';
```

Prilikom stvaranja objekta, dodjeljuje mu se vlasnik i najčešće je to uloga koja ga je stvorila. Za najveći broj objekata, ta uloga je jedini vlasnik (ili superkorisnik) i jedino ona može raditi operacije nad njim. Kako bi se dozvolilo drugim ulogama da je koriste, moraju im se dodijeliti prava (privilegije). Postoji nekoliko vrsta prava: SELECT, INSERT, UPDATE, DELETE, RULE, REFERENCES, TRIGGER, CREATE, TEMPORARY, EXECUTE, i USAGE.

Ukoliko se želi dodijeliti pravo, koristi se naredba GRANT. Primjer:

``` shell
# GRANT UPDATE ON ime_tablice TO ime_uloge;
```

Naredba PUBLIC se koristi za davanje prava svakoj ulozi u sustavu. Naredba ALL daje sva moguća prava ulozi nad nekim objektom. Za micanje svih prava se koristi REVOKE naredba:

``` shell
# REVOKE ALL ON ime_tablice FROM PUBLIC;
```

## Pohrana rezervnih kopija

Razlikujemo 3 moguća načina backupa sustava: SQL dump, backup datoteke sustava i on-line backup.

SQL dump označava generiranje tekstualne datoteke sa SQL naredbama na način da se o uvođenju na server ponovo kreira baza onakva kakva je bila u trenutku izvršenja SQL dumpa. PostgreSQL za ovu svrhu koristi pomoćni program `pg_dump`. Osnovna je naredba:

``` shell
# pg_dump bpime > outfile
```

Backup datoteke sustava označava direktno kopiranje datoteka koje PostgreSQL koristi za smještanje podataka u bazu. Za ovaj je backup značajno da server mora biti ugašen kako bi se dobio uporabljiv backup. Također, nije moguće backup-ovati samo određene dijelove baze.

Primjer je korištenja backup-a datoteke sustava sljedeći:

``` shell
# tar -cf backup.tar /usr/local/pgsql/data
```

Treći je način on-line backup i point-in-time recovery (PITR). PostgreSQL neprestano održava write ahead log (WAL) u pg_xlog/ direktoriju. Log opisuje svaku promjenu nad bazom i osnovni je razlog njegova postojanja sigurnost. Moguće je kombinirati backup datoteke sustava i backup WAL datoteka. Ukoliko je backup potreban, radi se restore backup-a, a potom se ponavljenjem događaja baza dovodi do posljednjeg stanja. U situacijama u kojima se zahtjeva visoka pouzdanost, ova je metoda najbolja. Njome je moguće obnoviti cijelu grupu baze podataka, no ne i podgrupe.

Prvi je korak backup-a baze provjera je li WAL arhiviranje omogućeno i funkcionira li ispravno. Spajamo se na bazu te unosimo sljedeću naredbu (kao superkorisnik):

``` shell
# SELECT pg_start_backup('label');
```

Pritom je 'label' ime pod kojim ćemo spremiti ovu operaciju izvršavanja backup-a. Pri izvođenju backup-a koriste se alati kao što su tar ili cpio. Ponovno se izvodi spajanje na bazu, kao superkorisnik, te se daje naredba:

``` shell
# SELECT pg_stop_backup();
```

Po završetku arhiviranja WAL segmenata datoteke kao redovne aktivnosti baze, proces je završen.

Obnavljanje se vrši u 9 koraka:

1. Ukoliko je postmaster pokrenut, potrebno ga je zaustaviti.
1. Kopirati cijelu grupu direktorija podataka i tablespaces na neku privremenu lokaciju za slučaj da zatrebaju naknadno.
1. Očistiti sve postojeće datoteke i poddirektorije u grupi direktorija podataka i u root direktoriju za svaki tablespaces koji se koristi.
1. Obnoviti bazu (ne kao root već kao korisnik baze podataka) .
1. Ukloniti bilo koji datoteku prisutan u pg_xlog/; ukoliko pg_xlog/nije arhiviran, potrebno ga je ponovo kreirati. Također, mora biti kreiran i poddirektorij pg_xlog/archive_status/.
1. Ukoliko postoje nearhivirani WAL segmenti datoteka koji su sačuvani u koraku 2, kopirati ih u pg_xlog/.
1. Kreirati datoteku za naredbu obnavljanja recovery.conf u grupi direktorija podataka.
1. Pokrenuti postmaster. On će preimenovati fajl recovery.conf u recovery.done.
1. Provjeriti sadržaj baze kako bismo se uvjerili da je obnavljanje uspješno napravljeno. Ukoliko nije, potrebno je vratiti se na korak 1.

## Literatura

!!! todo
    Reference treba pročistiti i povezati u tekstu.

1. https://youtu.be/dTVDsPt7rmU
1. https://library.linode.com/databases/postgresql/fedora-13
1. http://www.adempiere.com/Install_on_Fedora_10_with_PostgreSQL
1. https://www.postgresql.org/docs/9.3/static/admin.html
1. https://tecadmin.net/install-postgresql-on-centos-rhel-and-fedora/
1. https://www.if-not-true-then-false.com/2012/install-postgresql-on-fedora-centos-red-hat-rhel/
1. https://www.postgresql.org/download/linux/redhat/
1. https://admin.fedoraproject.org/pkgdb/acls/name/postgresql
1. https://wiki.postgresql.org/wiki/SEPostgreSQL_Administration
1. http://www.cert.hr/sites/default/files/CCERT-PUBDOC-2006-10-171.pdf
1. http://www.vtsnis.edu.rs/Predmeti/baze_podataka/BAZE%20PREDAVANJA%202%20-%20SUBP.pdf
1. http://www.vps.ns.ac.rs/nastavnici/Materijal/mat50.pdf
1. https://www.fer.unizg.hr/_download/repository/BazePodataka_SQLPredavanja.pdf
1. http://media.lukaperkov.net/lukaperkov.net/files/papers/Seminar[2009]Perkov_Luka.pdf
1. https://en.wikipedia.org/wiki/PostgreSQL
1. http://marjan.fesb.hr/~emudnic/Download/BazePodataka2/56315396-BP-VEST-Skripta.pdf
