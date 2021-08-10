---
author: Matko Abramović, Vedran Miletić
---

# LDAP poslužitelj OpenLDAP

!!! hint
    Za više informacija proučite [službenu dokumentaciju](https://www.openldap.org/doc/).

[OpenLDAP](https://www.openldap.org/) je popularno programsko rješenje otvorenog koda za implementaciju [Lightweight Directory Access Protocol (LDAP)](https://ldap.com/learn-about-ldap/) servisa.

Do verzije 2.3 OpenLDAP programskog rješenja predefinirane postavke su se nalazile u `slapd.conf` datoteci (tekstualni zapis) te su one bile učitavane u radnu memoriju prilikom podizanja LDAP servisa. Loša strana toga je bila ta što se takve informacije nisu mogle dinamički mijenjati nego je za svaku promjenu bilo potrebno ponovno pokrenuti OpenLDAP servis.

Prilikom prelaska na verziju 2.3 OpenLDAP programskog rješenja predstavljen je novi način pohranjivanja predefiniranih postavki. One se sada nalaze u LDAP bazi podataka koja je implementirana putem tekstualnoga zapisa na disku. Ovakav način korištenja predefiniranih postavki omogućuje djelomično mijenjanje istih tijekom rada samog servisa. Predefinirane postavke koje se učitavaju u takvu bazu nalaze se zapisane u obliku LDIF datoteke na disku.

Sam zapis tekstualne baze na disku nalazi se kao struktura direktorija u stablu podirektorija `slapd.d`. Prelaskom na novi način pohranjivanja konfiguracije došlo je do promjena načina rada s istom te se umjesto ažuriranja podataka u tekstualnoj datoteci promjene implementiraju koristeći standardne LDAP funkcije, a ručno ažuriranje LDIF datoteka u sustavu podirektorija slapd.d se nikako ne preporuča.

## Instalacija

Instalacija paketa OpenLDAP-a vrši se pomoću naredbe:

``` shell
# apt-get install slapd ldap-utils
```

te se korisnika prilikom instalacije traži da unese lozinku za administratora u LDAP imeniku.

## Konfiguracija

Nakon što se program uspješno instalirao, potrebno ga je konfigurirati. Konfiguraciju pokrećemo naredbom:

``` shell
# dpkg-reconfigure slapd
```

te nam se otvara prozor za konfiguriranje paketa.

Prvo pitanje koje nam postavlja je da li želimo da se preskoči konfiguracija servera (da se ne kreira inicijalna konfiguracija baze podataka). Odgovorio sam 'ne', znači želim da se kreira inicijalna konfiguracija baze podataka.

Sljedeće moramo upisati DNS ime domene. Ja sam se odlučio za `local`.

Nakon toga moramo upisati ime organizacije (također sam se odlučio za `local`) te nas ponovno pita da unesemo lozinku za administratora u LDAP imeniku.

Sljedeće na redu je konfiguracija baze podataka (odabir između BDB i HDB). Savjetuje nas da odaberemo HDB te sam ja to i napravio.

Nakon toga su još tri pitanja: želimo li da se baza podataka ukloni kada se ukloni `slapd` (odgovorio sam 'ne') te želimo li omogućiti LDAPv2 protokol (odgovorio sam 'ne').

Nakon što smo postavili početnu konfiguraciju, možemo pretražiti koje podatke za sada imamo u svome imeniku. No, kako bi vidjeli rezultate za ovu domenu koju smo naveli moramo navesti bazu u datoteci `/etc/ldap/ldap.conf`. Ime domene se piše s oznakom `dc`. Tako se i ime razdvaja ako je odvojeno točkom.

```
BASE    dc=local
URI     ldap://localhost

TLS_CACERT       /etc/ssl/certs/ca-certificates.crt
```

Nakon toga možemo pretražiti imenik naredbom

``` shell
$ ldapsearch -x
```

te ćemo vidjeti našu organizaciju.

## Rad u programu

U imeniku možemo dodavati organizacijske jedinice te sam se ja odlučio nadodati organizacijsku jedinicu grupa i organizacijsku jedinicu ljudi. Kako bi mogli nadodati te organizacijske jedinice, moramo ih prvo zapisati u LDIF datoteci u određenom formatu gdje napišemo kojoj organizaciji ih želimo pridružiti, njezino ime i klasu koja je organizacijska jedinica.

```
dn: ou=People,dc=local
ou: People
objectClass: organizationalUnit

dn: ou=Group,dc=local
ou: Group
objectClass: organizationalUnit
```

Kako bi nadodali zapise u LDIF datoteci, moramo pokrenuti naredbu za dodavanje zapisa:

``` shell
$ ldapadd -x -D "cn=admin,dc=local" -W -f ou.ldif
```

gdje kažemo koju akciju želimo napraviti (`add`), kojoj organiziciji radimo promjene, te koju LDIF datoteku koristimo.

Nakon toga možemo nadodati neku grupu i korisnika, primjerice ovakvom LDIF datotekom:

```
dn: cn=matko,ou=group,dc=local
cn: matko
gidNumber: 20000
objectClass: top
objectClass: posixGroup

dn: uid=matko,ou=people,dc=local
uid: matko
uidNumber: 20000
gidNumber: 20000
cn: Matko
sn: Matko
objectClass: top
objectClass: person
objectClass: posixAccount
objectClass: shadowAccount
loginShell: /bin/bash
homeDirectory: /home/matko
```

Za grupu navodimo njezino ime i id grupe (`gidNumber`) između ostaloga, a za korisnika uz ime i id korisnika navodimo i id grupe kojoj pripada kao i kućni direktorij. To su neke osnovne postavke koje se navode za grupu i korisnika.

Također možemo dodati neku grupu (npr. grupu `UMS` sa id-om grupe `30000`) te korisnike koji pripadaju toj grupi (navedemo da imaju taj id grupe). Tada naredbom

``` shell
$ ldapsearch -x gidNumber=30000
```

možemo pretražiti koji korisnici pripadaju toj grupi te dobiti informacije o tim korisnicima kao i o samoj grupi. Izlaz naredbe `ldapsearch` je oblika:

``` shell
# extended LDIF
#
# LDAPv3
# base <dc=local> (default) with scope subtree
# filter: gidNumber=30000
# requesting: ALL
#

# ums, Group, local
dn: cn=ums,ou=Group,dc=local
cn: ums
gidNumber: 30000
objectClass: top
objectClass: posixGroup

# irena, People, local
dn: uid=irena,ou=People, local
uid: irena
uidNumber: 20002
gidNumber: 30000
cn: Irena Hartmann
sn: Irena Hartmann
objectClass: top
objectClass: person
objectClass: posixAccount
objectClass: shadowAccount
loginShell: /bin/bash
homeDirectory: /home/irena

# domagoj, People, local
dn: uid=domagoj,ou=People, local
uid: domagoj
uidNumber: 20003
gidNumber: 30000
```

Također možemo modificirati podatke nekog već dodanog korisnika kako bi ga priključili nekoj drugoj grupi ili mu promjenili neke osobne podatke. Zapis u LDIF datoteci tada je drugačijeg formata. Moramo navesti za kojeg korisnika želimo mijenjati podatke, napisati tip promjene (dodavanje nove karakteristike, brisanje karakteristike ili izmjena postojeće karakteristike) te ovisno o tomu reći za koju karakteristiku to radimo i vrijednost koju želimo upisati.

```
dn: uid=toni,ou=people,dc=local
changetype: modify
replace: gidNumber
gidNumber: 30000
-
replace: cn
cn: Toni Butkovic
-
replace: sn
sn: Toni Butkovic
```

Korisnika tada mijenjamo naredbom:

``` shell
$ ldapmodify -x -D "cn=admin,dc=local" -W -f userToniToUms.ldif
```

iz čega vidimo da se mijenja samo naziv akcije.
