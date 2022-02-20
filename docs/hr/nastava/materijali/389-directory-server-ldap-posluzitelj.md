---
author: Ivo Bujan, Vedran Miletić
---

# LDAP poslužitelj 389 Directory Server

!!! hint
    Za više informacija proučite [Red Hat Directory Server Documentation](https://access.redhat.com/articles/5705531).

[389 Directory Server](https://directory.fedoraproject.org/) (prethodno nazvan Fedora Directory Server) je [Lightweight Directory Access Protocol (LDAP)](https://ldap.com/learn-about-ldap/) poslužitelj razvijen od strane organizacije [Red Hat](https://www.redhat.com/), a kao projekt podržan od strane zajednice okupljene oko [projekta Fedora](https://fedoraproject.org/). Ime je dobio po portu 389 preko kojega putuje LDAP promet.

Cilj projekta 389 Directory Server je brzo razviti nove značajke software‐a. Izvorni kod je dostupan pod općenitom GPLv2 licencom. 389 Directory Server razvija se na operativnom sustavu Fedora, ali podržava i mnogo drugih operacijskih sustava kao što je Red Hat Enterprise, Debian, Solaris i HP‐UX‐11i

## Namjena

Directory Server je servis koji omogućava vođenje centraliziranog imenika za intranet, ekstranet i mrežne informacije. Integrira se sa postojećim sustavima te kao takav djeluje kao središnje spremište za pohranu podataka o zaposlenicima, kupcima, dobavljačima, partnerima ili bilo kojim drugim informacijama. Također se može proširiti te služiti kao sustav za upravljanje korisničkim profilima ili ekstranet autentifikaciju korisnika.

## LDAP

LDAP je zajednički jezik koji klijentske aplikacije i poslužitelji koristite za međusobnu komunikaciju. LDAP je jednostavnija verzija Directory Access Protocol (DAP) koji koristi ISO X.500 standard. LDAP omogućava bilo kojoj aplikaciji pristup u imenik putem robusnog informacijskog okvira, ali je puno jeftiniji u odnosu na DAP.

LDAP koristi pojednostavljene metode kodiranja "open directory access" protokola koristeći TCP/IP vezu. Kao takav zadržava model ISO X.500 standardna te istovremeno može podržati milijune unosa uz mala ulaganja u hardware i mrežnu infrastrukturu.

### Struktura

Svaki zapis sastoji se od seta atributa, a svaki atribut ima ime (identifikator) i jednu ili više vrijednosti. Svaki zapis ima jedinstveni ključ po kojemu se razlikuje od svih drugih. Taj ključ se naziva Distinguished name (DN). Atributi se definiraju shemom. Primjer strukture LDAP‐a je sljedeći:

```
dn: cn=Marko Laca,dc=testzone,dc=local
cn: Marko Laca
givenName: Marko
sn: Laca
telephoneNumber: +385 51 123 675
telephoneNumber: +385 51 123 123
mail: marko@testzone.com
manager: cn=Petar Jozef,dc=testzone,dc=local
objectClass: inetOrgPerson
objectClass: organizationalPerson
objectClass: person
objectClass: top
```

Kod LDAP‐a redosljed nije bitan.

## Instalacija

Za platformu na kojoj će se pokretati 389 Directory Server (nadalje u tekstu 389) nametnula se distribucija Fedora 18 samim time što se 389 razvija na tom istom operativnom sustavu. Sam OS podignut je na virtualnom računalu pomoću VMware Workstation 9 virtualizacijskog paketa.

Pošto instalacija OS nije tema ovog seminara prelazimo samo na dijelove bitne za 389 Directory Server.

### Radnje koje prethode instalaciji

Prvo je potrebno promijenit hostname računala koje će biti LDAP server na način da odgovara domenskom zapisu. Prilikom instalacije Fedora hostname postavlja na `localhost`, a za naše potrebe moramo ga promijeniti u `<ime_racunala>.<domena>.local`, tj. u `ldap.testzone.local`. To radimo tako da u datoteku `/etc/sysconfig/network` dodamo:

```
HOSTNAME=ldap.testzone.local
```

Također je potrebno dodati novu liniju u `/etc/hosts`, a koja mora sadržavati IP adresu računala i hostname. Nakon tog koraka potrebno je ponovno pokrenuti servise kako bi se ažurirale izmjene.

Zadnji korak prije same instalacije 389 Directory Servera je dodavanje novog korisnika koji će biti potreban kasnije:

``` shell
# useradd -s /sbin/nologin ldapuser
```

### Sama instalacija

Instalaciju se pokreće sa samo jednom naredbom:

``` shell
# yum install 389-ds.
```

Nakon toga kreće preuzimanje podataka po čijem je završetku potrebno pokrenuti instalacijsku skriptu (super user):

``` shell
# setup-ds-admin.pl.
```

Skripta prvo provjerava moguće probleme ili nedostatke na računalu (potrebno ih je riješiti prije nastavka instalacije). U slučaju virtualnog računala RAM, procesor i NIC nisu standardni, nemaju identifikaciju, ali bez obzira na to može se nastaviti.

Prilikom instalacije ponuđene su tri opcije; brza instalacija, standardna instalacija te prilagođena instalacija. Standardna instalacija zadovoljava većinu uvjeta te je optimalan odabir.

Nakon toga slijedi postavljanje imena računala (ldap.testzone.local). Ukoliko je preimenovanje uspješno obavljeno instalacijska skripta će sama pronaći ime i domenu računala pomoću DNS lookupa.

Sljedeće što je potrebno unijeti je ime i grupu korisnika čiji je korisnički račun unaprijed izrađen (u oba slučaja unosi se `ldapuser`). Moguće je i pustit početne vrijednosti `nobody`, međutim nije dobra praksa.

Nakon toga instalacijska skripta pitati će da li je potrebno izraditi novi imenik ili registrirati postojeći. U ovome primjeru potrebno je izbrati novi imenik iz razloga što nema postojećih podatka.

Slijedi izrada administracijskog korisničkog računa, potvrda domene koja je prethodno definirana, odabir portova (389 za standardni promet LDAP‐a, te 9830 za administrativnu konzolu), jedinstvenog identifikatora imenika, sufiksa koji je prethodno definiran (.local) te stvaranje Directory Managera. Time je proces postavljanja početne databaze imenika završen i 389 Directory Server je konfiguriran.

Nakon toga je pokrenuti servis:

``` shell
# service dirsrv start
```

ili

``` shell
# systemctl start dirsrv.target
```

## Administracija

389 Directory Server imenik je moguće administrirati pomoću grafičkog sučelja (konzole, naredba `389-console`) ili pomoću sučelja naredbenog retka. Nije potrebno posebno isticati kako je GUI sučelje puno elegantnije od podužih LDAP naredbi koje nisu nimalo jednostavne niti lake za zapamtiti. Samu konzolu pozivamo jednostavnom naredbom:

``` shell
# /usr/bin/389-console
```

Konzola dozvoljava administraciju servera imenika (LDAP) i administracijskog servera. U oba slučaja u postavke se ulazi, kao i u postavke drugih elemenata, dvostrukim klikom. Osnovne radnje zajedničke za oba servera su pokretanje, zaustavljanje, resetiranje te izdavanje certifikata. Server imenika, uz to, ima i elegantno riješenu izradu sigurnosnih kopija, te vraćanje sustava na staro stanje iz istih. U nativnom izdanju LDAP metode izrade sigurnosnih kopija i vraćanje na staro stanje su puno kompliciranije.

U slučaju da je potrebno izraditi još jedan imenik (ili spojiti postojeći) u istu domenu to se može postići desnim klikom na `Server Group`, odabirom `Create Instance Of` te klikom na `389 Directory Server`. Nakon toga upisuju se isti podatci koji su se upisivali prilikom instalacije, ali ovaj put u preglednijem sučelju.

Prije popunjavanja imenika preporučeno je zaštiti određene unose, kao što je zaporka, algoritmom za enkripciju. Iz tog razloga potrebno je pristupiti serveru imenika, odabrati domenu na kojoj radi, te odabrati tab `Attribute Encryption` kao što je vidljivo na slici.

### Rad s imenikom

U slučaju rada i ažuriranja imenika često se javlja potreba za dodavanjem ili brisanjem novih organizacijskih jedinica, grupa i korisnika. U tim slučajevima u konzoli se bira server imenika (LDAP), te zatim tab `Directory`.

U tom pogledu vidljiva je domena u kojoj je imenik aktivan. Jednom kada je domena odabrana, sa desne strane otvara se sadržaj iste. U slučaju kada je potrebno dodati element u imenik dovoljno je desnim klikom na bilo koji prazni desni dio konzole pozvati izbornik. Izbornik nam nudi sve potrebne radnje za ažuriranje imenika.

Na primjer, u slučaju da je potrebno dodati novu organizacijsku jedinicu u domenu prvo je potrebno pozvati izbornik desnim klikom u prazan dio konzole, zatim odabrati opciju `New`, a zatim `Organizational Unit`. Tome slijedi odabir novo napravljene upravljačke jedinice, ponovo desni klik (ovaj put na jedinicu), `New`, `Group`. Na isti način stvaraju se i korisnici u samoj grupi.

U status bar‐u konzole vidljivi su osnovni pozadinski atributi koji se mijenjaju ovisno o tipu odabranog elementa. Za `Test grupa`, osim atributa o domeni, vidljiv je i atribut Organizational Unit (`ou`). Za `Grupa 1` uz već navedene atribute vidljiv je i atribut "cn" koji definira naziv same grupe. Korisnik pak uz navedene ima dodatni atribut "uid" koji sadrži korisničko ime. To naravno nisu svi atributi koje ima neki element. Elementi kao email, broj telefona, adresa, spol itd. također su pridodani tipu elementa ovisno o potrebi. Iz navedenog evidentna je hijerarhijska struktura atributa zapisanih u imenik, a na vrhu koje se nalazi domena (`testzone`).

Primjera radi, isti ovaj postupak obavljen putem command line sučelja izgleda ovako za samo jednog korisnika.

- stvaranje korisnika

    ``` shell
    # useradd test1
    ```

- migriranje korisnika u LDAP

    ``` shell
    # grep root /etc/passwd > /etc/openldap/passwd.test1
    ```

- pretvaranje postojeć zaporke u ldif format

    ``` shell
    # /usr/share/openldap/migration/migrate_passwd.pl /etc/openldap/passwd.test1 /etc/openldap/test1.ldif
    ```

- izmijena datoteke `root.ldif` ako želimo novu grupu, 'cn', inače izmijena `/etc/openldap/test1.ldif`

    ```
    #1 dn: uid=test1,ou=People,dc=testzone,dc=local
    #2 uid: test1
    #3 cn: Manager
    #4 objectClass: account
    ```

- stvaranje ldif datoteke `/etc/openldap/testzone.local.ldif` za domenu

    ```
    #dn: dc=testzone,dc=local
    #dc: testzone
    #description: LDAP Admin
    #objectClass: dcObject
    #objectClass: organizationalUnit
    #ou: rootobject
    #dn: ou=Grupa 1, dc=testzone,dc=local
    #ou: Test grupa
    #description: Users of testzone
    #objectClass: organizationalUnit
    ```

- uvođenje korisnika u stvorenu domenu

    ``` shell
    # ldapadd -x -D "cn=Manager,dc=testzone,dc=local" -W -f /etc/openldap/test1.ldif
    Enter LDAP Password:
    adding new entry "uid=test1,ou=People,dc=testzone,dc=local"
    ```

## SSL autorizacija

SSL autorizacija je korisna (potrebna) iz razloga što bez nje sva autentikacija mrežom prolazi u plain text formatu. Drugim riječima korisnička imena i šifre su čitljive uz pomoć network sniffera svakome tko ima fizički pristup toj istoj mreži.

### Izrada certifikata

Prije same konfiguracije 389 imenika potrebno je stvoriti certifikate korištenjem alata [OpenSSL](https://www.openssl.org/). Postupak se sastoji od sljedećih koraka:

1. Stvoriti privemene lokacije za certifikate:

    ``` shell
    # mkdir /tmp/ldap
    # mkdir /tmp/admingui
    ```

1. Stvoriti Certificate Authority (CA) za imenik (u direktoriju `/tmp/ldap`):

    ``` shell
    # openssl genrsa -des3 -out ca.key 4096
    # openssl req -new -x509 -days 365 -key ca.key -out ca.crt
    ```

1. Stvoriti Server key (u direktoriju `/tmp/ldap`):

    ``` shell
    # openssl genrsa -des3 -out server.key 4096
    # openssl req -new -key server.key -out server.csr
    ```

1. Ovjeriti serverski certifikat (`server.csr`) koristeći Certificate Authority iz drugog koraka:

    ``` shell
    # openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt
    ```

1. Stvoriti Certificate Authority (CA) za administracijsko sučelje (u direktoriju `/tmp/admingui`):

    ``` shell
    # openssl genrsa -des3 -out ca.key 4096
    # openssl req -new -x509 -days 365 -key ca.key -out ca.crt
    ```

1. Stvoriti Server key (u direktoriju `/tmp/admingui`):

    ``` shell
    # openssl genrsa -des3 -out server.key 4096
    # openssl req -new -key server.key -out server.csr
    ```

1. Ovjeriti serverski certifikat (`server.csr`) koristeći Certificate Authority iz petog koraka:

    ``` shell
    # openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt
    ```

1. Pretvoriti certifikate u format pkcs12 koji koristi 389 Directory Server (u direktoriju `/tmp/ldap`):

    ``` shell
    # openssl pkcs12 -export -in server.crt -inkey server.key -out server.p12 -nodes -name "DS-Server-Cert"
    # openssl pkcs12 -export -in ca.crt -inkey ca.key -out ca.p12 -nodes -name "DS-Cert"
    ```

1. Pretvoriti certifikate u format pkcs12 koji koristi 389 imenik (u direktoriju `/tmp/admingui`):

    ``` shell
    # openssl pkcs12 -export -in server.crt -inkey server.key -out server.p12 -nodes -name "Admin-Server-Cert"
    # openssl pkcs12 -export -in ca.crt -inkey ca.key -out ca.p12 -nodes -name "Admin-Cert"
    ```

### Unos SSL certifikata

Generirane i ovjerene certifikate unosimo pomoću već spomenute konzole. Postupak se sastoji od sljedećih koraka te je isti i za LDAP imenik i Administration Server konzolu:

1. Otvoriti konzolu
1. Izabrati `Console` > `Security` > `Manage Certificates` iz padajućeg izbornika
1. Unijeti lozinku za buduće promjene certifikata
1. Pronaći unaprijed stvorene `.p12` certifikate te ih uvesti

Alternativa ručnom stvaranju certifikata i njihovom unosu je korištenje već gotove shell skripte dostupne u [repozitoriju richm/scripts na GitHubu](https://github.com/richm/scripts/blob/master/setupssl2.sh). Skripta automatski stvara CA certifikat, certifikat imenika, certifikat administracijskog servera, `pin.txt` datoteku potrebnu za nenadzirano ponovno pokretanje, omogućuje korištenje SSL‐a u imeniku te izvozi certifikat za daljnje korištenje. Shell skripta predviđa da pravilno vraća potrebne podatke, da se pokreće kao super user, da je pokrenuta nakon svježe instalacije, da se pri korištenju koriste standardni portovi. U slučaju da skripta nema privilegije treba ih dozvoliti ( `# chmod +x setupssl2.sh` ). Prilikom pozivanja potrebno je proslijediti argument koji označava lokaciju imenika za koji se certifikati stvaraju ( `# ./setupssl2.sh /etc/dirsrv/admin-serv/` ).

## Kontrola pristupa

Kontrola pristupa podrazumijeva autorizaciju, autentifikaciju, reviziju i dozvoljavanje pristupa. Neki tipični primjeri kontrole pristupa su:

- Anonimni pristup za čitanje i pretraživanje
- Dozvoli zaposlenicima mijenjati svoje osnovne informacije (broj telefona, adresa)
- Dozvoli zaposlenicima čitanje, pisanje i pretraživanje
- ...

Definiranje kontrole pristupa u 389 imeniku vrši se pomoću konzole. U samoj konzoli potrebno je desnim klikom na organizacijsku jedinicu, grupu ili korisnika pozvati izbornik te na njemu odabrati `Set Access Permissions` (kratica ++control+l++). Ta naredba otvara `Access Control Editor`, tj. uređivač kontrole pristupa. Pomoću njega definiramo kako i kada korisnici pristupaju određenim informacijama.

### Stvaranje nove kontrole pristupa

Uređivač kontrole pristupa nudi nam opciju dodavanja (`New`) i mijenjanja (`Edit`) pravila. Odabir jedne od tih opcija rezultira otvaranjem nove konzole putem koje se mogu dodati korisnici kojima treba dodijeliti određena prava ili restrikcije. Dodavanje korisnika je ujedno i prvi korak nakon kojega je potrebno prijeći na sljedeći tab (`Rights`). Pod tim tabom dodjeljuju se dozvole čitanja, uspoređivanja, pretrage, pisanja, brisanja itd. Tab `Targets` prikazuje nad kojim atributima trenutno odabrani element ima definirane dozvole. Pomnoću `Hosts` definira se na koju domenu će se pravila odnositi. Tab `Times` definira u kojem će vremenu kontrola pristupa biti aktivna.

#### Anoniman pristup

Omogućavanje anonimnog pristupa vrši se na sljedeći način:

- Unutar imenika potrebno je odabrati domenu u navigacijskom stablu, te pozvati izbornik sa desnim klikom i sa njega odabrati `Set Access Permissions`.
- Odabrati unos novog pravila klikom na `New`.
- Dati opisni naziv novom pravilu (npr. `Anoniman pristup na testzone.local`).
- Provjeriti da se pod tabom `Users` nalaze svi korisnici `All Users`.
- Pod tabom prava (`Rights`) označiti SAMO read, compare i search prava.
- Pod tabom `Targets` kliknuti na tipku `This Entry`, a u tablici atributa maknuti oznaku sa atributa `userPassword`.
- Pod tabom `Hosts` dodati domenu u filtre u obliku `*.<domena>.<sufiks>` (`*.testzone.local`).

## SSH prijava koristeći LDAP autentifikaciju

Svrha SSH je sigurno spajanje na udaljeno računalo sa ciljem izdavanja određenih naredbi tom istom računalu. Ukoliko to udaljeno računalo koristi LDAP imenik logično je autentifikaciju vršiti uz pomoć LDAP računa, a ne na neki drugi, redundantni način.

Testni slučaj predstavlja Fedora 18 server sa LDAP imenikom i Ubuntu 12.10 računalo sa namještenim SSH‐om. Informacije koje su potrebne prije pristupanja rješavanju problema su:

- LDAP korisnički račun za autentifikaciju
- DNS LDAP servera ili IP adresa istog
- Korisnički račun kojim se šalju upiti LDAP‐u
- SuperUser pristup
- Lokaciju korisničkih računa u LDAP‐u (path)

Radnje se odvijaju na klijent računalu (u ovom slučaju Ubuntu 12.10), a koraci u omogućavanju SSH prijave koristeći LDAP autentifikaciju su:

1. Dodati LDAP korisnički račun kao standardnog korisnika na računalo:

    ``` shell
    # sudo adduser
    ```

    Lozinka koja se koristi uz korisnika se ne koristi, ali preporučeno je zabilježiti je i osigurati snagu iste.

1. Instalirati `ldap‐utils` i `libpam‐ldap`:

    ``` shell
    # sudo apt-get install ldap-utils libpam-ldap
    ```

1. Napraviti sigurnosnu kopiju datoteke `/etc/pam_ldap.conf`:

    ``` shell
    # sudo cp /etc/pam_ldap.conf /etc/pam_ldap.conf.bak
    ```

1. Otvoriti datoteku `/etc/pam_ldap.conf` i zakomentirati sve što se nalazi u datoteci (dodati # na početak svake linije) te na početak dodati:

    ```
    host ldap.testzone.local
    base ou=Users,dn=ldap,dn=testzone,dn=local
    ldap_version 3
    binddn cn=ldapauth,ou=Service accounts,dn=ldap,dn=testzone,dn=local
    bindpw <lozinka za ldapauth korisnički račun>
    pam_password crypt
    pam_login_attribute name
    ```

1. Napraviti sigurnosnu kopiju datoteke `/etc/pam.d/common‐auth`:

    ``` shell
    # sudo cp /etc/pam.d/common-auth /etc/pam.d/common-auth.bak
    ```

1. Otvoriti datoteku `/etc/pam.d/common‐auth` i iznad linije `auth required pam_unix.so` dodati:

    ``` shell
    # auth sufficient pam_ldap.so debug
    ```

    Redoslijed je bitan zbog toga što se pam metode čitaju od vrha prema dnu.
