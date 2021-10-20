---
author: Vedran Miletić
---

# Konfiguracija sustava za upravljanje bazom podataka MariaDB

[MariaDB](https://mariadb.org/) ([Wikipedia](https://en.wikipedia.org/wiki/MariaDB)) je sustav za upravljanje relacijskom bazom podataka, razvijan od strane zajednice i komercijalno podržan. Razvoj MariaDB-a započet je odvajanjem od [MySQL](https://www.mysql.com/)-a izvedenim od strane nekolicine izvornih programera MySQL-a zbog zabrinutosti oko njegove budućnosti povodom [akvizicije Suna od strane Oraclea](https://www.directionsmag.com/article/2357). Prvenstveni cilj stvaranja projekta neovisnog o Oracleu je želja da softvera ostane slobodan i otvorenog koda pod licencom GNU General Public License.

MariaDB zadržava visok nivo kompatibilnosti s MySQL-om, ali proširuje njegovu funkcionalnost tako da nudi i značajke koje MySQL nema. [Kao i MySQL](https://www.mysql.com/cloud/), MariaDB je ponuđena i u obliku usluge u oblaku (drugim riječima, sustava koji netko drugi održava) pod imenom [SkySQL](https://mariadb.com/products/skysql/).

Ime MariaDB dao je autor MySQL-a [Michael "Monty" Widenius](https://en.wikipedia.org/wiki/Michael_Widenius) prema svojoj mlađoj kćeri koja se zove Maria. Slično tome, ime MySQL dao je prema imenu starije kćeri koja se zove My.

## Konfiguracija poslužitelja MariaDB

!!! note
    Kako MariaDB zadržava što je moguće veću razinu kompatibilnosti s MySQL-om, uočit ćemo kako se u naredbama ljuske, konfiguracijskim naredbama i nazivima datoteka na mnogo mjesta može koristiti naziv MySQL umjesto MariaDB. Primjerice, klijent sučelja naredbenog retka dostupan je naredbama `mariadb` i `mysql`, a poslužitelj naredbama `mariadbd` (MariaDB daemon) i `mysqld` (MySQL daemon).

Nakon instalacije sva konfiguracija poslužitelja MariaDB nalazi se u `/etc/mysql`. Nama će u nastavku biti zanimljive samo datoteka `/etc/mysql/mariadb.cnf` i datoteke u direktorijima  `/etc/mysql/mariadb.conf.d` i `/etc/mysql/mariadb.conf.d`. Razmotrimo za početak sadržaj datoteke `/etc/mysql/mariadb.cnf`:

``` ini
# The MariaDB configuration file
#
# The MariaDB/MySQL tools read configuration files in the following order:
# 1. "/etc/mysql/mariadb.cnf" (this file) to set global defaults,
# 2. "/etc/mysql/conf.d/*.cnf" to set global options.
# 3. "/etc/mysql/mariadb.conf.d/*.cnf" to set MariaDB-only options.
# 4. "~/.my.cnf" to set user-specific options.
#
# If the same option is defined multiple times, the last one will apply.
#
# One can use all long options that the program supports.
# Run program with --help to get a list of available options and with
# --print-defaults to see which it would actually understand and use.

#
# This group is read both both by the client and the server
# use it for options that affect everything
#
[client-server]

# Import all .cnf files from configuration directory
!includedir /etc/mysql/conf.d/
!includedir /etc/mysql/mariadb.conf.d/
```

Uočimo kako gotovo čitav sadržaj datoteke čine komentari i prazni retci, osim odjeljka `[client-server]` i dvije naredbe `!includedir` kojima se uključuju konfiguracijske datoteke iz dva već ranije spomenuta direktorija.

U direktoriju `/etc/mysql/conf.d/` datoteka `mysql.cnf` sadrži samo oznaku odjeljka, a datoteka `mysqldump.cnf` tiče se konfiguracije naredbe `mysqldump` koja omogućuje pohranjivanje baza u tekstualnom zapisu kao niz SQL upita i koja nam u nastavku neće trebati.

U direktoriju `/etc/mysql/mariadb.conf.d/` nalazimo konfiguraciju klijenta (`50-client.cnf`), svih aplikacija za pristup poslužitelju (`50-mysql-clients.cnf`), konfiguraciju za sigurno pokretanje u slučaju problema (`50-mysqld_safe.cnf`) i konfiguraciju poslužitelja (`50-server.cnf`). Razmtorimo sadržaj posljednje datoteke:

``` ini
#
# These groups are read by MariaDB server.
# Use it for options that only the server (but not clients) should see
#
# See the examples of server my.cnf files in /usr/share/mysql

# this is read by the standalone daemon and embedded servers
[server]

# this is only for the mysqld standalone daemon
[mysqld]

#
# * Basic Settings
#
user                    = mysql
pid-file                = /run/mysqld/mysqld.pid
socket                  = /run/mysqld/mysqld.sock
#port                   = 3306
basedir                 = /usr
datadir                 = /var/lib/mysql
tmpdir                  = /tmp
lc-messages-dir         = /usr/share/mysql
#skip-external-locking

# Instead of skip-networking the default is now to listen only on
# localhost which is more compatible and is not less secure.
bind-address            = 127.0.0.1

# ...
```

Uočimo kako su u odjeljku `[mysqld]` koji se odnosi na poslužitelj MariaDB navedene brojne konfiguracijske naredbe, primjerice:

- `user`, kojom se postavlja ime korisnika koji pokreće poslužitelj,
- `pid`, kojom se postavlja putanja do datoteke u kojoj je naveden PID procesa poslužitelja,
- `datadir`, kojom se navodi putanja do podatkovnog direktorija u kojem se spremaju baze podataka i binarni logovi
- `tmpdir`, kojom se navodi putanja do direktorija s privremenim datotekama i
- `lc-messages-dir`, kojom se navodi putanja do direktorija s lokalizacijskim datotekama (prijevodima na različite jezike).

Povežimo se klijentom na poslužitelj:

``` shell
$ sudo mariadb
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 52
Server version: 10.3.25-MariaDB-0ubuntu0.20.04.1 Ubuntu 20.04

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]>
```

U ljusci klijenta MariaDB dostupne su nam naredbe jezika SQL i brojne druge specifične za MariaDB. Naredbe jezika SQL su slične među različitim sustavima za upravljanje bazama podataka i njih ćemo koristiti nešto kasnije. Jedna od naredbi specifičnih za MariaDB je `SHOW` ([dokumentacija](https://mariadb.com/kb/en/show/)) koja može prikazati popis autora MariaDB korištenjem parametra `AUTHORS` ([dokumentacija](https://mariadb.com/kb/en/show-authors/)):

``` sql
SHOW AUTHORS;
```

```
+--------------------------------+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Name                           | Location                              | Comment                                                                                                                                 |
+--------------------------------+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Michael (Monty) Widenius       | Tusby, Finland                        | Lead developer and main author                                                                                                          |
| Sergei Golubchik               | Kerpen, Germany                       | Architect, Full-text search, precision math, plugin framework, merges etc                                                               |
| Igor Babaev                    | Bellevue, USA                         | Optimizer, keycache, core work                                                                                                          |
| Sergey Petrunia                | St. Petersburg, Russia                | Optimizer                                                                                                                               |
| Oleksandr Byelkin              | Lugansk, Ukraine                      | Query Cache (4.0), Subqueries (4.1), Views (5.0)                                                                                        |
(...)
+--------------------------------+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
102 rows in set (0.000 sec)
```

Primjenom parametra `DATABASES` ([dokumentacija](https://mariadb.com/kb/en/show-databases/)) prikazat će popis baza podataka koje postoje na poslužitelju:

``` sql
SHOW DATABASES;
```

```
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
+--------------------+
3 rows in set (0.000 sec)
```

Naposlijetku, moguće je prikazati globalne varijable ili varijable trenutne sesije parametrom `VARIABLES` ([dokumentacija](https://mariadb.com/kb/en/show-variables/)). U slučaju da želimo prikazati globalnu varijablu, dodajemo i parametar `GLOBAL`, a kod navođenja imena varijable u obliku znakovnog niza koristimo standardni SQL operator `LIKE` ([dokumentacija](https://mariadb.com/kb/en/like/)). MariaDB [podržava jednostruke i dvostruke navodnike kod navođenja znakovnih nizova](https://mariadb.com/kb/en/string-literals/). Varijabla koja nas zanima je `have_ssl` pa je naredba oblika:

``` sql
SHOW GLOBAL VARIABLES LIKE 'have_ssl';
```

```
+---------------+----------+
| Variable_name | Value    |
+---------------+----------+
| have_ssl      | DISABLED |
+---------------+----------+
1 row in set (0.001 sec)
```

Uočimo kako MariaDB trenutno ne koristi SSL/TLS za šifriranje komunikacije klijenta i poslužitelja, što ćemo promijeniti u nastavku.

## Šifriranje podataka

MariaDB [podržava šifriranje](https://mariadb.com/kb/en/securing-mariadb-encryption/):

- [podataka u prijenosu](https://mariadb.com/docs/security/encryption/in-transit/) (engl. *data in transit*), odnosno upita poslanih od klijenta do poslužitelja i njihovih rezultata poslanih od poslužitelja do klijenta te
- [podataka na odmoru](https://mariadb.com/docs/security/encryption/at-rest/) (engl. *data at rest*), odnosno spremljenih sadržaja baza podataka i binarnih logova poslužitelja.

### Šifriranje podataka u prijenosu

#### Stvaranje asimetričnih ključeva

[Ključeve i X.509 certifikate](https://mariadb.com/docs/security/encryption/in-transit/create-self-signed-certificates-keys-openssl/#create-self-signed-certificates-keys-openssl) koje će MariaDB koristiti stvaramo korištenjem OpenSSL-a i Easy-RSA. Inicijalizirajmo infrastrukturu privatnog ključa i stvorimo ključ i certifikat autoriteta certifikata:

``` shell
$ easyrsa init-pki
$ easyrsa build-ca
```

Naravno, ako smo već ranije koristili Easy-RSA pa imamo uspostavljenu infrastrukturu privatnog ključa, neke od naredbi će nam javiti da već postoje napravljeni certifikati i ključevi. U tom slučaju, obzirom da samo učimo kako radi infrastruktura privatnog ključa u MariaDB, možemo iskorstiti certifikate i ključeve koje smo već koristili u drugim sustavima; u praksi to ne smijemo napraviti jer time smanjujemo sigurnost i naše instance MariaDB i tih drugih sustava.

Izgradimo ključ i certifikat za poslužitelj:

``` shell
$ easyrsa build-server-full moj-mariadb-posluzitelj nopass
```

Izgradimo ključ i certifikat koji će klijent koristiti:

``` shell
$ easyrsa build-client-full moj-mariadb-klijent nopass
```

Kopirajmo stvoreni certifikat autoriteta certifikata na mjesto s kojeg ćemo ga koristiti pa mu postavimo odgovarajućeg vlasnika i grupu te dozvole:

``` shell
$ sudo mkdir /etc/mysql/certs

$ sudo cp pki/ca.crt /etc/mysql/certs/ca-cert.pem
$ sudo chmod go-rwx /etc/mysql/certs/ca-cert.pem
```

Napravimo isto za poslužiteljski certifikat i ključ:

``` shell
$ sudo cp pki/issued/moj-mariadb-posluzitelj.crt /etc/mysql/certs/server-cert.pem
$ sudo chmod go-rwx /etc/mysql/certs/server-cert.pem

$ sudo cp pki/private/moj-mariadb-posluzitelj.key /etc/mysql/certs/server-key.pem
$ sudo chmod go-rwx /etc/mysql/certs/server-key.pem
```

Naposlijetku, postavimo na isto mjesto i klijentski certifikat i ključ:

``` shell
$ sudo cp pki/issued/moj-mariadb-klijent.crt /etc/mysql/certs/client-cert.pem
$ sudo chmod go-rwx /etc/mysql/certs/client-cert.pem

$ sudo cp pki/private/moj-mariadb-klijent.key /etc/mysql/certs/client-key.pem
$ sudo chmod go-rwx /etc/mysql/certs/client-key.pem
```

#### Uključivanje ključeva u konfiguracijsku datoteku

Nakon stvaranja privatnog ključa i X.509 certifikata te njihovog premještanja u `/etc/mysql/certs` uključit ćemo njihovo korištenje konfiguracijskim naredbama:

- `ssl_cert` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_cert/)), kojom ćemo navesti X.509 certifikat u formatu PEM
- `ssl_key` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_key/)), kojom ćemo navesti privatni ključ u formatu PEM
- `ssl_ca` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_ca/)), kojom ćemo, u slučaju korištenja samopotpisanih certifikata, navesti i certifikat autoriteta certifikata koji ga je potpisao

Više informacija o ovim konfiguracijskim naredbama može se naći u [službenoj dokumentaciji](https://mariadb.com/docs/security/encryption/in-transit/enable-tls-server/).

MariaDB ne preporuča modificiranje postojećih konfiguracijskih datoteka, već dodavanje novih. U direktoriju `/etc/mysql/mariadb.conf.d` korištenjem uređivača teksta GNU nano ili bilo kojeg drugog stvorimo datoteku `90-ssl.cnf`:

``` shell
$ sudo nano /etc/mysql/mariadb.conf.d/90-ssl.cnf
```

Jako je važno da nastavak bude `.cnf`, a ne `.conf` ili neka treća jer konfiguracijska naredba `!includedir` učitava u danom direktoriju samo datoteke s nastavkom `.cnf`. U njoj postavimo željene postavke poslužiteljskih i klijentskih ključeva i certifikata:

``` ini
[mariadb]

ssl_cert = /etc/mysql/certs/server-cert.pem
ssl_key = /etc/mysql/certs/server-key.pem
ssl_ca = /etc/mysql/certs/ca-cert.pem

[client-mariadb]

ssl_cert = /etc/mysql/certs/client-cert.pem
ssl_key = /etc/mysql/certs/client-key.pem
ssl_ca = /etc/mysql/certs/ca-cert.pem
```

Umjesto odjeljka `[mariadb]` također smo konfiguracijske naredbe mogli staviti i u odjeljak `[server]` ili `[mysqld]` koje poslužitelj čita kod pokretanja (detaljnije objašnjenje pojedinih odjeljaka moguće je [naći u dijelu službene dokumentacije Configuring MariaDB with Option Files](https://mariadb.com/kb/en/configuring-mariadb-with-option-files/#server-option-groups)), dok smo umjesto odjeljka `[client-mariadb]` također mogli koristiti `[client]` ([detalji](https://mariadb.com/kb/en/configuring-mariadb-with-option-files/#client-option-groups)).

Pokrenimo ponovno poslužitelj kako bi učitao nove postavke:

``` shell
$ sudo systemctl restart mariadb
```

Kako bismo se uvjerili da je TLS uključen, povežimo se klijentom (naredba `mariadb`) kao i ranije pa iskoristimo naredbu `SHOW` za prikaz vrijednosti globalne varijable `have_ssl`:

``` sql
SHOW GLOBAL VARIABLES LIKE 'have_ssl';
```

```
+---------------+-------+
| Variable_name | Value |
+---------------+-------+
| have_ssl      | YES   |
+---------------+-------+
1 row in set (0.001 sec)
```

#### Dodatne postavke TLS-a

Slično kao kod drugih softvera koji koriste TLS, možemo navesti dodatne postavke šifriranja. Neke od korisnih naredbi su:

- `require_secure_transport` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/require_secure_transport/)), postavljanjem koje na `ON` možemo tražiti da se svako povezivanje klijenta sa poslužiteljem događa uz korištenje TLS-a (zadana vrijednost je `OFF`) (zahtijeva MariaDB 10.5 ili noviju)
- `ssl_capath` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_crlpath/)), kojom navodimo putanju do direktorija s certifikatima autoriteta certifikata kojima se vjeruje, primjerice `/usr/share/ca-certificates/mozilla`
- `ssl_cipher` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_cipher/)), kojom navodimo popis šifrarnika koji će se koristiti
- `tls_version` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/tls_version/)), kojom navodimo verziju TLS-a koji će se koristiti (zahtijeva MariaDB 10.4 ili noviju)

!!! note
    Za generiranje adekvatnih sigurnosnih postavki možemo ponovno iskoristiti [Mozillin generator SSL konfiguracije](https://ssl-config.mozilla.org/) i kao poslužiteljski softver odabrati MySQL.

### Šifriranje podataka na odmoru

MariaDB podržava šifriranje podataka na odmoru korištenjem [HashiCorp Vaulta](https://www.hashicorp.com/products/vault), [Amazon Web Services (AWS) Key Management Servicea](https://aws.amazon.com/kms/) ili datoteke s ključevima. Kako nemamo trenutno postavljen vlastiti oblak koji koristi HashiCorp Vault i nemamo želju postavljati sve potrebne usluge na AWS-u, koristit ćemo [dodatak file_key_management](https://mariadb.com/docs/reference/mdb/plugins/file_key_management/) ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/)) koji upravlja šifriranjem podataka na odmoru [pomoću datoteke s ključevima](https://mariadb.com/docs/security/encryption/at-rest/encryption-plugins/file-key-management/understand-file-key-management/) za šifriranje baza podataka i binarnih logova.

#### Učitavanje dodatka

U direktoriju `/etc/mysql/mariadb.conf.d` stvorimo datoteku `91-file-key-management.cnf` u kojoj ćemo konfiguracijskom naredbom `plugin_load_add` ([dokumentacija](https://mariadb.com/docs/reference/mdb/cli/mariadbd/plugin-load-add/)) učitati dodatak `file_key_management`:

``` ini
[mariadb]

plugin_load_add = file_key_management
```

#### Stvaranje simetričnih ključeva

Korišteni algoritam za šifriranje je [Advanced Encryption Standard (AES)](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) i podržane veličine ključeva su 128, 192 i 256 bita. Ključeve je moguće navesti po želji ili generirati OpenSSL-om kao slučajne podatke određene duljine. Kako se duljina navodi u bajtovima, za stvaranje 256-bitnog ključa generirat ćemo 32-bajtni slučajni podatak naredbom `openssl rand`:

``` shell
$ openssl rand -hex 32
1cece9bfd9e0263da50bcf02d088b12889cf1eddeb7f8ffdd719b9ab23359be2
```

U slučaju da trebamo više ključeva, naredbu ćemo iskoristiti više puta. Dodatne informacije o ovoj naredbi i parametrima koje podržava moguće je pronaći u man stranici `rand(1ssl)` (naredba `man 1ssl rand`). Stvorimo direktorij za ključeve za šifriranje:

``` shell
$ sudo mkdir /etc/mysql/encryption
```

Stvorimo u njemu datoteku `/etc/mysql/encryption/keyfile` u kojoj su u svakom retku identifikator ključa (cijeli broj), znak točke sa zarezom i ključ:

```
1;1cece9bfd9e0263da50bcf02d088b12889cf1eddeb7f8ffdd719b9ab23359be2
2;df8762cdcd39d0e095a538ecad06ca94f55f805cc76841ed304ec26e4f46d2a0
11;6bf495bc15a970f8e51b7be49e0ebad5b74fc8febccd1ff45e259ca82e35a973
34;34e791122e8cb9fe983534d33bc45522d3e9ca2ec373e720eb08fc34e7f59b7b
```

Identifikatori ključa moraju biti međusobno različiti, ali ne moraju ići po redu. Dodijelimo tu datoteku korisniku `mysql` i grupi `mysql` pa je skrijmo od ostalih:

``` shell
$ sudo chown mysql:mysql /etc/mysql/encryption/keyfile
$ sudo chmod o-rwx /etc/mysql/encryption/keyfile
```

#### Uključivanje datoteke s ključevima

Dopunimo datoteku `91-file-key-management.cnf` dodavanjem konfiguracijske naredbe `file_key_management_filename` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_filename)) tako da bude oblika:

``` ini
[mariadb]

plugin_load_add = file_key_management
file_key_management_filename = /etc/mysql/encryption/keyfile
```

!!! tip
    Datoteku s ključevima je [moguće šifrirati](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#encrypting-the-key-file) pa navesti ključ za dešifriranje konfiguracijskom naredbom `file_key_management_filekey` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_filekey)).

#### Odabir algoritma za šifriranje

Podržane su dvije varijante algoritma AES:

- AES-CBC, koji koristi [Cipher block chaining (CBC)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Cipher_block_chaining_(CBC))
- AES-CTR, koji koristi kombinaciju [Counter (CTR)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)) i [Galois/Counter Mode (GCM)](https://en.wikipedia.org/wiki/Galois/Counter_Mode) (zahtijeva da korišteni OpenSSL podržava te algoritme)

Kombinacija CTR-a i GCM-a se smatra boljim odabirom i naša je verzija OpenSSL-a podržava pa ćemo njeno korištenje uključiti dodavanjem konfiguracijske naredbe `file_key_management_encryption_algorithm` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_encryption_algorithm)) u datoteku `91-file-key-management.cnf` tako da bude oblika:

``` ini
[mariadb]

plugin_load_add = file_key_management
file_key_management_filename = /etc/mysql/encryption/keyfile
file_key_management_encryption_algorithm = AES_CTR
```

Ako korišteni OpenSSL ne podržava taj algoritam, zamijenit ćemo posljednju liniju za:

``` ini
file_key_management_encryption_algorithm = AES_CBC
```

Pokrenimo ponovno poslužitelj kako bi učitao nove postavke:

``` shell
$ sudo systemctl restart mariadb
```

#### Šifriranje tablica u bazi podataka

[Šifriranje tablica u bazi podataka](https://mariadb.com/kb/en/innodb-encryption/) zahtijeva od nas da prvo stvorimo bazu podataka. Povežimo se na poslužitelj klijentom `mariadb` kao ranije pa stvorimo bazu SQL upitom `CREATE DATABASE` ([dokumentacija](https://mariadb.com/kb/en/create-database/)):

``` sql
CREATE DATABASE mojabaza;
```

```
Query OK, 1 row affected (0.000 sec)
```

Odaberimo je za korištenje naredbom `USE` ([dokumentacija](https://mariadb.com/kb/en/use/)):

``` sql
USE mojabaza;
```

```
Database changed
```

Uočimo kako nam piše ime baze unutar uglatih zagrada gdje je ranije pisalo `(none)`. Stvorimo šifriranu tablicu SQL upitom `CREATE TABLE` ([dokumentacija](https://mariadb.com/kb/en/create-table/)) uz navođenje opcije `ENCRYPTED` ([dokumentacija](https://mariadb.com/kb/en/create-table/#encrypted)) postavljene na `YES`:

``` sql
CREATE TABLE names ( id int PRIMARY KEY, str varchar(50) ) ENCRYPTED=YES;
```

```
Query OK, 0 rows affected (0.004 sec)
```

Ubacimo i nešto podataka u tablicu SQL upitom `INSERT` ([dokumentacija](https://mariadb.com/kb/en/insert/)) jer će nam dobro doći kasnije da nam tablica nije prazna:

``` sql
INSERT INTO names VALUES (1, 'Tomislav');
```

```
Query OK, 1 row affected (0.001 sec)
```

``` sql
INSERT INTO names VALUES (2, 'Arijana');
```

```
Query OK, 1 row affected (0.001 sec)
```

Uvjerimo se da su podaci u tablici SQL upitom `SELECT` ([dokumentacija](https://mariadb.com/kb/en/select/)):

``` sql
SELECT * FROM names;
```

```
+----+----------+
| id | str      |
+----+----------+
|  1 | Tomislav |
|  2 | Arijana  |
+----+----------+
2 rows in set (0.000 sec)
```

Ova tablica i svi uneseni podaci su šifrirani korištenjem ključa s identifikatorom 1, koji je zadani ključ. U to da je ključ s identifikatorom 1 zadani možemo se uvjeriti dohvaćanjem vrijednosti varijable sesije `innodb_default_encryption_key_id` ([dokumentacija](https://mariadb.com/kb/en/innodb-system-variables/#innodb_default_encryption_key_id)). ([InnoDB](https://mariadb.com/kb/en/innodb/) je zadani pogon za pohranu tablica u MariaDB, a dostupan je za korištenje i u MySQL-u.) Za dohvaćanje te varijable iskoristit ćemo naredbu `SHOW` na način:

``` sql
SHOW SESSION VARIABLES LIKE 'innodb_default_encryption_key_id';
```

```
+----------------------------------+-------+
| Variable_name                    | Value |
+----------------------------------+-------+
| innodb_default_encryption_key_id | 1     |
+----------------------------------+-------+
1 row in set (0.001 sec)
```

Korištenje ključa različitog od zadanog možemo postaviti kod stvaranja tablice navođenjem dodatne opcije `ENCRYPTION_KEY_ID` ([dokumentacija](https://mariadb.com/kb/en/create-table/#encryption_key_id)) postavljene na identifikator ključa:

``` sql
CREATE TABLE surnames ( id int PRIMARY KEY, str varchar(50) ) ENCRYPTED=YES ENCRYPTION_KEY_ID=2;
```

```
Query OK, 0 rows affected (0.004 sec)
```

Ponovno, ubacimo nešto podataka da imamo s čime raditi kasnije:

``` sql
INSERT INTO surnames VALUES (1, 'Lasić');
```

```
Query OK, 1 row affected (0.001 sec)
```

``` sql
INSERT INTO surnames VALUES (2, 'Rebić');
```

```
Query OK, 1 row affected (0.001 sec)
```

``` sql
INSERT INTO surnames VALUES (3, 'Kutleša');
```

```
Query OK, 1 row affected (0.001 sec)
```

Uvjerimo se ponovno da su podaci u tablici:

``` sql
SELECT * FROM surnames;
```

```
+----+----------+
| id | str      |
+----+----------+
|  1 | Lasić    |
|  2 | Rebić    |
|  3 | Kutleša  |
+----+----------+
3 rows in set (0.000 sec)
```

#### Šifriranje binarnih logova

[Šifriranje binarnih logova](https://mariadb.com/kb/en/encrypting-binary-logs/) moguće je uključiti dodavanjem konfiguracijske naredbe `encrypt_binlog` ([dokumentacija](https://mariadb.com/kb/en/replication-and-binary-log-system-variables/#encrypt_binlog)) postavljene na vrijednost `ON`:

``` ini
[mariadb]

encrypt_binlog = ON
```

## Privilegije korisnika

Dosad smo pokretali klijent MariaDB kao korijenski korisnik i zbog toga imali sva prava na svim bazama i svim tablicama u tim bazama. U praksi tu mogućnost želimo koristiti samo kad nam treba, a u redovnom radu ograničiti privilegije korisnika u radu s bazom i tablicama u njoj na one koje su zaista potrebne. Naime, u slučaju neograničenih prava zlonamjeran SQL upit može vrlo lako promijeniti ili izbrisati podatke kojima inače uopće ne pristupa i nema razloga pristupati. U aplikaciji želimo spriječiti da takav upit uopće dođe do sustava za upravljanje bazom podataka; ako ipak pronađe put do njega, u sustavu za upravljanje bazom podataka želimo spriječiti da bude uspješan.

### Korisnici

MariaDB omogućuje [upravljanje korisnicima](https://mariadb.com/kb/en/user-account-management/) korištenjem [naredbi sličnih SQL upitima](https://mariadb.com/kb/en/account-management-sql-commands/):

- `CREATE USER` ([dokumentacija](https://mariadb.com/kb/en/create-user/)) stvara novi korisnički račun
- `ALTER USER` ([dokumentacija](https://mariadb.com/kb/en/alter-user/)) mijenja postojeći korisnički račun
- `RENAME USER` ([dokumentacija](https://mariadb.com/kb/en/rename-user/)) preimenuje korisnika
- `DROP USER` ([dokumentacija](https://mariadb.com/kb/en/drop-user/)) briše jedan ili više korisničkih računa

Stvorimo korisnika sa svojim imenom koji će se prijavljivati s lokalnog računala korištenjem lozinke:

``` sql
CREATE USER 'ivanzhegalkin'@'localhost' IDENTIFIED BY 'ne3pr0v4lj1v4t4jn4l0z1nka';
```

```
Query OK, 0 rows affected (0.000 sec)
```

Sada se možemo prijaviti kao taj korisnik, što ćemo i napraviti u drugom terminalu bez prekidanja postojeće sesije korijenskog korisnika. Parametrom `-u` navodimo korisničko ime korisnika koji se prijavljuje, a parametrom `-p` navodimo da ćemo izvršiti prijavu korištenjem zaporke:

``` shell
$ mariadb -u ivanzhegalkin
ERROR 1045 (28000): Access denied for user 'ivanzhegalkin'@'localhost' (using password: NO)

$ mariadb -u ivanzhegalkin -p
Enter password:
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 58
Server version: 10.3.25-MariaDB-0ubuntu0.20.04.1 Ubuntu 20.04

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]>
```

Pokušamo li sada pristupiti bazi `mojabaza` od ranije, uočit ćemo kako to ne možemo izvesti:

``` sql
USE mojabaza;
```

```
ERROR 1044 (42000): Access denied for user 'ivanzhegalkin'@'localhost' to database 'mojabaza'
```

Ostavimo otvorena oba terminala jer ćemo u nastavku dodjeljivati i provjeravati dodijeljene privilegije.

### Privilegije

Korijenski korisnik može naredbom `GRANT` ([dokumentacija](https://mariadb.com/kb/en/grant/)) dodjeljivati privilegije drugim korisnicima, uključujući i privilegiju da mogu dalje dodjeljivati privilegije koje posjeduju. U nastavku ćemo raditi kao korijenski korisnik, a kao drugi korisnik možemo se eventualno uvjeriti da zaista posjedujemo dodijeljene privilegije.

Postoji više nivoa privilegija, a nama su zanimljive:

1. globalne privilegije ([popis u službenoj dokumentaciji](https://mariadb.com/kb/en/grant/#global-privileges)), u koje spadaju pregledavanje popisa svih baza koje postoje na sustavu naredbom `SHOW DATABASES`, stvaranje korisnika naredbom `CREATE USER` i dodjeljivanje privilegija drugom korisniku naredbom `GRANT OPTION`:
1. privilegije baze podataka ([popis u službenoj dokumentaciji](https://mariadb.com/kb/en/grant/#database-privileges)), u koje spadaju stvaranje baze naredbom `CREATE DATABASE`, brisanje baze naredbom `DROP DATABASE` i dodjeljivanje tih privilegija drugom korisniku naredbom `GRANT OPTION`;
1. privilegije tablice ([popis u službenoj dokumentaciji](https://mariadb.com/kb/en/grant/#table-privileges)), u koje spadaju stvaranje tablice naredbom `CREATE TABLE`, stvaranje pogleda naredbom `CREATE VIEW`, promjena strukture tablice naredbom `ALTER TABLE`, čitanje podataka iz tablice naredbom `SELECT`, umetanje podataka u tablicu naredbom `INSERT`, promjenu podataka u tablici naredbom `UPDATE`, brisanje tablice ili pogleda naredbama `DROP TABLE` i `DROP VIEW`, stvaranje indeksa tablice naredbom `CREATE INDEX` i druge.

Dodijeljene privilegije mogu prikazati naredbom `SHOW GRANTS` ([dokumentacija](https://mariadb.com/kb/en/show-grants/)).

``` sql
SHOW GRANTS FOR 'ivanzhegalkin'@'localhost';
```

```
+----------------------------------------------------------------------------------------------------------------------+
| Grants for ivanzhegalkin@localhost                                                                                   |
+----------------------------------------------------------------------------------------------------------------------+
| GRANT USAGE ON *.* TO `ivanzhegalkin`@`localhost` IDENTIFIED BY PASSWORD '*FB0DD9B1D2A1CF4B2203A2FBF5E1EAE4A2008788' |
+----------------------------------------------------------------------------------------------------------------------+
1 row in set (0.000 sec)
```

#### Dodjeljivanje privilegija

Privilegija `USAGE` ne omogućuje ništa specijalno, već samo označava da korisnik postoji. Dodijelimo korisniku pravo pregledavanja baza koje postoje na sustavu:

``` sql
GRANT SHOW DATABASES ON *.* TO 'ivanzhegalkin'@'localhost';
```

```
Query OK, 0 rows affected (0.000 sec)
```

Kako bismo bili sigurni da su naše promjene privilegija aktivne, moramo sprati dodijeljene privilegije naredbom `FLUSH` ([dokumentacija](https://mariadb.com/kb/en/flush/)):

``` sql
FLUSH PRIVILEGES;
```

```
Query OK, 0 rows affected (0.000 sec)
```

Uočimo promjenu kod izlaza naredbe `SHOW GRANTS`:

``` sql
SHOW GRANTS FOR 'ivanzhegalkin'@'localhost';
```

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Grants for ivanzhegalkin@localhost                                                                                            |
+-------------------------------------------------------------------------------------------------------------------------------+
| GRANT SHOW DATABASES ON *.* TO `ivanzhegalkin`@`localhost` IDENTIFIED BY PASSWORD '*FB0DD9B1D2A1CF4B2203A2FBF5E1EAE4A2008788' |
+-------------------------------------------------------------------------------------------------------------------------------+
1 row in set (0.000 sec)
```

Uvjerimo se kao obični korisnik da smo zaista dobili privilegiju pregledavanja popisa baza podataka:

``` sql
SHOW DATABASES;
```

```
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mojabaza           |
| mysql              |
| performance_schema |
+--------------------+
4 rows in set (0.000 sec)
```

Kako se čitanje podataka iz baze vrši naredbom `SELECT`, pravo čitanja podataka iz svih tablica korisniku ćemo dati na način:

``` sql
GRANT SELECT ON mojabaza.* to 'ivanzhegalkin'@'localhost';
```

```
Query OK, 0 rows affected (0.000 sec)
```

``` sql
FLUSH PRIVILEGES;
```

```
Query OK, 0 rows affected (0.000 sec)

```

``` sql
SHOW GRANTS FOR 'ivanzhegalkin'@'localhost';
```

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Grants for ivanzhegalkin@localhost                                                                                            |
+-------------------------------------------------------------------------------------------------------------------------------+
| GRANT SHOW DATABASES ON *.* TO `ivanzhegalkin`@`localhost` IDENTIFIED BY PASSWORD '*FB0DD9B1D2A1CF4B2203A2FBF5E1EAE4A2008788' |
| GRANT SELECT ON `mojabaza`.* TO `ivanzhegalkin`@`localhost`                                                                   |
+-------------------------------------------------------------------------------------------------------------------------------+
2 rows in set (0.000 sec)
```

Uvjerimo se kao obični korisnik da zaista imamo privilegije pregledavanja tablica i čitanja podataka iz njih:

``` sql
USE mojabaza;
```

```
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
```

``` sql
SHOW TABLES;
```

```
+--------------------+
| Tables_in_mojabaza |
+--------------------+
| names              |
| surnames           |
+--------------------+
2 rows in set (0.000 sec)
```

``` sql
DESCRIBE names;
```

```
+-------+-------------+------+-----+---------+-------+
| Field | Type        | Null | Key | Default | Extra |
+-------+-------------+------+-----+---------+-------+
| id    | int(11)     | NO   | PRI | NULL    |       |
| str   | varchar(50) | YES  |     | NULL    |       |
+-------+-------------+------+-----+---------+-------+
2 rows in set (0.001 sec)
```

``` sql
DESCRIBE surnames;
```

```
+-------+-------------+------+-----+---------+-------+
| Field | Type        | Null | Key | Default | Extra |
+-------+-------------+------+-----+---------+-------+
| id    | int(11)     | NO   | PRI | NULL    |       |
| str   | varchar(50) | YES  |     | NULL    |       |
+-------+-------------+------+-----+---------+-------+
2 rows in set (0.001 sec)
```

``` sql
SELECT * FROM names;
```

```
+----+----------+
| id | str      |
+----+----------+
|  1 | Tomislav |
|  2 | Arijana  |
+----+----------+
2 rows in set (0.000 sec)
```

``` sql
SELECT * FROM surnames;
```

```
+----+----------+
| id | str      |
+----+----------+
|  1 | Lasić    |
|  2 | Rebić    |
|  3 | Kutleša  |
+----+----------+
3 rows in set (0.000 sec)
```

Uvjerimo se i da nismo dobili druge privilegije:

``` sql
INSERT INTO names VALUES (5, 'Marko');
```

```
ERROR 1142 (42000): INSERT command denied to user 'ivanzhegalkin'@'localhost' for table 'names'
```

``` sql
DROP TABLE surnames;
```

```
ERROR 1142 (42000): DROP command denied to user 'ivanzhegalkin'@'localhost' for table 'surnames'
```

Ako to želimo, na analogan način bismo uz pomoć [popisa privilegija tablice](https://mariadb.com/kb/en/grant/#table-privileges) mogli dodijeliti te privilegije kao i mogućnost promjene podataka, promjene strukture tablice i druge.

#### Opozivanje privilegija

Dodijeljene privilegije se opozivaju naredbom `REVOKE` ([dokumentacija](https://mariadb.com/kb/en/revoke/)) čiji je način korištenja potpuno analogan načinu korištenja naredbe `GRANT`.

### Uloge

[Uloge](https://mariadb.com/kb/en/roles_overview/) se koriste za grupiranje većeg broja privilegija zajedno, što olakšava dodjeljivanje privilegijama korisnicima. Privilegije se dodjeljuju ulozi, a onda se uloga dodjeljuje proizvoljnom broju korisnika. Naredba `CREATE ROLE` ([dokumentacija](https://mariadb.com/kb/en/create-role/)) stvara ulogu, a naredba `DROP ROLE` ([dokumentacija](https://mariadb.com/kb/en/drop-role/)) briše ulogu.

Za ilustraciju, možemo zamisliti sustav u kojem studenti tek uče raditi s bazama podataka pa imaju pravo čitanja podataka iz svih tablica iz svih baza, ali ne i zapisivanja podataka u njih. Ulogu `student` ćemo stvoriti, dodijeliti joj privilegije i dodijeliti je korisnicima na način:

``` sql
CREATE ROLE student;

GRANT SHOW DATABASES ON *.* TO student;
GRANT SELECT ON *.* TO student;

GRANT student to filip;
GRANT student to laura;
```

S druge strane, profesori imaju potrebu i unositi novi sadržaj u baze podataka i uređivati postojeći sadržaj u njima pa za ulogu `professor` imamo:

``` sql
CREATE ROLE professor;

GRANT SHOW DATABASES ON *.* TO professor;
GRANT SELECT ON *.* TO professor;
GRANT INSERT ON *.* TO professor;
GRANT UPDATE ON *.* TO professor;

GRANT professor to lucia;
GRANT professor to danijela;
GRANT professor to kristian;
```

Naredba `DROP ROLE` koristi se analogno naredbi `CREATE ROLE`.
