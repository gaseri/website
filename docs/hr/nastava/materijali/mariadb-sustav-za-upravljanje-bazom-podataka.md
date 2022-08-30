---
author: Vedran Miletić
---

# Konfiguracija sustava za upravljanje bazom podataka MariaDB

[MariaDB](https://mariadb.org/) ([Wikipedia](https://en.wikipedia.org/wiki/MariaDB)) je sustav za upravljanje relacijskom bazom podataka, razvijan od strane zajednice i komercijalno podržan. Razvoj MariaDB-a započet je odvajanjem od [MySQL](https://www.mysql.com/)-a izvedenim od strane nekolicine izvornih programera MySQL-a zbog zabrinutosti oko njegove budućnosti povodom [akvizicije Suna od strane Oraclea](https://www.directionsmag.com/article/2357). Prvenstveni cilj stvaranja projekta neovisnog o Oracleu je želja da softvera ostane slobodan i otvorenog koda pod licencom GNU General Public License.

MariaDB zadržava visok nivo kompatibilnosti s MySQL-om, ali proširuje njegovu funkcionalnost tako da nudi i značajke koje MySQL nema. [Kao i MySQL](https://www.mysql.com/cloud/), MariaDB je ponuđena i u obliku usluge u oblaku (drugim riječima, sustava koji netko drugi održava) pod imenom [SkySQL](https://mariadb.com/products/skysql/).

Ime MariaDB dao je autor MySQL-a [Michael "Monty" Widenius](https://en.wikipedia.org/wiki/Michael_Widenius) prema svojoj mlađoj kćeri koja se zove Maria. Slično tome, ime MySQL dao je prema imenu starije kćeri koja se zove My.

!!! note
    Kako MariaDB zadržava što je moguće veću razinu kompatibilnosti s MySQL-om, uočit ćemo kako se u naredbama ljuske, konfiguracijskim naredbama i nazivima datoteka na mnogo mjesta može koristiti naziv MySQL umjesto MariaDB. Primjerice, klijent sučelja naredbenog retka dostupan je naredbama `mariadb` i `mysql`, a poslužitelj naredbama `mariadbd` (MariaDB daemon) i `mysqld` (MySQL daemon).

## Slika `mariadb` na Docker Hubu

Sustav za upravljanje bazom podataka MariaDB koristit ćemo u obliku [Docker](https://www.docker.com/) [kontejnera](https://www.docker.com/resources/what-container/). Na [Docker Hubu](https://hub.docker.com/) moguće je pronaći sliku [mariadb](https://hub.docker.com/_/mariadb), koja je jedna od [službenih slika](https://hub.docker.com/search?type=image&image_filter=official). Koristit ćemo [verziju 10.7](https://mariadb.com/kb/en/changes-improvements-in-mariadb-107/), koja je posljednja izdana stabilna verzija. Pokretanje kontejnera `maridab` izvodimo naredbom `docker run`:

``` shell
$ docker run mariadb:10.7
Unable to find image 'mariadb:10.7' locally
10.7: Pulling from library/mariadb
e0b25ef51634: Pull complete
8aa3f605beb6: Pull complete
c43298fa9eba: Pull complete
f565e2a61005: Pull complete
3b5a73a7467f: Pull complete
d219b4dd5889: Pull complete
008719f0a8ad: Pull complete
cdb2ef26c44d: Pull complete
16f6e068c19c: Pull complete
ecfd25d3e0e6: Pull complete
fc6e322e4875: Pull complete
Digest: sha256:9d2cde0e154989d499114bf468fab23497120cf889fb6965050c0f8fcf69d037
Status: Downloaded newer image for mariadb:10.7
2022-04-15 16:51:01+00:00 [Note] [Entrypoint]: Entrypoint script for MariaDB Server 1:10.7.3+maria~focal started.
2022-04-15 16:51:01+00:00 [Note] [Entrypoint]: Switching to dedicated user 'mysql'
2022-04-15 16:51:01+00:00 [Note] [Entrypoint]: Entrypoint script for MariaDB Server 1:10.7.3+maria~focal started.
2022-04-15 16:51:01+00:00 [ERROR] [Entrypoint]: Database is uninitialized and password option is not specified
        You need to specify one of MARIADB_ROOT_PASSWORD, MARIADB_ALLOW_EMPTY_ROOT_PASSWORD and MARIADB_RANDOM_ROOT_PASSWORD
```

Docker slika ne dozvoljava pokretanje bez navođenja zaporke korijenskog korisnika koja će se korstiti. Navedimo ju putem varijable okoline `MARIADB_ROOT_PASSWORD` parametrom `--env` i dodajmo parametar `--detach` kako bismo zadržali mogućnost daljnjeg rada u terminalu:

``` shell
$ docker run --detach --env MARIADB_ROOT_PASSWORD=m0j4z4p0rk4 mariadb:10.7
b618846ff70c0012813dc62c02a3e262f29d0ac6e54d4504ff042914e7f6d9e9
```

Naredbom `docker ps` možemo se uvjeriti da je kontejner pokrenut:

``` shell
$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS      NAMES
b618846ff70c   mariadb:10.7   "docker-entrypoint.s…"   2 minutes ago   Up 2 minutes   3306/tcp   cranky_jang
```

Naredbom `docker logs` koja kao argument prima identifikator kontejnera možemo se uvjeriti da je pokretanje poslužitelja bilo uspješno:

``` shell
$ docker logs b618846ff70c0012813dc62c02a3e262f29d0ac6e54d4504ff042914e7f6d9e9
2022-04-15 17:07:23+00:00 [Note] [Entrypoint]: Entrypoint script for MariaDB Server 1:10.7.3+maria~focal started.
2022-04-15 17:07:23+00:00 [Note] [Entrypoint]: Switching to dedicated user 'mysql'
2022-04-15 17:07:23+00:00 [Note] [Entrypoint]: Entrypoint script for MariaDB Server 1:10.7.3+maria~focal started.
2022-04-15 17:07:24+00:00 [Note] [Entrypoint]: Initializing database files
2022-04-15 17:07:24 0 [Warning] You need to use --log-bin to make --expire-logs-days or --binlog-expire-logs-seconds work.


PLEASE REMEMBER TO SET A PASSWORD FOR THE MariaDB root USER !
To do so, start the server, then issue the following command:

'/usr/bin/mysql_secure_installation'

which will also give you the option of removing the test
databases and anonymous user created by default.  This is
strongly recommended for production servers.

See the MariaDB Knowledgebase at https://mariadb.com/kb

Please report any problems at https://mariadb.org/jira

The latest information about MariaDB is available at https://mariadb.org/.

Consider joining MariaDB's strong and vibrant community:
https://mariadb.org/get-involved/

2022-04-15 17:07:25+00:00 [Note] [Entrypoint]: Database files initialized
2022-04-15 17:07:25+00:00 [Note] [Entrypoint]: Starting temporary server
2022-04-15 17:07:25+00:00 [Note] [Entrypoint]: Waiting for server startup
2022-04-15 17:07:25 0 [Note] mariadbd (server 10.7.3-MariaDB-1:10.7.3+maria~focal) starting as process 155 ...
2022-04-15 17:07:25 0 [Note] InnoDB: Compressed tables use zlib 1.2.11
2022-04-15 17:07:25 0 [Note] InnoDB: Number of transaction pools: 1
2022-04-15 17:07:25 0 [Note] InnoDB: Using crc32 + pclmulqdq instructions
2022-04-15 17:07:25 0 [Note] InnoDB: Using Linux native AIO
2022-04-15 17:07:25 0 [Note] InnoDB: Initializing buffer pool, total size = 134217728, chunk size = 134217728
2022-04-15 17:07:25 0 [Note] InnoDB: Completed initialization of buffer pool
2022-04-15 17:07:25 0 [Note] InnoDB: 128 rollback segments are active.
2022-04-15 17:07:25 0 [Note] InnoDB: Creating shared tablespace for temporary tables
2022-04-15 17:07:25 0 [Note] InnoDB: Setting file './ibtmp1' size to 12 MB. Physically writing the file full; Please wait ...
2022-04-15 17:07:25 0 [Note] InnoDB: File './ibtmp1' size is now 12 MB.
2022-04-15 17:07:25 0 [Note] InnoDB: 10.7.3 started; log sequence number 41361; transaction id 14
2022-04-15 17:07:25 0 [Note] Plugin 'FEEDBACK' is disabled.
2022-04-15 17:07:25 0 [Warning] You need to use --log-bin to make --expire-logs-days or --binlog-expire-logs-seconds work.
2022-04-15 17:07:25 0 [Warning] 'user' entry 'root@b618846ff70c' ignored in --skip-name-resolve mode.
2022-04-15 17:07:25 0 [Warning] 'proxies_priv' entry '@% root@b618846ff70c' ignored in --skip-name-resolve mode.
2022-04-15 17:07:25 0 [Note] mariadbd: ready for connections.
Version: '10.7.3-MariaDB-1:10.7.3+maria~focal'  socket: '/run/mysqld/mysqld.sock'  port: 0  mariadb.org binary distribution
2022-04-15 17:07:26+00:00 [Note] [Entrypoint]: Temporary server started.
Warning: Unable to load '/usr/share/zoneinfo/leap-seconds.list' as time zone. Skipping it.
Warning: Unable to load '/usr/share/zoneinfo/leapseconds' as time zone. Skipping it.
Warning: Unable to load '/usr/share/zoneinfo/tzdata.zi' as time zone. Skipping it.
2022-04-15 17:07:27+00:00 [Note] [Entrypoint]: Securing system users (equivalent to running mysql_secure_installation)

2022-04-15 17:07:27+00:00 [Note] [Entrypoint]: Stopping temporary server
2022-04-15 17:07:27 0 [Note] mariadbd (initiated by: root[root] @ localhost []): Normal shutdown
2022-04-15 17:07:27 0 [Note] InnoDB: FTS optimize thread exiting.
2022-04-15 17:07:27 0 [Note] InnoDB: Starting shutdown...
2022-04-15 17:07:27 0 [Note] InnoDB: Dumping buffer pool(s) to /var/lib/mysql/ib_buffer_pool
2022-04-15 17:07:27 0 [Note] InnoDB: Buffer pool(s) dump completed at 220419 17:07:27
2022-04-15 17:07:27 0 [Note] InnoDB: Removed temporary tablespace data file: "./ibtmp1"
2022-04-15 17:07:27 0 [Note] InnoDB: Shutdown completed; log sequence number 42335; transaction id 15
2022-04-15 17:07:27 0 [Note] mariadbd: Shutdown complete

2022-04-15 17:07:28+00:00 [Note] [Entrypoint]: Temporary server stopped

2022-04-15 17:07:28+00:00 [Note] [Entrypoint]: MariaDB init process done. Ready for start up.

2022-04-15 17:07:28 0 [Note] mariadbd (server 10.7.3-MariaDB-1:10.7.3+maria~focal) starting as process 1 ...
2022-04-15 17:07:28 0 [Note] InnoDB: Compressed tables use zlib 1.2.11
2022-04-15 17:07:28 0 [Note] InnoDB: Number of transaction pools: 1
2022-04-15 17:07:28 0 [Note] InnoDB: Using crc32 + pclmulqdq instructions
2022-04-15 17:07:28 0 [Note] InnoDB: Using Linux native AIO
2022-04-15 17:07:28 0 [Note] InnoDB: Initializing buffer pool, total size = 134217728, chunk size = 134217728
2022-04-15 17:07:28 0 [Note] InnoDB: Completed initialization of buffer pool
2022-04-15 17:07:28 0 [Note] InnoDB: 128 rollback segments are active.
2022-04-15 17:07:28 0 [Note] InnoDB: Creating shared tablespace for temporary tables
2022-04-15 17:07:28 0 [Note] InnoDB: Setting file './ibtmp1' size to 12 MB. Physically writing the file full; Please wait ...
2022-04-15 17:07:28 0 [Note] InnoDB: File './ibtmp1' size is now 12 MB.
2022-04-15 17:07:28 0 [Note] InnoDB: 10.7.3 started; log sequence number 42335; transaction id 14
2022-04-15 17:07:28 0 [Note] InnoDB: Loading buffer pool(s) from /var/lib/mysql/ib_buffer_pool
2022-04-15 17:07:28 0 [Note] Plugin 'FEEDBACK' is disabled.
2022-04-15 17:07:28 0 [Warning] You need to use --log-bin to make --expire-logs-days or --binlog-expire-logs-seconds work.
2022-04-15 17:07:28 0 [Note] InnoDB: Buffer pool(s) load completed at 220419 17:07:28
2022-04-15 17:07:28 0 [Note] Server socket created on IP: '0.0.0.0'.
2022-04-15 17:07:28 0 [Note] Server socket created on IP: '::'.
2022-04-15 17:07:28 0 [Note] mariadbd: ready for connections.
Version: '10.7.3-MariaDB-1:10.7.3+maria~focal'  socket: '/run/mysqld/mysqld.sock'  port: 3306  mariadb.org binary distribution
```

Poslužitelj zaustavljamo naredbom `docker kill`:

``` shell
$ docker kill b618846ff70c0012813dc62c02a3e262f29d0ac6e54d4504ff042914e7f6d9e9
b618846ff70c0012813dc62c02a3e262f29d0ac6e54d4504ff042914e7f6d9e9
```

Naredba `docker ps` ukazuje da poslužitelj više nije pokrenut:

``` shell
$ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

Uvjerili smo se da možemo pokrenuti i zaustaviti poslužitelj, ali [klijent](https://youtu.be/lMxDPYraXG4) se neće moći povezati na ovako pokrenut poslužitelj. Naime, kako bi se mogla ostvariti veza klijenta i poslužitelja, oni moraju biti u istoj mreži i poslužitelj mora biti imenovan. Stvorimo mrežu naredbom `docker network`:

``` shell
$ docker network create db-network
ef5139aa1ec739e9c5da903581328a537380095b9117b555d72a1b33678836b7
```

Pokrenimo poslužitelj na toj mreži i nazovimo ga `fidit-mariadb` korištenjem parametra `--name`:

``` shell
$ docker run --detach --network db-network --name fidit-mariadb --env MARIADB_ROOT_PASSWORD=m0j4z4p0rk4 mariadb:10.7
39b77a3679fd4e5c8638d63d3d577464b46c3ee0b9cc34a965b100355b9bb6aa
```

Pokrenimo klijent na istoj mreži i iskoristimo parametar `-h` naredbe `mariadb` za navođenje imena poslužitelja, parametar `-u` za navođenje imena korisnika i parametar `-p` za uključivanje upita za zaporkom. Uočimo pritom kako naredba `mariadb` ne očekuje razmak između parametra i njegove vrijednosti.

``` shell
$ docker run -it --network db-network mariadb:10.7 mariadb -hfidit-mariadb -uroot -p
Enter password:
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 3
Server version: 10.7.3-MariaDB-1:10.7.3+maria~focal mariadb.org binary distribution

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]>
```

Nakon unošenja zaporke vidimo da smo se uspješno povezali na poslužitelj.

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
| sys                |
+--------------------+
4 rows in set (0.001 sec)
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

!!! note
    U nastavku stvaramo ključeve korištenjem [Easy-RSA](https://community.openvpn.net/openvpn/wiki/EasyRSA), koji je skup skripti za olakšavanje stvaranja RSA ključeva korištenjem OpenSSL-a. Alternativno, moguće je certifikate napraviti i ručnim korištenjem OpenSSL-a, kako je to opisano u koracima 3, 4, 5 i 6 u [članku How to set up MariaDB SSL and secure connections from clients na nixCraftu](https://www.cyberciti.biz/faq/how-to-setup-mariadb-ssl-and-secure-connections-from-clients/).

[Ključeve i X.509 certifikate](https://mariadb.com/docs/security/encryption/in-transit/create-self-signed-certificates-keys-openssl/#create-self-signed-certificates-keys-openssl) koje će MariaDB koristiti stvaramo korištenjem OpenSSL-a i Easy-RSA. Inicijalizirajmo infrastrukturu privatnog ključa i kopirajmo direktorij s informacijama o vrstama X.509 certifikata i datoteke s postavkama OpenSSL-a:

``` shell
$ easyrsa init-pki
$ cp -r /etc/easy-rsa/x509-types pki
$ cp /etc/easy-rsa/openssl-easyrsa.cnf pki
$ cp /etc/easy-rsa/vars pki
```

Izgradimo prvo ključ i certifikat autoriteta certifikata:

``` shell
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

Pripremit ćemo direktorije s konfiguracijom i certifikatima za poslužiteljski i klijentski kontejner. Kopirajmo stvoreni certifikat autoriteta certifikata, poslužiteljski certifikat i ključ na mjesto s kojeh ćemo ih koristiti te dozvolimo ostalim korisnicima (među kojima je i korisnik `mysql` koji pokreće MariaDB poslužitelj) čitanje datoteka:

``` shell
$ mkdir -p mariadb-server-conf/certs
$ cp pki/ca.crt mariadb-server-conf/certs/ca-cert.pem
$ cp pki/issued/moj-mariadb-posluzitelj.crt mariadb-server-conf/certs/server-cert.pem
$ cp pki/private/moj-mariadb-posluzitelj.key mariadb-server-conf/certs/server-key.pem
$ chmod a+r mariadb-server-conf/certs/*.pem
```

Analogno pripremimo certifikate i ključeve za klijentski kontejner:

``` shell
$ mkdir -p mariadb-client-conf/certs
$ cp pki/ca.crt mariadb-client-conf/certs/ca-cert.pem
$ cp pki/issued/moj-mariadb-klijent.crt mariadb-client-conf/certs/client-cert.pem
$ cp pki/private/moj-mariadb-klijent.key mariadb-client-conf/certs/client-key.pem
$ chmod a+r mariadb-client-conf/certs/*.pem
```

#### Uključivanje ključeva u konfiguracijsku datoteku

Nakon stvaranja privatnog ključa i X.509 certifikata te njihovog kopiranja u direktorije iz kojih će se koristiti uključit ćemo njihovo korištenje konfiguracijskim naredbama:

- `ssl_cert` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_cert/)), kojom ćemo navesti X.509 certifikat u formatu PEM
- `ssl_key` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_key/)), kojom ćemo navesti privatni ključ u formatu PEM
- `ssl_ca` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/ssl_ca/)), kojom ćemo, u slučaju korištenja samopotpisanih certifikata, navesti i certifikat autoriteta certifikata koji ga je potpisao

Više informacija o ovim konfiguracijskim naredbama može se naći u [službenoj dokumentaciji](https://mariadb.com/docs/security/encryption/in-transit/enable-tls-server/).

MariaDB ne preporuča modificiranje postojećih konfiguracijskih datoteka, već dodavanje novih. U direktoriju za konfiguraciju poslužitelja korištenjem uređivača teksta GNU nano ili bilo kojeg drugog stvorimo datoteku `ssl.cnf`:

``` shell
$ nano mariadb-server-conf/ssl.cnf
```

Jako je važno da nastavak bude `.cnf`, a ne `.conf` ili neka treća jer konfiguracijska naredba `!includedir` koju MariaDB koristi učitava u danom direktoriju samo datoteke s nastavkom `.cnf`. U njoj postavimo putanje poslužiteljskih ključeva i certifikata:

``` ini
[mariadb]

ssl_cert = /etc/mysql/conf.d/certs/server-cert.pem
ssl_key = /etc/mysql/conf.d/certs/server-key.pem
ssl_ca = /etc/mysql/conf.d/certs/ca-cert.pem
```

Analogno stvorimo konfiguracijsku datoteku klijenta i u njoj postavimo putanje za klijentske ključeve i certifikate:

``` shell
$ nano mariadb-client-conf/ssl.cnf
```

``` ini
[client-mariadb]

ssl_cert = /etc/mysql/conf.d/certs/client-cert.pem
ssl_key = /etc/mysql/conf.d/certs/client-key.pem
ssl_ca = /etc/mysql/conf.d/certs/ca-cert.pem
```

!!! note
    Umjesto u odjeljak `[mariadb]`, konfiguracijske smo naredbe mogli staviti u `[server]` ili `[mysqld]` koje poslužitelj čita kod pokretanja (detaljnije objašnjenje pojedinih odjeljaka moguće je [naći u dijelu službene dokumentacije Configuring MariaDB with Option Files](https://mariadb.com/kb/en/configuring-mariadb-with-option-files/#server-option-groups)), dok smo umjesto u odjeljak `[client-mariadb]` konfiguracijske naredbe mogli staviti u odjeljak `[client]` ([detalji](https://mariadb.com/kb/en/configuring-mariadb-with-option-files/#client-option-groups)).

Pokrenimo ponovno poslužitelj tako da dodatno parametrom `-v` montiramo direktorij `/home/korisnik/mariadb-server-conf` na `/etc/mysql/conf.d`, iz kojeg će MariaDB poslužitelj čitati konfiguraciju:

``` shell
$ docker run --detach --network db-network --name fidit-mariadb -v /home/korisnik/mariadb-server-conf:/etc/mysql/conf.d --env MARIADB_ROOT_PASSWORD=m0j4z4p0rk4 mariadb:10.7
fd5bc31673cb7f87d632ef9d59bf515bafa944ba4c1058f5776f3d0dc74a402d
```

Kako bismo se uvjerili da je TLS uključen, povežimo se klijentom tako da i na njegovoj strani montiramo direktorij s konfiguracijskim datotekama:

``` shell
$ docker run -it --network db-network -v /home/korisnik/mariadb-client-conf:/etc/mysql/conf.d mariadb:10.7 mariadb -hfidit-mariadb -uroot -p
Enter password:
```

Zatim iskoristimo naredbu `SHOW` za prikaz vrijednosti globalne varijable `have_ssl`:

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

U konfiguracijskom direktoriju poslužitelja stvorimo datoteku `file-key-management.cnf` u kojoj ćemo konfiguracijskom naredbom `plugin_load_add` ([dokumentacija](https://mariadb.com/docs/reference/mdb/cli/mariadbd/plugin-load-add/)) učitati dodatak `file_key_management`:

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
$ mkdir mariadb-server-conf/encryption
```

Stvorimo u tom direktoriju datoteku `keyfile` u kojoj su u svakom retku identifikator ključa (cijeli broj), znak točke sa zarezom i ključ:

```
1;1cece9bfd9e0263da50bcf02d088b12889cf1eddeb7f8ffdd719b9ab23359be2
2;df8762cdcd39d0e095a538ecad06ca94f55f805cc76841ed304ec26e4f46d2a0
11;6bf495bc15a970f8e51b7be49e0ebad5b74fc8febccd1ff45e259ca82e35a973
34;34e791122e8cb9fe983534d33bc45522d3e9ca2ec373e720eb08fc34e7f59b7b
```

Identifikatori ključa moraju biti međusobno različiti, ali ne moraju ići po redu.

#### Uključivanje datoteke s ključevima

Dopunimo datoteku `file-key-management.cnf` dodavanjem konfiguracijske naredbe `file_key_management_filename` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_filename)) tako da bude oblika:

``` ini
[mariadb]

plugin_load_add = file_key_management
file_key_management_filename = /etc/mysql/conf.d/encryption/keyfile
```

!!! tip
    Datoteku s ključevima je [moguće šifrirati](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#encrypting-the-key-file) pa navesti ključ za dešifriranje konfiguracijskom naredbom `file_key_management_filekey` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_filekey)).

#### Odabir algoritma za šifriranje

Podržane su dvije varijante algoritma AES:

- AES-CBC, koji koristi [Cipher block chaining (CBC)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Cipher_block_chaining_(CBC))
- AES-CTR, koji koristi kombinaciju [Counter (CTR)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)) i [Galois/Counter Mode (GCM)](https://en.wikipedia.org/wiki/Galois/Counter_Mode) (zahtijeva da korišteni OpenSSL podržava te algoritme)

Kombinacija CTR-a i GCM-a se smatra boljim odabirom i naša je verzija OpenSSL-a podržava pa ćemo njeno korištenje uključiti dodavanjem konfiguracijske naredbe `file_key_management_encryption_algorithm` ([dokumentacija](https://mariadb.com/kb/en/file-key-management-encryption-plugin/#file_key_management_encryption_algorithm)) u datoteku `file-key-management.cnf` tako da bude oblika:

``` ini
[mariadb]

plugin_load_add = file_key_management
file_key_management_filename = /etc/mysql/conf.d/encryption/keyfile
file_key_management_encryption_algorithm = AES_CTR
```

Ako korišteni OpenSSL ne podržava taj algoritam, zamijenit ćemo posljednju liniju za:

``` ini
file_key_management_encryption_algorithm = AES_CBC
```

Zaustavimo i pokrenimo ponovno poslužitelj kako bi učitao nove postavke pa se povežimo na njega klijentom.

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

## Dodatak: pregled strukture konfiguracije poslužitelja MariaDB

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
