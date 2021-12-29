---
author: Vedran Miletić
---

# Replikacija sadržaja poslužitelja sustava za upravljanje bazom podataka MariaDB

[Replikacija](https://mariadb.com/kb/en/replication-overview/) je značajka sustava za upravljanje bazom podataka koja omogućuje da sadržaji (baze podataka i tablice u bazama podataka) jednog ili više poslužitelja (koje nazivamo **primarnim**, engl. *primary*) budu dodatno dostupni na nekom drugom poslužitelju ili više njih (koje nazivamo **replikama**, engl. *replica*). Replikacija se koristi u više slučajeva, a nama su najzanimljiviji:

- skalabilnost: veći broj poslužitelja omogućuje da se čitanja iz baze podataka raspodijele na veći broj poslužitelja, što smanjuje opterećenje primarnog poslužitelja i naročito je primjenjivo za sustave koji imaju velik broj čitanja u odnosu na broj pisanja, npr. [Wikipedia](https://commons.wikimedia.org/wiki/File:Wikipedia_webrequest_flow_2020.png)
- [pomoć u izradi rezervnih kopija podataka](https://mariadb.com/kb/en/replication-as-a-backup-solution/): rezervne kopije je lakše izraditi kad se poodaci ne mijenjaju pa je tipičan postupak izrada replikacije, odspajanje replike od primarnog poslužitelja (čime ona prestaje izvoditi promjene na podacima) i izrada pohrane podataka na replici

Mehanizam koji omogućuje replikaciju je [binarni zapisnik](https://mariadb.com/kb/en/binary-log/) koji sprema sve promjene na bazi podataka i omogućuje replikama da izvedu iste promjene na svojim kopijama podataka. Prilikom izvođenja replikacije stvara se [relejni zapisnik](https://mariadb.com/kb/en/relay-log/) na svakoj od replika koji je istog oblika kao binarni zapisnik. Primarni poslužitelj i replike komuniciraju kontinuirano; u slučaju privremenog prekida komunikacije, replikacija će se zaustaviti. Nakon ponovnog uspostavljanja komunikacije replikacija će se nastaviti tamo gdje je stala.

Standardna replikacija uključuje jedan primarni poslužitelj i jednu ili više replika. Time se dobiva neograničeno skaliranje operacija čitanja podataka, a dodatno se može jedna od replika po potrebi pretvoriti u primarni poslužitelj i time postići visoka raspoloživost usluge u slučaju prekida rada primarnog poslužitelja.

## Postavljanje replikacije

U nastavku pretpostavljamo da je primarni poslužitelj pokrenut na virtualnom stroju s adresom 192.168.122.100, a replika na virtualnom stroju s adresom 192.168.122.101.

### Postavljanje primarnog poslužitelja

Na primarnom poslužitelju postavit ćemo redom:

- stvaranje binarnog zapisnika konfiguracijskom naredbom `bin_log` ([dokumentacija](https://mariadb.com/kb/en/replication-and-binary-log-system-variables/#log_bin), [više detalja](https://mariadb.com/kb/en/activating-the-binary-log/))
- jedinstveni identifikator poslužitelja konfiguracijskom naredbom `server_id` ([dokumentacija](https://mariadb.com/kb/en/replication-and-binary-log-system-variables/#server_id)); replike će također imati identifikator i to različit od primarnog poslužitelja
- jedinstveno ime replikacijskog zapsnika naredbom `log_basename` ([dokumentacija](https://mariadb.com/kb/en/mysqld-options/#-log-basename))
- postaviti oblik zapisan na konfiguracijskom naredbom `binlog_format` ([dokumentacija](https://mariadb.com/kb/en/replication-and-binary-log-system-variables/#binlog_format)); mogućnosti su zapis temeljen na retcima koji su promijenjeni, zapis temeljen na naredbama koje mijenjaju podatke te kombinacija ta dva zapisa koju ćemo i koristiti ([više detalja](https://mariadb.com/kb/en/binary-log-formats/))

Stvorimo datoteku `/etc/mysql/mariadb.conf.d/90-replication.cnf` tako da je njen sadržaj:

``` ini
[mariadb]
log_bin
server_id = 1
log_basename = primary1
binlog_format = mixed
```

Stvorimo korisnika koji će se koristiti za replikaciju:

``` sql
CREATE USER 'victorglushkov'@'%' IDENTIFIED BY 'h4k3r1n3c3pr0b1t10v0';
```

```
Query OK, 0 rows affected (0.000 sec)
```

Uočimo da smo korisniku omogućili pristup od bilo kuda navođenjem znaka postotka (`%`) u imenu domaćina. Dodijelimo korisniku privilegiju repliciranja baze ([dokumentacija](https://mariadb.com/kb/en/grant/#replication-slave)):

``` sql
GRANT REPLICATION SLAVE ON *.* TO 'victorglushkov'@'%';
```

```
Query OK, 0 rows affected (0.000 sec)
```

Naposlijetku, postavimo da se MariaDB poslužitelj veže i na adrese do kojih je moguće doći izvan lokalnog računala. To ćemo učiniti promjenom vrijednosti konfiguracijske naredbe `bind-address` ([dokumentacija](https://mariadb.com/kb/en/server-system-variables/#bind_address)) u datoteci `/etc/mysql/mariadb.conf.d/50-server.cnf` iz vezivanja na lokalnu adresu:

``` ini
bind-address            = 127.0.0.1
```

u vezivanje na sve adrese:

``` ini
bind-address            = 0.0.0.0
```

Nakon promjena postavki pokrenimo ponovno poslužitelj:

``` shell
$ sudo systemctl restart mariadb
```

U nastavku uzimamo da je korisnik koji pristupa bazi podataka na primarnom poslužitelju i na replikama `ivanzhegalkin`, a baza podataka `mojabaza`. Zaključajmo tablice za promjene SQL upitom `FLUSH` ([dokumentacija](https://mariadb.com/kb/en/flush/)) tako da prvo spremimo sve podatke u tablice, a potom ih zaključamo:

``` sql
FLUSH TABLES WITH READ LOCK;
```

```
Query OK, 0 rows affected (0.001 sec)
```

``` sql
SHOW MASTER STATUS;
```

```
+--------------------+----------+--------------+------------------+
| File               | Position | Binlog_Do_DB | Binlog_Ignore_DB |
+--------------------+----------+--------------+------------------+
| master1-bin.000096 |      568 |              |                  |
+--------------------+----------+--------------+------------------+
```

Zapamtimo ime datoteke `master1-bin.000096` i mjesto u njoj `568` koji će nam trebati kod postavljanja replike.

Iskoristimo u novom terminalu naredbu ljuske `mariadb-dump` ([dokumentacija](https://mariadb.com/kb/en/mariadb-dump/)) za spremanje podataka iz baze `mojabaza` u datoteku `mojabaza-dump.sql` koja će sadržavati niz SQL upita kojima će se stvoriti baza iste strukture s istim podacima na drugom mjestu. Naredba je oblika:

``` shell
$ mariadb-dump -u ivanzhegalkin -p mojabaza > mojabaza-dump.sql
```

Otključajmo tablice SQL upitom `UNLOCK TABLES` ([dokumentacija](https://mariadb.com/kb/en/lock-tables/)):

``` sql
UNLOCK TABLES;
```

```
Query OK, 0 rows affected (0.001 sec)
```

### Postavljanje replike

Na replici postavit ćemo `server_id` na vrijednost različitu od one na primarnom poslužitelju, npr. `server_id = 2`. U nastavku pretpostavljamo da su korisnik `ivanzhegalkin` i baza podataka `mojabaza` stvoreni na isti način kao na primarnom poslužitelju. Uočimo da to nisu isti korisnik i ista baza jer se nalaze u drugoj instanci sustava za upravljanje bazom podataka.

Iskoristimo MariaDB klijent `mariadb` ([dokumentacija](https://mariadb.com/kb/en/mariadb-command-line-client/)) za uvoz podataka iz datoteke:

``` shell
$ mariadb -u ivanzhegalkin -p mojabaza < mojabaza-dump.sql
```

Nakon uvoza podataka povežimo repliku na primarni poslužitelj SQL upitom `CHANGE MASTER TO` ([dokumentacija](https://mariadb.com/kb/en/change-master-to/)):

``` sql
CHANGE MASTER TO
  MASTER_HOST = '192.168.122.100',
  MASTER_PORT = 3306,
  MASTER_CONNECT_RETRY = 10,
  MASTER_USER = 'victorglushkov',
  MASTER_PASSWORD = 'h4k3r1n3c3pr0b1t10v0',
  MASTER_LOG_FILE = 'master1-bin.000096',
  MASTER_LOG_POS = 568,

  MASTER_USE_GTID = slave_pos;
```

```
Query OK, 0 rows affected (0.001 sec)
```

Ovdje smo naveli redom:

- adresu domaćina na kojem se nalazi primarni poslužitelj: `MASTER_HOST = '192.168.122.100'`
- vrata na kojima je pokrenut primarni poslužitelj: `MASTER_PORT = 3306`
- broj pokušaja povezivanja na primarni poslužitelj prije odustajanja: `MASTER_CONNECT_RETRY = 10`
- ime korisnika na primarnom poslužitelju: `MASTER_USER = 'victorglushkov'`
- zaporku korisnika na primarnom poslužitelju: `MASTER_PASSWORD = 'h4k3r1n3c3pr0b1t10v0'`
- ime datoteke zapisnika koje smo ranije očitali na primarnom poslužitelju: `MASTER_LOG_FILE = 'master1-bin.000096'`
- mjesto u datoteci zapisnika koje smo ranije očitali na primarnom poslužitelju `MASTER_LOG_POS = 568`
- uključivanje korištenja [globalnih transakcijskih identifikatora (GTID)](https://mariadb.com/kb/en/gtid/): `MASTER_USE_GTID = slave_pos`

Pokrenimo repliku SQL upitom `START SLAVE` ([dokumentacija](https://mariadb.com/kb/en/start-replica/)):

``` sql
START SLAVE;
```

```
Query OK, 0 rows affected (0.001 sec)
```

Uvjerimo se da je replikacija uspješno pokrenuta SQL upitom `SHOW SLAVE STATUS` ([dokumentacija](https://mariadb.com/kb/en/show-replica-status/)). Ovdje ćemo iskoristiti završetak upita `\G` umjesto `;` kako bismo dobili čitljivi vertikalni ispis umjesto horizontalnog ispisa u retku tablice. Naredba je oblika:

``` sql
SHOW SLAVE STATUS\G
```

```
*************************** 1. row ***************************
               Slave_IO_State: Waiting for master to send event
                  Master_Host: db01.example.com
                  Master_User: replicant
                  Master_Port: 3306
                Connect_Retry: 60
              Master_Log_File: mariadb-bin.000096
          Read_Master_Log_Pos: 568
             Slave_IO_Running: Yes
            Slave_SQL_Running: Yes
...
```

Uočimo `Yes` u stupcima `Slave_IO_Running` i `Slave_SQL_Running`, koji označavaju da je replikacija uspješno postavljena.

Više detalja može se pronaći u [službenoj dokumentaciji o postavljanju replikacije](https://mariadb.com/kb/en/setting-up-replication/), uključujući i [pregled naredbi](https://mariadb.com/kb/en/replication-commands/).

## Ograničavanje replike na čitanje

Replike je moguće [ograničiti samo na čitanje](https://mariadb.com/kb/en/read-only-replicas/) postavljanjem konfiguracijske naredbe `read_only` ([dokumentacija](https://mariadb.com/docs/reference/mdb/system-variables/read_only/)) na vrijednost 1 na način:

``` ini
[mariadb]
read_only = 1
```
