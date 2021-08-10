---
author: Vedran Miletić
---

# Praćenje događaja operacijskog sustava

!!! hint
    Za više informacija proučite [Chapter 18. Viewing and Managing Log Files](https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/7/html/System_Administrators_Guide/ch-Viewing_and_Managing_Log_Files.html) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/7/html/System_Administrators_Guide/).

- log datoteke sadrže informacije o operacijskom sustavu: jezgri, uslugama i aplikacijama
- log datoteke nalaze se u direktoriju `/var/log`

    - `/var/log/messages` -- poruke operacijskog sustava
    - `/var/log/yum.log` -- poruke upravitelja paketima `yum`
    - `/var/log/Xorg.0.log` -- poruke koje javlja X Window System

- log datoteke se **rotiraju** da ne bi postale prevelike

    - datoteka `xyz.log` postaje `xyz.log.1` ili `xyz.log-20140326`
    - stvara se nova prazna datoteka `xyz.log`

!!! admonition "Zadatak"
    - Izdvojite iz `/var/log/messages` poruke koje se odnose na `ntpd`.
    - Pronađite u `/var/log/yum.log` podatke o nekoliko zadnjih paketa koje ste instalirali.

## Syslog

- narefba `rsyslog`
- konfiguracijska datoteka `/etc/rsyslog.conf`

    - `$` -- globalne naredbe
    - `$ModLoad`

!!! admonition "Zadatak"
    - U konfiguracijskoj datoteci `rsyslog.conf` uključite file sync.
    - Proučite `man rsyslog.conf` i pronađite način da stvorite datoteku `/var/log/mojzapis.log` koja koristi [RFC 3339](http://www.ietf.org/rfc/rfc3339.txt) vremenske pečate.

- `<FACILITY>.<PRIORITY>` zapis filtera

    - `<FACILITY>` je jedan od `auth`, `authpriv`, `cron`, `daemon`, `kern`, `lpr`, `mail`, `news`, `syslog`, `user`, `uucp` i `local0` through `local7`.
    - `<PRIORITY>` je jedan od `debug`, `info`, `notice`, `warning`, `err`, `crit`, `alert`, and `emerg`.

- nakon svake promjene potrebno je ponovno pokrenuti uslugu `rsyslog`

!!! admonition "Zadatak"
    - Uključite zapisivanje poruka s `debug` prioritetom koje ispisuje `syslog` facility.
    - Isključite zapisivanje `authpriv` poruka.
    - Promijenite u `/etc/ssh/sshd_config` postavku da `rsyslog` koristi `auth` facility umjesto `authpriv`, a zatim stvorite novu direktivu u kojoj će se sve `auth` poruke spremati u `/var/log/mojauth.log`.

## Rotacija log datoteka

- naredba `logrotate`
- konfiguracijska datoteka `/etc/logrotate.conf`

    - direktive koje određuju koliko često se događa rotacija: `daily`, `weekly`, `monthly`, `yearly`
    - direktive koje određuju kompresiju: `compress`, `nocompress`, `compresscmd`, `uncompresscmd`, `compressext`, `compressoptions`, `delaycompress`
    - direktiva `rotate BROJ` čini da se čuva `BROJ` rotiranih datoteka, odnosno da log datoteka prođe `BROJ` rotiranja prije brisanja
    - direktiva `mail ADRESA` omogućuje slanje log datoteke mailom na adresu `ADRESA` kod rotacije neposredno prije brisanja

!!! admonition "Zadatak"
    - Uključite kompresiju rotiranih log datoteka.
    - Učinite da se log datoteke rotiraju mjesečno, te da se čuva zadnjih 6 datoteka.
    - Za `/var/log/mojauth.log` postavite dnevno rotiranje, te čuvanje zadnjih 3 datoteke. Datoteka za log neka se stvara s dozvolama `rw-rw---`.

## Systemd Journal

- Journal je komponenta `systemd`-a zadužena za pregledavanje i upravljanje log datotekama
- koristi se paralelno sa Syslogom, a može ga i zamijeniti
- usluga `journald` sakuplja i pohranjuje log podatke i pripadne metapodatke
- alat `journalctl` služi za pregledavanje log datoteka

    - struktura izlaza slična `/var/log/messages`, način rada sličan `less`-u (vrlo neočekivano!)
    - vizualno su označene poruke većeg prioriteta (crvenom bojom, podebljano)
    - vremena se prevode u lokalnu vremensku zonu
    - prikazuju se svi podaci, uključujući rotirane log datoteke

- `journalctl -n BROJ` prikazuje samo zadnjih `BROJ` poruka
- `journalctl -o OBLIK` prikazuje izlaz u obliku ispisa `OBLIK`, pri čemu `OBLIK` može biti `verbose`, `export`, `json`, ...
- `journalctl -f` prati poruke kako nastaju

!!! admonition "Zadatak"
    - Usporedite zadnjih 40 unosa u `verbose` i `json` obliku.
    - Pratite poruke kako nastaju dok korištenjem `yum`-a obrišete pa instalirate paket `nano`.

- `journalctl -p PRIORITET` prikazuje samo poruke prioriteta `PRIORITET` ili višeg, pri čemu je `PRIORITET` jedan od `debug`, `info`, `notice`, `warning`, `err`, `crit`, `alert`, `emerg`
- `journalctl -b` prikazuje poruke od zadnjeg pokretanja
- `journalctl --since=OD --until=DO` prikazuje poruke u vremenskom rasponu od `OD` do `DO`, pri čemu su `OD` i `DO` oblika `"2013-3-16 23:59:59"`

!!! admonition "Zadatak"
    Prikažite samo poruke prioriteta `warning` ili višeg u rasponu od 1. listopada 2014. 10 ujutro do 1. prosinca 2014. 10 ujutro.
