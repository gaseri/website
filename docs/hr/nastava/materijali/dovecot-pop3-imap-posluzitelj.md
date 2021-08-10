---
author: Ivan Ivakić, Vedran Miletić
---

# POP3 i IMAP: Dovecot

[Dovecot](https://www.dovecot.org/) je jednostavan i brz POP3 i IMAP server.

## Instalacija Dovecota

Jednostavno pokrenimo sljedeću naredbu u terminalu:

``` shell
# yum install dovecot
```

## Osnovna konfiguracija

Dovecot će većinu stvari konfigurirati pri instalaciji, no možda nije loše uključiti loging da bi lakše pronašli eventualne pogreške pri daljnjoj konfiguraciji.

Otvorimo datoteku `/etc/dovecot/dovecot.conf` i pronađimo sljedeće linije:

``` shell
# Log file to use for error messages, instead of sending them to syslog.
# /dev/stderr can be used to log into stderr.
log_path = /var/dovecot/debug/log
```

Navedimo proizvoljnu putanju do datoteke (paziti pritom da direktoriji i sama datoteka postoje) te ponovno pokrenimo dovecot servis.

``` shell
# systemctl restart dovecot
```

Ukoliko sve prođe u redu dobit ćemo poruku:

```
Restarting IMAP/POP3 mail server: dovecot.
```

a ukoliko postoji greška u konfiguracijskoj datoteci nešto slično:

```
Can't open log file /var/dovecot/debug: Is a directory failed!
```

Primjetimo da se uglavnom javi i u čemu je problem što uvelike pomaže pri uspješnom konfiguriranju. Linije koje nas potom zanimaju su postavljanje protokola i definiranje putanja do `inbox`-ova koje ćemo koristiti. Za potrebe testiranja kreirao sam novog korisnika `test` sa istom šifrom `test`.

``` shell
# Protocols we want to be serving: imap imaps pop3 pop3s managesieve
# If you only want to use dovecot-auth, you can set this to "none".
#protocols = imap imaps
protocols = imap imaps
```

Dodajmo u protokole `pop3`:

```
protocols = imap imaps pop3
```

Sljedeće što treba učiniti je postaviti putanje do inboxova, prije svega treba pronaći na kojoj su "fizičkoj" lokaciji mailovi pa potom pronaći liniju u konfiguracijskoj datoteci Dovecota te ju uputiti na prethodno pronađenu lokaciju:

```
mail_location = /var/mail/%u
```

Primjetite `%u` koji dinamički označava korisničko ime (engl. **u**ser). Mogućih opcija je još nekoliko, primjerice ako bi htjeli imati više domena sa više korisnika koristili bi konfiguraciju sličniju sljedećoj:

```
mail_location = /var/%d/mail/%u
```

ili pak:

```
mail_location = /%h/var/mail/%d/%n
```

ukoliko bi htjeli referencirati na home direktorij (`%h`) koristeći i domenu (`%d`) i korisnički dio mail adrese (`%n`). U slučaju adrese `korisnik@domena`, `korisnik` je `%n`, a `domena` je `%d`.

Podesimo još mogućnost autentificiranja lozinkama u plain text obliku u liniji:

```
disable_plaintext_auth = no
```

!!! tip
    Ovo u realnom okruženju treba biti postavljeno na `yes` te se autentifikacija treba vršiti korištenjem [SSL/TLS](https://en.wikipedia.org/wiki/Secure_Sockets_Layer).

## Testiranje instalacije

Testirajmo možemo li uspješno poslati mail sljedećom naredbom:

``` shell
$ echo "Ovo je mail samom sebi" | mail -s "Dovecot naslov" ivan
```

Pregledajmo inbox kao i ranije naredbom `mail` i potvrdimo primitak gore navedenog maila.

Dalje trebamo testirati jesu li POP3 i IMAP serveri pokrenuti i rade li kako treba, a za to ćemo ponovno koristiti `telnet`.

``` shell
$ telnet localhost 110
```

će nas spojiti na POP3 pri čemu ćemo dobiti ispis:

```
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
+OK Dovecot ready.
```

Prijavimo se kao `test`, dohvatimo inbox te pročitajmo prvu poruku.

Ključna riječ `user` nakon koje slijedi korisničko ime:

```
user test
+OK
```

Ključna riječ `pass` nakon koje slijedi lozinka

```
pass test
+OK Logged in.
```

Primjer pogrešne naredbe

```
dummy-naredba
-ERR Unknown command: DUMMY-NAREDBA
```

Ključna riječ `list` koja ispisuje broj poruka i pripadajuće veličine

```
 list
 +OK 3 messages:
1 632
2 666
3 681
.
```

Ključna riječ `retr` nakon koje slijedi id poruke koju želimo dohvatiti

```
retr 1
+OK 632 octets
```

Nakon čega se ispisuje cijelo tijelo poruke. Naredbom `quit` gasimo vezu.

Testiranje IMAPa možemo izvesti spajanjem na port 143 i vrlo sličnim postupkom kao gore dohvatiti poruke.
