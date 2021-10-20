---
author: Ivan Ivakić, Vedran Miletić
---

# Spam filter: SpamAssassin

[SpamAssassin](https://spamassassin.apache.org/) je jedan od mnogih dostupnih spam filtera koji ima za cilj obilježiti neželjenu (spam) poštu te ju filtrirati po zadanim kriterijima.

## Instalacija SpamAssassina

Vrlo jednostavno:

``` shell
# yum install spamassassin
```

Zanimljivo je pri kraju instalacije primjetiti:

```
SpamAssassin Mail Filter Daemon: disabled, see /etc/default/spamassassin
```

što nam jasno daje do znanja da ćemo morati ručno uključiti Spamassassin Daemon proces.

Nadalje, nije poželjno da nam se Spam filter pokreće sa root privilegijama pa ćemo mu dodijeliti specifičnog korisnika i grupu te ograničiti djelovanje na određene datoteke i direktorije. Dodajmo novu grupu sa ID-ijem 5001 i nazovimo ju spamd

``` shell
# groupadd -g 5001 spamd
```

Dodajmo novog korisnika `spamd` u grupu `spamd`, definiramo da ne postoji login za tog korisnika i postavljamo kućni direktorij na `/var/lib/spamassassin`

``` shell
# useradd -u 5001 -g spamd -s /sbin/nologin -d /var/lib/spamassassin spamd
```

Kreiramo sam kućni direktorij korisnika

``` shell
# mkdir /var/lib/spamassassin
```

Dodjeljujemo vlasnička prava korisniku `spamd` iz grupe `spamd` nad direktorijem `/var/lib/spamassassin`

``` shell
# chown spamd:spamd /var/lib/spamassassin
```

Nakon što smo obavili pripreme možemo krenuti na samu konfiguraciju osnovnih postavki.

## Osnovna konfiguracija

Otvorivši datoteku `/etc/default/spamassassin` prvo što ćemo primjetiti je vjerojatno:

``` shell
# Change to one to enable spamd
ENABLED=0
```

pa promijenimo ovu `0` (nulu) u `1`. Time smo uključili spam deamon.

Radi jednostavnosti definirajmo jednu varijablu `HOME` i dodijelimo joj vrijednost kućnog direktorija `spamd` korisnika.

```
HOME="/var/lib/spamassassin"
```

Promijenimo sada još:

```
OPTIONS="--create-prefs --max-children 5 --username spamd --helper-home-dir ${HOME} -s ${HOME}/spamd.log"
```

Ovime dodjeljujemo korisnika `spamd`, definiramo kućni direktorij i definiramo lokaciju datoteke `spamd.log` koja će sadržavati log podatke. Zatim definiramo gdje će se pratiti [PID](https://en.wikipedia.org/wiki/Process_identifier) opcijom

```
PIDFILE="${HOME}/spamd.pid
```

## Pravila filtriranja

Krenimo sada na definiranje pravila filtriranja u datoteci `/etc/spamassassin/local.cf`:

```
rewrite_header Subject *****UPOZORENJE SPAM*****
```

Ovime definiramo modifikaciju zaglavlja da bi prepoznali spam poruke.

```
report_safe 1
```

Ovime definiramo izgled filtrirane poruke, uobičajena vrijednost je 1. Postoje tri mogućnosti:

- `0` -- promjene spam poruka će se izvršavati samo nad zaglavljem, tijelo poruke ostaje isto
- `1` -- tijelo poruke se sprema kao prilog filtriranoj poruci, time se onemogućava lako izvršavanje/čitanje te poruke, no ujedno pruža mogućnost lakog dohvaćanja samog sadržaja
- `2` -- kao i 1, samo što se prilog sprema u plain textu; ovo se radi ukoliko očekujemo automatsko učitavanje priloga od strane mail klijenta

Uključimo još i Bayesov klasifikator i podesimo ga na samoučenje.

```
use_bayes 1
bayes_auto_learn 1
```

Podešavanje "osjetljivosti" filtera radimo na idući način:

```
required_score 2.0
```

čime kažemo koliku minimalnu ocjenu treba postići određena poruka da bi bila klasificirana kao spam poruka. Nakon toga možemo pokrenuti spamassasin daemon proces:

``` shell
# systemctl start spamassasin
```

i ukoliko smo sve podesili kako treba ispis će biti oblika:

```
Starting SpamAssassin Mail Filter Daemon: spamd.
```

## Spajanje SpamAssassina i Postfixa

Za kraj je potrebno Postfixu dati do znanja da imamo novi spam filter i da ga želimo koristiti.

Otvorimo stoga datoteku drugu konfiguracijsku datoteku Postfixa `/etc/postfix/master.cf` te liniju:

```
smtp inet n -- -- -- -- smtpd
```

promijenimo u

```
smtp inet n -- -- -- -- smtpd -o content_filter=spamassassin
```

čime našem smtp daemonu dajemo do znanja da koristi filter `spamassassin`.

Preostaje još samo definirati vanjsku aplikaciju za filtriranje što činimo dodavanjem sljedećih linija na kraj datoteke:

```
spamassassin unix -- n n -- -- pipe
user=spamd argv=/usr/bin/spamc -f -e
/usr/sbin/sendmail -oi -f ${sender} ${recipient}
```

Nakon toga valja ponovno učitati postavke postfixa da bi se promjene primjenile.

``` shell
# systemctl reload postfix
```
