---
author: Ivan Ivakić, Vedran Miletić
---

# Message transfer agent: Postfix

## Predispozicije i teorijska podloga

Mail poslužitelj koji ćemo konfigurirati sastoji se od četiri komponente:

- [Message transfer agent](https://en.wikipedia.org/wiki/Message_transfer_agent) omogućuje prijenos mailova sa jednog računala na drugo po modelu klijent-server, koristi [SMTP](https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol)
- [Mail delivery agent](https://en.wikipedia.org/wiki/Mail_delivery_agent) vrši dostavu mailova u korisnički sandučić (engl. *mailbox*)
- [POP3](https://en.wikipedia.org/wiki/POP3) i [IMAP](https://en.wikipedia.org/wiki/IMAP) poslužitelj koji omogućuje klijentima dohvaćanje mailova
- [Spam filter](https://en.wikipedia.org/wiki/Anti-spam_techniques_(e-mail)) koji filtrira nepoželjne mailove po zadanim kriterijima

[Postfix](http://www.postfix.org/) je, kako kažu autori, alternativa popularnom [Sendmailu](https://www.proofpoint.com/us/products/email-protection/open-source-email-solution) u pokušaju da bude brz, jednostavan za korištenje i siguran agent za prijenos e-mail poruka.

## Instalacija Postfixa

Uvjerimo se za početak da MTA nije već instaliran.

``` shell
$ telnet localhost 25
```

!!! tip
    [Telnet](https://en.wikipedia.org/wiki/Telnet) koristimo za pristupanje trenutnom mailserveru da bi potvrdili njegovo postojanje; pritom je *localhost* je oznaka domaćina, *25* je oznaka vrata na kojima očekujemo mail server -- ukoliko pogledamo [popis dobro poznatih vrata](https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers) primjetit ćemo da za port 25 stoji SMTP.

Ukoliko MTA, već postoji, ispis koji ćemo dobiti je oblika:

```
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
220 Ivan-debian.dummy.porta.siemens.net ESMTP Exim 4.72 Sat, 19 Jan 2013 14:22:01 +0100
```

U ovom slučaju u četvrtoj liniji jasno je vidljivo da je trenutno instaliran `Exim`. U nastavku ćemo pretpostaviti da ne postoji instalirani MTA na sustavu.

Postfix instaliramo naredbom

``` shell
# yum install postfix
```

Korisniku se nudi nekoliko izbornika u kojima odabiremo neke od predefiniranih konfiguracija, za potrebe ovog rada koristit ćemo predložak "local only". Također je potrebno postaviti "mail-name", odnosno domenu (dio mail adrese iza znaka *@*). Postavljena vrijednost koju ćemo koristiti je *localhost.loc*.

Provjerimo uspješnost instalacije sa:

``` shell
$ telnet localhost 25
```

ispisuje:

```
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
220 Ivan-debian.dummy.porta.siemens.net ESMTP Postfix (Debian/GNU)
```

gdje sada vidimo `Postfix` što govori da je instalacija prošla uspješno. Uz to moguće je provjeriti datoteke u kojima se čuvaju logovi mail servera. datoteke se nalaze u `/var/log`, a one koje nas zanimaju su tipa `mail.*`, specifično recimo `mail.info`.

Ispišimo ju sa:

``` shell
$ cat /var/log/mail.info
Jan 19 14:56:46 Ivan-debian postfix/master[4817]: daemon started -- version 2.7.1, configuration /etc/postfix
Jan 19 14:56:58 Ivan-debian postfix/smtpd[4875]: connect from localhost[127.0.0.1]
Jan 19 14:57:06 Ivan-debian postfix/smtpd[4875]: disconnect from localhost[127.0.0.1]
```

Pokušajmo sada poslati testni mail sljedećim slijedom naredbi i teksta. Spajanje na SMTP telnetom:

``` shell
$ telnet localhost 25
```

Upišimo sada nekoliko naredbi primajući interaktivno odgovor:

```
mail from: root@localhost.loc
250 2.1.0 Ok
rcpt to: ivan@localhost.loc
250 2.1.5 Ok
data
354 End data with <CR><LF>.<CR><LF>
```

Nakon toga pišemo tijelo poruke. Primjerice:

```
To: ivan@localhost.loc
From: root@localhost.loc
Subject: Naslov

Ovo je tekst poruke
```

Poruku, kako je navedeno završavamo sa novim retkom i točkom (.).

```
.
250 2.0.0 Ok: queued as B162E2E0AF
quit
221 2.0.0 Bye
```

Ukoliko je sve prošlo u redu, možemo provjeriti mailbox tako da kao korisnik `ivan` upišemo naredbu:

``` shell
$ mail
```

čime bi se trebalo ispisati nešto slično sljedećem tekstu ovisno naravno o samim testnim podacima:

```
Mail version 8.1.2 01/15/2001.  Type ? for help.
"/var/mail/ivan": 1 message 1 new
>N  1 root@localhost.lo  Sat Jan 19 16:35   15/543   naslov
&
```

Ukucavši `1` možemo pročitati taj mail i očekivani ispis je:

```
Message 1:
From root@localhost.loc  Sat Jan 19 16:35:33 2013
X-Original-To: ivan@localhost.loc
To: ivan@localhost.loc
From: root@localhost.loc
Subject: naslov
Date: Sat, 19 Jan 2013 16:34:49 +0100 (CET)

Ovo je tekst poruke
&
```

## Konfiguracija

Osnovna konfiguracija podešena je automatski s obzirom da smo koristili predložak za lokalnu upotrebu. Pogledajmo sada neke od zanimljivih linija u konfiguracijskim datotekama `/etc/postfix/main.cf` i `/etc/postfix/master.cf`.

U datoteci `/etc/postfix/main.cf` imamo redom

```
myhostname = Ivan-debian
```

može biti proizvoljno ime, zatim

```
alias_maps = hash:/etc/aliases
```

mapa aliasa (alternativnih imena), zatim

```
alias_database = hash:/etc/aliases
```

baza aliasa. Aliase koristimo da bi imali nekoliko mogućih naziva za istu mail adresu.

```
myorigin = /etc/mailname
```

domena koja se dodaje, zatim

```
mydestination = localhost.loc, Ivan-debian.dummy.porta.siemens.net, localhost.dummy.porta.siemens.net, localhost
```

predefinirana odredišta za localhost, može se definirati mnogo aliasa, svi koji su navedeni impliciraju slanje na lokalno računalo, bez prosljeđivanja na neka druga, zatim

```
relayhost =
```

autorizirane ne-lokalne domene, zatim

```
mynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128
```

autoriziranje na osnovu IP adrese i pripadajuće maske u [CIDR notaciji](https://en.wikipedia.org/wiki/CIDR_notation), zatim

```
mailbox_command = procmail -a "$EXTENSION"
```

pozivanje procmail servisa i prosljeđivanje parametra mailboxa, zatim

```
mailbox_size_limit = 0
```

ograničenje veličine mailboxa, 0 znači neograničeno, zatim

```
recipient_delimiter = +
```

znak za odvajanje primaoca.

## Mail delivery agent: procmail

Pri instalaciji postfixa automatski se instalira i [procmail](https://www.procmail.org/) bez ikakvih konfiguracijskih parametara. Sam sustav radi "out-of-the-box", a za modificiranje konfiguracije prvo je potrebno kreirati same konfiguracijske datoteke.
