---
author: Vedran Miletić, Edvin Močibob
---

# Rad s protokolima aplikacijske razine

## Pregled protokola aplikacijske razine

### Pojam URL-a

**URL (uniform resource locator)** je znakovni niz koji predstavlja vezu na neki resurs. URL se često koristi za web stranice (HTTP protokol) no, između ostalih, podržava FTP i e-mail (mailto) protokole.

Sintaksu URL-a čine: protokol, domena ili IP adresa, broj porta (opcionalno) i putanja do resursa. Primjerice:

```
http://www.example.org/page/subpage
```

### Protokoli HTTP i HTTPS

**HTTP (Hypertext Transfer Protocol)** je aplikacijski protokol i temelj podatkovne komunikacije za WWW. Internet Engineering Task Force (IETF) i the World Wide Web Consortium (W3C) stoje iza njegovog razvijanja. HTTP radi na request-response paradigmi po kljent-poslužitelj modelu. Jedna HTTP sesija sastoji se od niza zahtjeva (klijent) i odgovora (poslužitelj).

**HTTPS** je proširenje HTTP protokola SSL/TLS slojem. HTTPS se koristi za sigurnu komunikaciju preko računalne mreže. U srži TLS-a je korištenje privatnih i javnih ključeva te enkripcija podataka. HTTPS još podrazumijeva autentifikaciju i korištenje certifikata za dokazivanje identiteta.

### Protokol FTP

**FTP (File Transfer Protocol)** je mrežni protokol koji omogućuje transfer datoteka sa jednog domaćina na drugi preko TCP veze. Radi na klijent-poslužitelj principu. FTP može, ali ne mora koristiti autentifikaciju.

**FTPS** proširuje FTP tako da uz autentifikaciju omogućuje i enkripciju (SSL/TLS usluge). FTPS treba razlikovati od SFTP protokola koji je vezan uz SSH (Secure Shell).

## Osnovne značajke i način korištenja alata cURL

**cURL** (naredba `curl`) je komandno linijski alat za prijenos podataka korištenjem URL sintakse. Podržava brojne protokole (DICT, FILE, FTP, FTPS, Gopher, HTTP, HTTPS, IMAP, IMAPS, LDAP, LDAPS, POP3, POP3S, RTMP, RTSP, SCP, SFTP, SMTP, SMTPS, Telnet i TFTP), od kojih ćemo u nastavku koristiti manji dio. cURL je slobodan softver dostupan pod [MIT licencom](https://en.wikipedia.org/wiki/MIT_License).

cURL podržava SSL certifikate, HTTP naredbe POST i PUT, FTP upload, HTTP upload zasnovan na obrascima, proxy poslužitelje, keksiće, autentifikaciju korištenjem korisničkog imena i zaproke (Basic, Digest, NTLM, Negotiate, kerberos i druge), nastavljanje prijenosa datoteke, tuneliranje putem proxy poslužitelja i još mnogo toga. Mi ćemo se ograničiti na osnovnu funkcionalnost, ali [službena dokumentacija](https://curl.se/docs/manual.html) dostupna na [cURL-ovim stranicama](https://curl.se/) ima više detalja.

### Preuzimanje stranice ili datoteke

Preuzimanje se vrši navođenjem URL-a.

``` shell
$ curl http://inf2.uniri.hr/
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8">
    <title>Naslovna stranica poslužitelja inf2</title>
    <link rel="stylesheet" href="style.css">
    <!-- <script src="script.js"></script> -->
  </head>
  <body>
    <h1>Poslužitelj inf2 je zvijer</h1>
    ...
  </body>
</html>
```

Ukoliko želimo spremiti izlaz u datoteku umjesto ispisati na standardni izlaz, koristimo parametar `-o` i navodimo ime datoteke:

``` shell
$ curl -o inf2-index.html http://inf2.uniri.hr/
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2033  100  2033    0     0  74324      0 --:--:-- --:--:-- --:--:-- 75296
```

Parametrom `-O` izlaz možemo spremiti u lokalnu datoteku istog imena kao datoteka na poslužitelju:

``` shell
$ curl -O http://inf2.uniri.hr/index.html
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2033  100  2033    0     0   120k      0 --:--:-- --:--:-- --:--:--  124k
$ ls
index.html
```

### Preuzimanje dijela datoteke

Parametrom `-r` moguće je specificirati raspon podataka datoteke koji će bit preuzet:

``` shell
$ curl -r 0-120 http://inf2.uniri.hr/index.html
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8">
    <title>Naslovna stranica poslužitelja inf2</tit%
```

### Mjerenje napretka

Kod većih datoteka može se uočiti da je mjerenje napretka interaktivno:

``` shell
$ curl -O https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0 554M    0 2985k    0     0  9896k      0  0:03:49 --:--:--  0:03:49 9886k
```

Sa lijeva na desno stupci redom imaju sljedeće značenje.

| Stupac | Značenje |
| ------ | -------- |
| `%` | postotak obavljenog ukupnog prijenosa |
| `Total` | ukupna veličina očekivanog prijenosa |
| `%` | postotak obavljenog prijenosa (download) |
| `Received` | trenutno preuzetih bajtova |
| `%` | postatak obavljenog prijenosa (upload) |
| `Xferd` | trenutno prenesenih bajtova |
| `Average Speed: Dload` | srednja bzina prijenosa (download) |
| `Average Speed: Upload` | srednja bzina prijenosa (upload) |
| `Time Total` | očekivano vrijeme za ukupni prijenos |
| `Time Current` | proteklo vrijeme |
| `Time Left` | preostalo vrijeme |
| `Current Speed` | srednja brzina prijenosa zadnjih 5 sekundi |

Jednostavniji prikaz napretka parametrom `-#` prikazuje samo postotak obavljenog prijenosa numerički i znakom `#`:

``` shell
$ curl -O -# https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img
#                                                                          1.6%
```

### Ograničenje brzine

Ograničavanje se vrši parametrom `-Y`. Pritom se brzina prijenosa navodi u bajtovima po sekundi.

``` shell
$ curl -Y 3000 http://inf2.uniri.hr/
```

U ovom primjeru smo brzinu prijenosa ograničili na približno 3 kilobajta po sekundi.

### Rječit način rada

!!! note
    Poslužitelj `inf2.uniri.hr` danas koristi [nginx](https://nginx.org/) tako da će ispis biti malo drugačiji kad isprobate iduće naredbe.

Parametrom `-v` moguće je dobiti više detalja kod prijenosa. Sami detalji variraju ovisno o protokolu:

``` shell
$ curl -v http://inf2.uniri.hr/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to inf2.uniri.hr (193.198.209.42) port 80 (#0)
> GET / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: inf2.uniri.hr
> Accept: */*
>
< HTTP/1.1 200 OK
< Date: Thu, 20 Mar 2014 13:12:35 GMT
* Server Apache/2.4.7 (Debian) is not blacklisted
< Server: Apache/2.4.7 (Debian)
< Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
< ETag: "7f1-4f3b7016528d0"
< Accept-Ranges: bytes
< Content-Length: 2033
< Vary: Accept-Encoding
< Content-Type: text/html
<
<!DOCTYPE html>
<html lang="hr">
...
</html>
* Connection #0 to host inf2.uniri.hr left intact
```

## Rad sa specifičnim značajkama protokola HTTP u alatu cURL

!!! todo
    Osvježiti primjerima koji koriste [httpbin.org](https://httpbin.org/).

### HTTP metoda GET

Korištenjem parametra `-X` moguće je specificirati tip zahtjeva koji će biti napravljen na poslužitelj. HTTP sa zadanim postavkama koristi GET, tako da su iduće dvije naredbe ekvivalentne:

``` shell
$ curl http://inf2.uniri.hr/
$ curl -X GET http://inf2.uniri.hr/
```

### HTTP metoda HEAD

HTTP metoda HEAD dohvaća metapodatke iz HTTP zaglavlja. Parametrom `-i` uključujemo prikaz dohvaćenog HTTP zaglavlja.

``` shell
$ curl -X HEAD -i http://inf2.uniri.hr/
HTTP/1.1 200 OK
Date: Thu, 20 Mar 2014 13:18:14 GMT
Server: Apache/2.4.7 (Debian)
Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
ETag: "7f1-4f3b7016528d0"
Accept-Ranges: bytes
Content-Length: 2033
Vary: Accept-Encoding
Content-Type: text/html

curl: (18) transfer closed with 2033 bytes remaining to read
```

Greška koju cURL javlja je očekivana i posljedica je činjenice da HEAD dohvaća samo zaglavlje HTTP odgovora, ne i tijelo. Ona se može izbjeći korištenjem parametra `-I` za dohvaćanje zaglavlja umjesto `-X HEAD`:

``` shell
$ curl -I http://inf2.uniri.hr/
HTTP/1.1 200 OK
Date: Thu, 20 Mar 2014 13:19:30 GMT
Server: Apache/2.4.7 (Debian)
Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
ETag: "7f1-4f3b7016528d0"
Accept-Ranges: bytes
Content-Length: 2033
Vary: Accept-Encoding
Content-Type: text/html
```

### HTTP metoda POST

Kod metode POST potrebno je parametrom `-d` navesti podatke koji se šalju u obliku `varijabla=vrijednost`. Ukoliko se navodi više varijabli, parametar `-d` navodi se više puta:

``` shell
$ curl -X POST -d "ime=Ivan" -d "prezime=Horvat" -d "dob=23" http://inf2.uniri.hr/postexperiment.php
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8">
    <title>Naslovna stranica poslužitelja inf2</title>
    <link rel="stylesheet" href="style.css">
    <!-- <script src="script.js"></script> -->
  </head>
  <body>
    <h1>Stranica za eksperimentiranje s HTTP POST zahtjevima</h1>
    <p>POST varijabla ime ima vrijednost Ivan</p>
    <p>POST varijabla prezime ima vrijednost Horvat</p>
    <p>POST varijabla dob ima vrijednost 23</p>
  </body>
</html>
```

Parametar `-d` implicira metodu POST, tako da je gornja naredba ekvivalentna naredbi:

``` shell
$ curl -d "ime=Ivan" -d "prezime=Horvat" -d "dob=23" http://inf2.uniri.hr/postexperiment.php
```

### HTTP metoda PUT

Za postavljanje datoteka na poslužitelj koristi se HTTP metoda PUT. Međutim, iz sigurnosnih razloga ona je većinom nedozvoljena na poslužiteljima:

``` shell
$ curl -T lokalnadatoteka.txt http://inf2.uniri.hr/podaci.txt
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>405 Method Not Allowed</title>
</head><body>
<h1>Method Not Allowed</h1>
<p>The requested method PUT is not allowed for the URL /podaci.txt.</p>
<hr>
<address>Apache/2.4.7 (Debian) Server at inf2.uniri.hr Port 80</address>
</body></html>
```

### HTTP referer

HTTP referer naveden u zahtjevu moguće je navesti parametrom `-e`. Da bi vidjeli promjenu, potrebno je uključiti rječiti način rada.

``` shell
$ curl -v -e www.google.hr -I http://inf2.uniri.hr/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to inf2.uniri.hr (193.198.209.42) port 80 (#0)
> HEAD / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: inf2.uniri.hr
> Accept: */*
> Referer: www.google.hr
>
< HTTP/1.1 200 OK
HTTP/1.1 200 OK
< Date: Thu, 20 Mar 2014 13:45:30 GMT
Date: Thu, 20 Mar 2014 13:45:30 GMT
* Server Apache/2.4.7 (Debian) is not blacklisted
< Server: Apache/2.4.7 (Debian)
Server: Apache/2.4.7 (Debian)
< Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
< ETag: "7f1-4f3b7016528d0"
ETag: "7f1-4f3b7016528d0"
< Accept-Ranges: bytes
Accept-Ranges: bytes
< Content-Length: 2033
Content-Length: 2033
< Vary: Accept-Encoding
Vary: Accept-Encoding
< Content-Type: text/html
Content-Type: text/html

<
* Connection #0 to host inf2.uniri.hr left intact
```

### HTTP user agent

HTTP user agent koji cURL koristi u zadanim postavkama je `curl/7.35.0` (pri čemu je 7.35.0 verzija cURL-a) i moguće ga je promijeniti parametrom `-A`. Ponovno koristimo rječit način rada kako bi u zaglavlju vidjeli razliku:

``` shell
$ curl -v -A 'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0' -I http://inf2.uniri.hr/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to inf2.uniri.hr (193.198.209.42) port 80 (#0)
> HEAD / HTTP/1.1
> User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0
> Host: inf2.uniri.hr
> Accept: */*
>
< HTTP/1.1 200 OK
HTTP/1.1 200 OK
< Date: Thu, 20 Mar 2014 13:47:25 GMT
Date: Thu, 20 Mar 2014 13:47:25 GMT
* Server Apache/2.4.7 (Debian) is not blacklisted
< Server: Apache/2.4.7 (Debian)
Server: Apache/2.4.7 (Debian)
< Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
Last-Modified: Mon, 03 Mar 2014 17:20:34 GMT
< ETag: "7f1-4f3b7016528d0"
ETag: "7f1-4f3b7016528d0"
< Accept-Ranges: bytes
Accept-Ranges: bytes
< Content-Length: 2033
Content-Length: 2033
< Vary: Accept-Encoding
Vary: Accept-Encoding
< Content-Type: text/html
Content-Type: text/html

<
* Connection #0 to host inf2.uniri.hr left intact
```

Primjeri user agenata koji se također mogu koristiti:

- `Mozilla/3.0 (Win95; I)`
- `Mozilla/3.04 (Win95; U)`
- `Mozilla/2.02 (OS/2; U)`
- `Mozilla/4.04 [en] (X11; U; AIX 4.2; Nav)`
- `Mozilla/4.05 [en] (X11; U; Linux 2.0.32 i586)`

Više primjera moguće je naći na [WhatIsMyBrowser-ovim stranicama](https://developers.whatismybrowser.com/useragents/explore/).

### HTTP kolačići (cookies)

HTTP kolačiće koje stranica nudi moguće je spremiti u datoteku korištenjem parametra `-c` i navođenjem imena datoteke.

``` shell
$ curl -c cookies.txt http://www.google.hr/
...
$ cat cookies.txt
# Netscape HTTP Cookie File
# https://curl.se/docs/http-cookies.html
# This file was generated by libcurl! Edit at your own risk.

.google.hr      TRUE    /       FALSE   1458395398      PREF    ID=66d2d46d264532f6:FF=0:TM=1395323398:LM=1395323398:S=AZv39tEXo7wyBIxv
#HttpOnly_.google.hr    TRUE    /       FALSE   1411134598      NID     67=s6f-BTMOKNbJ8iGJe_51mp7JaQH2mDHhS-zRTcQiEq1CBUL1o7XgNo8087-szuFre2SZ1u6NNOTPVbNussrGdSLWysuhK-INU7sIuJ2SQUTFzsZkg31ilyB3uYwc6Qdf
```

Uočimo nakon tri komentara i praznog retka kolačić dva retka s kolačićima, jedan koji se koristi i jedan zakomentirani koji se ne koristi. Stupci u retku s kolačićem su redom:

- domena, u našem slučaju `.google.hr`
- uključuje li se i poddomene, u našem slučaju `TRUE`, što ima smisla obzirom da smo kolačić dobili s `www.google.hr`
- putanja, u našem slučaju `/`, što je u skladu s putanjom na koju smo uputili zahtjev
- ograničava li se slanje i primanje na HTTPS, u našem slučaju `FALSE`, što ima smisla obzirom da smo kolačić dobili s `http://`
- vrijeme isteka kolačića izraženo u Unix epohi, odnosno broju sekundi od 1. siječnja 1970. u ponoć, u našem slučaju `1458395398`
- ime kolačića, u našem slučaju `PREF`
- vrijednost kolačića, u našem slučaju `ID=66d2d46d264532f6:FF=0:TM=1395323398:LM=1395323398:S=AZv39tEXo7wyBIxv`

Uočimo u retku ispod još jedan, ali zakomentirani kolačić.

Kolačiće je moguće iskoristiti kod ponovnog pristupa stranici navođenjem imena datoteke parametrom `-b`

``` shell
$ curl -b cookies.txt http://www.google.hr/
...
```

ili navođenjem kolačića u obliku:

``` shell
$ curl -b "ID=66d2d46d264532f6; FF=0; TM=1395323398; LM=1395323398; S=AZv39tEXo7wyBIxv" http://www.google.hr/
...
```

Parametrom `-v` vidimo koji kolačići se šalju:

``` shell
$ curl -v -I -b "ID=66d2d46d264532f6; FF=0; TM=1395323398; LM=1395323398; S=AZv39tEXo7wyBIxv" http://www.google.hr/
* Hostname was NOT found in DNS cache
*   Trying 173.194.70.94...
* Connected to www.google.hr (173.194.70.94) port 80 (#0)
> HEAD / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: www.google.hr
> Accept: */*
> Cookie: ID=66d2d46d264532f6; FF=0; TM=1395323398; LM=1395323398; S=AZv39tEXo7wyBIxv
>
< HTTP/1.1 200 OK
HTTP/1.1 200 OK
< Date: Thu, 20 Mar 2014 13:54:57 GMT
Date: Thu, 20 Mar 2014 13:54:57 GMT
< Expires: -1
Expires: -1
< Cache-Control: private, max-age=0
Cache-Control: private, max-age=0
< Content-Type: text/html; charset=ISO-8859-2
Content-Type: text/html; charset=ISO-8859-2
< Set-Cookie: PREF=ID=c15f2cf143ce0e16:FF=0:TM=1395323697:LM=1395323697:S=hADpF-Ww5RNTpWhG; expires=Sat, 19-Mar-2016 13:54:57 GMT; path=/; domain=.google.hr
Set-Cookie: PREF=ID=c15f2cf143ce0e16:FF=0:TM=1395323697:LM=1395323697:S=hADpF-Ww5RNTpWhG; expires=Sat, 19-Mar-2016 13:54:57 GMT; path=/; domain=.google.hr
< Set-Cookie: NID=67=M_DGSRCzXOL0vR5WYiCeuoitrNy23wKTuag6Zs-IkQRo6fTjm-ERQyR6obpfgtHpaUncQFED5rXaVE9LvVOGvlfLIHVGr4xywZhlw0mZZdByCofUPihRQLQ1rwXszRkQ; expires=Fri, 19-Sep-2014 13:54:57 GMT; path=/; domain=.google.hr; HttpOnly
Set-Cookie: NID=67=M_DGSRCzXOL0vR5WYiCeuoitrNy23wKTuag6Zs-IkQRo6fTjm-ERQyR6obpfgtHpaUncQFED5rXaVE9LvVOGvlfLIHVGr4xywZhlw0mZZdByCofUPihRQLQ1rwXszRkQ; expires=Fri, 19-Sep-2014 13:54:57 GMT; path=/; domain=.google.hr; HttpOnly
< P3P: CP="This is not a P3P policy! See http://www.google.com/support/accounts/bin/answer.py?hl=en&answer=151657 for more info."
P3P: CP="This is not a P3P policy! See http://www.google.com/support/accounts/bin/answer.py?hl=en&answer=151657 for more info."
* Server gws is not blacklisted
< Server: gws
Server: gws
< X-XSS-Protection: 1; mode=block
X-XSS-Protection: 1; mode=block
< X-Frame-Options: SAMEORIGIN
X-Frame-Options: SAMEORIGIN
< Alternate-Protocol: 80:quic
Alternate-Protocol: 80:quic
< Transfer-Encoding: chunked
Transfer-Encoding: chunked

<
* Connection #0 to host www.google.hr left intact
```

## Dodatak: alternativno korisničko sučelje Curlie

[Curlie](https://curlie.io/) je alternativno korisničko sučelje za alat cURL napravljeno po uzoru na [HTTPie](https://httpie.io/). Curlie podržava sve cURL-ove značajke, ali nudi jednostavnije sučelje naredbenog retka i bojenje izlaza.

Uvjerimo se da imamo instaliran Curlie:

``` shell
$ curlie -V
curl 7.72.0 (x86_64-pc-linux-gnu) libcurl/7.72.0 OpenSSL/1.1.1h zlib/1.2.11 brotli/1.0.9 libidn2/2.3.0 libpsl/0.21.0 (+libidn2/2.3.0) libssh2/1.8.0 nghttp2/1.41.0 librtmp/2.3
Release-Date: 2020-08-19
Protocols: dict file ftp ftps gopher http https imap imaps ldap ldaps pop3 pop3s rtmp rtsp scp sftp smb smbs smtp smtps telnet tftp
Features: AsynchDNS brotli GSS-API HTTP2 HTTPS-proxy IDN IPv6 Kerberos Largefile libz NTLM NTLM_WB PSL SPNEGO SSL TLS-SRP UnixSockets
```

Uočimo da se Curlie nama predstavlja kao cURL jer njega i koristi u pozadini. Iz istog razloga će se i web poslužiteljima na koje se budemo povezivali predstavljati kao cURL, u što se možemo uvjeriti promatranjem zaglavlja `User-Agent` u narednim primjerima.

Osnovno korištenje je identično kao i kod cURL-a:

``` shell
$ curlie http://httpbin.org/headers
{
    "headers": {
        "Accept": "application/json, */*",
        "Host": "httpbin.org",
        "User-Agent": "curl/7.72.0",
        "X-Amzn-Trace-Id": "Root=1-5fb8f92c-31c8504941f54a534445099f"
    }
}
HTTP/1.1 200 OK
Date: Sat, 21 Nov 2020 11:25:33 GMT
Content-Type: application/json
Content-Length: 191
Connection: keep-alive
Server: gunicorn/19.9.0
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true

$ curlie https://httpbin.org/headers
HTTP/2 200
date: Sat, 21 Nov 2020 11:26:43 GMT
content-type: application/json
content-length: 191
server: gunicorn/19.9.0
access-control-allow-origin: *
access-control-allow-credentials: true

{
    "headers": {
        "Accept": "application/json, */*",
        "Host": "httpbin.org",
        "User-Agent": "curl/7.72.0",
        "X-Amzn-Trace-Id": "Root=1-5fb8f973-5070f2961ed624467990ebdc"
    }
}
```

Složenije korištenje, npr. HTTP metoda PUT s navođenjem sadržaja zaglavlja i tijela poruke je dostupna putem značajno jednostavnije sintakse nego što je to slučaj kad koristimo cURL:

``` shell
$ curlie -v PUT httpbin.org/status/201 X-API-Token:123 name=John
*   Trying 34.198.212.59:80...
* Connected to httpbin.org (34.198.212.59) port 80 (#0)
PUT /status/201 HTTP/1.1
Host: httpbin.org
User-Agent: curl/7.72.0
X-API-Token:123
Content-Type: application/json
Accept: application/json, */*
Content-Length: 15

{
    "name": "John"
}


* upload completely sent off: 15 out of 15 bytes
* Mark bundle as not supporting multiuse
HTTP/1.1 201 CREATED
Date: Sat, 21 Nov 2020 11:28:36 GMT
Content-Type: text/html; charset=utf-8
Content-Length: 0
Connection: keep-alive
Server: gunicorn/19.9.0
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true

* Connection #0 to host httpbin.org left intact
```

## Dodatak: specifične značajke protokola FTP i ostalih protokola u alatu cURL

### Dohvaćanje datoteke sa poslužitelja korištenjem FTP-a

Dohvaćanje FTP URL-a se izvodi slično kao kod HTTP-a:

``` shell
$ curl ftp://inf2.uniri.hr/
drwxr-xr-x    2 0        0            4096 Mar 30 23:21 pub
```

U slučaju da u direktoriju postoje datoteke, one će biti ispisane:

``` shell
$ curl ftp://inf2.uniri.hr/pub/
-rw-r--r--    1 0        0              18 Mar 30 23:21 cake.txt
```

U slučaju da preuzimamo datoteku, na standardni izlaz ispisuje se njen sadržaj:

``` shell
$ curl ftp://inf2.uniri.hr/pub/cake.txt
THE CAKE IS A LIE
```

### Podizanje datoteka na poslužitelj korištenjem FTP-a

Postavljanje datoteke na FTP poslužitelj vrši se parametrom `-T`:

``` shell
$ curl -T lokalnadatoteka.txt ftp://inf2.uniri.hr/datoteka.txt
```

Postavljanje datoteke uz prijavu vrši se parametrom `-u` i navođenjem korisničkog imena i zaporke:

``` shell
$ curl -T lokalnadatoteka.txt -u vedranm:l33th4x0rp4ssw0rd ftp://inf2.uniri.hr/datoteka.txt
```

Ukoliko je nakon `-u` navedeno samo korisničko ime, cURL će tražiti unos zaporke:

``` shell
$ curl -T lokalnadatoteka.txt -u vedranm ftp://inf2.uniri.hr/datoteka.txt
Enter host password for user 'vedranm':
```

### Korištenje protokola SCP i SFTP

U cURL-u se SCP i SFTP koriste slično kao FTP; razlika je da postoji mogućnost korištenja privatnog ključa umjesto lozinke. Ponovno parametrom `-u` navodimo korisničko ime kojim se prijavljujemo na poslužitelj. Primjer korištenja SCP-a je oblika:

``` shell
$ curl -u vedranm scp://inf2.uniri.hr/home/vedranm/epic-battle.txt
Enter host password for user 'vedranm':

Tacgnol vs Longcat
On a scale from 1 to epic, I'd probably say EPIC
```

Primjer s korištenjem SFTP-a je oblika:

``` shell
$ curl -u vedranm sftp://inf2.uniri.hr/~/protip.txt
Enter host password for user 'vedranm':

Doom II protip: To defeat the Cyberdemon, shoot at it until it dies.
```
