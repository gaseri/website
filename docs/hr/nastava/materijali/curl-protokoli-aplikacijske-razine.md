---
author: Vedran Miletić, Edvin Močibob
---

# Rad s protokolima aplikacijske razine korištenjem cURL-a

## Pregled protokola aplikacijske razine

### Pojam URL-a

[Uniform Resource Locator (URL)](https://en.wikipedia.org/wiki/URL) je znakovni niz koji predstavlja vezu na neki resurs. URL se često koristi za pristupanje web stranicama korištenjem protokola HTTP (`http://`) i HTTPS (`https://`), ali podržava i druge protokole, npr. FTP (`ftp://`) i e-mail (`mailto:`).

Primjerice, URL može biti `http://example.group.miletic.net:80/category/page.html`. Njegovi su dijelovi:

- protokol (`http://`),
- domena (`example.group.miletic.net`), umjesto domene se može koristiti i IP adresa,
- broj vrata (`80`, opcionalan) i
- putanja do resursa (`/category/page.html`).

Pritom resurs može biti HTML datoteka kao u primjeru, ali i bilo što drugo (npr. CSS datoteka, slika ili video).

### Protokoli HTTP i HTTPS

[Hypertext Transfer Protocol (HTTP)](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol) je aplikacijski protokol i temelj podatkovne komunikacije za WWW. Internet Engineering Task Force (IETF) i the World Wide Web Consortium (W3C) stoje iza njegovog razvijanja. HTTP radi na request-response paradigmi po kljent-poslužitelj modelu. Jedna HTTP sesija sastoji se od niza zahtjeva (klijent) i odgovora (poslužitelj).

[Hypertext Transfer Protocol Secure (HTTPS)](https://en.wikipedia.org/wiki/HTTPS) je proširenje HTTP protokola SSL/TLS slojem. HTTPS se koristi za sigurnu komunikaciju preko računalne mreže. U srži TLS-a je korištenje privatnih i javnih ključeva te enkripcija podataka. HTTPS još podrazumijeva autentifikaciju i korištenje certifikata za dokazivanje identiteta.

### Protokol FTP

[File Transfer Protocol (FTP)](https://en.wikipedia.org/wiki/File_Transfer_Protocol) je mrežni protokol koji omogućuje transfer datoteka sa jednog domaćina na drugi preko TCP veze. Radi na klijent-poslužitelj principu. FTP može, ali ne mora koristiti autentifikaciju.

[FTP-SSL (FTPS, poznat kao i FTP Secure)](https://en.wikipedia.org/wiki/FTPS) proširuje FTP tako da uz autentifikaciju omogućuje i enkripciju (SSL/TLS usluge). FTPS treba razlikovati od SFTP protokola koji je vezan uz SSH (Secure Shell).

## Osnovne značajke i način korištenja alata cURL

[cURL](https://en.wikipedia.org/wiki/CURL) (naredba `curl`) je komandno linijski alat za prijenos podataka korištenjem URL sintakse. Podržava brojne protokole (DICT, FILE, FTP, FTPS, Gopher, HTTP, HTTPS, IMAP, IMAPS, LDAP, LDAPS, POP3, POP3S, RTMP, RTSP, SCP, SFTP, SMTP, SMTPS, Telnet i TFTP), od kojih ćemo u nastavku koristiti manji dio. cURL je slobodan softver dostupan pod [MIT licencom](https://en.wikipedia.org/wiki/MIT_License).

cURL podržava SSL certifikate, HTTP naredbe POST i PUT, FTP upload, HTTP upload zasnovan na obrascima, proxy poslužitelje, keksiće, autentifikaciju korištenjem korisničkog imena i zaproke (Basic, Digest, NTLM, Negotiate, kerberos i druge), nastavljanje prijenosa datoteke, tuneliranje putem proxy poslužitelja i još mnogo toga. Mi ćemo se ograničiti na osnovnu funkcionalnost, ali [službena dokumentacija](https://curl.se/docs/manual.html) dostupna na [cURL-ovim stranicama](https://curl.se/) ima više detalja.

### Preuzimanje stranice ili datoteke

Preuzimanje se vrši navođenjem URL-a. Pristupimo HTTP poslužitelju na domeni `example.group.miletic.net` za primjer:

``` shell
$ curl http://example.group.miletic.net/
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <title>Apache2 Debian Default Page: It works</title>
    <style type="text/css" media="screen">
  * {
    margin: 0px 0px 0px 0px;
    padding: 0px 0px 0px 0px;
  }

  body, html {
    padding: 3px 3px 3px 3px;

    background-color: #D8DBE2;

    font-family: Verdana, sans-serif;
    font-size: 11pt;
    text-align: center;
  }

  div.main_page {
    position: relative;
    display: table;

    width: 800px;

    margin-bottom: 3px;
    margin-left: auto;
    margin-right: auto;
    padding: 0px 0px 0px 0px;

    border-width: 2px;
    border-color: #212738;
    border-style: solid;

    background-color: #FFFFFF;

    text-align: center;
  }

  div.page_header {
    height: 99px;
    width: 100%;

    background-color: #F5F6F7;
  }

  div.page_header span {
    margin: 15px 0px 0px 50px;

    font-size: 180%;
    font-weight: bold;
  }

  div.page_header img {
    margin: 3px 0px 0px 40px;

    border: 0px 0px 0px;
  }

  div.table_of_contents {
    clear: left;

    min-width: 200px;

    margin: 3px 3px 3px 3px;

    background-color: #FFFFFF;

    text-align: left;
  }

  div.table_of_contents_item {
    clear: left;

    width: 100%;

    margin: 4px 0px 0px 0px;

    background-color: #FFFFFF;

    color: #000000;
    text-align: left;
  }

  div.table_of_contents_item a {
    margin: 6px 0px 0px 6px;
  }

  div.content_section {
    margin: 3px 3px 3px 3px;

    background-color: #FFFFFF;

    text-align: left;
  }

  div.content_section_text {
    padding: 4px 8px 4px 8px;

    color: #000000;
    font-size: 100%;
  }

  div.content_section_text pre {
    margin: 8px 0px 8px 0px;
    padding: 8px 8px 8px 8px;

    border-width: 1px;
    border-style: dotted;
    border-color: #000000;

    background-color: #F5F6F7;

    font-style: italic;
  }

  div.content_section_text p {
    margin-bottom: 6px;
  }

  div.content_section_text ul, div.content_section_text li {
    padding: 4px 8px 4px 16px;
  }

  div.section_header {
    padding: 3px 6px 3px 6px;

    background-color: #8E9CB2;

    color: #FFFFFF;
    font-weight: bold;
    font-size: 112%;
    text-align: center;
  }

  div.section_header_red {
    background-color: #CD214F;
  }

  div.section_header_grey {
    background-color: #9F9386;
  }

  .floating_element {
    position: relative;
    float: left;
  }

  div.table_of_contents_item a,
  div.content_section_text a {
    text-decoration: none;
    font-weight: bold;
  }

  div.table_of_contents_item a:link,
  div.table_of_contents_item a:visited,
  div.table_of_contents_item a:active {
    color: #000000;
  }

  div.table_of_contents_item a:hover {
    background-color: #000000;

    color: #FFFFFF;
  }

  div.content_section_text a:link,
  div.content_section_text a:visited,
   div.content_section_text a:active {
    background-color: #DCDFE6;

    color: #000000;
  }

  div.content_section_text a:hover {
    background-color: #000000;

    color: #DCDFE6;
  }

  div.validator {
  }
    </style>
  </head>
  <body>
    <div class="main_page">
      <div class="page_header floating_element">
        <img src="/icons/openlogo-75.png" alt="Debian Logo" class="floating_element"/>
        <span class="floating_element">
          Apache2 Debian Default Page
        </span>
      </div>
<!--      <div class="table_of_contents floating_element">
        <div class="section_header section_header_grey">
          TABLE OF CONTENTS
        </div>
        <div class="table_of_contents_item floating_element">
          <a href="#about">About</a>
        </div>
        <div class="table_of_contents_item floating_element">
          <a href="#changes">Changes</a>
        </div>
        <div class="table_of_contents_item floating_element">
          <a href="#scope">Scope</a>
        </div>
        <div class="table_of_contents_item floating_element">
          <a href="#files">Config files</a>
        </div>
      </div>
-->
      <div class="content_section floating_element">


        <div class="section_header section_header_red">
          <div id="about"></div>
          It works!
        </div>
        <div class="content_section_text">
          <p>
                This is the default welcome page used to test the correct
                operation of the Apache2 server after installation on Debian systems.
                If you can read this page, it means that the Apache HTTP server installed at
                this site is working properly. You should <b>replace this file</b> (located at
                <tt>/var/www/html/index.html</tt>) before continuing to operate your HTTP server.
          </p>


          <p>
                If you are a normal user of this web site and don't know what this page is
                about, this probably means that the site is currently unavailable due to
                maintenance.
                If the problem persists, please contact the site's administrator.
          </p>

        </div>
        <div class="section_header">
          <div id="changes"></div>
                Configuration Overview
        </div>
        <div class="content_section_text">
          <p>
                Debian's Apache2 default configuration is different from the
                upstream default configuration, and split into several files optimized for
                interaction with Debian tools. The configuration system is
                <b>fully documented in
                /usr/share/doc/apache2/README.Debian.gz</b>. Refer to this for the full
                documentation. Documentation for the web server itself can be
                found by accessing the <a href="/manual">manual</a> if the <tt>apache2-doc</tt>
                package was installed on this server.

          </p>
          <p>
                The configuration layout for an Apache2 web server installation on Debian systems is as follows:
          </p>
          <pre>
/etc/apache2/
|-- apache2.conf
|       `--  ports.conf
|-- mods-enabled
|       |-- *.load
|       `-- *.conf
|-- conf-enabled
|       `-- *.conf
|-- sites-enabled
|       `-- *.conf
          </pre>
          <ul>
                        <li>
                           <tt>apache2.conf</tt> is the main configuration
                           file. It puts the pieces together by including all remaining configuration
                           files when starting up the web server.
                        </li>

                        <li>
                           <tt>ports.conf</tt> is always included from the
                           main configuration file. It is used to determine the listening ports for
                           incoming connections, and this file can be customized anytime.
                        </li>

                        <li>
                           Configuration files in the <tt>mods-enabled/</tt>,
                           <tt>conf-enabled/</tt> and <tt>sites-enabled/</tt> directories contain
                           particular configuration snippets which manage modules, global configuration
                           fragments, or virtual host configurations, respectively.
                        </li>

                        <li>
                           They are activated by symlinking available
                           configuration files from their respective
                           *-available/ counterparts. These should be managed
                           by using our helpers
                           <tt>
                                a2enmod,
                                a2dismod,
                           </tt>
                           <tt>
                                a2ensite,
                                a2dissite,
                            </tt>
                                and
                           <tt>
                                a2enconf,
                                a2disconf
                           </tt>. See their respective man pages for detailed information.
                        </li>

                        <li>
                           The binary is called apache2. Due to the use of
                           environment variables, in the default configuration, apache2 needs to be
                           started/stopped with <tt>/etc/init.d/apache2</tt> or <tt>apache2ctl</tt>.
                           <b>Calling <tt>/usr/bin/apache2</tt> directly will not work</b> with the
                           default configuration.
                        </li>
          </ul>
        </div>

        <div class="section_header">
            <div id="docroot"></div>
                Document Roots
        </div>

        <div class="content_section_text">
            <p>
                By default, Debian does not allow access through the web browser to
                <em>any</em> file apart of those located in <tt>/var/www</tt>,
                <a href="http://httpd.apache.org/docs/2.4/mod/mod_userdir.html" rel="nofollow">public_html</a>
                directories (when enabled) and <tt>/usr/share</tt> (for web
                applications). If your site is using a web document root
                located elsewhere (such as in <tt>/srv</tt>) you may need to whitelist your
                document root directory in <tt>/etc/apache2/apache2.conf</tt>.
            </p>
            <p>
                The default Debian document root is <tt>/var/www/html</tt>. You
                can make your own virtual hosts under /var/www. This is different
                to previous releases which provides better security out of the box.
            </p>
        </div>

        <div class="section_header">
          <div id="bugs"></div>
                Reporting Problems
        </div>
        <div class="content_section_text">
          <p>
                Please use the <tt>reportbug</tt> tool to report bugs in the
                Apache2 package with Debian. However, check <a
                href="http://bugs.debian.org/cgi-bin/pkgreport.cgi?ordering=normal;archive=0;src=apache2;repeatmerged=0"
                rel="nofollow">existing bug reports</a> before reporting a new bug.
          </p>
          <p>
                Please report bugs specific to modules (such as PHP and others)
                to respective packages, not to the web server itself.
          </p>
        </div>




      </div>
    </div>
    <div class="validator">
    </div>
  </body>
</html>
```

Ukoliko želimo spremiti izlaz u datoteku umjesto ispisati na standardni izlaz, koristimo parametar `-o` i navodimo ime datoteke:

``` shell
$ curl -o example-index.html http://example.group.miletic.net/
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2033  100  2033    0     0  74324      0 --:--:-- --:--:-- --:--:-- 75296
```

Parametrom `-O` izlaz možemo spremiti u lokalnu datoteku istog imena kao datoteka na poslužitelju:

``` shell
$ curl -O http://example.group.miletic.net/index.html
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2033  100  2033    0     0   120k      0 --:--:-- --:--:-- --:--:--  124k
$ ls
index.html
```

### Preuzimanje dijela datoteke

Parametrom `-r` moguće je specificirati raspon podataka datoteke koji će bit preuzet:

``` shell
$ curl -r 0-350 http://example.group.miletic.net/index.html
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <title>Apache2 Debian Default Page: It works</title>
    <style type="text/css" media="screen">
```

### Mjerenje napretka

Kod većih datoteka može se uočiti da je mjerenje napretka interaktivno:

``` shell
$ curl -O https://mirror.pkgbuild.com/images/latest/Arch-Linux-x86_64-basic-20220215.47946.qcow2
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  5  459M    5 23.7M    0     0  1777k      0  0:04:24  0:00:13  0:04:11 2849k
```

!!! tip
    Poveznica u primjeru iznad je na verziju od 15. veljače 2022. i vjerojatno je zastarjela. Provjerite [mirror.pkgbuild.com/images/latest/](https://mirror.pkgbuild.com/images/latest/) za točno ime aktualne verzije.

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
$ curl -O -# https://mirror.pkgbuild.com/images/latest/Arch-Linux-x86_64-cloudimg-20220215.47946.qcow2
#####################                                                                          23,0%
```

### Ograničenje brzine

Ograničavanje se vrši parametrom `-Y`. Pritom se brzina prijenosa navodi u bajtovima po sekundi.

``` shell
$ curl -Y 3000 http://example.group.miletic.net/
```

U ovom primjeru smo brzinu prijenosa ograničili na približno 3 kilobajta po sekundi.

### Rječit način rada

!!! note
    Poslužitelj `example.group.miletic.net` danas koristi noviju verziju [HTTP poslužitelja Apache](https://httpd.apache.org/) tako da će ispis biti malo drugačiji kad isprobate iduće naredbe.

Parametrom `-v` moguće je dobiti više detalja kod prijenosa. Sami detalji variraju ovisno o protokolu:

``` shell
$ curl -v http://example.group.miletic.net/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to example.group.miletic.net (193.198.209.42) port 80 (#0)
> GET / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: example.group.miletic.net
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
* Connection #0 to host example.group.miletic.net left intact
```

## Rad sa specifičnim značajkama protokola HTTP u alatu cURL

!!! todo
    Osvježiti primjerima koji koriste [httpbin.org](https://httpbin.org/).

### HTTP metoda GET

Korištenjem parametra `-X` moguće je specificirati tip zahtjeva koji će biti napravljen na poslužitelj. HTTP sa zadanim postavkama koristi GET, tako da su iduće dvije naredbe ekvivalentne:

``` shell
$ curl http://example.group.miletic.net/
$ curl -X GET http://example.group.miletic.net/
```

### HTTP metoda HEAD

HTTP metoda HEAD dohvaća metapodatke iz HTTP zaglavlja. Parametrom `-i` uključujemo prikaz dohvaćenog HTTP zaglavlja.

``` shell
$ curl -X HEAD -i http://example.group.miletic.net/
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
$ curl -I http://example.group.miletic.net/
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
$ curl -X POST -d "ime=Ivan" -d "prezime=Horvat" -d "dob=23" https://apps.group.miletic.net/request/
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8">
    <title>Stranica za eksperimentiranje s HTTP POST zahtjevima</title>
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
$ curl -d "ime=Ivan" -d "prezime=Horvat" -d "dob=23" https://apps.group.miletic.net/request/
```

### HTTP metoda PUT

Za postavljanje datoteka na poslužitelj koristi se HTTP metoda PUT. Međutim, iz sigurnosnih razloga ona je većinom nedozvoljena na poslužiteljima:

``` shell
$ curl -T lokalnadatoteka.txt http://example.group.miletic.net/podaci.txt
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>405 Method Not Allowed</title>
</head><body>
<h1>Method Not Allowed</h1>
<p>The requested method PUT is not allowed for the URL /podaci.txt.</p>
<hr>
<address>Apache/2.4.7 (Debian) Server at example.group.miletic.net Port 80</address>
</body></html>
```

### HTTP referer

HTTP referer naveden u zahtjevu moguće je navesti parametrom `-e`. Da bi vidjeli promjenu, potrebno je uključiti rječiti način rada.

``` shell
$ curl -v -e www.google.hr -I http://example.group.miletic.net/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to example.group.miletic.net (193.198.209.42) port 80 (#0)
> HEAD / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: example.group.miletic.net
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
* Connection #0 to host example.group.miletic.net left intact
```

### HTTP user agent

HTTP user agent koji cURL koristi u zadanim postavkama je `curl/7.35.0` (pri čemu je 7.35.0 verzija cURL-a) i moguće ga je promijeniti parametrom `-A`. Ponovno koristimo rječit način rada kako bi u zaglavlju vidjeli razliku:

``` shell
$ curl -v -A 'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0' -I http://example.group.miletic.net/
* Hostname was NOT found in DNS cache
*   Trying 193.198.209.42...
* Connected to example.group.miletic.net (193.198.209.42) port 80 (#0)
> HEAD / HTTP/1.1
> User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0
> Host: example.group.miletic.net
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
* Connection #0 to host example.group.miletic.net left intact
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

## Pojednostavljeno korisničko sučelje Curlie

[Curlie](https://curlie.io/) je pojednostavljeno korisničko sučelje za alat cURL inspirirano sučeljem naredbenog retka alata [HTTPie](https://httpie.io/) (cURL + HTTPie = Curlie). Curlie podržava sve cURL-ove značajke, ali nudi i jednostavnost korištenja sučelja naredbenog retka i ljepše oblikovanje izlaznih podataka (specifično, zaglavlja HTTP odgovora i JSON-a u tijelu HTTP odgovora) uz korištenje boje.

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
$ curl ftp://example.group.miletic.net/
drwxr-xr-x    2 0        0            4096 Mar 30 23:21 pub
```

U slučaju da u direktoriju postoje datoteke, one će biti ispisane:

``` shell
$ curl ftp://example.group.miletic.net/pub/
-rw-r--r--    1 0        0              18 Mar 30 23:21 cake.txt
```

U slučaju da preuzimamo datoteku, na standardni izlaz ispisuje se njen sadržaj:

``` shell
$ curl ftp://example.group.miletic.net/pub/cake.txt
THE CAKE IS A LIE
```

### Podizanje datoteka na poslužitelj korištenjem FTP-a

Postavljanje datoteke na FTP poslužitelj vrši se parametrom `-T`:

``` shell
$ curl -T lokalnadatoteka.txt ftp://example.group.miletic.net/datoteka.txt
```

Postavljanje datoteke uz prijavu vrši se parametrom `-u` i navođenjem korisničkog imena i zaporke:

``` shell
$ curl -T lokalnadatoteka.txt -u vedranm:l33th4x0rp4ssw0rd ftp://example.group.miletic.net/datoteka.txt
```

Ukoliko je nakon `-u` navedeno samo korisničko ime, cURL će tražiti unos zaporke:

``` shell
$ curl -T lokalnadatoteka.txt -u vedranm ftp://example.group.miletic.net/datoteka.txt
Enter host password for user 'vedranm':
```

### Korištenje protokola SCP i SFTP

U cURL-u se SCP i SFTP koriste slično kao FTP; razlika je da postoji mogućnost korištenja privatnog ključa umjesto lozinke. Ponovno parametrom `-u` navodimo korisničko ime kojim se prijavljujemo na poslužitelj. Primjer korištenja SCP-a je oblika:

``` shell
$ curl -u vedranm scp://example.group.miletic.net/home/vedranm/epic-battle.txt
Enter host password for user 'vedranm':

Tacgnol vs Longcat
On a scale from 1 to epic, I'd probably say EPIC
```

Primjer s korištenjem SFTP-a je oblika:

``` shell
$ curl -u vedranm sftp://example.group.miletic.net/~/protip.txt
Enter host password for user 'vedranm':

Doom II protip: To defeat the Cyberdemon, shoot at it until it dies.
```
