---
author: Vedran Miletić
---

# Konfiguracija HTTPS-a u web poslužitelju Apache HTTP Server

Modul [mod_ssl](https://httpd.apache.org/docs/2.4/mod/mod_ssl.html) implementira podršku za SSLv3 i TLSv1.x u Apacheju (SSLv2 više nije podržan jer se smatra nesigurnim) koji se koriste kod ostvarivanja HTTPS veza. Za izvođenje kriptografskih algoritama koristi se [OpenSSL](https://www.openssl.org/).

!!! hint
    Više detalja o SSL-u i TLS-u u Apacheju moguće je pronaći u službenoj dokumentaciji u dijelu [Apache SSL/TLS Encryption](https://httpd.apache.org/docs/2.4/ssl/)

## Stvaranje i postavljanje certifikata i privatnog ključa

Stvorimo samopotpisani certifikat `server.crt` koji traje 30 dana i njegov pripadni tajni ključ `server.key` (takva imena datoteka će nam trebati kasnije):

``` shell
$ openssl req -x509 -nodes -days 30 -newkey rsa:4096 -keyout server.key -out server.crt
Generating a RSA private key
................................................................................................................................................................++++
......................................................................................................................................................................................................++++
writing new private key to 'server.key'
-----
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Country Name (2 letter code) [AU]:
State or Province Name (full name) [Some-State]:
Locality Name (eg, city) []:
Organization Name (eg, company) [Internet Widgits Pty Ltd]:
Organizational Unit Name (eg, section) []:
Common Name (e.g. server FQDN or YOUR name) []:
Email Address []:
```

Dodajmo dvije nove naredbe `COPY` u `Dockerfile` kojima ćemo kopirati certifikat i tajni ključ u direktorij `/usr/local/apache2/conf` unutar slike:

``` dockerfile hl_lines="4-5"
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
COPY ./www /var/www
COPY server.crt /usr/local/apache2/conf
COPY server.key /usr/local/apache2/conf
```

U datoteci `my-httpd.conf` odkomentirajmo linije:

``` apacheconf
#...
#LoadModule socache_shmcb_module modules/mod_socache_shmcb.so
#...
#LoadModule ssl_module modules/mod_ssl.so
#...
#Include conf/extra/httpd-ssl.conf
#...
```

Modul mod_ssl za svoj rad zahtijeva modul [mod_socache_shmcb](https://httpd.apache.org/docs/2.4/mod/mod_socache_shmcb.html) pa prva od tri linije učitava taj modul. Druga linija učitava mod_ssl, a treća njegovu konfiguraciju u kojoj se nalaze naredbe kao što je `Listen 443` koja uključuje HTTPS na vratima 443. Konfiguraciju ćemo nešto kasnije uređivati.

Izgradimo sliku i pokrenimo kontejner:

``` shell
$ docker build -t "my-httpd:2.4-4" .
Sending build context to Docker daemon  32.26kB
Step 1/5 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/5 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> 765440898954
Step 3/5 : COPY ./www /var/www
---> a0c0158083fb
Step 4/5 : COPY server.key /usr/local/apache2/conf
---> 6ffd3334243b
Step 5/5 : COPY server.crt /usr/local/apache2/conf
---> d5b1f004015f
Successfully built d5b1f004015f
Successfully tagged my-httpd:2.4-4
$ docker run my-httpd:2.4-4
[Sun May 10 17:02:53.475776 2020] [ssl:warn] [pid 1:tid 140698297431168] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 17:02:53.476093 2020] [ssl:warn] [pid 1:tid 140698297431168] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 17:02:53.478932 2020] [ssl:warn] [pid 1:tid 140698297431168] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 17:02:53.478939 2020] [ssl:warn] [pid 1:tid 140698297431168] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 17:02:53.480043 2020] [mpm_event:notice] [pid 1:tid 140698297431168] AH00489: Apache/2.4.43 (Unix) OpenSSL/1.1.1d configured -- resuming normal operations
[Sun May 10 17:02:53.480070 2020] [core:notice] [pid 1:tid 140698297431168] AH00094: Command line: 'httpd -D FOREGROUND'
```

Ukoliko se kod izdavanja certifikata pod `Common Name` unese `apache-primjer.rm.miletic.net`, upozorenja `server certificate does NOT include an ID which matches the server name` ne bi trebalo biti. Ovo drugo upozorenje, `server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)`, Apache daje zato što je certifikat samopotpisan. Zbog toga kod testiranja naredba `curl` treba parametar `-k` i imamo:

``` shell
$ curl -k https://172.17.0.2/
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>403 Forbidden</title>
</head><body>
<h1>Forbidden</h1>
<p>You don't have permission to access this resource.</p>
</body></html>
```

## Konfiguracijske naredbe HTTPS poslužitelja

Vidimo da HTTPS poslužitelj odgovara na upit, ali ne sadržajem datoteke `index.html` koji smo se nadali dobiti. Na strani poslužitelja ispisuju se poruke o pogreškama:

```
[Sun May 10 17:02:56.688057 2020] [authz_core:error] [pid 9:tid 140698145773312] [client 172.17.0.1:54160] AH01630: client denied by server configuration: /usr/local/apache2/htdocs/
172.17.0.1 - - [10/May/2020:17:02:56 +0000] "GET / HTTP/1.1" 403 199
```

Morat ćemo konfigurirati `DocumentRoot` i za HTTPS poslužitelj. Dohvatimo konfiguracijsku datoteku:

``` shell
$ docker run --rm httpd:2.4 cat /usr/local/apache2/conf/extra/httpd-ssl.conf > my-httpd-ssl.conf
```

!!! todo
    Nedostaje kratak opis sadržaja datoteke.

Uredimo datoteku; uočimo da u njoj postoji niz konfiguracijskih naredbi koje počinju nizom slova `SSL` i određuju način rada SSL-a. Naredbom `SSLCipherSuite HIGH:MEDIUM:!MD5:!RC4:!3DES` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#sslciphersuite)) navode se dozvoljeni šifrarnici. Podsjetimo se da se kod SSL-a koriste četiri vrste šifrarnika:

- algoritam za razmjenu ključeva: RSA, Diffie-Hellman, Elliptic Curve Diffie-Hellman, Secure Remote Password
- algoritam za autentifikaciju: RSA, Diffie-Hellman, DSS, ECDSA, a specijalno može i nijedan
- algoritam za šifrirane poruka: AES, DES, Triple-DES, RC4, RC2, IDEA, itd.
- algoritam za računanje sažetka koda autentifikacije poruke: MD5, SHA ili SHA1, SHA256, SHA384

Prve dvije vrste algoritma se u nekim podjelama šifrirarnika navode zajedno pa onda kažemo da su tri različite vrste šifrarnika. U našem slučaju vrijednost `HIGH:MEDIUM:!MD5:!RC4:!3DES` ima značenje:

- `HIGH` -- svi šifrarnici koji koriste trostruki DES ili bolji algoritam (npr. AES)
- `MEDIUM` -- svi šifrarnici koji koriste 128-bitno ili bolje šifriranje
- `!MD5` -- ne MD5
- `!RC4` -- ne RC4
- `!3DES` -- ne trostruki DES (bolji algoritmi od trostrukog DES-a su dozvoljeni)

!!! note
    Za današnje standarde korištenje ovoliko širokog skupa šifrarnika (koji onda uključuje i neke slabe šifrarnike) ne bi bilo preporučeno. Popise preporučenih šifrarnika možete pročitati [na Remy van Elstovom blogu u članku Strong SSL Security on Apache2](https://raymii.org/s/tutorials/Strong_SSL_Security_On_Apache2.html) i [na Mozillinom wikiju u članku Security/Server Side TLS](https://wiki.mozilla.org/Security/Server_Side_TLS). Pored toga, Mozilla je razvila [generator konfiguracija SSL-a](https://ssl-config.mozilla.org/) koji je vrlo jednostavan za korištenje, a podržava i Apache.

Naredbom `openssl ciphers` možemo provjeriti o kojim se točno šifrarnicima radi:

``` shell
$ openssl ciphers -v 'HIGH:MEDIUM:!MD5:!RC4:!3DES'
TLS_AES_256_GCM_SHA384  TLSv1.3 Kx=any      Au=any  Enc=AESGCM(256) Mac=AEAD
TLS_CHACHA20_POLY1305_SHA256 TLSv1.3 Kx=any      Au=any  Enc=CHACHA20/POLY1305(256) Mac=AEAD
TLS_AES_128_GCM_SHA256  TLSv1.3 Kx=any      Au=any  Enc=AESGCM(128) Mac=AEAD
ECDHE-ECDSA-AES256-GCM-SHA384 TLSv1.2 Kx=ECDH     Au=ECDSA Enc=AESGCM(256) Mac=AEAD
ECDHE-RSA-AES256-GCM-SHA384 TLSv1.2 Kx=ECDH     Au=RSA  Enc=AESGCM(256) Mac=AEAD
DHE-DSS-AES256-GCM-SHA384 TLSv1.2 Kx=DH       Au=DSS  Enc=AESGCM(256) Mac=AEAD
DHE-RSA-AES256-GCM-SHA384 TLSv1.2 Kx=DH       Au=RSA  Enc=AESGCM(256) Mac=AEAD
ECDHE-ECDSA-CHACHA20-POLY1305 TLSv1.2 Kx=ECDH     Au=ECDSA Enc=CHACHA20/POLY1305(256) Mac=AEAD
ECDHE-RSA-CHACHA20-POLY1305 TLSv1.2 Kx=ECDH     Au=RSA  Enc=CHACHA20/POLY1305(256) Mac=AEAD
(...)
PSK-CAMELLIA128-SHA256  TLSv1 Kx=PSK      Au=PSK  Enc=Camellia(128) Mac=SHA256
DHE-RSA-SEED-SHA        SSLv3 Kx=DH       Au=RSA  Enc=SEED(128) Mac=SHA1
DHE-DSS-SEED-SHA        SSLv3 Kx=DH       Au=DSS  Enc=SEED(128) Mac=SHA1
ADH-SEED-SHA            SSLv3 Kx=DH       Au=None Enc=SEED(128) Mac=SHA1
SEED-SHA                SSLv3 Kx=RSA      Au=RSA  Enc=SEED(128) Mac=SHA1
```

!!! note
    Skup algoritama za šifriranje čije korištenje se preporuča kod rada sa TLS/SSL certifikatima mijenja se iz godine u godinu kako se pronalaze sigurnosni propusti u njima i kako procesna moć računala raste pa je dobro kod postavljanja TLS-a/SSL-a provjeriti aktualne najbolje prakse, primjerice [one koje navodi Qualys SLL Labs](https://www.ssllabs.com/projects/best-practices/index.html), autor [SSL Server Testa](https://www.ssllabs.com/ssltest/index.html) i [SSL Client Testa](https://www.ssllabs.com/ssltest/viewMyClient.html).

Naredba `SSLProtocol all -SSLv3` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#sslprotocol)) definira dozvoljene verzije TLS-a i SSL-a. Vrijednost `all -SSLv3` uključuje TLSv1, TLSv1.1, TLSv1.2, TLSv1.3, a isključuje SSLv3.

Naredbe `SSLCertificateFile "/usr/local/apache2/conf/server.crt"` i `SSLCertificateKeyFile "/usr/local/apache2/conf/server.key"` navode certifikat i pripadni tajni ključ koji se koriste (respektivno).

Osim konfiguracijskih naredbi koje počinju nizom slova `SSL`, unutar bloka `<VirtualHost></VirtualHost>` vidimo i već poznate naredbe `DocumentRoot`, `ServerName` i `ServerAdmin`:

``` apacheconf
<VirtualHost _default_:443>

#   General setup for the virtual host
DocumentRoot "/usr/local/apache2/htdocs"
ServerName www.example.com:443
ServerAdmin you@example.com
#...
</VirtualHost>
```

Detaljnije se načinom rada virtualnih domaćina (naredba `<VirtualHost>`, [dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#virtualhost)) bavimo drugdje. Postavimo `DocumentRoot "/var/www/html"` i `ServerName apache-primjer.rm.miletic.net:443`, a `ServerAdmin` na vlastitu e-mail adresu. Dodajmo u `Dockerfile` kopiranje izmijenjene datoteke na odgovarajuće mjesto:

``` dockerfile hl_lines="6"
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
COPY ./www /var/www
COPY server.crt /usr/local/apache2/conf
COPY server.key /usr/local/apache2/conf
COPY ./my-httpd-ssl.conf /usr/local/apache2/conf/extra/httpd-ssl.conf
```

Izgradimo sliku i pokrenimo Docker kontejner:

``` shell
$ docker build -t "my-httpd:2.4-5" .
Sending build context to Docker daemon  72.19kB
Step 1/6 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/6 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> Using cache
---> 765440898954
Step 3/6 : COPY ./www /var/www
---> Using cache
---> a0c0158083fb
Step 4/6 : COPY server.crt /usr/local/apache2/conf
---> baafa15fa976
Step 5/6 : COPY server.key /usr/local/apache2/conf
---> eebe8db06676
Step 6/6 : COPY ./my-httpd-ssl.conf /usr/local/apache2/conf/extra/httpd-ssl.conf
---> 0514261bb6fe
Successfully built 0514261bb6fe
Successfully tagged my-httpd:2.4-5
$ docker run my-httpd:2.4-5
[Sun May 10 18:57:41.346009 2020] [ssl:warn] [pid 1:tid 139678405715072] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 18:57:41.346345 2020] [ssl:warn] [pid 1:tid 139678405715072] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 18:57:41.349298 2020] [ssl:warn] [pid 1:tid 139678405715072] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 18:57:41.349305 2020] [ssl:warn] [pid 1:tid 139678405715072] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 18:57:41.350403 2020] [mpm_event:notice] [pid 1:tid 139678405715072] AH00489: Apache/2.4.43 (Unix) OpenSSL/1.1.1d configured -- resuming normal operations
[Sun May 10 18:57:41.350429 2020] [core:notice] [pid 1:tid 139678405715072] AH00094: Command line: 'httpd -D FOREGROUND'
```

Uvjerimo se da sada radi ispravno:

``` shell
$ curl -k https://172.17.0.2/
<html><body><h1>Radi!</h1></body></html>
```

!!! hint
    U praksi je ponekad moguće automatizirati čitav ovdje opisani postupak. Naime, Apache [od verzije 2.4.30 nadalje](https://blog.des.no/2021/10/lets-encrypt-apache-mod-md/) sadrži [mod_md](https://httpd.apache.org/docs/2.4/mod/mod_md.html) koji dohvaća certifikate s [besplatnog i otvorenog](https://letsencrypt.org/about/) [autoriteta certifikata Let's Encrypt](https://letsencrypt.org/) za domene navedene pod konfiguracijskom naredbom `MDomain`. Nažalost, taj postupak ovdje ne možemo koristiti jer Let's Encrypt zahtijeva da imamo registrirane domene na internetu kako bi mogao izdati certifikat za njih. Pored toga, čak i kad možemo automatizirati postavljanje certifikata, dobro je znati kako čitav postupak izgleda kako bismo se mogli snaći u situacijama kad neki dio tog automatiziranog postupka zakaže.

## Server Name Indication

Ranije prikazani način dohvaćanja domaćina kojeg želimo navođenjem njegovog imena u zaglavlju `Host` neće raditi kad se koristi HTTPS:

``` shell
$ curl --header "Host: www.math.uniri.hr" https://www.biotech.uniri.hr/hr/
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>400 Bad Request</title>
</head><body>
<h1>Bad Request</h1>
<p>Your browser sent a request that this server could not understand.<br />
</p>
<hr>
<address>Apache/2.4.10 (Debian) Server at www.math.uniri.hr Port 443</address>
</body></html>
```

Razlog je da više HTTPS poslužitelja na jednoj IP adresi zahtijeva [Server Name Indication](https://en.wikipedia.org/wiki/Server_Name_Indication) (kraće SNI, dokumentiran u [RFC-u 6066: Transport Layer Security (TLS) Extensions: Extension Definitions u odjeljku 3. Server Name Indication](https://datatracker.ietf.org/doc/html/rfc6066#page-6)). Za tu svrhu cURL ima parametre `--resolve` i `--connect-to`, od kojih prvi koristimo u nastavku.

## Konfiguracija virtualnih domaćina za HTTPS

Želimo li virtualne domaćine korisiti u kombinaciji s HTTPS-om, dodat ćemo u `my-httpd-vhosts.conf` blokove za virtualne domaćine na HTTPS vratima 443:

``` apacheconf
<VirtualHost *:443>
    DocumentRoot "/var/www/html"
    SSLCertificateFile "/usr/local/apache2/conf/server.crt"
    SSLCertificateKeyFile "/usr/local/apache2/conf/server.key"
</VirtualHost>

<VirtualHost *:443>
    DocumentRoot "/var/www/epimetej/html"
    <Directory "/var/www/epimetej/html">
        Require all granted
    </Directory>
    ServerName epimetej.rm.miletic.net
    SSLCertificateFile "/usr/local/apache2/conf/epimetej.crt"
    SSLCertificateKeyFile "/usr/local/apache2/conf/epimetej.key"
</VirtualHost>

<VirtualHost *:443>
    DocumentRoot "/var/www/prometej/html"
    <Directory "/var/www/prometej/html">
        Require all granted
    </Directory>
    ServerName prometej.rm.miletic.net
    SSLCertificateFile "/usr/local/apache2/conf/prometej.crt"
    SSLCertificateKeyFile "/usr/local/apache2/conf/prometej.key"
</VirtualHost>
```

Vidimo da opet imamo zadani blok i onda po jedan blok za svako web sjedište. Ovi blokovi imaju dodatne konfiguracijske naredbe `SSLCertificateFile` i `SSLCertificateKeyFile`. Certifikate i privatne ključeve ćemo kao i ranije generirati OpenSSL-om i paziti da pod `Common Name` navedemo imena domena:

``` shell
$ openssl req -x509 -nodes -days 30 -newkey rsa:4096 -keyout epimetej.key -out epimetej.crt
(...)
Common Name (e.g. server FQDN or YOUR name) []:epimetej.rm.miletic.net
(...)

$ openssl req -x509 -nodes -days 30 -newkey rsa:4096 -keyout prometej.key -out prometej.crt
(...)
Common Name (e.g. server FQDN or YOUR name) []:prometej.rm.miletic.net
(...)
```

`Dockerfile` ćemo dodati 4 nove naredbe `COPY` i sada je oblika:

``` dockerfile hl_lines="8-11"
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
COPY ./www /var/www
COPY server.crt /usr/local/apache2/conf
COPY server.key /usr/local/apache2/conf
COPY ./my-httpd-ssl.conf /usr/local/apache2/conf/extra/httpd-ssl.conf
COPY ./my-httpd-vhosts.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf
COPY epimetej.crt /usr/local/apache2/conf
COPY epimetej.key /usr/local/apache2/conf
COPY prometej.crt /usr/local/apache2/conf
COPY prometej.key /usr/local/apache2/conf
```

Izgradit ćemo sliku i pokrenut Docker kontejner:

``` shell
$ docker build -t "my-httpd:2.4-6" .
Sending build context to Docker daemon   72.7kB
Step 1/11 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/11 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> Using cache
---> e475515ada38
Step 3/11 : COPY ./www /var/www
---> Using cache
---> c764360e0255
Step 4/11 : COPY server.crt /usr/local/apache2/conf
---> Using cache
---> fc1bfa23df69
Step 5/11 : COPY server.key /usr/local/apache2/conf
---> Using cache
---> b1b86d1c6b3d
Step 6/11 : COPY ./my-httpd-ssl.conf /usr/local/apache2/conf/extra/httpd-ssl.conf
---> Using cache
---> 9c09539265cd
Step 7/11 : COPY ./my-httpd-vhosts.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf
---> 34ae4d97114f
Step 8/11 : COPY epimetej.crt /usr/local/apache2/conf
---> 96fd7f0e9a88
Step 9/11 : COPY epimetej.key /usr/local/apache2/conf
---> 5a63ce1a7686
Step 10/11 : COPY prometej.crt /usr/local/apache2/conf
---> eb5005e4dbfe
Step 11/11 : COPY prometej.key /usr/local/apache2/conf
---> 3b4c03a5e682
Successfully built 3b4c03a5e682

$ docker run my-httpd:2.4-6
[Sun May 10 22:49:09.438791 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.439105 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 22:49:09.439521 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: prometej.rm.miletic.net:80:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.439933 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: epimetej.rm.miletic.net:80:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.443353 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: www.example.com:443:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.443360 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01909: www.example.com:443:0 server certificate does NOT include an ID which matches the server name
[Sun May 10 22:49:09.443707 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: prometej.rm.miletic.net:80:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.444046 2020] [ssl:warn] [pid 1:tid 140677310489728] AH01906: epimetej.rm.miletic.net:80:0 server certificate is a CA certificate (BasicConstraints: CA == TRUE !?)
[Sun May 10 22:49:09.445125 2020] [mpm_event:notice] [pid 1:tid 140677310489728] AH00489: Apache/2.4.43 (Unix) OpenSSL/1.1.1d configured -- resuming normal operations
[Sun May 10 22:49:09.445153 2020] [core:notice] [pid 1:tid 140677310489728] AH00094: Command line: 'httpd -D FOREGROUND'
```

Sad možemo cURL-om isprobati da virtualni domaćini rade ispravno kad se koristi HTTPS (zbog SNI-ja nije dovoljno koristiti `--header Host: ...` pa koristimo `--resolve`):

``` shell
$ curl -k --resolve prometej.rm.miletic.net:443:172.17.0.2 https://prometej.rm.miletic.net/
<html><body><h1>Prometej</h1></body></html>
$ curl -k --resolve epimetej.rm.miletic.net:443:172.17.0.2 https://epimetej.rm.miletic.net/
<html><body><h1>Epimetej</h1></body></html>
```

Mi ovdje cURL-u parametrom `--resolve` kažemo da preskoči DNS pretragu i zatraži URL-ove `https://prometej.rm.miletic.net/` i `https://epimetej.rm.miletic.net/` na adresi 172.17.0.2 i vratima 443. Uvjerimo se da ostali zahtjevi završavaju na zadanom virtualnom domaćinu:

``` shell
$ curl -k --resolve atlas.rm.miletic.net:443:172.17.0.2 https://atlas.rm.miletic.net/
<html><body><h1>Radi!</h1></body></html>
$ curl -k https://172.17.0.2/
<html><body><h1>Radi!</h1></body></html>
```
