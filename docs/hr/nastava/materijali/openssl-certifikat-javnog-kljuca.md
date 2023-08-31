---
author: Domagoj Margan, Vedran Miletić
---

# Certifikat javnog ključa, certifikacijska tijela i sigurni poslužitelj

!!! note
    Skup algoritama za šifriranje čije korištenje se preporuča kod rada sa TLS/SSL certifikatima mijenja se iz godine u godinu kako se pronalaze sigurnosni propusti u njima i kako procesna moć računala raste pa je dobro kod postavljanja TLS-a/SSL-a provjeriti aktualne najbolje prakse, primjerice [one koje navodi Qualys SLL Labs](https://www.ssllabs.com/projects/best-practices/index.html), autor [SSL Server Testa](https://www.ssllabs.com/ssltest/index.html) i [SSL Client Testa](https://www.ssllabs.com/ssltest/viewMyClient.html).

## Certifikat javnog ključa

Certifikat javnog ključa je elektronički dokument kojim se dokazuje posjedovanje javnog ključa. Certifikat sadrži informacije o ključu, subjektu koji je vlasnik ključa i digitalni potpis tijela koje je provjerilo sadržaj certifikata (tzv. izdavatelj certifikata, engl. *certificate issuer*). Ako je potpis valjan i ako softver koji radi s certifikatima vjeruje izdavatelju, tada je moguće ostvariti sigurnu komunikaciju sa subjektom koji je vlasnik ključa. Popisi izdavatelja kojima se vjeruje variraju među softverima; primjerice, u službenim razvojnim dokumentima web preglednika moguće je pronaći popise izdavatelja certifikata kojima vjeruju [Mozilla Firefox](https://wiki.mozilla.org/CA/Included_Certificates) i [Google Chrome](https://www.chromium.org/Home/chromium-security/root-ca-policy). Također, popis izdavatelja kojima se vjeruje varira i kroz vrijeme pa, primjerice, izdavatelj besplatnih TLS/SSL certifikata [Let's Encrypt](https://letsencrypt.org/) od kraja srpnja 2018. godine ima [povjerenje svih važnijih programa, uključujući Microsoft, Google, Apple, Mozillu, Oracle i Blackberry](https://letsencrypt.org/2018/08/06/trusted-by-all-major-root-programs.html).

Certifikati javnog ključa standardizirani su od strane [Sektora za standardizaciju](https://www.itu.int/en/ITU-T/) [Međunarodne telekomunikacijske unije](https://www.itu.int/) (ITU-T) 1988. godine pod nazivom [X.509](https://www.itu.int/rec/T-REC-X.509). Standard X.509 je također opisan [u RFC-u 5280 pod naslovom Internet X.509 Public Key Infrastructure Certificate and Certificate Revocation List (CRL) Profile](https://datatracker.ietf.org/doc/html/rfc5280). Više informacija o standardu može se pronaći [na Wikipedijinoj stranici X.509](https://en.wikipedia.org/wiki/X.509).

### Stvaranje samopotpisanih certifikata

Generiranje samopotpisanog certifikata (engl. *self-signed certificate*) zbog složenosti zahtjeva uporabu više opcija i parametara.

Idući primjer pokazuje generiranje datoteke privatnog ključa `mojkljuc.pem` i datoteke `mojcertifikat.pem` u kojoj su sadržani javni ključ i certifikat javnog ključa. Generiranje zahtjeva za potpisivanjem i potpisivanje zahtjeva vršimo opcijom `req`, a parametar `-x509` znači da se radi o certifikatima koji slijede ITU-T standard X.509. Moguće je odrediti broj dana valjanosti certifikata (u ovom slučaju jedna godina, znači 365), te hoće li ključ biti šifriran (opcija `-nodes` isključuje šifriranje). Pri stvaranju certifikata potrebno je odgovoriti na nekoliko pitanja kojima se u certifikat unose informacije o vlasniku.

``` shell
$ openssl req -x509 -nodes -days 365 -newkey rsa:4096 -keyout mojkljuc.pem -out mojcertifikat.pem
Generating a RSA private key
.............................................++++
...................................................++++
writing new private key to 'mojcertifikat.pem'
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

Mi ćemo se ograničiti samo na osnovno korištenje ove naredbe, a za dodatne opcije proučite `req(1ssl)`.

### Pregled i provjera valjanosti certifikata

Stvoreni certifikat možemo pregledati opcijom `x509`:

``` shell
$ openssl x509 -in mojcertifikat.pem -noout -text
Certificate:
 Data:
     Version: 3 (0x2)
     Serial Number:
         6f:63:28:30:2c:2b:10:f3:ab:5c:91:50:27:76:97:7c:72:fc:a0:9c
     Signature Algorithm: sha256WithRSAEncryption
     Issuer: C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
     Validity
         Not Before: Jun  8 17:13:57 2019 GMT
         Not After : Jun  7 17:13:57 2020 GMT
     Subject: C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
     Subject Public Key Info:
         Public Key Algorithm: rsaEncryption
             RSA Public-Key: (4096 bit)
             Modulus:
                 00:aa:b7:ca:a7:9d:23:dc:fc:1d:c8:6c:00:64:d7:
                 27:f1:91:c5:2a:e9:13:0d:ce:b3:3c:02:68:0e:70:
                 9c:33:fa:3c:f0:dc:3d:e4:cc:1f:73:f3:a1:46:d9:
                 (...)
             Exponent: 65537 (0x10001)
     X509v3 extensions:
         X509v3 Subject Key Identifier:
             39:E7:75:1C:26:F9:E7:0B:34:8A:A8:9D:AF:A9:43:B4:78:19:98:2D
         X509v3 Authority Key Identifier:
             keyid:39:E7:75:1C:26:F9:E7:0B:34:8A:A8:9D:AF:A9:43:B4:78:19:98:2D

         X509v3 Basic Constraints: critical
             CA:TRUE
 Signature Algorithm: sha256WithRSAEncryption
      40:dc:7e:ad:3f:1d:b6:22:08:2c:ac:71:a7:d0:46:32:66:9c:
      ce:24:c8:83:50:c2:8b:30:69:a4:40:b7:17:ce:8f:ab:75:ee:
      71:68:d2:61:3d:55:07:6c:e8:52:c6:88:f4:09:86:3a:98:de:
      (...)
```

Za provjeru valjanosti certifikata koristi se opcija `verify`. Ukoliko lokalna instalacija OpenSSL-a prepozna certifikat i on ima potpis izdavatelja kojem se vjeruje, vraća se povratna poruka `OK`.

``` shell
$ openssl verify valjani-certifikat.pem
OK
```

Ukoliko se pak pronađe problem sa certifikatom, javlja se obavijest o tome uz kratak opis problema, primjerice:

``` shell
$ openssl verify sampotpisani-certifikat.pem
error 18 at 0 depth lookup:self signed certificate
```

Ovdje certifikat nije valjan jer je samopotpisan. Ukoliko se ne napravi iznimka, OpenSSL neće sampotpisani certifikat označiti kao valjan.

Certifikati su najčešće predodređeni za konkretni vremenski period, te OpenSSL javlja grešku ukoliko je taj period istekao.

``` shell
$ openssl verify istekli-certifikat.pem
error 10 at 0 depth lookup:certificate has expired
```

Dio poruke `at 0 depth` znači da je pogreška pronađena na samom certifikatu koji se provjerava, a ne negdje dalje u lancu potpisa. Primjerice, može se dogoditi i da istekne certifikat nekog od autoriteta certifikata koji je potpisao certifikat koji se provjerava, certifikat autoriteta certifikata koji je potpisao certifikat autoriteta certifikata koji je potpisao certifikat koji se provjerava itd.

### Pretvorba formata certifikata

#### Standard PKCS #8

[PKCS #8: Private-Key Information Syntax Standard](https://en.wikipedia.org/wiki/PKCS_8) je standardna sintaksa za pohranu privatnih ključeva. Posljednja verzija 1.2 standardizirana je u [RFC-u 5208 pod naslovom Public-Key Cryptography Standards (PKCS) #8: Private-Key Information Syntax Specification Version 1.2](https://datatracker.ietf.org/doc/html/rfc5208). Proširenje standarda predloženo je u [RFC-u 5958](https://datatracker.ietf.org/doc/html/rfc5958).

OpenSSL u zadanim postavkama već zapisuje privatne ključeve u formatu PKCS #8, a dodatno je moguće kod zapisivanja ključeve šifrirati.

Ako nismo odabrali šifrirati privatni ključ kod stvaranja, pretvorbu u šifrirani oblik vršimo naredbom:

``` shell
$ openssl pkcs8 -topk8 -in mojkljuc.pem -out mojkljuc.pem.enc
Enter Encryption Password:
Verifying - Enter Encryption Password:
$ cat mojkljuc.pem.enc
-----BEGIN ENCRYPTED PRIVATE KEY-----
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
-----END ENCRYPTED PRIVATE KEY-----
```

#### Formati PEM i DER

Ključevi koje smo stvorili zapisani su u zadanom formatu [Privacy-Enhanced Mail](https://en.wikipedia.org/wiki/Privacy-Enhanced_Mail) (PEM), standardiziranom u [RFC-u 7468 pod naslovom Textual Encodings of PKIX, PKCS, and CMS Structures](https://datatracker.ietf.org/doc/html/rfc7468). Ključeve je moguće zapisati i u formatu [Distinguished Encoding Rules](https://en.wikipedia.org/wiki/Distinguished_Encoding_Rules) (DER), koji koristi [ASN.1](https://en.wikipedia.org/wiki/ASN.1) za zapis struktura podataka kompatibilan s formatom [PKCS #1](https://en.wikipedia.org/wiki/PKCS_1) RSAPrivateKey za tajni ključ i SubjectPublicKeyInfo za certifikat. ([Public Key Cryptography Standards](https://en.wikipedia.org/wiki/PKCS) (PKCS) je skup standarada za kriptografiju javnog ključa koje je objavila kompanija [RSA Security](https://en.wikipedia.org/wiki/RSA_Security).)

Pretvorbu ključeva iz formata PEM u format DER vršimo opcijom `rsa` na način:

``` shell
$ openssl rsa -inform PEM -outform DER -in mojkljuc.pem -out mojkljuc.der
writing RSA key
```

Pretvorbu certifikata iz formata PEM u format DER vršimo opcijom `x509` na način:

``` shell
$ openssl x509 -inform PEM -outform DER -text -in mojcertifikat.pem -out mojcertifikat.der
```

Za razliku od PEM-a koji je kodiran u formatu Base64, DER je binarni format pa dobivene zapise ključa i certifikata ne ispisujemo na ekran.

#### Standard PKCS #7

[PKCS #7: Cryptographic Message Syntax](https://en.wikipedia.org/wiki/PKCS_7) je standardna sintaksa za pohranu potpisanih i/ili šifriranih podataka. Posljednja verzija 1.5 standardizirana je u [RFC-u 2315 pod naslovom PKCS #7: Cryptographic Message Syntax Version 1.5](https://datatracker.ietf.org/doc/html/rfc2315), a proširenja standarda su dostupna u RFC-ima [2630](https://datatracker.ietf.org/doc/html/rfc2630), [3369](https://datatracker.ietf.org/doc/html/rfc3369), [3852](https://datatracker.ietf.org/doc/html/rfc3852) i [5652](https://datatracker.ietf.org/doc/html/rfc5652).

Pretvorbu certifikata iz formata PEM u format PKCS #7 vršimo opcijom `crl2pkcs7` na način:

``` shell
$ openssl crl2pkcs7 -nocrl -certfile mojcertifikat.pem -out mojcertifikat.pem.p7b
```

Dobiveni certifikat je kodiran u formatu Base64 pa ga možemo ispisati:

``` shell
$ cat mojcertifikat.pem.p7b
-----BEGIN PKCS7-----
MIIFnAYJKoZIhvcNAQcCoIIFjTCCBYkCAQExADALBgkqhkiG9w0BBwGgggVvMIIF
azCCA1OgAwIBAgIUW6yMgcfWNMkf+ZggBGYXLxcai+wwDQYJKoZIhvcNAQELBQAw
RTELMAkGA1UEBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoMGElu
dGVybmV0IFdpZGdpdHMgUHR5IEx0ZDAeFw0yMTAzMjUwOTUxMzFaFw0yMjAzMjUw
OTUxMzFaMEUxCzAJBgNVBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYD
VQQKDBhJbnRlcm5ldCBXaWRnaXRzIFB0eSBMdGQwggIiMA0GCSqGSIb3DQEBAQUA
A4ICDwAwggIKAoICAQDFBcX5wQafk2nfTQRz0ZIP5pi6tqFsKzRsn7UBCavClYcm
somVwZOxoPciEmoKnM8sUeaSPffZV6JCxg772ctKw40YKGhwqiwYhgrO3dAgbTdg
8BC5HkzlKWNldEzhqa1qkQm2J6fln/okEZuXJ1ygrDB7nJeL+VTx2AIdBssyWqoH
drY320dlQBWZjMLn2ods5lUOSjRm4XPHFqgAvQCvyUNOsFy1RBs4o4DlB+ckzTT0
PaYzSfOIK1lScVhgAU+EbH/vhT6zFzZIRcDiZKWQXYM6q5sGejnNXbREpR6HsVeO
z6TN55eNByCIX3sZoGbYdeDWJXOe4LDD9mZfCLs1B5U7FR0mwme1MemkGigs1V+0
KBqu6IuT7e1AAGp0VYr6gBysoFsWDa9iSOSWbpcZBBZiiPx5opJUVMwt6imacywU
Nk40ueqfQFAFEpLZNic5XiLIHOdYqKfEbeJUy2oPHPPWoBjpnxbE96EGZmcbClO3
O9n48G1gj94wdK8wE3pk1nWpcDu/8zu7h7HMkb7uPnEW9zvfe9/vKekZbvw1/Ykz
uV5I5ci4QL6KciwpT7mwUesSV97I5YHDdFmnVLIGsl2FMyVpxSl+Y1O6icbaN91c
AdfXvCIT6ezc07ymZAfSm1073uGEmptyUCyQh+nys/7Id9I7zzOg12hrsfX1bQID
AQABo1MwUTAdBgNVHQ4EFgQUp/K4o5XCVrYnu5hW19/dr54a2BQwHwYDVR0jBBgw
FoAUp/K4o5XCVrYnu5hW19/dr54a2BQwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG
9w0BAQsFAAOCAgEAd3Oe/nSbJvL3+KFZk8M3DEgkCE3bug4SPmDkMRrRMI0inlHV
eeORDCNte90aBFlTIjchRLqWVvZbqrn8KyWE3ZdiNDTGWZyOIZnUCzmgsTMYwLJ5
dfDsUcY6nHzgGiJ/QBX1lQFhmxTmliw30zdAsLaRJoMsTdc3EkavbNr7hkytNhwP
7SgzhSTZRCWvsmG50koUjuXMK/JY2E8A7F52Ib7SdNl43JfS8pjE3T3GCzmU/6nc
a8QU5msHOEwEdDDvLa7pQjWfd2DJV5yFE9J1t0oFRzGQJtAYuKh3BI+/kDPd7mR5
33Nz2Dw64edowMDkNP+aqVHGl3a9/7/uLWyoyEUkU1YSB+oaebVCzEpP2zYdY1mG
CUnpnVdx8sIKKyN03wRMOCYCAJysX1NoGmmKFKFcGoLcqhaB/5fXXYnZ373XvX7B
Lhq3/iOrLrrEO8YNSxBvLTSYTgBjdZQIVhY8ehDssqMonXmqd4Xq9bxem/UhRxcA
3iciahvM+lAMNYD51ucefckg+b+iwWFho7ZNze38KL2lJ7eLN+s9GjyzvEVdwtuf
zpjkrTf48Xg0LEIOQxoVUsnAB17xMAvlCqohocHrenhDTKgTYl8Kyx8Ziy/D9Uw2
Lxnd4Be1X6IHPgMhxe4s00koDQgekZCqX3CeL4Zy4gOyiy5nVnXtRC1DcnmhADEA
-----END PKCS7-----
```

Opcijom `pkcs7` možemo ispisati podatke o certifikatu zapisanom u formatu PKCS #7 na način:

``` shell
$ openssl pkcs7 -in mojcertifikat.pem.p7b -noout -print_certs
subject=C = AU, ST = Some-State, O = Internet Widgits Pty Ltd

issuer=C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
```

#### PKCS #12

Format PKCS#12, poznat i pod nazivom PFX, zapisuje u istu datoteku i privatni ključ i certifikat u binarnom obliku. Format je značajan zato što ga [koristi Microsoftov web poslužitelj Internet Information Services](https://www.ssls.com/knowledgebase/how-to-install-an-ssl-certificate-on-microsoft-iis7/) (kraće IIS, [službena web stranica](https://www.iis.net/)).

Pretvorbu iz formata PEM u format PKCS#12 vršimo opcijom `pkcs12` na način:

``` shell
$ openssl pkcs12 -export -out kljuc-certifikat.pfx -inkey mojkljuc.pem -in mojcertifikat.pem
Enter Export Password:
Verifying - Enter Export Password:
```

Ako imamo i certifikat autoriteta certifikata koji želimo uključiti u lanac certifikata, možemo ga dodati parametrom `-certfile` na način:

``` shell
$ openssl pkcs12 -export -out kljuc-certifikat.pfx -inkey mojkljuc.pem -in mojcertifikat.pem -certfile DigiCertCA.crt
```

## Certifikacijsko tijelo

!!! warning
    U nastavku navodimo postupak stvaranja certifikata certifikacijskog tijela u pojednostavljenoj varijanti kako bi razumijeli koji certifikati se gdje koriste, ali ne i kako se štite od krađe. U praksi bi bilo potrebno provesti dodatne mehanizme zaštite kako bi bili sigurni da su tajni ključevi zaista zaštićeni od krađe.

### Stvaranje korijenskog ključa

Za korijenski ključ certifikacijskog tijela biramo 4096-bitni RSA:

``` shell
$ openssl genrsa -out korijenskikljuc.pem 4096
```

### Samopotpisivanje korijenskog certifikata

Potpisujemo sami sebi korijenski certifikat:

``` shell
$ openssl req -x509 -new -nodes -key korijenskikljuc.pem -sha256 -days 3650 -out korijenskicertifikat.pem
```

Sada ovim certifikatom možemo potpisivati tuđe zahtjeve za potpisom certifikata. Ako bismo sada htjeli da naš certifikat i certifikati koje potpišemo bude valjani, organizacije kao Mozilla i Google bi ga morale dodati u popis izdavatelja kojima se vjeruje.

### Potpisivanje certifikata putem zahtjeva za potpis certifikata

Kod primitka zahtjeva za potpisom certifikata, prvo provjeravamo zahtjev korištenjem opcije `req`:

``` shell
$ openssl req -in zahtjev.pem -noout -text
```

Tuđe zahtjeve za potpisom certifikata možemo potpisati korištenjem opcije `ca`:

``` shell
$ openssl ca -out potpisanicertifikat.pem -in zahtjev.pem
```

Potpuno ekvivalentno, za potpisivanje možemo koristiti korištenjem opcije `x509`:

``` shell
$ openssl x509 -req -in zahtjev.pem -CA korijenskicertifikat.pem -CAkey korijenskikljuc.pem -CAcreateserial -out potpisanicertifikat.pem
```

### Provjera potpisa

Potpisani certifikat pregledati korištenjem opcije `x509`:

``` shell
$ openssl x509 -in potpisanicertifikat.pem -noout -text
```

Valjanost certifikata sada možemo provjeriti korištenjem opcije `verify`:

``` shell
$ openssl verify -CAfile korijenskicertifikat.pem potpisanicertifikat.pem
```

## Sigurni poslužitelj

Komunikaciju na internetu danas nemoguće je zamisliti bez HTTPS-a i TLS/SSL certifikata koji ga omogućuju. Danas se HTTPS uključuje rutinski (u [pojedinim poslužiteljima, kao što je Caddy, čak i automatski](https://caddyserver.com/docs/automatic-https)), a [TLS/SSL šifriranje radi brzo](https://istlsfastyet.com/) i pouzdano. Međutim, nije uvijek bilo tako.

U travnju 2014. godine otkriven je ozbiljan propust u OpenSSL-u popularno nazvan [Heartbleed](https://heartbleed.com/), a službeno [CVE-2014-0160](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-0160). Zahvaljujući Hearbleedu bilo je moguće ukrasti privatne ključeve pridružene certifikatima. Iste je godine otkriven još jedan ozbiljan propust, ovaj put na razini protokola. Propust popularno nazvan [POODLE](https://access.redhat.com/articles/1232123), a službeno [CVE-2014-3566](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-3566), pogađa SSL verziju 3.0 i omogućuje dešifriranje šifriranog sadržaja od treće strane. Četiri godine prije tih propusta, Marco Peereboom, direktor u tvrtci [Company Zero](https://www.companyzero.com/), dao je zanimljiv osvrt na stanje koda OpenSSL-a. U svom eseju pod naslovom [OpenSSL is written by monkeys](https://web.archive.org/web/20100906031647/https://www.peereboom.us/assl/assl/html/openssl.html) piše:

> After messing around with this code for about a month I decided to write this up for the tubes in the hope that I can save some souls. I have come to the conclusion that OpenSSL is equivalent to monkeys throwing feces at the wall. It is, bar none, the worst library I have ever worked with. I can not believe that the internet is running on such a ridiculous complex and gratuitously stupid piece of code. Since circa 1998 the whole world has been trusting their secure communications to this impenetrable morass that calls itself the "OpenSSL" project.

U narednim mjesecima OpenSSL je donio [novu sigurnosnu politiku](https://www.openssl.org/policies/secpolicy.html), a zatim je tijekom 2015. godine [doživio čišćenje koda](https://www.openssl.org/blog/blog/2015/02/11/code-reformat-finished/), [otkriveni su i popravljeni i drugi sigurnosni propusti](https://www.openssl.org/blog/blog/2015/03/19/security-updates/), [i još sigurnosnih propusta](https://www.openssl.org/blog/blog/2015/05/20/logjam-freak-upcoming-changes/), [i još čišćenja koda](https://www.openssl.org/blog/blog/2015/07/28/code-cleanup/). Krajem 2015. godine [stanje projekta i koda bilo je bitno bolje nego pred godinu dana](https://www.openssl.org/blog/blog/2015/09/01/openssl-security-a-year-in-review/).

S rastom optimizma oko najkorištenije implementacije TLS-a/SSL-a porasla je i njena primjena. Od početka 2016. godine do polovine 2019. godine [postotak web stranica koje se učitavaju korištenjem HTTPS-a se udvostručio](https://letsencrypt.org/stats/#percent-pageloads), a paralelno s time [nastavlja rasti i broj (besplatnih) TLS/SSL certifikata koje izdaje Let's Encrypt](https://letsencrypt.org/stats/#daily-issuance).

### Klijentska strana

Za ostvarivanje TLS/SSL veze s poslužiteljima i testiranje istih, koristimo opciju `s_client` i parametar `-connect`. Poslužitelj mora imati omogućeno pristupanje putem TLS-a/SSL-a.

Primjerice, za HTTP preko TLS-a/SSL-a:

``` shell
$ openssl s_client -connect example.com:443
```

Za SMTP preko TLS-a/SSL-a:

``` shell
$ openssl s_client -connect example.com:465
```

Za IMAP preko TLS-a/SSL-a:

``` shell
$ openssl s_client -connect example.com:993
```

Za POP-3 preko TLS-a/SSL-a:

``` shell
$ openssl s_client -connect example.com:995
```

Za LDAP preko TLS-a/SSL-a:

``` shell
$ openssl s_client -connect example.com:636
```

Nakon spajanja mougće je unositi naredbe. Konkretno, ukoliko smo ostvarili HTTPS vezu, moguće je unositi uobičajene HTTP naredbe (npr. GET i POST).

Za primjer se možemo spojiti na HTTPS poslužitelj na domeni `example.group.miletic.net` naredbom:

``` shell
$ openssl s_client -connect example.group.miletic.net:443
CONNECTED(00000003)
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = example.group.miletic.net
verify return:1
---
Certificate chain
 0 s:CN = example.group.miletic.net
   i:C = US, O = Let's Encrypt, CN = R3
   a:PKEY: rsaEncryption, 2048 (bit); sigalg: RSA-SHA256
   v:NotBefore: Aug  9 18:27:00 2023 GMT; NotAfter: Nov  7 18:26:59 2023 GMT
 1 s:C = US, O = Let's Encrypt, CN = R3
   i:C = US, O = Internet Security Research Group, CN = ISRG Root X1
   a:PKEY: rsaEncryption, 2048 (bit); sigalg: RSA-SHA256
   v:NotBefore: Sep  4 00:00:00 2020 GMT; NotAfter: Sep 15 16:00:00 2025 GMT
 2 s:C = US, O = Internet Security Research Group, CN = ISRG Root X1
   i:O = Digital Signature Trust Co., CN = DST Root CA X3
   a:PKEY: rsaEncryption, 4096 (bit); sigalg: RSA-SHA256
   v:NotBefore: Jan 20 19:14:03 2021 GMT; NotAfter: Sep 30 18:14:03 2024 GMT
---
Server certificate
-----BEGIN CERTIFICATE-----
MIIE/zCCA+egAwIBAgISBCZ5mZelKNKW6GBmJBs+iAZUMA0GCSqGSIb3DQEBCwUA
MDIxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1MZXQncyBFbmNyeXB0MQswCQYDVQQD
EwJSMzAeFw0yMzA4MDkxODI3MDBaFw0yMzExMDcxODI2NTlaMCQxIjAgBgNVBAMT
GWV4YW1wbGUuZ3JvdXAubWlsZXRpYy5uZXQwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQCrxxsM7cYB+Oqps88IF0+iy3w0xGYS5u/zmBd5yWXuZkwfmpJ9
M+4H+i4VYve08x/VTy6xZ6hJQr/jzJq3MEbCaPUoqWRpb0xLZCTJ3O1Gn6Qfwu9v
NtC8aSe44tYYcEAstPXuj/cNjG4Dkudd1j68u8lbKBCgWvY39eGeFSNybo5pAQmk
jKTJ19sFAZBIS5AgjDh6CmB0eRgmMI5gCxe5JKCA3z8UANMJ5zRHNWN8VNKgneFX
0csT0zwwJJeO6jQAn8xsDGr3VLxeYNxGMcIJ3tnD42MejxzFkJDo2oa+ffHDHxqG
aZsL4LIMRwjIklkrZi/6oTihLxBl9pf9FoczAgMBAAGjggIbMIICFzAOBgNVHQ8B
Af8EBAMCBaAwHQYDVR0lBBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMAwGA1UdEwEB
/wQCMAAwHQYDVR0OBBYEFGNOFYVWWqSUAsIWQqSll5o4AleXMB8GA1UdIwQYMBaA
FBQusxe3WFbLrlAJQOYfr52LFMLGMFUGCCsGAQUFBwEBBEkwRzAhBggrBgEFBQcw
AYYVaHR0cDovL3IzLm8ubGVuY3Iub3JnMCIGCCsGAQUFBzAChhZodHRwOi8vcjMu
aS5sZW5jci5vcmcvMCQGA1UdEQQdMBuCGWV4YW1wbGUuZ3JvdXAubWlsZXRpYy5u
ZXQwEwYDVR0gBAwwCjAIBgZngQwBAgEwggEEBgorBgEEAdZ5AgQCBIH1BIHyAPAA
dgC3Pvsk35xNunXyOcW6WPRsXfxCz3qfNcSeHQmBJe20mQAAAYnbxEZ7AAAEAwBH
MEUCIFFral5b+gCOTXd1Y4Rvjrpc8NKEv0GqinCR/Ky9J1BtAiEAlpHsePglrJbk
DSKdQW2gvKA4uu45xL7rl5AQQ/kNKMgAdgDoPtDaPvUGNTLnVyi8iWvJA9PL0RFr
7Otp4Xd9bQa9bgAAAYnbxEaDAAAEAwBHMEUCIG9KQZhLaBQQPvnqOAn8KUDOVurK
VptEEP69j4v++jRMAiEAtGuFVOkbIa+0nVmGDf5DBeoUyVEi01s8vkxERN+eQWYw
DQYJKoZIhvcNAQELBQADggEBAEntYka/0HRu3YVgFWseOrq/sNkjyE+kSJdCUStA
h8SsqVuMz3/hj1FZk0d8tprfGCLR23s/3Jyo/qIEOl7XFl0wTrjQTyNl1jjJVe2+
HY14QIQNhXH9TZrB7tYogAtcAyC7mkhUaMLLSEM6JO96RYCYow2vqqQiBU2jSW8n
PcUrQx8lc2jXbkWboOc8IVXMvsTlThVl5YaKZXXcFrZHWNHZt1QosvUtcsKaxNOs
WmtBuCf5MEEjSbFbwmbcUtHEfpd6mtlYsiUFrTf7L0K303SDwfS+tBUOqf8S7g4q
RbXbf3Ky1FjEiz8aodnKRTO0AXsLbBTIsK/HrL8XjyRDQ24=
-----END CERTIFICATE-----
subject=CN = example.group.miletic.net
issuer=C = US, O = Let's Encrypt, CN = R3
---
No client certificate CA names sent
Peer signing digest: SHA256
Peer signature type: RSA-PSS
Server Temp Key: X25519, 253 bits
---
SSL handshake has read 4539 bytes and written 407 bytes
Verification: OK
---
New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384
Server public key is 2048 bit
This TLS version forbids renegotiation.
Compression: NONE
Expansion: NONE
No ALPN negotiated
Early data was not sent
Verify return code: 0 (ok)
---
---
Post-Handshake New Session Ticket arrived:
SSL-Session:
    Protocol  : TLSv1.3
    Cipher    : TLS_AES_256_GCM_SHA384
    Session-ID: 6726B67429BF086E4CCCF142D43FDBB9AAD525103AB8F55223E7721D0274F5EF
    Session-ID-ctx: 
    Resumption PSK: AA9E603E57C8F56EEFFE4797A64416B2E197D5BAF52F9232626EEEA0845650AD591AD1DE3C2D5CC07944BDB57A2426CD
    PSK identity: None
    PSK identity hint: None
    SRP username: None
    TLS session ticket lifetime hint: 7200 (seconds)
    TLS session ticket:
    0000 - 1a ae af 52 1f 73 d0 63-c9 a6 bc 38 5a 25 c1 f3   ...R.s.c...8Z%..
    0010 - e3 9d 0e 6a f0 29 0a e5-f6 87 b9 8b 44 00 e0 0c   ...j.)......D...
    0020 - 36 39 e3 82 f2 73 72 cb-02 88 e8 e6 30 f2 31 6d   69...sr.....0.1m
    0030 - 72 1e f5 0c 9b 19 53 0a-3e cc 3f 4b c2 1e 20 b5   r.....S.>.?K.. .
    0040 - 47 c5 64 3c 4e 88 68 c9-46 5e 01 6a 66 ef 51 31   G.d<N.h.F^.jf.Q1
    0050 - 3f e6 63 2c cf c3 c2 8c-7f 36 ba 3d e9 0b 1b fa   ?.c,.....6.=....
    0060 - 1e bd ff ad 5d 54 ea 96-dc 5d 3d b3 75 82 68 37   ....]T...]=.u.h7
    0070 - 77 45 1e 46 98 12 c6 c9-0a 2f 7c 94 24 8a 87 94   wE.F...../|.$...
    0080 - d3 90 1f c3 88 e8 b5 8a-12 f0 9b f5 61 bb 24 46   ............a.$F
    0090 - 2c f5 dc ed 76 05 e6 9a-e3 8f db 92 0e 9d 73 c4   ,...v.........s.
    00a0 - 09 17 2d b1 c5 23 00 ba-10 56 78 bc e1 a1 8d 72   ..-..#...Vx....r
    00b0 - 0a 2a ff 1e 73 0c a9 a3-fd d5 70 b6 78 84 fe 39   .*..s.....p.x..9
    00c0 - 11 d9 07 39 ae 91 5c ad-c1 e4 54 96 64 78 40 65   ...9..\...T.dx@e
    00d0 - 99 f2 94 0d be 9c 38 0a-e3 ad e0 5c 45 2d e5 76   ......8....\E-.v
    00e0 - ae 37 52 f7 f1 8f e5 db-39 35 13 8d a9 db 9a a0   .7R.....95......

    Start Time: 1693473818
    Timeout   : 7200 (sec)
    Verify return code: 0 (ok)
    Extended master secret: no
    Max Early Data: 0
---
read R BLOCK
---
Post-Handshake New Session Ticket arrived:
SSL-Session:
    Protocol  : TLSv1.3
    Cipher    : TLS_AES_256_GCM_SHA384
    Session-ID: E3B857CB3885A36593984AB113FC661355E70959DD1A97BAFFE81FAF60BB7520
    Session-ID-ctx: 
    Resumption PSK: 7DC9FB1027C19A3D19A9FCAD0A47D4B5DCE6275FE2F88F30CE69136197AD7240E7F82A513B04C6BF6E09C68A65101D58
    PSK identity: None
    PSK identity hint: None
    SRP username: None
    TLS session ticket lifetime hint: 7200 (seconds)
    TLS session ticket:
    0000 - 1a ae af 52 1f 73 d0 63-c9 a6 bc 38 5a 25 c1 f3   ...R.s.c...8Z%..
    0010 - 4c 0c 5c dd b9 a8 5d 46-fc aa c7 c2 5b cc 9e 44   L.\...]F....[..D
    0020 - 93 6c f7 b3 c2 75 32 c7-c5 66 1f ce cc d4 03 d5   .l...u2..f......
    0030 - 37 69 e6 cd d5 69 4c 9f-38 04 3d 1b b6 1c 3f 22   7i...iL.8.=...?"
    0040 - 4f 59 ef de 65 99 a2 c3-dc 40 1a 0e ce 5a 9b e9   OY..e....@...Z..
    0050 - 21 80 27 6a c4 3e 39 8b-f0 72 5b c7 42 04 e4 ef   !.'j.>9..r[.B...
    0060 - 88 39 9e cd 04 a7 59 5a-83 68 d8 f1 ef b2 f1 a0   .9....YZ.h......
    0070 - ca e0 f2 1b 60 20 b7 c3-9e b4 29 3f 51 57 05 c2   ....` ....)?QW..
    0080 - 5c f1 8b 72 85 d2 d9 cd-1a 4e 25 1b c1 9c 42 22   \..r.....N%...B"
    0090 - 89 c8 bd 37 ea ba bb 78-00 dd cb 3b 3a aa fe 93   ...7...x...;:...
    00a0 - bf a6 f0 9c a6 e2 57 8e-4b 42 a8 2d ae fc bd e0   ......W.KB.-....
    00b0 - 21 7c f0 42 9c 55 5c 9e-14 b9 97 10 6d 89 0a 6c   !|.B.U\.....m..l
    00c0 - e6 f4 be 57 bb 6f 9a 46-e6 b2 50 b9 ef 6a 68 bd   ...W.o.F..P..jh.
    00d0 - 9e 6a e3 e3 f1 df 69 54-10 59 d0 94 ba d6 bf 21   .j....iT.Y.....!
    00e0 - 66 29 7c d4 0b f3 23 ac-e3 af e8 8f 46 89 67 72   f)|...#.....F.gr

    Start Time: 1693473818
    Timeout   : 7200 (sec)
    Verify return code: 0 (ok)
    Extended master secret: no
    Max Early Data: 0
---
read R BLOCK
```

Prekid veze možemo izvesti kombinacijom tipki ++control+c++.

Ovdje možemo primijetiti brojne informacije. Odmah na početku vidimo podatke o lancu izdavatelja certifikata (na dubini 0 `depth=0 CN = example.group.miletic.net`, na dubini 1 `depth=1 C = US, O = Let's Encrypt, CN = R3` i na dubini 2 `depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1`). Svi certifikati su uspješno prošli provjeru valjanosti(`verify=1`).

Zatim je ispisan certifikat poslužitelja (koji je u lancu na dubini 0) u formatu PEM (`-----BEGIN CERTIFICATE----- (...) -----END CERTIFICATE-----`) i podaci o subjektu (`subject=CN = example.group.miletic.net`) i izdavaču (`issuer=C = US, O = Let's Encrypt, CN = R3`).

Navedeni su algoritmi koji se koriste za potpis hashiranog sadržaja (`SHA256`, `RSA-PSS`), broj bajtova koje poslanih i primljenih kod ostvarivanja veze (`4539` i `407`) i certifikat je označen kao valjan (`Verification: OK`).

Slijede podaci o korištenoj verziji TLS-a/SSL-a (`TLSv1.3`), kombinaciji algoritama za šifriranje koja se koristi (`TLS_AES_256_GCM_SHA384`), duljini ključa poslužitelja (`2048 bit`), kompresiji, [TLS ekstenziji Application-Layer Protocol Negotiation (ALPN)](https://www.keycdn.com/support/alpn) i drugim značajkama ostvarene veze. Na ovaj način opciju `s_client` možemo koristiti pri dijagnosticiranju problema sigurnih poslužitelja jer pokazuje brojne podatke koje ne vidimo kod normalnog korištenja web preglednika ili klijenata za elektroničku poštu.

Više o opciji `s_client` i parametrima koje prima moguće je naći u pripadnoj stranici priručnika `s_client(1ssl)`.

Informacije iz certifikata moguće je ispisati korištenjem već ranije spomenute naredbe `openssl x509` i parametra `-text`:

``` shell
$ openssl s_client -connect example.group.miletic.net:443 | openssl x509 -noout -text
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = example.group.miletic.net
verify return:1
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            04:26:79:99:97:a5:28:d2:96:e8:60:66:24:1b:3e:88:06:54
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: C = US, O = Let's Encrypt, CN = R3
        Validity
            Not Before: Aug  9 18:27:00 2023 GMT
            Not After : Nov  7 18:26:59 2023 GMT
        Subject: CN = example.group.miletic.net
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:ab:c7:1b:0c:ed:c6:01:f8:ea:a9:b3:cf:08:17:
                    4f:a2:cb:7c:34:c4:66:12:e6:ef:f3:98:17:79:c9:
                    65:ee:66:4c:1f:9a:92:7d:33:ee:07:fa:2e:15:62:
                    f7:b4:f3:1f:d5:4f:2e:b1:67:a8:49:42:bf:e3:cc:
                    9a:b7:30:46:c2:68:f5:28:a9:64:69:6f:4c:4b:64:
                    24:c9:dc:ed:46:9f:a4:1f:c2:ef:6f:36:d0:bc:69:
                    27:b8:e2:d6:18:70:40:2c:b4:f5:ee:8f:f7:0d:8c:
                    6e:03:92:e7:5d:d6:3e:bc:bb:c9:5b:28:10:a0:5a:
                    f6:37:f5:e1:9e:15:23:72:6e:8e:69:01:09:a4:8c:
                    a4:c9:d7:db:05:01:90:48:4b:90:20:8c:38:7a:0a:
                    60:74:79:18:26:30:8e:60:0b:17:b9:24:a0:80:df:
                    3f:14:00:d3:09:e7:34:47:35:63:7c:54:d2:a0:9d:
                    e1:57:d1:cb:13:d3:3c:30:24:97:8e:ea:34:00:9f:
                    cc:6c:0c:6a:f7:54:bc:5e:60:dc:46:31:c2:09:de:
                    d9:c3:e3:63:1e:8f:1c:c5:90:90:e8:da:86:be:7d:
                    f1:c3:1f:1a:86:69:9b:0b:e0:b2:0c:47:08:c8:92:
                    59:2b:66:2f:fa:a1:38:a1:2f:10:65:f6:97:fd:16:
                    87:33
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Key Usage: critical
                Digital Signature, Key Encipherment
            X509v3 Extended Key Usage: 
                TLS Web Server Authentication, TLS Web Client Authentication
            X509v3 Basic Constraints: critical
                CA:FALSE
            X509v3 Subject Key Identifier: 
                63:4E:15:85:56:5A:A4:94:02:C2:16:42:A4:A5:97:9A:38:02:57:97
            X509v3 Authority Key Identifier: 
                14:2E:B3:17:B7:58:56:CB:AE:50:09:40:E6:1F:AF:9D:8B:14:C2:C6
            Authority Information Access: 
                OCSP - URI:http://r3.o.lencr.org
                CA Issuers - URI:http://r3.i.lencr.org/
            X509v3 Subject Alternative Name: 
                DNS:example.group.miletic.net
            X509v3 Certificate Policies: 
                Policy: 2.23.140.1.2.1
            CT Precertificate SCTs: 
                Signed Certificate Timestamp:
                    Version   : v1 (0x0)
                    Log ID    : B7:3E:FB:24:DF:9C:4D:BA:75:F2:39:C5:BA:58:F4:6C:
                                5D:FC:42:CF:7A:9F:35:C4:9E:1D:09:81:25:ED:B4:99
                    Timestamp : Aug  9 19:27:00.731 2023 GMT
                    Extensions: none
                    Signature : ecdsa-with-SHA256
                                30:45:02:20:51:6B:6A:5E:5B:FA:00:8E:4D:77:75:63:
                                84:6F:8E:BA:5C:F0:D2:84:BF:41:AA:8A:70:91:FC:AC:
                                BD:27:50:6D:02:21:00:96:91:EC:78:F8:25:AC:96:E4:
                                0D:22:9D:41:6D:A0:BC:A0:38:BA:EE:39:C4:BE:EB:97:
                                90:10:43:F9:0D:28:C8
                Signed Certificate Timestamp:
                    Version   : v1 (0x0)
                    Log ID    : E8:3E:D0:DA:3E:F5:06:35:32:E7:57:28:BC:89:6B:C9:
                                03:D3:CB:D1:11:6B:EC:EB:69:E1:77:7D:6D:06:BD:6E
                    Timestamp : Aug  9 19:27:00.739 2023 GMT
                    Extensions: none
                    Signature : ecdsa-with-SHA256
                                30:45:02:20:6F:4A:41:98:4B:68:14:10:3E:F9:EA:38:
                                09:FC:29:40:CE:56:EA:CA:56:9B:44:10:FE:BD:8F:8B:
                                FE:FA:34:4C:02:21:00:B4:6B:85:54:E9:1B:21:AF:B4:
                                9D:59:86:0D:FE:43:05:EA:14:C9:51:22:D3:5B:3C:BE:
                                4C:44:44:DF:9E:41:66
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        49:ed:62:46:bf:d0:74:6e:dd:85:60:15:6b:1e:3a:ba:bf:b0:
        d9:23:c8:4f:a4:48:97:42:51:2b:40:87:c4:ac:a9:5b:8c:cf:
        7f:e1:8f:51:59:93:47:7c:b6:9a:df:18:22:d1:db:7b:3f:dc:
        9c:a8:fe:a2:04:3a:5e:d7:16:5d:30:4e:b8:d0:4f:23:65:d6:
        38:c9:55:ed:be:1d:8d:78:40:84:0d:85:71:fd:4d:9a:c1:ee:
        d6:28:80:0b:5c:03:20:bb:9a:48:54:68:c2:cb:48:43:3a:24:
        ef:7a:45:80:98:a3:0d:af:aa:a4:22:05:4d:a3:49:6f:27:3d:
        c5:2b:43:1f:25:73:68:d7:6e:45:9b:a0:e7:3c:21:55:cc:be:
        c4:e5:4e:15:65:e5:86:8a:65:75:dc:16:b6:47:58:d1:d9:b7:
        54:28:b2:f5:2d:72:c2:9a:c4:d3:ac:5a:6b:41:b8:27:f9:30:
        41:23:49:b1:5b:c2:66:dc:52:d1:c4:7e:97:7a:9a:d9:58:b2:
        25:05:ad:37:fb:2f:42:b7:d3:74:83:c1:f4:be:b4:15:0e:a9:
        ff:12:ee:0e:2a:45:b5:db:7f:72:b2:d4:58:c4:8b:3f:1a:a1:
        d9:ca:45:33:b4:01:7b:0b:6c:14:c8:b0:af:c7:ac:bf:17:8f:
        24:43:43:6e
```

Parametrom `-noout` izbjegava se dodatno ispisivanje certifikata u formatu PEM.

#### Server Name Indication

[Server Name Indication](https://en.wikipedia.org/wiki/Server_Name_Indication) (SNI) je proširenje TLS-a kojim klijent navodi u procesu rukovanja ime poslužitelja na koji se povezuje, što dozvoljava da na jednoj IP adresi postoji više od jednog HTTPS poslužitelja. Kako gotovo sve novije implementacije TLS-a podržavaju SNI, njegova [uporaba je u porastu posljednjih godina](https://developer.akamai.com/blog/2017/10/20/encrypting-web-all-need-support-tls-sni-remaining-clients).

Iskoristimo parametar `-servername` da navedemo ime poslužitelja na koji se povezujemo. Za ilustraciju, usporedimo rezultat povezivanja na poslužitelj `mileticnet.github.io` bez navođenja parametra `-servername` s rezultatima izvođenja kad su navedene vrijednosti tog parametra `vedran.miletic.net`, odnosno `www.miletic.net`:

``` shell
$ openssl s_client -connect mileticnet.github.io:443
CONNECTED(00000004)
depth=2 C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root CA
verify return:1
depth=1 C = US, O = DigiCert Inc, CN = DigiCert TLS RSA SHA256 2020 CA1
verify return:1
depth=0 C = US, ST = California, L = San Francisco, O = "GitHub, Inc.", CN = *.github.io
verify return:1
(...)
```

``` shell
$ openssl s_client -connect mileticnet.github.io:443 -servername www.miletic.net
CONNECTED(00000004)
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = www.miletic.net
verify return:1
(...)
```

``` shell
$ openssl s_client -connect mileticnet.github.io:443 -servername vedran.miletic.net
CONNECTED(00000004)
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = vedran.miletic.net
verify return:1
(...)
```

Dodatno se možemo uvjeriti da se radi o različitim certifikatima promatranjem njihovog ispisa u formatu PEM.

Spojili smo se spojili na isti poslužitelj u sva tri slučaja, ali možemo uočiti razlike u vrijednosti zajedničkog imena (`CN =`) na dubini 0, što indicira da smo u svakom od povezivanja dobili različit certifikat. Uočimo kako je u posljednja dva slučaja ime domene `mileticnet.github.io` bilo iskorišteno samo za otkrivanje IP adrese na koju će se `s_client` povezati jer je odgovarajući certifikat zatražen putem vrijednosti navedene u parametru `-servername`; u prvom slučaju je ta vrijednost implicitno jednaka imenu poslužitelja na koji se povezujemo.

Postoje i situacije u kojima više poslužitelja na jednoj adresi ima zajednički certifikat. Primjerice, na jednom od [računala grupe](../../../en/blog/posts/2023-06-23-what-hardware-software-and-cloud-services-do-we-use.md#cloud-services) nalaze se web sjedišta:

- `apps.group.miletic.net`: [aplikacije i usluge](https://apps.group.miletic.net/) koje je grupa razvila i
- `staging.group.miletic.net`: probna verzija [web sjedišta grupe](../../index.md).

Povežimo se na prvi poslužitelj:

``` shell
$ openssl s_client -connect apps.group.miletic.net:443
CONNECTED(00000004)
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = afrodite.miletic.net
verify return:1
(...)
```

Uočimo da je ovo ekvivalentno dodatnom navođenju parametra `-servername` s vrijednošću `apps.group.miletic.net` jer se ta vrijednost implicitno postavlja iz imena poslužitelja koji smo naveli u parametru `-connect`.

``` shell
$ openssl s_client -connect apps.group.miletic.net:443 -servername staging.group.miletic.net
CONNECTED(00000004)
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = afrodite.miletic.net
verify return:1
(...)
```

Uočimo kako smo dobili certifikat s istom vrijednosti zajedničkog imena na dubini 0. Kako bismo se uvjerili da je navedeni certifikat valjan i za ime domene `apps.group.miletic.net` i za ime domene `staging.group.miletic.net`, iskoristimo ponovno naredbu `openssl x509` na način:

``` shell
$ openssl s_client -connect apps.group.miletic.net:443 -servername staging.group.miletic.net | openssl x509 -noout -text
depth=2 C = US, O = Internet Security Research Group, CN = ISRG Root X1
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = afrodite.miletic.net
verify return:1
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            04:74:e5:1e:0f:4f:8c:bb:7e:16:a0:f0:4a:85:8e:a8:60:14
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: C = US, O = Let's Encrypt, CN = R3
        Validity
            Not Before: Aug 22 15:54:17 2023 GMT
            Not After : Nov 20 15:54:16 2023 GMT
        Subject: CN = afrodite.miletic.net
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                RSA Public-Key: (2048 bit)
                Modulus:
                    00:c7:c3:75:04:67:15:a0:40:61:1f:d9:ac:48:a5:
                    82:29:5c:56:c1:ed:b2:8d:04:ff:e1:ee:4f:ff:bb:
                    78:51:e9:66:ba:da:5e:12:41:e9:a6:20:8f:41:d5:
                    c7:23:64:b3:d1:80:16:6a:80:f8:9d:ba:fe:62:c0:
                    e3:4d:a3:7d:2c:cf:eb:f8:3f:1d:bf:32:1e:31:2e:
                    6f:da:8c:5e:cb:7c:3f:09:54:cd:35:17:38:a8:3d:
                    98:52:b7:c4:67:f5:34:7e:4e:22:51:e1:9f:fc:c2:
                    18:29:65:e0:2f:89:c5:a7:32:93:0a:24:f7:9e:e6:
                    40:47:72:62:37:b6:62:27:47:cd:e8:e6:17:8f:3e:
                    38:fb:e2:65:d9:91:f5:e6:f9:69:83:00:0a:37:cc:
                    35:83:35:aa:45:4e:e2:2c:c0:45:fe:9e:e1:45:98:
                    c5:36:12:e8:3e:fc:f6:08:57:5e:67:72:60:69:d9:
                    7f:6b:1e:8f:64:9f:d0:b6:df:0d:42:c9:af:f1:68:
                    89:57:aa:84:39:3c:55:c3:14:fb:75:f8:e5:c5:0a:
                    5c:89:56:6f:fd:63:b7:be:e2:77:69:8a:3f:ee:eb:
                    bb:3f:84:92:36:e7:68:ea:21:44:9f:0a:66:7b:d2:
                    74:54:04:ce:75:ea:fb:1d:a3:4d:41:66:28:21:e4:
                    02:bd
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Key Usage: critical
                Digital Signature, Key Encipherment
            X509v3 Extended Key Usage: 
                TLS Web Server Authentication, TLS Web Client Authentication
            X509v3 Basic Constraints: critical
                CA:FALSE
            X509v3 Subject Key Identifier: 
                E6:2F:30:C4:68:70:E7:41:BE:43:74:2E:3A:62:C2:10:9F:BE:CE:96
            X509v3 Authority Key Identifier: 
                keyid:14:2E:B3:17:B7:58:56:CB:AE:50:09:40:E6:1F:AF:9D:8B:14:C2:C6

            Authority Information Access: 
                OCSP - URI:http://r3.o.lencr.org
                CA Issuers - URI:http://r3.i.lencr.org/

            X509v3 Subject Alternative Name: 
                DNS:afrodite.miletic.net, DNS:apps.group.miletic.net, DNS:code.group.miletic.net, DNS:group.miletic.net, DNS:lab.miletic.net, DNS:staging.group.miletic.net
            X509v3 Certificate Policies: 
                Policy: 2.23.140.1.2.1

(...)
```

Uočimo u dijelu `X509v3 Subject Alternative Name:` vrijednosti `DNS:apps.group.miletic.net` i `DNS:staging.group.miletic.net`, iz čega zaključujemo da oba poslužitelja koriste zajednički certifikat. U praksi se to nekad koristi, ali često se sreću i razdvojeni certifikati za pojedine poslužitelje.

### Poslužiteljska strana

Za testiranje stvorenih certifikata možemo koristiti opciju `s_server` s parametrima `-key` i `-cert`. Da bi pokrenuli poslužitelj koji se ponaša kao web poslužitelj, koristimo opciju `-www`, a vrata na kojima radi navodimo parametrom `-port`.

``` shell
$ openssl s_server -key mojkljuc.pem -cert mojcertifikat.pem -port 49152 -www
```

Ovako pokrenuti web poslužitelj sluša dokle god ga ne zaustavimo. Naredbom `nestat -ln` možemo se uvjeriti da je pokrenut, a testiranje možemo vršiti cURL-om ili ranije spomenutom opcijom `s_client`. Ako koristimo `curl`, prvo se možemo uvjeriti da nismo pokrenuli poslužitelj koji odgovara na HTTP zahtjeve, već isključivo HTTPS:

``` shell
$ curl http://localhost:49152/
curl: (52) Empty reply from server

$ curl https://localhost:49152/
curl: (60) SSL certificate problem: Invalid certificate chain
More details here: https://curl.se/docs/sslcerts.html

curl performs SSL certificate verification by default, using a "bundle"
 of Certificate Authority (CA) public keys (CA certs). If the default
 bundle file isn't adequate, you can specify an alternate file
 using the --cacert option.
If this HTTPS server uses a certificate signed by a CA represented in
 the bundle, the certificate verification probably failed due to a
 problem with the certificate (it might be expired, or the name might
 not match the domain name in the URL).
If you'd like to turn off curl's verification of the certificate, use
 the -k (or --insecure) option.
```

Parametrom `-k` cURL-u kažemo da prihvaćamo samopotpisani certifikat koji nije prošao provjeru treće strane. Dobivamo sadržaj stranice koju OpenSSL-ova naredba `s_server` generira:

``` shell
$ curl -k https://localhost:49152/
<HTML><BODY BGCOLOR="#ffffff">
<pre>

s_server -key mojkljuc.pem -cert mojcertifikat.pem -port 49152 -www
Secure Renegotiation IS supported
Ciphers supported in s_server binary
TLSv1.3    :TLS_AES_256_GCM_SHA384    TLSv1.3    :TLS_CHACHA20_POLY1305_SHA256
TLSv1.3    :TLS_AES_128_GCM_SHA256    TLSv1.2    :ECDHE-ECDSA-AES256-GCM-SHA384
TLSv1.2    :ECDHE-RSA-AES256-GCM-SHA384 TLSv1.2    :DHE-RSA-AES256-GCM-SHA384
(...)
Ciphers common between both SSL end points:
TLS_AES_256_GCM_SHA384     TLS_CHACHA20_POLY1305_SHA256 TLS_AES_128_GCM_SHA256
ECDHE-ECDSA-AES256-GCM-SHA384 ECDHE-RSA-AES256-GCM-SHA384 DHE-RSA-AES256-GCM-SHA384
(...)
Signature Algorithms: ECDSA+SHA256:ECDSA+SHA384:ECDSA+SHA512:Ed25519:Ed448:RSA-PSS+SHA256:RSA-PSS+SHA384:RSA-PSS+SHA512:RSA-PSS+SHA256:RSA-PSS+SHA384:RSA-PSS+SHA512:RSA+SHA256:RSA+SHA384:RSA+SHA512:ECDSA+SHA224:RSA+SHA224:DSA+SHA224:DSA+SHA256:DSA+SHA384:DSA+SHA512
Shared Signature Algorithms: ECDSA+SHA256:ECDSA+SHA384:ECDSA+SHA512:Ed25519:Ed448:RSA-PSS+SHA256:RSA-PSS+SHA384:RSA-PSS+SHA512:RSA-PSS+SHA256:RSA-PSS+SHA384:RSA-PSS+SHA512:RSA+SHA256:RSA+SHA384:RSA+SHA512:ECDSA+SHA224:RSA+SHA224
Supported Elliptic Groups: X25519:P-256:X448:P-521:P-384
Shared Elliptic groups: X25519:P-256:X448:P-521:P-384
---
No server certificate CA names sent
---
New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384
SSL-Session:
    Protocol  : TLSv1.3
    Cipher    : TLS_AES_256_GCM_SHA384
    Session-ID: 6B58EB40672EBBDACCF5CE8130A1D3C42FE7E6E25C7A1687ADD6748E40062244
    Session-ID-ctx: 01000000
    Resumption PSK: 54FB47D2656F5B3321B0595D1C4FAF927AC124AED1A49EEA0E76EAAF5A99341CEFB6645CE5B8A30265E6B2EA3F896039
    PSK identity: None
    PSK identity hint: None
    SRP username: None
    Start Time: 1560018096
    Timeout   : 7200 (sec)
    Verify return code: 0 (ok)
    Extended master secret: no
    Max Early Data: 0
---
0 items in the session cache
0 client connects (SSL_connect())
0 client renegotiates (SSL_connect())
0 client connects that finished
9 server accepts (SSL_accept())
0 server renegotiates (SSL_accept())
6 server accepts that finished
0 session cache hits
0 session cache misses
0 session cache timeouts
0 callback cache hits
0 cache full overflows (128 allowed)
---
no client certificate available
</pre></BODY></HTML>
```

Za usporedbu, povezivanje na tako pokrenuti poslužitelj opcijom `s_client` izveli bi na način:

``` shell
$ openssl s_client -connect localhost:49152
CONNECTED(00000003)
Can't use SSL_get_servername
depth=0 C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
verify error:num=18:self signed certificate
verify return:1
depth=0 C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
verify return:1
---
Certificate chain
0 s:C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
  i:C = AU, ST = Some-State, O = Internet Widgits Pty Ltd
---
Server certificate
-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIUb2MoMCwrEPOrXJFQJ3aXfHL8oJwwDQYJKoZIhvcNAQEL
BQAwRTELMAkGA1UEBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoM
GEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDAeFw0xOTA2MDgxNzEzNTdaFw0yMDA2
(...)
```

Opcija `s_server` može se ponašati kao pravi web poslužitelj i posluživati HTML datoteke iz trenutnog direktorija korištenjem parametra `-WWW` umjesto `-www`. Tada OpenSSL neće sam generirati nikakav HTML.

``` shell
$ openssl s_server -key mojkljuc.pem -cert mojcertifikat.pem -port 49152 -WWW
```

Ako u direktoriju gdje je `s_server` pokrenut imamo datoteku `index.html`, zahtjev ćemo napraviti cURL-om na način:

``` shell
$ curl -k https://localhost:49152/index.html
```

Više o opciji `s_server` moguće je naći u pripadnoj stranici priručnika `s_server(1ssl)`.
