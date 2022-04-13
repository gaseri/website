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

```
$ openssl s_client -connect example.group.miletic.net:443
CONNECTED(00000003)
depth=2 O = Digital Signature Trust Co., CN = DST Root CA X3
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = Let's Encrypt Authority X3
verify return:1
depth=0 CN = miletic.net
verify return:1
---
Certificate chain
0 s:CN = miletic.net
  i:C = US, O = Let's Encrypt, CN = Let's Encrypt Authority X3
1 s:C = US, O = Let's Encrypt, CN = Let's Encrypt Authority X3
  i:O = Digital Signature Trust Co., CN = DST Root CA X3
---
Server certificate
-----BEGIN CERTIFICATE-----
MIIGNDCCBRygAwIBAgISA97SuFP8MdJDGngVxAaK1xqTMA0GCSqGSIb3DQEBCwUA
MEoxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1MZXQncyBFbmNyeXB0MSMwIQYDVQQD
ExpMZXQncyBFbmNyeXB0IEF1dGhvcml0eSBYMzAeFw0xOTA1MjAxMDA2NTRaFw0x
OTA4MTgxMDA2NTRaMBYxFDASBgNVBAMTC21pbGV0aWMubmV0MIIBIjANBgkqhkiG
(...)
-----END CERTIFICATE-----
subject=CN = miletic.net

issuer=C = US, O = Let's Encrypt, CN = Let's Encrypt Authority X3

---
No client certificate CA names sent
Peer signing digest: SHA256
Peer signature type: RSA
Server Temp Key: ECDH, P-256, 256 bits
---
SSL handshake has read 3475 bytes and written 433 bytes
Verification: OK
---
New, TLSv1.2, Cipher is ECDHE-RSA-AES256-GCM-SHA384
Server public key is 2048 bit
Secure Renegotiation IS supported
Compression: NONE
Expansion: NONE
No ALPN negotiated
SSL-Session:
    Protocol  : TLSv1.2
    Cipher    : ECDHE-RSA-AES256-GCM-SHA384
    Session-ID: F1D513CE8C25CE7B6055A68FBB67A71D3F2A37CFCBEAEAA2AC3054A91A0E6048
    Session-ID-ctx:
    Master-Key: 9991DF56B54564C72EEAE4F08B40146BC5CDCDB06A3F7CB1EA92588D3E829EC82005DE618DD9587942FB1CDC80176D69
    PSK identity: None
    PSK identity hint: None
    SRP username: None
    TLS session ticket lifetime hint: 300 (seconds)
    TLS session ticket:
    0000 - 5c 0a a8 4c c2 06 86 83-3b b5 83 10 4e 09 47 ea   \..L....;...N.G.
    0010 - 31 ef da dd 9a f9 9e f5-35 a2 69 a4 64 dc 3d cd   1.......5.i.d.=.
    0020 - 69 4d 7f 32 b8 59 18 37-57 3f ad 29 68 5e 5f 4c   iM.2.Y.7W?.)h^_L
    0030 - b1 49 ef 4e 77 c2 b0 3a-94 5c e0 bd 60 2c d8 99   .I.Nw..:.\..`,..
    0040 - 5c d3 92 ee 66 d5 07 68-0a eb 8e e2 e0 7b 63 fe   \...f..h.....{c.
    0050 - 32 6f 1d b2 9e 0d b1 a3-31 d2 df 07 07 b7 33 fb   2o......1.....3.
    0060 - 75 50 a7 31 b8 ba 27 ad-24 ae 29 4b 17 8f 91 37   uP.1..'.$.)K...7
    0070 - c4 2b 3b e4 9b 6b c8 f1-75 c9 db 5a 52 f1 7f 4b   .+;..k..u..ZR..K
    0080 - e3 6c 87 97 37 95 ca 4b-ca c4 ef cb 9b 00 dd b4   .l..7..K........
    0090 - 50 b3 c4 a7 14 25 cc e8-0c 69 77 2e 1a ce 0a a0   P....%...iw.....
    00a0 - 05 7d fe 0e fd 77 65 46-68 06 40 07 d0 cf 4e f4   .}...weFh.@...N.
    00b0 - 19 58 86 37 09 f8 45 76-38 40 70 6b e8 16 83 ef   .X.7..Ev8@pk....
    00c0 - 19 c2 8f c3 44 9f c9 df-19 6b 3f 87 4d 5e 24 ed   ....D....k?.M^$.

    Start Time: 1560016165
    Timeout   : 7200 (sec)
    Verify return code: 0 (ok)
    Extended master secret: no
---
```

Ovdje možemo primijetiti brojne informacije. Odmah na početku vidimo podatke o certifikatu (`CN = miletic.net`) i pripadnom lancu izdavatelja certifikata (`CN = Let's Encrypt Authority X3` i `CN = DST Root CA X3`). Navedeni su algoritmi koji se koriste za potpis hashiranog sadržaja (`SHA256`, `RSA`), broj bajtova koje poslanih i primljenih kod ostvarivanja veze (`3475` i `433`) i certifikat je označen kao valjan (`Verification: OK`). Slijede podaci o korištenoj verziji TLS-a/SSL-a (`TLSv1.2`), kombinaciji algoritama za šifriranje koja se koristi (`ECDHE-RSA-AES256-GCM-SHA384`), duljini ključa poslužitelja (`2048 bit`), kompresiji, [TLS ekstenziji Application-Layer Protocol Negotiation (ALPN)](https://www.keycdn.com/support/alpn) i drugim značajkama ostvarene veze. Na ovaj način opciju `s_client` možemo koristiti pri dijagnosticiranju problema sigurnih poslužitelja jer pokazuje brojne podatke koje ne vidimo kod normalnog korištenja web preglednika ili klijenata za elektroničku poštu.

#### Server Name Indication

[Server Name Indication](https://en.wikipedia.org/wiki/Server_Name_Indication) (SNI) je proširenje TLS-a kojim klijent navodi u procesu rukovanja ime poslužitelja na koji se povezuje, što dozvoljava da na jednoj IP adresi postoji više od jednog HTTPS poslužitelja. Kako gotovo sve novije implementacije TLS-a podržavaju SNI, njegova [uporaba je u porastu posljednjih godina](https://developer.akamai.com/blog/2017/10/20/encrypting-web-all-need-support-tls-sni-remaining-clients).

Na istom poslužitelju na kojem je `example.group.miletic.net` su i [usluge i aplikacije koje je grupa razvila](https://apps.group.miletic.net/) `apps.group.miletic.net`. Iskoristimo parametar `-servername` da navedemo ime poslužitelja na koji se povezujemo:

``` shell
$ openssl s_client -connect example.group.miletic.net:443 -servername apps.group.miletic.net
CONNECTED(00000003)
depth=2 O = Digital Signature Trust Co., CN = DST Root CA X3
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = apps.group.miletic.net
verify return:1
---
Certificate chain
 0 s:CN = apps.group.miletic.net
   i:C = US, O = Let's Encrypt, CN = R3
 1 s:C = US, O = Let's Encrypt, CN = R3
   i:O = Digital Signature Trust Co., CN = DST Root CA X3
---
Server certificate
-----BEGIN CERTIFICATE-----
MIIFKzCCBBOgAwIBAgISA8luzKRXoYhXtGaHLyfJn6byMA0GCSqGSIb3DQEBCwUA
MDIxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1MZXQncyBFbmNyeXB0MQswCQYDVQQD
EwJSMzAeFw0yMTAzMDkxMDAxNDRaFw0yMTA2MDcxMDAxNDRaMBUxEzARBgNVBAMT
CnJ4ZG9jay5vcmcwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDjvGBq
hsYUlahsp0t14GuW5OGZsqZMRth0CySQJicG++vGrcj7d5R1PGfDSsTHzYeKP1SQ
w72Bu1bcm7E/oHqWQQbJgFrEtI6mlQ8tMbY9tF1ynE7SSMDzwXl3Ag4dHZoW2RIP
xOcq//uBlzbw+vkbiggkWBO1012pPoxU13blUuWondjjC/2OUYEde10h6yzv7tno
g/qI5Xl9uKeOK1LQHDBzeyIqkPlq0vRwTWIRDfACOl/bNVY1qE2MNFwvmYLzH8G2
pS0n7/aHaYP+b6Ys5daPKR+7SGyFIGw2i5PCHjX0QC8nTKmO1a+MchEAIKSk9tDB
4jODt+9VX0dvrW1RAgMBAAGjggJWMIICUjAOBgNVHQ8BAf8EBAMCBaAwHQYDVR0l
BBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMAwGA1UdEwEB/wQCMAAwHQYDVR0OBBYE
FMV8yV46fxwn3KHu2p7XqfcSds5NMB8GA1UdIwQYMBaAFBQusxe3WFbLrlAJQOYf
r52LFMLGMFUGCCsGAQUFBwEBBEkwRzAhBggrBgEFBQcwAYYVaHR0cDovL3IzLm8u
bGVuY3Iub3JnMCIGCCsGAQUFBzAChhZodHRwOi8vcjMuaS5sZW5jci5vcmcvMCUG
A1UdEQQeMByCCnJ4ZG9jay5vcmeCDnd3dy5yeGRvY2sub3JnMEwGA1UdIARFMEMw
CAYGZ4EMAQIBMDcGCysGAQQBgt8TAQEBMCgwJgYIKwYBBQUHAgEWGmh0dHA6Ly9j
cHMubGV0c2VuY3J5cHQub3JnMIIBBQYKKwYBBAHWeQIEAgSB9gSB8wDxAHcARJRl
LrDuzq/EQAfYqP4owNrmgr7YyzG1P9MzlrW2gagAAAF4FqZdyQAABAMASDBGAiEA
5yJnraRlL6AEBfQYxjvTaVYtZHP6cp2WEqduYcC0zhwCIQDkMAqYPN7gQ2yREBD4
SD0JFNbAH6RmowOhRY6btoH7HQB2APZclC/RdzAiFFQYCDCUVo7jTRMZM7/fDC8g
C8xO8WTjAAABeBamXbcAAAQDAEcwRQIgR0kFwclJ/36jRMAsLPSxy+ZP1ZVhd5FD
oWNA/0a6ID8CIQDZbHmc3qLlV+eNaFDLd/w0LsEfctDYyootev7ZyOYmdzANBgkq
hkiG9w0BAQsFAAOCAQEAgzflds7We2BJvgRzX+gIguTbkxeFe0qqpoE4kxAMwqXu
aKvIhHOGdjmAdIWBEsUxO5fkRn0FZx13UFtio9jJGWgynHCh9X/aZOZ6r69tIlJA
vG/6ZVr+XZv60D4NgQWMDrHKbwG8RMsnX0BiQ3Eo0JjDcqn8gLS47cG720gmCAfH
n2ufgLXWBUg2m4e4cjN7tPdZqlQdR8KKV/ssiUbDj8L/yOuxkjm8NYBMRUOKrMN5
f4dnt69CcpNmTTcw6qENs3mcz5IYmVoB4PRD2DBQ0swsPIxcpViaHbcxEgfIIFFU
xvZop7QEC07dsp1+T5+4WP4EWDHfpiuqxcJHnndjPg==
-----END CERTIFICATE-----
subject=CN = apps.group.miletic.net

issuer=C = US, O = Let's Encrypt, CN = R3
(...)
```

Uočimo kako smo dobili drugi certifikat te je ime domene `example.group.miletic.net` bilo iskorišteno samo za otkrivanje IP adrese na koju će se `s_client` povezati.

Više o opciji `s_client` i parametrima koje prima moguće je naći u pripadnoj stranici priručnika `s_client(1ssl)`.

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
