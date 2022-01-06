---
author: Vedran Miletić
---

# HTTP autentifikacija u jeziku PHP

HTTP nudi okvir za upravljanje pravom pristupa i autentifikacijom. Mi ćemo se ovdje ograničiti na osnovnu autentifikaciju pristupa (engl. *basic access authentication*, [Wikipedia](https://en.wikipedia.org/wiki/Basic_access_authentication)) koja je standardizirana u [RFC-u 7617 pod naslovom The 'Basic' HTTP Authentication Scheme](https://datatracker.ietf.org/doc/html/rfc7617).

## Rudimentarna provjera autorizacije

Kako smo ranije dodavali druga zaglavlja, možemo u HTTP zahtjevu možemo dodati zaglavlje `Authorization` koje sadrži tip i vjerodajnice autorizacije za izvođenje operacije na poslužitelju ([više detalja o HTTP zaglavlju Authorization na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization)). To zaglavlje možemo na poslužitelju pronaći u polju koje sadrži sva zaglavlja dohvaćenom funkcijom `getallheaders()`. Uzmimo da je za uspješnu prijavu nužno koristiti autorizaciju osnovnog tipa `Basic` i vjerodajnicu (u ovom slučaju kodirani zapis korisničkog imena i zaporke) `YWxhZGRpbjpvcGVuc2VzYW1l`. (Kasnije ćemo naučiti generirati vjerodajnice iz unesenog korisničkog imena i zaporke.) Imamo kod oblika:

``` php
<?php

$request_headers = getallheaders();
if (array_key_exists("Authorization", $request_headers) && $request_headers["Authorization"] == "Basic YWxhZGRpbjpvcGVuc2VzYW1l") {
    echo "<p>Dobrodošli.</p>\n";
} else {
    http_response_code(401);
    echo "<p>Nemate pravo pristupa.</p>\n";
}
```

U slučaju da provjera autorizacije ne prođe uspješno, odgovor ima statusni kod 401 Unauthorized.

Uvjerimo se da ova rudimentarna provjera autorizacije radi ispravno:

``` shell
$ curl -v -H "Authorization: Ruski haker" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Authorization: Ruski haker
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 401 Unauthorized
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 00:56:39 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
<p>Nemate pravo pristupa.</p>
* Closing connection 0

$ curl -v -H "Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 00:57:11 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
<p>Dobrodošli.</p>
* Closing connection 0
```

Naravno, mi želimo da korisnik na strani klijenta unosi korisničko ime i zaporku umjesto niza znakova `YWxhZGRpbjpvcGVuc2VzYW1l` pa ćemo se u nastavku baviti stvarnom implementacijom osnovne varijante HTTP autentifikacije.

## Implementacija autentifikacije

[RFC 7235 naslovljen Hypertext Transfer Protocol (HTTP/1.1): Authentication](https://datatracker.ietf.org/doc/html/rfc7235) definira općeniti autenfikacijski okvir na način:

1. Klijent šalje zahtjev za resursom, a poslužitelj šalje odgovor sa statusnim kodom [401 Unauthorized](https://http.cat/401) ([više detalja o HTTP statusnom kodu 401 Unauthorized na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401)) koji sadrži informacije o tome kako se autentificirati i zaglavlje `WWW-Authenticate` s najmanje jednim izazovom ([više detalja o HTTP zaglavlju WWW-Authenticate na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/WWW-Authenticate)).
1. Klijent koji se želi autentificirati s poslužiteljem će poslati zahtjev koji sadrži zaglavlje `Authorization` s tipom vjerodajnicama autorizacije.

Više detalja o HTTP autentifikaciji [može se naći na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication).

[U jeziku PHP moguće je implementirati HTTP autentifikaciju](https://www.php.net/manual/en/features.http-auth.php) korištenjem funkcije `header()` za slanje HTTP zaglavlja `WWW-Authenticate` s odgovarajućom vrijednosti i, kao i ranije, funkcije `http_response_code()` za postavljanje statusnog koda 401 Unauthorized u odgovoru na zahtjev. Nakon primanja idućeg zahtjeva koji sadrži zaglavlje `Authorization` s odgovarajućom vrijednosti, u polju `$_SERVER` su dodane vrijednosti `$_SERVER["AUTH_TYPE"]`, `$_SERVER["PHP_AUTH_USER"]` i `$_SERVER["PHP_AUTH_PW"]` koje sadrže tip autentifikacije, korisničko ime i zaporku (respektivno).

``` php
<?php

if (isset($_SERVER["PHP_AUTH_USER"]) && isset($_SERVER["PHP_AUTH_PW"])) {
    $user = $_SERVER["PHP_AUTH_USER"];
    $pw = $_SERVER["PHP_AUTH_PW"];
    echo "<p>Pozdrav $user, unijeli ste $pw kao zaporku.</p>\n";
} else {
    http_response_code(401);
    header("WWW-Authenticate: Basic realm=\"Tajni laboratorij Odjela za informatiku\"");
    echo "<p>Niste prijavljeni.</p>\n";
}
```

U HTTP zaglavlju `WWW-Authenticate` navodimo tip autentifikacije koji se koristi, u našem slučaju `Basic` i opis zaštićenog dijela web sjedišta (`realm="Tajni laboratorij Odjela za informatiku"`). Pritom uočimo da se dvostruki navodnici (znak `"`) koji su dio znakovnog niza (i nisu oznaka za njegov početak ili kraj) unose korištenjem znaka silazne kose crte i dvostrukog navodnika (znakovi `\"`).

Napravimo zahtjev:

``` shell
$ curl -v http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 401 Unauthorized
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 17:09:28 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< WWW-Authenticate: Basic realm="Tajni laboratorij Odjela za informatiku"
< Content-Type: text/html; charset=UTF-8
<
<p>Niste prijavljeni.</p>
* Closing connection 0
```

Uočimo zaglavlje `WWW-Authenticate` u odgovoru koje ima sadržaj opisan iznad. U cURL-u se možemo prijaviti korištenjem parametra `--user`, odnosno `-u` na način:

``` shell
$ curl -v -u ivanhorvat:m0jazap0rka http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
* Server auth using Basic with user 'ivanhorvat'
> GET / HTTP/1.1
> Host: localhost:8000
> Authorization: Basic aXZhbmhvcnZhdDptMGphemFwMHJrYQ==
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 17:15:58 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
<p>Pozdrav ivanhorvat, unijeli ste m0jazap0rka kao zaporku.</p>
* Closing connection 0
```

Vidimo da smo se uspješno prijavili. U zahtjevu postoji zaglavlje `Authorization` koje ima vrijednost `Basic aXZhbmhvcnZhdDptMGphemFwMHJrYQ==`; `Basic` je tip autentifikacije, a niz znakova `aXZhbmhvcnZhdDptMGphemFwMHJrYQ==` je kodiran shemom [Base64](https://en.wikipedia.org/wiki/Base64) ([više detalja o shemi kodiranja Base64 na MDN-u](https://developer.mozilla.org/en-US/docs/Glossary/Base64)). Sad kad smo naučili da se radi o shemi Base64, ostavljamo čitatelju da odgonetne koje korisničko ime i zaporku kodira ranije naveden niz znakova `Basic YWxhZGRpbjpvcGVuc2VzYW1l`.

## Dekodiranje Base64 zapisa

Kako se ne radi o šifriranju, već o jednostavnom kodiranju, lako je znakovni niz dekodirati korištenjem funkcije `base64_decode()` ([dokumentacija](https://www.php.net/manual/en/function.base64-decode.php)).

Interpreter jezika PHP od [verzije 5.1.0](https://www.php.net/releases/5_1_0.php) nadalje omogućuje korištenje interaktivnog načina rada koji je primarno namijenjen za brzo isprobavanje kako manji dijelovi koda rade, što je vrlo korisno i u procesu učenja. Pokrenimo interaktivni način rada interpretera PHP-a korištenjem parametra `--interactive`, odnosno `-a` te izvedimo Base64 dekodiranje:

``` shell
$ php -a
Interactive mode enabled

php > echo base64_decode("aXZhbmhvcnZhdDptMGphemFwMHJrYQ==");
ivanhorvat:m0jazap0rka
```

Zatim izvedimo kodiranje funkcijom `base64_encode()` ([dokumentacija](https://www.php.net/manual/en/function.base64-encode.php)):

```
php > echo base64_encode("ivanhorvat:m0jazap0rka");
aXZhbmhvcnZhdDptMGphemFwMHJrYQ==
```

!!! warning
    Slati u HTTP zahtjevu zapise kodirane shemom Base64 bez dodatnog šifriranja nije dobra sigurnosna praksa. U konkretnoj primjeni se koriste složeniji tipovi autentifikacije od ovog osnovnog i HTTPS koji uspostavlja šifrirani kanal za poruke (zahtjeve i odgovore). Mi se ovim temama bavimo s ciljem razumijevanja načina rada HTTP-a pa se ovdje ograničavamo na osnovni tip autentifikacije.
