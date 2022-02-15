---
author: Vedran Miletić
---

# Slanje podataka u HTTP zahtjevu metodama GET i POST u jeziku PHP

## Specifičnosti metode GET

Varijable navedene kod slanja zahtjeva HTTP metodom GET možemo dohvatiti putem varijable `$_GET` ([dokumentacija](https://www.php.net/manual/en/reserved.variables.get.php)). Primjerice, recimo da skripta index.php očekuje ime i prezime na način da dajete zatjeve na putanju `/profil?ime=Ivan&prezime=Horvat`. Možemo se ponadati da bismo obradu takvih zahtjeva mogli izvesti kodom:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["REQUEST_URI"] == "/profil") {
    $ime = $_GET["ime"];
    $prezime = $_GET["prezime"];
    echo "<p>Vi ste $ime $prezime.</p>\n";
}
```

Međutim, `$_SERVER["REQUEST_URI"]` će uključivati čitavu putanju u kojoj se nalazi i upit (u našem slučaju `?ime=Ivan&prezime=Horvat`) pa nam kod neće raditi kako očekujemo. Putanja bez upita je sadržana u varijabli `$_SERVER["PATH_INFO"]` pa je ispravan kod oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["PATH_INFO"] == "/profil") {
    $ime = $_GET["ime"];
    $prezime = $_GET["prezime"];
    echo "<p>Vi ste $ime $prezime.</p>\n";
}
```

Kako ljuska tretira `?` i `&` kao posebne znakove, URL je kod izvođenja zahtjeva potrebno staviti pod navodnike na način:

``` shell
$ curl -v "http://localhost:8000/profil?ime=Ivan&prezime=Horvat"
*   Trying 127.0.0.1:8000...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET /profil?ime=Ivan&prezime=Horvat HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.68.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 03 Nov 2020 12:55:51 GMT
< Connection: close
< X-Powered-By: PHP/7.4.3
< Content-Type: text/html; charset=UTF-8
<
<p>Vi ste Ivan Horvat.</p>
* Closing connection 0
```

Vrijednosti pojedinih GET varijabli mogu sadržavati znak razmaka. U tom slučaju je potrebno znak razmaka kodirati znakom plusa (`+`; [više detalja o kodiranju znakova u URL-ima na MDN-u](https://developer.mozilla.org/en-US/docs/Glossary/percent-encoding)) na način:

``` shell
$ curl -v "http://localhost:8000/profil?ime=Ivan+Tomislav&prezime=Horvat"
*   Trying 127.0.0.1:8000...
* connect to 127.0.0.1 port 8000 failed: Veza odbijena
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /profil?ime=Ivan+Tomislav&prezime=Horvat HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.79.1
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Thu, 04 Nov 2021 16:51:19 GMT
< Connection: close
< X-Powered-By: PHP/8.0.12
< Content-type: text/html; charset=UTF-8
<
<p>Vi ste Ivan Tomislav Horvat.</p>
* Closing connection 0
```

## Specifičnosti metode POST

Podatke možemo primati i metodom POST i dohvatiti ih putem varijable `$_POST` ([dokumentacija](https://www.php.net/manual/en/reserved.variables.post.php)) na način:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/profil") {
    $ime = $_POST["ime"];
    $prezime = $_POST["prezime"];
    echo "<p>Vi ste $ime $prezime.</p>\n";
}
```

Ovdje putanja ne sadrži upit pa nam `$_SERVER["REQUEST_URI"]` ostaje `/profil` pa ne moramo koristiti `$_SERVER["PATH_INFO"]`. Zahtjev izvodimo na način:

``` shell
$ curl -v -X POST -d "ime=Ivan" -d "prezime=Horvat" http://localhost:8000/profil
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying 127.0.0.1:8000...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8000 (#0)
> POST /profil HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.68.0
> Accept: */*
> Content-Length: 23
> Content-Type: application/x-www-form-urlencoded
>
* upload completely sent off: 23 out of 23 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 03 Nov 2020 13:02:47 GMT
< Connection: close
< X-Powered-By: PHP/7.4.3
< Content-Type: text/html; charset=UTF-8
<
<p>Vi ste Ivan Horvat.</p>
* Closing connection 0
```

Uočimo također kako je sada po prvi put postavljeno HTTP zaglavlje `Content-Type` i u zahtjevu te ima vrijednost `application/x-www-form-urlencoded`, što je MIME tip podataka poslanih web formom. Drugim riječima, ovdje cURL-om šaljemo podatke isto kao što bismo na web sjedištu nekog sustava za učenje slali nakon popunjavanja podataka na svom profilu i pristiska na gumb za spremanje.

## Provjera poslanih varijabli u zahtjevima metodom GET i POST

Dodatno možemo provjeriti jesu li u zahtjevu poslane varijable koje želimo iskoristiti, što ćemo izvesti funkcijom `isset()` ([dokumentacija](https://www.php.net/manual/en/function.isset.php)). Zatim ćemo formirati adekvatan odgovor u slučaju da varijable nisu postavljene; primjerice, postavit ćemo već ranije korišteni statusni kod 400 Bad Request i poslati pripadnu poruku:

``` php
<?php

if ($_SERVER["REQUEST_URI"] == "/profil" && $_SERVER["REQUEST_METHOD"] == "POST") {
    $ime = $_POST["ime"];
    $prezime = $_POST["prezime"];
    if (isset($ime) && isset($prezime)) {
        echo "<p>Vi ste $ime $prezime.</p>\n";
    } else {
        http_response_code(400);
        echo "<p>Niste poslali ime ili prezime.</p>\n";
    }
}
```

Uvjerimo se da smo uhvatili sve situacije kad u zahtjevu nedostaje neka od varijabli:

``` shell
$ curl -X POST http://localhost:8000/profil
<p>Niste poslali ime ili prezime.</p>
$ curl -X POST -d "ime=Ivan" http://localhost:8000/profil
<p>Niste poslali ime ili prezime.</p>
$ curl -X POST -d "prezime=Horvat" http://localhost:8000/profil
<p>Niste poslali ime ili prezime.</p>
```

Na analogan način provjerili bismo poslane varijable u slučaju kad se koristi metoda GET.

Dodatno, varijable navedene kod slanja zahtjeva HTTP metodama GET i POST možemo dohvatiti putem varijable `$_REQUEST` ([dokumentacija](https://www.php.net/manual/en/reserved.variables.request.php)), što omogućuje korištenje istog koda za dohvaćanje varijabli bez obzira na metodu. To spominjemo radi potpunosti i dalje nećemo koristiti.
