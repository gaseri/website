---
author: Vedran Miletić
---

# Kodiranje HTTP sadržaja i kompresija u jeziku PHP

HTTP zaglavlje `Accept-Encoding` u zahtjevu navodi koja kodiranja sadržaja (u konkretnom slučaju najčešće algoritme kompresije sadržaja) klijent može primiti i razumijeti ([više detalja o HTTP zaglavlju Accept-Encoding na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Encoding)). Primjerice, zaglavlje može biti oblika `Accept-Encoding: br, gzip;q=1.0, *;q=0.5` i reći da klijent razumije [Brotli](https://brotli.org/) ([Wikipedia](https://en.wikipedia.org/wiki/Brotli)) (implicitno s kvalitetom 1.0), [Gzip](https://www.gzip.org/) ([Wikipedia](https://en.wikipedia.org/wiki/Gzip)) s kvalitetom 1.0 i sve ostale oblike kodiranja sadržaja s kvalitetom 0.5.

Poslužitelj na temelju primljene vrijednosti odabire jedno od predloženih kodiranja sadržaja pa njime kodira (u konkretnom slučaju najčešće komprimira) sadržaj i navodi korištenu metodu u HTTP zaglavlju `Content-Encoding` u odgovoru koji šalje klijentu ([više detalja o HTTP zaglavlju Content-Encoding na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding)).

U jeziku PHP kompiranje sadržaja je dostupno funkcijom `gzcompress()` ([dokumentacija](https://www.php.net/manual/en/function.gzcompress.php)), a dekomprimiranje funkcijom `gzuncompress()` ([dokumentacija](https://www.php.net/manual/en/function.gzuncompress.php)). Međutim, te funkcije samo komprimiraju dani sadržaj, a kako bi komprimirani sadržaj bio prepoznat na klijentu i dekomprimiran potrebno mu je dodati i zaglavlje i kontrolni zbroj. Za tu svrhu postoji funkcija `gzencode()` ([dokumentacija](https://www.php.net/manual/en/function.gzencode.php)) koja komprimira sadržaj i kodira ga u ispravnom obliku za slanje. Potpunosti radi spomenut ćemo da postoji i funkcija za dekodiranje i dekompresiju sadržaja `gzdecode()` ([dokumentacija](https://www.php.net/manual/en/function.gzdecode.php)) koju dalje nećemo koristiti.

## Jednostavan odabir kodiranja sadržaja

Radi jednostavnosti, uzmimo da klijent ne navodi kvalitetu razumijevanja.

``` php
<?php

$encodings = $_SERVER["HTTP_ACCEPT_ENCODING"];
$encodings_split = explode(",", $encodings);
$encodings_trimmed = array_map("trim", $encodings_split);

$contents = "<p>Ovo je čitav sadržaj koji će biti poslan klijentu.</p>\n";
if (in_array("gzip", $encodings_trimmed)) {
    header('Content-Encoding: gzip');
    $contents_compressed = gzencode($contents);
    echo $contents_compressed;
}
else {
    echo $contents;
}
```

Provjerimo dobiveni sadržaj u situaciji kad se kompresija ne koristi:

``` shell
$ curl -v http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Wed, 05 May 2021 14:30:05 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: text/html; charset=UTF-8
<
<p>Ovo je čitav sadržaj koji će biti poslan klijentu.</p>
* Closing connection 0
```

Zatražimo cURL-om komprimirani sadržaj korištenjem parametra `--compressed` koji će za nas navesti podržane formate kodiranja sadržaja u zaglavlju Accept-Encoding:

``` shell
$ curl -v --compressed http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Encoding: deflate, gzip, br
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Wed, 05 May 2021 14:30:06 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Encoding: gzip
< Content-Type: text/html; charset=UTF-8
<
<p>Ovo je čitav sadržaj koji će biti poslan klijentu.</p>
* Closing connection 0
```

Vidimo da je sadržaj uspješno primljen i dekodiran (dekomprimiran).

Na ovaj način svakom klijentu koji ne navede `gzip` kao podržani format kodiranja sadržaja poslati sadržaj bez kompresije. To nije u skladu sa standardom koji zahtijeva da, u slučaju da klijent navede `identity;q=0` (nepromijenjen sadržaj na strani klijenta ima kvalitetu razumijevanja 0) ili `*;q=0` (svi ostali sadržaji osim ranije navedenih na klijentu imaju kvalitetu razumijevanja 0) uz nepodržane formate, potrebno je vratiti odgovor s HTTP statusnim kodom 406 Not Acceptable. To ćemo postići kodom oblika:

``` php
<?php

$encodings = $_SERVER["HTTP_ACCEPT_ENCODING"];
$encodings_split = explode(",", $encodings);
$encodings_trimmed = array_map("trim", $encodings_split);

$contents = "<p>Ovo je čitav sadržaj koji će biti poslan klijentu.</p>\n";
if (in_array("gzip", $encodings_trimmed)) {
    header('Content-Encoding: gzip');
    $contents_compressed = gzencode($contents);
    echo $contents_compressed;
}
elseif (in_array("identity;q=0", $encodings_trimmed) || in_array("*;q=0", $encodings_trimmed)) {
    http_response_code(406);
    echo "<p>Poslani popis kodiranja nije prihvatljiv.</p>\n";
}
else {
    echo $contents;
}
```

## Opći odabir kodiranja sadržaja

U situaciji kad klijent navodi kvalitetu razumijevanja za pojedina kodiranja postupak odabira je sličan kao kod odabira jezika.
