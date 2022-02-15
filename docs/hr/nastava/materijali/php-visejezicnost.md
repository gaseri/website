---
author: Vedran Miletić
---

# Višejezičnost u jeziku PHP

Standard [ISO 639-1](https://www.iso.org/iso-639-language-codes.html) definira [dvoslovne kodove](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) za označavanje jezika koji postoje u svijetu. Donekle povezan standard [ISO 3166-1](https://www.iso.org/iso-3166-country-codes.html) u odjeljku alpha-2 definira [dvoslovne kodove](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) za označavanje država. Kako bismo lakše razlikovali jezične kodove od državnih, konvencija je da se ovi prvi pišu malim slovima, a ovi drugi velikim.

HTTP zaglavlje `Accept-Language` u zahtjevu oglašava koje jezike klijent može razumijeti i koja lokalna varijanta jezika je preferirana ([više detalja o HTTP zaglavlju Accept-Language na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language)). Jezici se navode odvojeni zarezom, a znak zvjezdice (`*`) označava sve jezike. Opcionalno se može navesti i kvaliteta razumijevanja pojedinih jezika (vrijednost između 0 i 1) dodavanjem `;q=`, npr. `Accept-Language: hr, en;q=0.9, de-AT;q=0.7, *;q=0.5` znači da klijent razumije hrvatski, engleski s kvalitetom 0.9, njemački kakav se govori u Austriji s kvalitetom 0.7, a ostale jezike s kvalitetom 0.5. U PHP-u nam je sadržaj tog zaglavlja dostupan u polju `$_SERVER`, konkretno kao `$_SERVER["HTTP_ACCEPT_LANGUAGE"]`.

HTTP zaglavlje `Content-Language` u odgovoru navodi u kojem je jeziku napisan sadržaj koji se šalje ([više detalja o HTTP zaglavlju Content-Language na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Language)). Primjerice, `Content-Language: de-AT` znači da je sadržaj na njemačkom jeziku kakav se govori u Austriji. Najčešće se koristi za tekstualni sadržaj (čisti tekst i HTML), ali može se koristiti za bilo koji medij.

## Rad s poljima u jeziku PHP

Dosad smo već koristili vrijednosti iz polja `$_SERVER` kao što su `$_SERVER["REQUEST_URI"]` i `$_SERVER["REQUEST_METHOD"]`. Vidjeli smo da su te vrijednosti znakovni nizovi, a istog su tipa i ključevi `"REQUEST_URI"` i `"REQUEST_METHOD"` putem kojih ih dohvaćamo. Dakle, ključevi polja u jeziku PHP ne moraju biti poredani cijeli brojevi koji kreću od nule kako smo navikli kod rada s poljima u C/C++-u. Polje u PHP-u ([dokumentacija](https://www.php.net/manual/en/language.types.array.php)) je zapravo poredano preslikavanje (engl. *ordered map*). Za ilustraciju načina rada s poljima u PHP-u od [verzije 5.4.0](https://www.php.net/releases/5_4_0.php) nadalje, definirajmo dva polja: prvo slično poljima u C/C++-u, a drugo polju `$_SERVER`:

``` php
<?php

$arr1 = ["moja vrijednost", 1, 3.5, true]; // ekvivalentno [0 => "moja vrijednost", 1 => 8, 2 => 3.5, 3 => true]
$arr2 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true];
```

Vrijednosti elemenata prvog polja možemo dohvatiti na način `$arr1[0]`,  `$arr1[1]`,  `$arr1[2]` i  `$arr1[3]`, a drugog na način `$arr1["moj kljuc"]`,  `$arr1["broj"]`,  `$arr1["drugi broj"]` i  `$arr1["je li istina"]`.

Pokrenimo interaktivni način rada interpretera PHP-a korištenjem parametra `--interactive`, odnosno `-a` te definirajmo ta dva polja `$arr1` i `$arr2` kao iznad, a zatim provjerimo njihov sadržaj funkcijom ispisa `print_r()` ([dokumentacija](https://www.php.net/manual/en/function.print-r.php)):

``` shell
$ php -a
Interactive mode enabled

php > $arr1 = ["moja vrijednost", 1, 3.5, true];
php > $arr2 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true];
php > print_r($arr1);
Array
(
    [0] => moja vrijednost
    [1] => 1
    [2] => 3.5
    [3] => 1
)
php > print_r($arr2);
Array
(
    [moj kljuc] => moja vrijednost
    [broj] => 8
    [drugi broj] => 3.5
    [je li istina] => 1
)
```

Možemo se uvjeriti i da uspješno dohvaćamo pojedine vrijednosti iz polja na način koji smo naveli:

```
php > echo $arr1[0];
moja vrijednost
php > echo $arr2["broj"];
8
php > echo $arr2["drugi broj"];
3.5
```

## Jednostavan odabir jezika

Implementirajmo jednostavan odabir jezika koji omogućuje primanje odabira na hrvatskom, engleskom ili poruke o pogrešci. Ako klijent navede da prihvaća hrvatski, dobit će odgovor na hrvatskom bez obzira je li naveo engleski jer je hrvatski preferirani jezik poslužitelja. Ako klijent navede da prihvaća engleski ili navede da prihvaća sve jezike bez da eksplicitno navede hrvatski ili engleski, dobit će odgovor na engleskom. Ako klijent navede popis jezika koji ne uključuje niti hrvatski, niti engleski, niti sve ostale jezike, dobit će odgovor s postavljenim HTTP statusnim kodom 406 Not Acceptable.

Kako bismo po nakon izdvajanja sadržaja zaglavlja `Accept-Language` po znaku zareza razdvojili dobiveni popis jezika, trebat će nam funkcija `explode()`. Ako je popis jezika znakovni niz oblika `"en,hr"`, cijepanjem po zarezu dobivamo polje `["en", "hr"]`, što nam odgovara.

Ako je pak klijent ostavio razmak nakon svakog zareza, što je dozvoljeno, ostat će nam razmaci nakon cijepanja; naime, cijepanjem po zarezu znakovnog niza `"en, hr"` dobivamo polje `["en", " hr"]`. Stoga ćemo na dobivenom polju funkcijom `array_map()` ([dokumentacija](https://www.php.net/manual/en/function.array-map.php)) primijeniti funkciju `trim()` ([dokumentacija](https://www.php.net/manual/en/function.trim.php)) kako bismo očistili znak razmaka s početka i kraja svakog elementa polja, ako ih ima. Naposlijetku ćemo funkcijom `in_array()` ([dokumentacija](https://www.php.net/manual/en/function.in-array.php)) provjeriti nalazi li se neka vrijednost za hrvatski jezik u polju:

``` php
<?php

$languages = $_SERVER["HTTP_ACCEPT_LANGUAGE"];
$languages_split = explode(",", $languages);
$languages_trimmed = array_map("trim", $languages_split);

if (in_array("hr", $languages_trimmed)) {
    header("Content-Language: hr");
    echo "<p>Pozdrav, svijete!</p>\n";
} elseif (in_array("en", $languages_trimmed) || in_array("*", $languages_trimmed)) {
    header("Content-Language: en");
    echo "<p>Hello, world!</p>\n";
} else {
    http_response_code(406);
    echo "<p>Poslani popis jezika nije prihvatljiv.</p>\n";
}
```

Isprobajmo odabir jezika:

``` shell
$ curl -v -H "Accept-Language: en,es,pt" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Accept-Language: en,es,pt
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Fri, 01 Jan 2021 20:14:25 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Language: en
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0

$ curl -v -H "Accept-Language: hr, ru" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Accept-Language: hr, ru
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Fri, 01 Jan 2021 20:14:31 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Language: hr
< Content-Type: text/html; charset=UTF-8
<
<p>Pozdrav, svijete!</p>
* Closing connection 0

$ curl -v -H "Accept-Language: de, fr" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: de, fr
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 406 Not Acceptable
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 09:41:56 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: text/html; charset=UTF-8
<
<p>Poslani popis jezika nije prihvatljiv.</p>
* Closing connection 0
```

Dobit ćemo odogovor na hrvatskom čak i u slučaju da je hrvatski jezik naveden u `Accept-Language` na nekom mjestu osim prvog:

``` shell
$ curl -v -H "Accept-Language: de, hr" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Accept-Language: de, hr
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Fri, 01 Jan 2021 20:17:18 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Language: hr
< Content-Type: text/html; charset=UTF-8
<
<p>Pozdrav, svijete!</p>
* Closing connection 0
```

## Opći odabir jezika

Evidentno je da kod iznad ne podržava odabir jezika oblika `hr-HR, en;q=0.9, de-AT;q=0.7, *;q=0.5`, a možemo se u to uvjeriti i slanjem zahtjeva:

```
curl -v -H "Accept-Language: hr-HR, en;q=0.9, de-AT;q=0.7, *;q=0.5" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: hr-HR, en;q=0.9, de-AT;q=0.7, *;q=0.5
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 406 Not Acceptable
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 09:48:12 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: text/html; charset=UTF-8
<
<p>Poslani popis jezika nije prihvatljiv.</p>
* Closing connection 0
```

Vidjeli smo na početku kako je taj način navođenja jezika legitiman i bilo bi ga potrebno podržati. Izgradimo korak po korak kod koji ga podržava.

### Navođenje lokalne varijante jezika

Podržimo odabir jezika koji navodi lokalnu varijantu, ali ne i kvalitetu razumijevanja, primjerice `hr-HR, en, de-AT, *`. Pomognimo se [popisom službenih jezika po državama](https://wiki.openstreetmap.org/wiki/Nominatim/Country_Codes) s [OpenStreetMap](https://www.openstreetmap.org/) [wikija](https://wiki.openstreetmap.org/wiki/Main_Page) kako bismo uočili da se hrvatski jezik govori kao službeni u Hrvatskoj te Bosni i Hercegovini. Imamo kod oblika:

``` php
<?php

$languages = $_SERVER["HTTP_ACCEPT_LANGUAGE"];
$languages_split = explode(",", $languages);
$languages_trimmed = array_map("trim", $languages_split);

if (in_array("hr", $languages_trimmed) || in_array("hr-HR", $languages_trimmed) || in_array("hr-BA", $languages_trimmed)) {
    header("Content-Language: hr");
    echo "<p>Pozdrav, svijete!</p>\n";
} elseif (in_array("en", $languages_trimmed) || in_array("*", $languages_trimmed)) {
    header("Content-Language: en");
    echo "<p>Hello, world!</p>\n";
} else {
    http_response_code(406);
    echo "<p>Poslani popis jezika nije prihvatljiv.</p>\n";
}
```

Isprobajmo kod:

``` shell
$ curl -v -H "Accept-Language: hr-HR, en, de-AT" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: hr-HR, en, de-AT
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 10:41:24 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Language: hr
< Content-Type: text/html; charset=UTF-8
<
<p>Pozdrav, svijete!</p>
* Closing connection 0
```

Na isti način možemo napraviti i za engleski, iako je popis država u kojima je jedan od službenih jezika podosta duži. Uz rizik da podržimo i neku neslužbenu varijantu engleskog jezika, možemo filtrirati primljeni popis jezika funkcijom `array_filter()` ([dokumentacija](https://www.php.net/manual/en/function.array-filter.php)) tako da iz njega izdvojimo samo lokalne varijante engleskog, odnosno sve jezike čija su prva dva znaka `en`. Tu ćemo provjeru napraviti funkcijom izdvajanjem znakovnog podniza funkcijom `substr()` ([dokumentacija](https://www.php.net/manual/en/function.substr.php)) od početnog znaka (indeks 0) duljine 2. Prosljeđivanje funkcije funkciji smo već izveli iznad kod korištenja funkcije `trim()`, a ovdje ćemo izvesti istu stvar uz tu razliku da ime funkcije nećemo navesti kao znakovni niz, već ćemo čitavu funkciju postaviti u argument.

Prihvaćanje engleskog jezika od strane klijenta se sada svodi na provjeru duljine popisa svih lokalnih varijanti engleskog jezika. Kod je oblika:

``` php
<?php

$languages = $_SERVER["HTTP_ACCEPT_LANGUAGE"];
$languages_split = explode(",", $languages);
$languages_trimmed = array_map("trim", $languages_split);

$languages_english = array_filter($languages_trimmed, function($value) { return substr($value, 0, 2) == "en"; });
$accepts_english = count($languages_english) > 0;

if (in_array("hr", $languages_trimmed) || in_array("hr-HR", $languages_trimmed) || in_array("hr-BA", $languages_trimmed)) {
    header("Content-Language: hr");
    echo "<p>Pozdrav, svijete!</p>\n";
} elseif ($accepts_english || in_array("*", $languages_trimmed)) {
    header("Content-Language: en");
    echo "<p>Hello, world!</p>\n";
} else {
    http_response_code(406);
    echo "<p>Poslani popis jezika nije prihvatljiv.</p>\n";
}
```

Isprobajmo kod navođenjem danske varijante engleskog jezika koja službeno ne postoji:

``` shell
$ curl -v -H "Accept-Language: en-DK, de-AT" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: en-DK, de-AT
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 10:42:31 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Language: en
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0
```

### Navođenje kvalitete razumijevanja

Uzmimo sad da poslužitelj može vratiti odgovor na hrvatskom, slovenskom, češkom ili ruskom i odabrat će onaj za koji klijent navede da ga najbolje razumije. Ako klijent ne navede nijedan od ta četiri jezika, poslužitelj će vratiti odgovor s postavljenim HTTP statusnim kodom 406 Not Acceptable.

Poslužitelj će primiti popis jezika od klijenta, opcionalno s navedenim kvalitetama razumijevanja. Ukoliko kvaliteta razumijevanja za neki jezik nije navedena, uzet će da ona iznosi 1.0 za taj jezik. Pretvorba primljenog znakovnog niza u popis koja koristi `explode()` i `trim()` je ista kao ranije. Od tog popisa želimo dobiti popis jezika koje korisnik razumije poredan po kvaliteti razumijevanja. Za tu svrhu se prvo svaki od jezika u dobivenom popisu (iskoristit ćemo konstrukt `foreach` ([dokumentacija](https://www.php.net/manual/en/control-structures.foreach.php)) za prolaz po čitavom polju) razdvaja od svoje kvalitete po znaku točke sa zarezom.

Ako su razdvajanjem dobivena dva elementa, onda se prvi uzima kao ime jezika, a drugi kao njegova kvaliteta zapisana u obliku znakovnog niza `q=x.yzw`. Funkcijom `substr()` moguće je preskočiti prva dva znaka pa potom korištenjem pretvorbe tipa `(float)` ([dokumentacija](https://www.php.net/manual/en/language.types.type-juggling.php#language.types.typecasting)) dobiveni broj zapisan u obliku znakovnog niza zapisati kao broj s pomičnim zarezom.

Ako razdvajanjem nisu dobivena dva elementa, uzima se naziv jezika kako je naveden i postavlja mu se kvaliteta razumijevanja na 1.0.

``` php
<?php

$languages = $_SERVER["HTTP_ACCEPT_LANGUAGE"];
$languages_split = explode(",", $languages);
$languages_trimmed = array_map("trim", $languages_split);

$languages_to_quality = [];
foreach ($languages_trimmed as $language_item) {
    $language_quality_pair = explode(";", $language_item);
    if (count($language_quality_pair) == 2) {
        $language = $language_quality_pair[0];
        $quality = (float) substr($language_quality_pair[1], 2);
        $languages_to_quality[$language] = $quality;
    }
    else {
        $languages_to_quality[$language_item] = 1.0;
    }
}
```

Jezike ćemo najčešće dobiti sortirane u silaznom poretku od najkvalitetnije razumljivog do najmanje kvalitetno razumljivog jer je konvencija da se tako navode na klijentskoj strani, a to nam i odgovora. Ipak, specifikacija to ne zahtijeva pa ćemo izvršiti sortiranje prije nastavka odabira jezika za odgovor.

Funkcija koja vrši poredavanje polja u silaznom poretku je `rsort()` ([dokumentacija](https://www.php.net/manual/en/function.rsort.php)), ali ona ne čuva udruženje ključa i vrijednosti polja. Primjerice, poredavanjem polja `["en" => 0.6, "hr" => 0.8]` funkcijom `rsort()` dobili bismo polje `[0.8, 0.6]` iz kojeg ne saznajemo ništa o preferiranim jezicima na strani klijenta.

Funkcija koja vrši poredavanje polja u silaznom poretku i pritom čuva udruženje ključa i vrijednosti polja je `arsort()` ([dokumentacija](https://www.php.net/manual/en/function.arsort.php)). Poredavanjem polja `["en" => 0.6, "hr" => 0.8]` tom funkcijom dobili bismo polje `["hr" => 0.8, "en" => 0.6]`, što smo i htjeli dobiti.

``` php
<?php

// ...
arsort($languages_to_quality);
```

Prolazit ćemo silazno poredani popis jezika konstruktom `foreach` dok u tom popisu ne pronađemo jedan od jezika koji poslužitelj podržava, a zatim poslati odgovor i prekinuti obradu zahtjeva naredbom `exit`. Ako nijedan u popisu jezika nije jezik koji poslužitelj podržava, vratit ćemo odgovor s HTTP statusnim kodom 406 Not Acceptable kao i ranije.

``` php
<?php

// ...

foreach ($languages_to_quality as $language => $quality) {
    if ($language == "hr") {
        header("Content-Language: hr");
        echo "<p>Pozdrav, svijete!</p>\n";
        exit;
    } elseif ($language == "sl") {
        header("Content-Language: sl");
        echo "<p>Pozdravljen, svet!</p>\n";
        exit;
    } elseif ($language == "cs") {
        header("Content-Language: cs");
        echo "<p>Ahoj světe!</p>\n";
        exit;
    } elseif ($language == "ru") {
        header("Content-Language: ru");
        echo "<p>Привет мир!</p>\n";
        exit;
    }
}

http_response_code(406);
echo "<p>Poslani popis jezika nije prihvatljiv.</p>\n";
```

Dodatno bismo se ovdje mogli pozabaviti prihvaćanjem i lokalnih varijanti pojedinih jezika kao i slanjem odgovora na nekom od podržanih jezika u slučaju kad je naveden znak zvjezdice, ali to prepuštamo čitatelju kao vježbu.

Čitav kod koji smo dosad opisali je oblika:

``` php
<?php

$languages = $_SERVER["HTTP_ACCEPT_LANGUAGE"];
$languages_split = explode(",", $languages);
$languages_trimmed = array_map("trim", $languages_split);

$languages_to_quality = [];
foreach ($languages_trimmed as $language_item) {
    $language_quality_pair = explode(";", $language_item);
    if (count($language_quality_pair) == 2) {
        $language = $language_quality_pair[0];
        $quality = (float) substr($language_quality_pair[1], 2);
        $languages_to_quality[$language] = $quality;
    }
    else {
        $languages_to_quality[$language_item] = 1.0;
    }
}

arsort($languages_to_quality);

foreach ($languages_to_quality as $language => $quality) {
    if ($language == "hr") {
        header("Content-Language: hr");
        echo "<p>Pozdrav, svijete!</p>\n";
        exit;
    } elseif ($language == "sl") {
        header("Content-Language: sl");
        echo "<p>Pozdravljen, svet!</p>\n";
        exit;
    } elseif ($language == "cs") {
        header("Content-Language: cs");
        echo "<p>Ahoj světe!</p>\n";
        exit;
    } elseif ($language == "ru") {
        header("Content-Language: ru");
        echo "<p>Привет мир!</p>\n";
        exit;
    }
}

http_response_code(406);
echo "<p>Poslani popis jezika nije prihvatljiv.</p>\n";
```

Isprobajmo kod:

``` shell
$ curl -v -H "Accept-Language: en;q=0.6, hr;q=0.8" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: en;q=0.6, hr;q=0.8
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 16:36:02 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Language: hr
< Content-Type: text/html; charset=UTF-8
<
<p>Pozdrav, svijete!</p>
* Closing connection 0

$ curl -v -H "Accept-Language: ru, en;q=0.6, hr;q=0.8" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: ru, en;q=0.6, hr;q=0.8
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 16:36:09 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Language: ru
< Content-Type: text/html; charset=UTF-8
<
<p>Привет мир!</p>
* Closing connection 0

$ curl -v -H "Accept-Language: de;q=0.9, fr;q=0.5" http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Accept-Language: de;q=0.9, fr;q=0.5
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 406 Not Acceptable
< Host: localhost:8000
< Date: Tue, 27 Apr 2021 16:36:20 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: text/html; charset=UTF-8
<
<p>Poslani popis jezika nije prihvatljiv.</p>
* Closing connection 0
```

## Prevođenje teksta

Funkcije `gettext()` ([dokumentacija](https://www.php.net/manual/en/function.setlocale.php)) i `setlocale()` ([dokumentacija](https://www.php.net/manual/en/function.setlocale.php)) omogućuju prevođenje teksta.

!!! todo
    Ovaj dio treba napisati.
