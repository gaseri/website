---
author: Vedran Miletić
---

# Implementacija REpresentational State Transfer (REST) aplikacijskog programskog sučelja u jeziku PHP

[REpresentational State Transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) je danas najkorištenija softverska arhitektura koja koristi HTTP i namijenjena je za aplikacije temeljene na web servisima ([više detalja o REST-u na MDN-u](https://developer.mozilla.org/en-US/docs/Glossary/REST)). Za web servis koji implementira REST kažemo da je [RESTful](https://restfulapi.net/); mi ćemo se u nastavku baviti takvim servisima, a aplikacijama koje se na njima temelje kasnije na predmetima iz područja razvoja web aplikacija. RESTful web servisi su [postali u praksi jako zastupljeni tijekom 2010-ih](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Evolution_of_HTTP#using_http_for_complex_applications).

REST je razvio Roy Fielding 2000. godine u [svom doktoratu](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm) i pritom kaže da se njegov razvoj temelji na HTTP-ovom modelu objekta iz 1994. godine (više detalja se može naći u [poglavlju 5 pod naslovom Representational State Transfer (REST)](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm) i [poglavlju 6 pod naslovom Experience and Evaluation](https://www.ics.uci.edu/~fielding/pubs/dissertation/evaluation.htm)).

Podsjetimo se da HTTP odredište zahtjeva naziva resursom i pritom ne definira detaljnije što resurs može biti ([više detalja o identificiranju resursa na webu može se pronaći na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Identifying_resources_on_the_Web)). Kako REST koristi HTTP, on specijalno koristi i pojam resursa iz HTTP-a.

Ključna apstrakcija podataka u REST-u je resurs; on može biti slika, dokument, zvuk, reprezentacija objekta iz stvarnosti (npr. osobe ili institucije), zbirka drugih resursa i slično. Stanje resursa u bilo kojem trenutku naziva se reprezentacija; ona sadrži podatke, metapodatke i poveznice prema drugim resursima. Za navođenje oblika reprezentacije koristi se već ranije spomenuti MIME tip.

Resursi se identificiraju korištenjem [Uniform Resource Identifiera (URI)](https://en.wikipedia.org/wiki/Uniform_Resource_Identifier). Recimo, možemo imati RESTful web servis koji putem URI-ja `/persons` (`http://localhost:8000/persons`) korištenjem HTTP metode GET omogućuje dohvaćanje popisa svih osoba koji je oblika:

``` json
{
  "count": 2,
  "results": [
    {
      "name": "Dennis MacAlistair Ritchie",
      "birth_year": 1941,
      "known_for": [
        "http://localhost:8000/technologies/1",
        "http://localhost:8000/technologies/2"
      ],
      "created": "2020-12-09T14:50:32+0100",
      "edited": "2020-12-20T21:14:07+0100",
      "url": "http://localhost:8000/persons/1"
    },
    {
      "name": "Kenneth Lane Thompson",
      "birth_year": 1943,
      "known_for": [
        "http://localhost:8000/technologies/2"
      ],
      "created": "2020-12-10T15:10:41+0100",
      "edited": "2020-12-20T21:17:55+0100",
      "url": "http://localhost:8000/persons/2"
    }
  ]
}
```

Za usporedbu, putem URI-ja `/persons/1` može se dohvatiti osoba s identifikatorom 1 i primljeni odgovor će biti oblika:

``` json
{
  "name": "Dennis MacAlistair Ritchie",
  "birth_year": 1941,
  "known_for": [
    "http://localhost:8000/technologies/1",
    "http://localhost:8000/technologies/2"
  ],
  "created": "2020-12-09T14:50:32+0100",
  "edited": "2020-12-20T21:14:07+0100",
  "url": "http://localhost:8000/persons/1"
}
```

Napomene radi, tehnologija s identifikatorom 1 je [programski jezik C](https://en.wikipedia.org/wiki/C_(programming_language)), a tehnologija s identifikatorom 2 je [Unix](https://en.wikipedia.org/wiki/Unix). Njihove reprezentacije možete zamisliti po želji.

U nastavku ćemo implementirati web servis koji na upite na te URI-je vraća odgovore s navedenim sadržajima i postavlja njihov MIME tip na `application/json`. Naravno, pored čitanja podataka, naš će web servis omogućavati i druge radnje nad podacima kao što su stvaranje, osvježavanje i brisanje (engl. [create, read, update, and delete](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete), kraće CRUD; [više detalja o CRUD-u na MDN-u](https://developer.mozilla.org/en-US/docs/Glossary/CRUD)).

## Radnje create, read, update i delete (CRUD) i pripadne HTTP metode

Korištenjem HTTP metoda i URI-ja možemo izvesti sve četiri navedene radnje nad podacima:

- stvaranje (engl. *create*) vrši se:

    - HTTP metodom POST na URI `/persons` stvara se osoba s prvim slobodnim identifikatorom ili
    - HTTP metodom PUT na URI `/persons/{id}`, pri čemu je `{id}` dotad neiskorišten identifikator osobe, stvara se nova osoba s navedenim identifikatorom

- čitanje (engl. *read*) vrši se:

    - HTTP metodom GET na URI `/persons` dohvaćaju se sve osobe ili
    - HTTP metodom GET na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe, dohvaća se osoba s navedenim identifikatorom

- osvježavanje (engl. *update*) vrši se:

    - HTTP metodom PUT na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe, osvježava se čitava osoba ili
    - HTTP metodom PATCH na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe, osvježavaju se dijelovi osobe

- brisanje (engl. *delete*) vrši se HTTP metodom DELETE na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe

REST ne definira način pohrane podataka i prepušta to implementaciji. Podaci mogu biti pohranjeni u relacijskoj bazi, nerelacijskoj bazi ili datoteci, a mi ćemo ih pohranjivati u datoteci radi jednostavnosti.

!!! note
    Implementacija koju koristimo u nastavku ne koristi gotove komponente (npr. [Symfony](https://symfony.com/)) i okvire (npr. [Lumen](https://lumen.laravel.com/)) jer je naš cilj ovdje razumijeti kako radi HTTP i kako implementirati REST, a ne razviti veliku praktično primjenjivu aplikaciju koju je kasnije potrebno održavati i proširivati. Primjer sličnog pristupa tumačenju pojmova u domeni weba je [Shopifyjev](https://www.shopify.com/) članak [How to Build a Web App with and without Rails Libraries](https://shopify.engineering/building-web-app-ruby-rails) u okviru kojeg je dan prikaz implementacije aplikacije u [jeziku Ruby](https://www.ruby-lang.org/) bez korištenja [okvira Rails](https://rubyonrails.org/).

## Čitanje podataka

Recimo da u istom direktoriju `public` gdje se nalazi `index.php` postoji datoteka `persons.json` sadržaja:

``` json
{
  "1": {
    "name": "Dennis MacAlistair Ritchie",
    "birth_year": 1941,
    "known_for": [
      "http://localhost:8000/technologies/1",
      "http://localhost:8000/technologies/2"
    ],
    "created": "2020-12-09T14:50:32+0100",
    "edited": "2020-12-20T21:14:07+0100",
    "url": "http://localhost:8000/persons/1"
  },
  "2": {
    "name": "Kenneth Lane Thompson",
    "birth_year": 1943,
    "known_for": [
      "http://localhost:8000/technologies/2"
    ],
    "created": "2020-12-10T15:10:41+0100",
    "edited": "2020-12-20T21:17:55+0100",
    "url": "http://localhost:8000/persons/2"
  }
}
```

!!! warning
    Iz sigurnosne perspektive, spremanje podataka u datoteku koja se nalazi na mjestu s kojeg je klijenti mogu dohvatiti HTTP zahtjevima (u ovom slučaju naredbom `curl http://localhost:8000/persons.json`) je katastrofalno loša praksa. U procesu učenja datoteku pohranjujemo na tom mjestu samo radi jednostavnosti.

Mi, naravno, ne možemo biti sigurni da će ta datoteka uvijek postojati pa ćemo kod pokretanja funkcijom `file_exists()` ([dokumentacija](https://www.php.net/manual/en/function.file-exists.php)) provjeriti njeno postojanje. Ako postoji, uzet ćemo da ona ima spremljene podatke od ranije te je učitati funkcijom `file_get_contents()` ([dokumentacija](https://www.php.net/manual/en/function.file-get-contents.php)) pa funkcijom `json_decode()` pretvoriti dobiveni sadržaj u obliku JSON u polje s podacima o osobama.

Ako datoteka ne postoji, popis osoba mogli bismo inicijalizirati na prazno polje. U tom slučaju bi prva osoba koju kasnije dodamo bila postavljena u polje kao element na indeksu 0 i imala isti taj broj kao identifikator za dohvaćanje putem URL-a. Kako ljudi intuitivno preferiraju brojati od 1, dodat ćemo u to polje proizvoljni element da zauzme indeks 0 pa će prva kasnije dodana osoba biti postavljena na indeks 1. Kako element na indeksu 0 nećemo koristiti, njegov nam sadržaj nije bitan pa možemo iskoristiti vrijednost `NULL`.

!!! note
    Uočimo da prazno polje, vrijednost `NULL` i polje koje sadrži vrijednost `NULL` nisu iste vrijednosti: prvo je polje bez elemenata, drugo nije polje, a treće je polje s jednim elementom; tip tog elementa ne mijenja činjenicu da polje ima element.

Kod u datoteci `index.php` je oblika:

``` php
<?php

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}
```

### Dohvaćanje svih osoba

Klijentu koji vrši dohvaćanje svih osoba HTTP metodom GET na URI `/persons` potrebno je vratiti dva podatka, broj osoba u odgovoru i podatke o osobama. Broj osoba ćemo dobiti funkcijom `count()` ([dokumentacija](https://www.php.net/manual/en/function.count.php)), a podatke o osobama možemo iz asocijativnog polja (proizvoljni indeksi, u našem slučaju identifikatori) pretvoriti u oblik indeksiranog polja (indeksi su cijeli brojevi od nule nadalje) funkcijom `array_values()` ([dokumentacija](https://www.php.net/manual/en/function.array-values.php)).

Time smo formirali odgovor i ostaje nam samo pretvoriti ga u JSON funkcijom `json_encode()` te ga ispisati. Kako bismo olakšali razlikovanje neispravnih zahtjeva od zahtjeva koji rezultiraju praznim odgovorom, postavit ćemo poslužitelj da na sve zahtjeve koji nisu eksplicitno dozvoljeni (na način kako smo upravo dozvolili HTTP zahtjev metodom GET na `/persons`) vraća HTTP statusni kod [400 Bad Request](https://http.cat/400) ([više detalja o HTTP statusnom kodu 400 Bad Request na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400)). Kod je oblika:

``` php
<?php

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["REQUEST_URI"] == "/persons") {
    $response_body_array = ["count" => count($persons), "results" => array_values($persons)];
    $response_body = json_encode($response_body_array);
    echo $response_body;
} else {
    http_response_code(400);
}
```

Podsjetimo se da je HTTP metoda GET zadana pa isprobajmo dohvaćanje URI-ja `/persons` na način:

``` shell
$ curl -v localhost:8000/persons
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /persons HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 15 Feb 2021 14:25:54 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
{"count":2,"results":[{"name":"Dennis MacAlistair Ritchie","birth_year":1941,"known_for":["http:\/\/localhost:8000\/technologies\/1","http:\/\/localhost:8000\/technologies\/2"],"created":"2020-12-09T14:50:32+0100","edited":"2020-12-20T21:14:07+0100","url":"http:\/\/localhost:8000\/persons\/1"},{"name":"Kenneth Lane Thompson","birth_year":1943,"known_for":["http:\/\/localhost:8000\/technologies\/2"],"created":"2020-12-10T15:10:41+0100","edited":"2020-12-20T21:17:55+0100","url":"http:\/\/localhost:8000\/persons\/2"}]}

$ curl -v http://localhost:8000/scientists
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /scientists HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 400 Bad Request
< Host: localhost:8000
< Date: Tue, 23 Mar 2021 23:38:33 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Ako želimo da odgovori izgledaju lijepo, možemo u pozivu funkcije `json_encode()` dodati `JSON_PRETTY_PRINT` na isti način kao i ranije.

### Dohvaćanje pojedine osobe

Uočimo kako smo kod dohvaćanja svih osoba imali samo jedan valjani URI, a kod dohvaćanja pojedine osobe imamo beskonačno mnogo valjanih URI-ja i ne možemo ih navesti pojedinačno pa provjeriti operatorom `==`. U slučaju kao što je ovaj koriste se regularni izrazi kojima možemo navesti oblik URI-ja koji nam odgovara. To ćemo učiniti funkcijom `preg_match()` ([dokumentacija](https://www.php.net/manual/en/function.preg-match.php)), a regularni izraz koji ćemo koristiti je `/^\/persons\/[1-9][0-9]*$/`. Raščlanimo njegove elemente:

- `/` -- znak za početak regularnog izraza
- `^` -- znak za početak znakovnog niza; želimo da se naš navedeni regularni izraz poklapa s danim znakovnim nizom od početka
- `\/` -- znak kose crte ima specijalno značenje pa, kad nam doslovno treba znak kose crte, unosimo ga na ovaj način
- `persons` -- doslovni tekst, niz znakova persons
- `\/` -- isto kao iznad
- `[1-9]` -- jedna znamenka od 1 do 9
- `[0-9]*` -- znamenka od 0 do 9 ponovljena proizvoljan broj puta, uključujući i nijednom
- `$` -- znak za kraj znakovnog niza; želimo da se naš navedeni regularni izraz poklapa s danim znakovnim nizom do kraja
- `/` -- znak za kraj regularnog izraza

Uočimo kako smo ovim regularnim izrazom učinili da su valjani npr. izrazi `/persons/9817` i `/persons/5`, a da nisu valjani npr. izrazi `/persons/007` i `/persons/Vespasian` jer identifikator osobe mora biti broj čija je prva znamenka od 1 do 9. Kod je oblika:

``` php
<?php

// ...

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["REQUEST_URI"] == "/persons") {
  // ...
} else if ($_SERVER["REQUEST_METHOD"] == "GET" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
  // ...
}
```

Kako bismo razdijelili URI, iskoristit ćemo funkciju `explode()` ([dokumentacija](https://www.php.net/manual/en/function.explode.php)); primjerice, eksplozijom izraza `/persons/1` po znaku kose crte (`/`) dobivamo polje `["", "persons", "1"]`. Dobiveni elementi polja su svi znakovni nizovi, ali PHP će ih po potrebi automatski pretvoriti u druge odgovarajuće tipove podataka ([više detalja o automatskoj pretvorbi tipova u službenoj dokumentaciji](https://www.php.net/manual/en/language.types.type-juggling.php)). Konkretno, treći element ovog polja, znakovni niz `"1"` postat će cijeli broj `1` kad ga iskoristimo za dohvaćanje osobe iz polja `$persons` preko njenog identifikatora.

Naposlijetku, kod stvaranja odgovora moramo se pobrinuti da postavimo HTTP statusni kod 404 Not Found i vratimo prazno tijelo odgovora ako ne postoji osoba s traženim identifikatorom. Provjeru ćemo izvršiti funkcijom `array_key_exists()` ([dokumentacija](https://www.php.net/manual/en/function.array-key-exists.php)). Dopunimo kod od ranije na način:

``` php
<?php

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["REQUEST_URI"] == "/persons") {
    $response_body_array = ["count" => count($persons), "results" => array_values($persons)];
    $response_body = json_encode($response_body_array);
    echo $response_body;
} else if ($_SERVER["REQUEST_METHOD"] == "GET" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        $response_body_object = $persons[$id];
        $response_body = json_encode($response_body_object);
        echo $response_body;
    } else {
        http_response_code(404);
    }
} else {
    http_response_code(400);
}
```

Isprobajmo dohvaćanje nekoliko URI-ja:

``` shell
$ curl -v http://localhost:8000/persons/1
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /persons/1 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Tue, 23 Mar 2021 23:41:08 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
{"name":"Dennis MacAlistair Ritchie","birth_year":1941,"known_for":["http:\/\/localhost:8000\/technologies\/1","http:\/\/localhost:8000\/technologies\/2"],"created":"2020-12-09T14:50:32+0100","edited":"2020-12-20T21:14:07+0100","url":"http:\/\/localhost:8000\/persons\/1"}

$ curl -v http://localhost:8000/persons/599
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /persons/599 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 404 Not Found
< Host: localhost:8000
< Date: Tue, 23 Mar 2021 23:41:10 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0

$ curl -v http://localhost:8000/persons/007
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /persons/007 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 400 Bad Request
< Host: localhost:8000
< Date: Tue, 23 Mar 2021 23:41:17 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Za stvaranje novih osoba moramo prvo naučiti obraditi tijelo zahtjeva.

## Obrada tijela zahtjeva i stvaranje tijela odgovora

Tijelo HTTP zahtjeva koje nam je u jeziku PHP dostupno na već ranije korištenoj putanji `php://input`. Kako se radi o putanjama s kojima se radi kao s datotekama, sadržaj tijela zahtjeva dohvatit ćemo već ranije korištenom funkcijom `file_get_contents()`, a sadržaj tijela odgovora na putanji `php://output` puniti analognom funkcijom `file_put_contents()` ([dokumentacija](https://www.php.net/manual/en/function.file-put-contents.php)).

Napravimo poslužitelj koji na zahtjev tipa POST sa sadržajem `Kako ide?` koji je MIME tipa text/plain odgovara sadržajem `A evo, dobro.` također MIME tipa text/plain. Kod je oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["HTTP_CONTENT_TYPE"] == "text/plain") {
    $request_body = file_get_contents("php://input");
    if ($request_body == "Kako ide?") {
        header("Content-Type: text/plain");
        $response_body = "A evo, dobro.";
        file_put_contents("php://output", $response_body);
    }
}
```

Napravimo zahtjev cURL-om metodom POST MIME tipa `text/plain` sa sadržajem tijela zahtjeva `"Kako ide?"`:

``` shell
$ curl -v -X POST -H "Content-Type: text/plain" -d "Kako ide?" http://localhost:8000/
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Content-Type: text/plain
> Content-Length: 9
>
* upload completely sent off: 9 out of 9 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 19:20:13 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/plain;charset=UTF-8
<
* Closing connection 0
A evo, dobro.
```

## Stvaranje podataka

Podsjetimo se da je HTTP protokol koji ne održava stanje (engl. *stateless*) pa se svaki zahtjev obrađuje neovisno o prethodnima.

U konkretnom slučaju to znači da eventualna dopuna polja `$persons` još jednom osobom ili izmjena podataka neke od postojećih osoba izvedena u jednom zahtjevu neće biti vidljiva u sljedećem zahtjevu. Razlog tome je što to polje i time svi podaci u njemu prestaju postojati brisanjem sadržaja memorije nakon slanja odgovora na primljeni zahtjev. Kako se prilikom obrade svakog zahtjeva isti kod izvršava ispočetka i pritom inicijalizira polje `$persons` ili iz datoteke ili kao polje s elementom `NULL`, podaci dodani ili promijenjeni u memoriji kod obrane prethodnog zahtjeva neće biti vidljivi.

Kako bi nove osobe ili izmjene osoba bile trajno sačuvane, potrebno je nakon obrade zahtjeva i slanja odgovora promijenjeni sadržaj spremiti iz memorije u datoteku koju koristimo kao izvor podataka. Ponovno ćemo iskoristiti serijalizaciju u oblik JSON funkcijom `json_encode()` te pohraniti dobiveni JSON zapis u datoteku funkcijom `file_put_contents()`. Kod je oblika:

``` php
<?php

// ...

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Poslužitelj očekuje od klijenta primiti zahtjeve za stvaranjem podataka čiji je sadržaj u formatu JSON i koji imaju u zaglavlju naveden odgovarajući MIME tip. U suprotnom, tj.ako HTTP zaglavlje `Content-Type` ne postoji u zahtjevu ili ako postoji i ima vrijednost koja nije application/json, poslužitelj će postaviti HTTP statusni kod odgovora na [415 Unsupported Media Type](https://http.cat/415) ([više detalja o HTTP statusnom kodu 415 Unsupported Media Type na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/415)) i potom prekinuti obradu zahtjeva naredbom `exit` ([dokumentacija](https://www.php.net/manual/en/function.exit.php)).

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

// ...
```

Sad smo izveli sve potrebne pripreme pa možemo naučiti stvarati (engl. *create*) podatke metodama POST i PUT.

### Stvaranje podataka HTTP metodom POST

Poslužitelj prihvaća zahtjeve HTTP metodom POST na URI `/persons` kojima stvara novu osobu s podacima koje klijent navede i prvim slobodnim identifikatorom. Za dodavanje nove osobe u polje `$persons` iskoristit ćemo operator za pristup elementima polja (`[]`, [dokumentacija](https://www.php.net/manual/en/language.types.array.php)) koji će, u slučaju da indeks nije naveden, postaviti element na prvi sljedeći slobodni indeks, a taj indeks će kod pretvorbe u JSON postati identifikator osobe. Nakon uspješnog stvaranja osobe poslužitelj će postaviti statusni kod odgovora na [201 Created](https://http.cat/201) ([više detalja o HTTP statusnom kodu 201 Created na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/201)).

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    $request_body = file_get_contents("php://input");
    $person = json_decode($request_body, true);
    $persons[] = $person;
    http_response_code(201);
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Poslužitelj trenutno prihvaća od klijenta bilo kakav sadržaj za koji klijent tvrdi da je u obliku JSON, čak i ako sadržaj to nije. U tom slučaju dekodiranje ne uspijeva pa će funkcija `json_decode()` vratiti `NULL` i ta "osoba" će biti dodana pod sljedećim slobodnim identifikatorom, a klijentu će biti poslana poruka o uspjehu. To očito nije željeni način rada pa je potrebno provjeriti je li primjeni JSON sadržaj uspješno dekodiran prije dodavanja osobe i vratiti HTTP statusni kod 400 Bad Request ako nije. Navedeno ćemo izvesti kodom:

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    $request_body = file_get_contents("php://input");
    $person = json_decode($request_body, true);
    if ($person != NULL) {
        $persons[] = $person;
        http_response_code(201);
    } else {
        http_response_code(400);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Klijent sada može dodati kao osobu bilo koji valjani JSON zapis, npr. `{"institution": "Odjel za informatiku", "address": "Radmile Matejčić 2"}` i taj će podatak biti spremljen. Kako bismo to izbjegli, izvest ćemo dvije promjene:

- Poslužitelj će provjeriti da su navedeni podaci `"name"` tipa `string`, `"birth_year"` tipa `int` i `"known_for"` tipa `array` (u praksi bi trebalo izvršiti i provjeru pojedinih elemenata polja `"known_for"`, ali ovdje radi jednostavnosti to nećemo raditi). Tip podatka saznat ćemo funkcijom `gettype()` ([dokumentacija](https://www.php.net/manual/en/function.gettype.php)).
- Iz klijentskog sadržaja poslužitelj će kod stvaranja osobe prihvatiti samo podatke `"name"`, `"birth_year"` i `"known_for"`, a sve ostale zanemariti. Primjerice, ako klijent pošalje JSON zapis `{"name": "Michael J. Karels", "worked_on_bsd_unix": true}`, onda će podatak `"worked_on_bsd_unix"` biti zanemaren.

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    $request_body = file_get_contents("php://input");
    $person = json_decode($request_body, true);
    if ($person != NULL &&
        array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
        array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
        array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array") {
        $persons[] = ["name" => $person["name"],
                      "birth_year" => $person["birth_year"],
                      "known_for" => $person["known_for"]];
        http_response_code(201);
    } else {
        http_response_code(400);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Naposlijetku nam ostaje dodati još datum i vrijeme stvaranja i uređivanja (`"created"` i `"edited"`) te URL zapisa (`"url"`).  Datum i vrijeme stvaranja i uređivanja će ovdje biti isto jer tek stvaramo osobu, a oblik u kojem su ta vremena zapisana u našim podacima je [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). U jeziku PHP je moguće dohvatiti trenutno vrijeme i datum u nekom obliku funkcijom `date()` ([dokumentacija](https://www.php.net/manual/en/function.date.php)), a [popis podržanih oblika](https://www.php.net/manual/en/class.datetimeinterface.php) uključuje `DATE_ISO8601` koji nam odgovara.

Kako bismo dobili URL, iskoristit ćemo funkciju `array_key_last()` ([dokumentacija](https://www.php.net/manual/en/function.array-key-last.php)) da saznamo identifikator posljednjeg elementa polja, a to je onaj upravo stvoren. Zatim ćemo operatorom za spajanje znakovnih nizova (`.`) spojiti zadani početak URL-a `http://localhost:8000/persons/` s dobivenim identifikatorom. PHP će pritom za nas izvesti pretvorbu identifikatora iz cjelobrojnog tipa u znakovni niz pa o tome ne moramo brinuti.

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    $request_body = file_get_contents("php://input");
    $person = json_decode($request_body, true);
    if ($person != NULL &&
        array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
        array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
        array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array") {
    $date_time = date(DATE_ISO8601);
        $persons[] = ["name" => $person["name"],
                      "birth_year" => $person["birth_year"],
                      "known_for" => $person["known_for"],
                      "created" => $date_time,
                      "edited" => $date_time];
        $id = array_key_last($persons);
        $persons[$id]["url"] = "http://localhost:8000/persons/" . $id;
        http_response_code(201);
    } else {
        http_response_code(400);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Isprobajmo stvaranje osobe na način koji smo upravo razvili. Za početak se uvjerimo da zahtjevi čiji sadržaj nije tipa application/json dobivaju odgovor sa statusnim kodom 415 Unsupported Media Type:

``` shell
$ curl -v -X POST http://localhost:8000/persons
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST /persons HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 415 Unsupported Media Type
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 20:40:10 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: text/html; charset=UTF-8
<
* Closing connection 0
```

Zatim se uvjerimo da poslužitelj vraća statusni kod 400 Bad Request za sadržaj koji tvrdi da je JSON, a nije:

``` shell
$ curl -v -X POST -H 'Content-Type: application/json' -d 'Ovaj sadržaj je 100% JSON' http://localhost:8000/persons
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST /persons HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 26
>
* upload completely sent off: 26 out of 26 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 400 Bad Request
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 20:42:27 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Rezultat je isti kad sadržaj poslanog zahtjeva nema odgovarajuće podatke:

``` shell
$ curl -v -X POST -H 'Content-Type: application/json' -d '{"institution": "Odjel za informatiku", "address": "Radmile Matejčić 2"}' http://localhost:8000/persons
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST /persons HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 74
>
* upload completely sent off: 74 out of 74 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 400 Bad Request
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 20:44:30 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Naposlijetku, ako su svi podaci navedeni, zahtjev je prihvaćen i stvara se nova osoba:

``` shell
$ curl -v -X POST -H 'Content-Type: application/json' -d '{"name": "Donald Ervin Knuth", "birth_year": 1938, "known_for": ["http://localhost:8000/technologies/3", "http://localhost:8000/technologies/4"]}' http://localhost:8000/persons
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST /persons HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 142
>
* upload completely sent off: 142 out of 142 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 201 Created
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 20:55:37 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Tehnologije su ovdje [TeX](https://en.wikipedia.org/wiki/TeX) i [Metafont](https://en.wikipedia.org/wiki/Metafont). Kao i ranije, zamislimo da već postoje i da o njima ne moramo brinuti.

Kako bismo se uvjerili da je dodavanje bilo uspješno, možemo ručno pogledati sadržaj datoteke `persons.json`.

### Stvaranje podataka HTTP metodom PUT

Poslužitelj prihvaća zahtjev HTTP metodom PUT na URI `/persons/{id}`, pri čemu je `{id}` dotad neiskorišten identifikator osobe i njime se stvara nova osoba s navedenim identifikatorom. Po primitku zahtjeva od klijenta provjerit će postoji li već objekt s danim identifikatorom:

- ako ne postoji, stvorit će objekt na gotovo isti način kao kod korištenja metode POST: jedina je razlika što u ovom slučaju identifikator ne dohvaća kao posljednji ključ u polju `$persons`, već ga ima navedenog u URL-u zahtjeva,
- ako postoji, bit će vraćen odgovor sa statusnim kodom 403 Forbidden.

Čitav kod je oblika:

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    // ...
} else if ($_SERVER["REQUEST_METHOD"] == "PUT" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (! array_key_exists($id, $persons)) {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array") {
        $date_time = date(DATE_ISO8601);
            $persons[$id] = ["name" => $person["name"],
                             "birth_year" => $person["birth_year"],
                             "known_for" => $person["known_for"],
                             "created" => $date_time,
                             "edited" => $date_time,
                             "url" => "http://localhost:8000/persons/" . $id];
            http_response_code(201);
        } else {
            http_response_code(400);
        }
    } else {
        http_response_code(403);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Isprobajmo stvaranje osobe pod rednim brojem 4 na način koji smo upravo razvili.

``` shell
$ curl -v -X PUT -H 'Content-Type: application/json' -d '{"name": "Grace Brewster Murray Hopper", "birth_year": 1906, "known_for": ["http://localhost:8000/technologies/5"]}' http://localhost:8000/persons/4
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> PUT /persons/4 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 115
>
* upload completely sent off: 115 out of 115 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 201 Created
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 21:04:27 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Tehnologija 5 je [COBOL](https://en.wikipedia.org/wiki/COBOL).

## Osvježavanje podataka

Osvježavanje (engl. *update*) podataka vrši se HTTP metodama PUT i PATCH.

### Osvježavanje podataka HTTP metodom PUT

HTTP metodom PUT na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe, osvježava se čitav objekt osobe. U tom smislu imamo gotovo isti pristup kao kod stvaranja osobe, ali ovaj put je logika obrnuta:

- ako postoji osoba s danim identifikatorom i poslani objekt sadrži sve podatke, podaci i njihovo vrijeme posljednje promjene će biti osvježeni,
- ako ne postoji osoba s tim identifikatorom, bit će vraćen odgovor sa statusnim kodom 403 Forbidden.

Zasad radimo dio po dio; u konačnici ćemo ujediniti te dvije situacije (stvaranje i osvježavanje podataka HTTP metodom PUT) i eliminirat ćemo potrebu za vraćanjem odgovora sa statusom 403 Forbidden.

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "PUT" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array" &&
            array_key_exists("created", $person) && gettype($person["created"]) == "string" &&
            array_key_exists("url", $person) && gettype($person["url"]) == "string") {
        $date_time = date(DATE_ISO8601);
            $persons[$id] = ["name" => $person["name"],
                             "birth_year" => $person["birth_year"],
                             "known_for" => $person["known_for"],
                             "created" => $person["created"],
                             "edited" => $date_time,
                             "url" => $person["url"]];
            http_response_code(200);
        } else {
            http_response_code(400);
        }
    } else {
        http_response_code(403);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

### Osvježavanje podataka HTTP metodom PATCH

HTTP metodom PATCH na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe, osvježavaju se dijelovi objekta osobe. Mi ćemo ovdje, radi jednostavnosti, dozvoliti i osvježavanje čitavog objekta. Provjeru je li u zahtjevu unesen neki od podataka koji se mogu osvježiti izvest ćemo operator ili (`||`) pa osvježiit samo unesene podatke i vrijeme zadnje promjene. Kod je oblika:

``` php
<?php

if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
    http_response_code(415);
    exit;
}

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "PATCH" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            (array_key_exists("name", $person) && gettype($person["name"]) == "string" ||
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" ||
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array")) {
        if (array_key_exists("name", $person)) {
                $persons[$id]["name"] = $person["name"];
        }
        if (array_key_exists("birth_year", $person)) {
                $persons[$id]["birth_year"] = $person["birth_year"];
        }
        if (array_key_exists("known_for", $person)) {
                $persons[$id]["known_for"] = $person["known_for"];
        }
        $date_time = date(DATE_ISO8601);
            $persons[$id]["edited"] = $date_time;
            http_response_code(200);
        } else {
            http_response_code(400);
        }
    } else {
        http_response_code(403);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Isprobajmo osvježavanje osobe na način koji smo upravo razvili. Osvježimo popis tehnologija koje zbog koji je [Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth) poznat tako da dodamo tehnologiju 6, odnosno [Computer Modern](https://en.wikipedia.org/wiki/Computer_Modern):

``` shell
$ curl -v -X PATCH -H 'Content-Type: application/json' -d '{"known_for": ["http://localhost:8000/technologies/3", "http://localhost:8000/technologies/4", "http://localhost:8000/technologies/6"]}' http://localhost:8000/persons/3
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> PATCH /persons/3 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 135
>
* upload completely sent off: 135 out of 135 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 21:21:51 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

## Brisanje podataka

Brisanje (engl. *delete*) podataka vrši se HTTP metodom DELETE na URI `/persons/{id}`, pri čemu je `{id}` identifikator osobe koju želim obrisati. Brisanje podataka iz polja `$persons` izvest ćemo funkcijom `unset()` ([dokumentacija](https://www.php.net/manual/en/function.unset.php)). Ako je brisanje uspješno, odgovor će imati HTTP statusni kod [410 Gone](https://http.cat/410) ([više detalja o HTTP statusnom kodu 410 Gone na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/410)). Ako brisanje nije uspješno zato što ne postoji osoba s danim identifikatorom, vratit ćemo odgovor sa statusnim kodom 404 Not Found.

``` php
<?php

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "DELETE" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        unset($persons[$id]);
        http_response_code(410);
    } else {
        http_response_code(404);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Isprobajmo brisanje osobe na način koji smo upravo razvili, prvo tako da pokušamo izbrisati osobu koja ne postoji:

``` shell
$ curl -v -X DELETE http://localhost:8000/persons/5
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> DELETE /persons/5 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 404 Not Found
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 21:33:58 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Izbrišimo sad osobu koju smo ranije dodali:

``` shell
$ curl -v -X DELETE http://localhost:8000/persons/3
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> DELETE /persons/3 HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.74.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 410 Gone
< Host: localhost:8000
< Date: Sat, 03 Apr 2021 21:34:09 GMT
< Connection: close
< X-Powered-By: PHP/7.4.15
< Content-Type: application/json
<
* Closing connection 0
```

Kako nemamo kod za čitanje podataka, trenutno je jedini način da se uvjerimo da je brisanje bilo uspješno da ručno pogledamo sadržaj datoteke `persons.json`. U nastavku je dan kod koji spaja sve dosad razvijene fragmente tako da poslužitelj nudi mogućnost izvođenja svih CRUD operacija.

## Kod poslužitelja koji izvodi sve CRUD operacije

``` php
<?php

$datoteka = "persons.json";
if (file_exists($datoteka)) {
    $j = file_get_contents($datoteka);
    $persons = json_decode($j, true);
} else {
    $persons = [NULL];
}

header("Content-Type: application/json");

if ($_SERVER["REQUEST_METHOD"] == "GET" && $_SERVER["REQUEST_URI"] == "/persons") {
    $response_body_array = ["count" => count($persons), "results" => array_values($persons)];
    $response_body = json_encode($response_body_array);
    echo $response_body;
} else if ($_SERVER["REQUEST_METHOD"] == "GET" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        $response_body_object = $persons[$id];
        $response_body = json_encode($response_body_object);
        echo $response_body;
    } else {
        http_response_code(404);
    }
} else if ($_SERVER["REQUEST_METHOD"] == "POST" && $_SERVER["REQUEST_URI"] == "/persons") {
    if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
        http_response_code(415);
        exit;
    }
    $request_body = file_get_contents("php://input");
    $person = json_decode($request_body, true);
    if ($person != NULL &&
        array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
        array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
        array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array") {
    $date_time = date(DATE_ISO8601);
        $persons[] = ["name" => $person["name"],
                      "birth_year" => $person["birth_year"],
                      "known_for" => $person["known_for"],
                      "created" => $date_time,
                      "edited" => $date_time];
        $id = array_key_last($persons);
        $persons[$id]["url"] = "http://localhost:8000/persons/" . $id;
        http_response_code(201);
    } else {
        http_response_code(400);
    }
} else if ($_SERVER["REQUEST_METHOD"] == "PUT" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
        http_response_code(415);
        exit;
    }
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (! array_key_exists($id, $persons)) {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array") {
        $date_time = date(DATE_ISO8601);
            $persons[$id] = ["name" => $person["name"],
                             "birth_year" => $person["birth_year"],
                             "known_for" => $person["known_for"],
                             "created" => $date_time,
                             "edited" => $date_time,
                             "url" => "http://localhost:8000/persons/" . $id];
            http_response_code(201);
        } else {
            http_response_code(400);
        }
    } else {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            array_key_exists("name", $person) && gettype($person["name"]) == "string" &&
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" &&
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array" &&
            array_key_exists("created", $person) && gettype($person["created"]) == "string" &&
            array_key_exists("url", $person) && gettype($person["url"]) == "string") {
        $date_time = date(DATE_ISO8601);
            $persons[$id] = ["name" => $person["name"],
                             "birth_year" => $person["birth_year"],
                             "known_for" => $person["known_for"],
                             "created" => $person["created"],
                             "edited" => $date_time,
                             "url" => $person["url"]];
            http_response_code(200);
        } else {
            http_response_code(400);
        }
    }
} else if ($_SERVER["REQUEST_METHOD"] == "PATCH" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    if (! array_key_exists("HTTP_CONTENT_TYPE", $_SERVER) || $_SERVER["HTTP_CONTENT_TYPE"] != "application/json") {
        http_response_code(415);
        exit;
    }
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        $request_body = file_get_contents("php://input");
        $person = json_decode($request_body, true);
        if ($person != NULL &&
            (array_key_exists("name", $person) && gettype($person["name"]) == "string" ||
            array_key_exists("birth_year", $person) && gettype($person["birth_year"]) == "integer" ||
            array_key_exists("known_for", $person) && gettype($person["known_for"]) == "array")) {
        if (array_key_exists("name", $person)) {
                $persons[$id]["name"] = $person["name"];
        }
        if (array_key_exists("birth_year", $person)) {
                $persons[$id]["birth_year"] = $person["birth_year"];
        }
        if (array_key_exists("known_for", $person)) {
                $persons[$id]["known_for"] = $person["known_for"];
        }
        $date_time = date(DATE_ISO8601);
            $persons[$id]["edited"] = $date_time;
            http_response_code(200);
        } else {
            http_response_code(400);
        }
    } else {
        http_response_code(403);
    }
} else if ($_SERVER["REQUEST_METHOD"] == "DELETE" && preg_match("/^\/persons\/[1-9][0-9]*$/", $_SERVER["REQUEST_URI"])) {
    $uri_parts = explode("/", $_SERVER["REQUEST_URI"]);
    $id = $uri_parts[2];

    if (array_key_exists($id, $persons)) {
        unset($persons[$id]);
        http_response_code(410);
    } else {
        http_response_code(404);
    }
} else {
    http_response_code(400);
}

$j = json_encode($persons);
file_put_contents($datoteka, $j);
```

Dodatno je moguće pojedine dijelove koda izvući u funkcije i na taj način skratiti konačni rezultat. Takvi postupci se ne tiču načina rada HTTP-a pa se njima nećemo ovdje baviti.
