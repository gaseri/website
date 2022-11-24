---
author: Vedran Miletić
---

# Postavljanje datoteka na poslužitelj u jeziku PHP

Jedna od funkcionalnosti HTTP poslužitelja je primanje datoteka koje korisnici žele postaviti, primjerice fotografija i audiovizualnih snimaka koje postavljaju na društvene mreže. Interpreter PHP-a podržava [postavljanje datoteka korištenjem HTTP metoda POST i PUT](https://www.php.net/manual/en/features.file-upload.php). U nastavku prezentiramo oba načina te korištenje HTTP metode DELETE koje se koristi za brisanje datoteke.

## Postavljanje datoteke metodom POST

Slično kako na poslužitelj metodom POST šaljemo podatke, [možemo i postavljati datoteke](https://www.php.net/manual/en/features.file-upload.post-method.php). U cURL-u to činimo parametrom `--form`, odnosno `-F`, koji se za postavljanje datoteke imena `pokloni.txt` pod ključem `popis_poklona` na poslužitelj na adresi `http://localhost:8000/` koristi na način:

``` shell
$ curl -F 'popis_poklona=@pokloni.txt' http://localhost:8000/
```

Na poslužiteljskoj strani u PHP-ovom polju `$_FILES` ([dokumentacija](https://www.php.net/manual/en/reserved.variables.files.php)) pojavit će se unos pod ključem `"popis_poklona"` koji predstavlja datoteku `pokloni.txt`. Iskoristit ćemo funkciju `move_uploaded_file()` ([dokumentacija](https://www.php.net/manual/en/function.move-uploaded-file.php)) da pomaknemo postavljenu datoteku s njezinog privremenog mjesta na mjesto na kojem želimo da bude. Trebat će nam i funkcija `getcwd()` ([dokumentacija](https://www.php.net/manual/en/function.getcwd.php)) kojom ćemo dohvatiti radni direktorij ugrađenog web poslužitelja u koji ćemo datoteke i spremati. Kod poslužitelja je sad oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $webroot = getcwd();
    move_uploaded_file($_FILES["popis_poklona"]["tmp_name"], $webroot . "/" . "popis.txt");
    http_response_code(201);
}
```

Nakon uspješnog postavljanja datoteke poslužitelj će postaviti statusni kod odgovora na [201 Created](https://http.cat/201) ([više detalja o HTTP statusnom kodu 201 Created na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/201)).

Ovdje smo operatorom konkatenacije (znak točke, `.`) ([dokumentacija](https://www.php.net/manual/en/language.operators.string.php)) spojili dijelove putanje. Primjerice, ako varijabla `$webroot` ima vrijednost `/home/ahilej/public`, onda `"/home/ahilej/public" . "/" . "popis.txt"` postaje `"/home/ahilej/public/popis.txt"`. Postavimo datoteku na poslužitelj:

``` shell
$ curl -v -F 'popis_poklona=@pokloni.txt' http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Content-Length: 59347
> Content-Type: multipart/form-data; boundary=------------------------715051f75fe2b85e
>
* We are completely uploaded and fine
* Mark bundle as not supporting multiuse
< HTTP/1.1 201 Created
< Host: localhost:8000
< Date: Mon, 28 Dec 2020 21:58:46 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
* Closing connection 0
```

Lako se možemo uvjeriti da su naša datoteka `pokloni.txt` i postavljena datoteka `popis.txt` istog sadržaja. Uočimo također kako je HTTP zaglavlje `Content-Type` u zahtjevu sada postavljeno na vrijednost `multipart/form-data` (umjesto na vrijednost `application/x-www-form-urlencoded` koju smo imali kod slanja podataka), što je MIME tip poslanih podataka kod postavljanja datoteke na poslužitelj. Na isti način, ali vjerojatno uz nešto bolji dizajn korisničkog sučelja, radi forma za promjenu slike profila na popularnim društvenim mrežama.

### Postavljanje datoteke s danim imenom

Na ovaj način preimenovat ćemo svaku datoteku koju klijent postavi na ime koje zadamo. Želimo li uzeti ime koje je datoteka imala na računalu klijenta, iskoristit ćemo varijablu `$_FILES["popis_poklona"]["name"]` umjesto `"popis.txt"` na način:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $webroot = getcwd();
    move_uploaded_file($_FILES["popis_poklona"]["tmp_name"], $webroot . "/" . $_FILES["popis_poklona"]["name"]);
    http_response_code(201);
}
```

### Postavljanje datoteke na danu putanju

Dodatno možemo uzeti u obzir i putanju koju je korisnik naveo u zahtjevu tako da `"/"` zamijenimo s `$_SERVER["REQUEST_URI"]`. Naravno, da bi postavljanje datoteke u direktorij uspjelo, taj direktorij mora postojati; direktorij u PHP-u stvaramo funkcijom `mkdir()` ([dokumentacija](https://www.php.net/manual/en/function.mkdir.php)) pa je kod oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $webroot = getcwd();
    mkdir($webroot . $_SERVER["REQUEST_URI"]);
    move_uploaded_file($_FILES["popis_poklona"]["tmp_name"], $webroot . $_SERVER["REQUEST_URI"] . $_FILES["popis_poklona"]["name"]);
    http_response_code(201);
}
```

Postavimo datoteku `pokloni.txt` u direktorij `datoteke-za-djeda-mraza` unutar radnog direktorija web poslužitelja naredbom:

``` shell
$ curl -v -F 'popis_poklona=@pokloni.txt' http://localhost:8000/datoteke-za-djeda-mraza/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST /datoteke-za-djeda-mraza/ HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Content-Length: 59347
> Content-Type: multipart/form-data; boundary=------------------------afd307b6d042449a
>
* We are completely uploaded and fine
* Mark bundle as not supporting multiuse
< HTTP/1.1 201 Created
< Host: localhost:8000
< Date: Wed, 30 Dec 2020 01:31:50 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
* Closing connection 0
```

Uočite kako smo u zahtjevu naveli direktorij znakom `/` na kraju putanje u URL-u. Ako ne želimo da zahtjevi navode taj znak `/`, kod treba biti oblika:

``` php hl_lines="5"
<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $webroot = getcwd();
    mkdir($webroot . $_SERVER["REQUEST_URI"]);
    move_uploaded_file($_FILES["popis_poklona"]["tmp_name"], $webroot . $_SERVER["REQUEST_URI"] . "/" . $_FILES["popis_poklona"]["name"]);
    http_response_code(201);
}
```

Jedina je promjena u ovom kodu u odnosu na prethodni dodan znak `"\"` između `$_SERVER["REQUEST_URI"]` i `$_FILES["popis_poklona"]["name"])`.

Izlistavanjem sadržaja radnog direktorija web poslužitelja možemo se uvjeriti da je u njemu stvoren direktorij `datoteke-za-djeda-mraza` i da se u tom direktoriju nalazi datoteka `pokloni.txt`.

!!! warning
    Kako sad direktorij postoji, PHP-ov ugrađeni poslužitelj će ga posluživati kao statički sadržaj bez obzira na korištenu metodu pa neće biti moguće postaviti još jednu datoteku u isti direktorij (postavljanje datoteka se uredno može izvršiti u drugi direktorij). Studentima prepuštamo za istraživanje radi li se o svojstvu standarda HTTP ili implementacijskom odabiru PHP-ovih programera.

## Postavljanje datoteke metodom PUT

Osim HTTP metodom POST, postavljanje datoteke na poslužitelj [moguće je izvesti i HTTP metodom PUT](https://www.php.net/manual/en/features.file-upload.put-method.php) ([više detalja o HTTP metodi PUT na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/PUT)). Kod postavljanja datoteke HTTP metodom PUT sadržaj datoteke nalazit će se u tijelu HTTP zahtjeva.

Sadržaj tijela HTTP zahtjeva nam je u jeziku PHP dostupan na putanji `php://input` ([dokumentacija](https://www.php.net/manual/en/wrappers.php.php)) i ponaša se kao datoteka iz koje možemo čitati. Ovaj zapis putanje tijela HTTP zahtjeva ne treba mistificirati jer naprosto radi o konvenciji koja se koristi; postoji analogna putanja `php://output` u koju možemo kao u datoteku zapisivati sadržaj tijela odgovora, odnosno na drugačiji način izvesti isto što već rutinski radimo naredbom `echo`. Štoviše, na sličnim putanjama dostupni su i drugi ulazno-izlazni tokovi: standardni ulaz, standardni izlaz i standardni izlaz za greške operacijskog sustava redom pod `php://stdin`, `php://stdout` i `php://stderr` ([dokumentacija](https://www.php.net/manual/en/features.commandline.io-streams.php)), opisnici otvorenih datoteka pod `php://fd` itd.

Sadržaj tijela zahtjeva ćemo u PHP-u dohvatiti s putanje `php://input` funkcijom `file_get_contents()` ([dokumentacija](https://www.php.net/manual/en/function.file-get-contents.php)), a zatim ćemo funkcijom `file_put_contents()` ([dokumentacija](https://www.php.net/manual/en/function.file-put-contents.php)) spremiti u datoteku `popis.txt` dohvaćeni sadržaj. Kako se koristi HTTP metoda PUT, očekujemo da `$_SERVER["REQUEST_METHOD"]` ima vrijednost `"PUT"` i da klijent šalje zahtjeve na putanju `http://localhost:8000/upload` pa je kod oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "PUT" && $_SERVER["REQUEST_URI"] == "/upload") {
    $putdata = file_get_contents('php://input');
    $webroot = getcwd();
    file_put_contents($webroot . "/" . "popis.txt", $putdata);
    http_response_code(201);
}
```

U cURL-u ćemo zahtjeve HTTP metodom PUT izvesti korištenjem parametra `--upload-file`, odnosno `-T`.

``` shell
$ curl -v -T pokloni.txt http://localhost:8000/upload
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0*   Trying ::1:8000...
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0* Connected to localhost (::1) port 8000 (#0)
> PUT /upload HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
> Content-Length: 59149
> Expect: 100-continue
>
* Done waiting for 100-continue
  0 59149    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0} [59149 bytes data]
* We are completely uploaded and fine
* Mark bundle as not supporting multiuse
< HTTP/1.1 201 Created
< Host: localhost:8000
< Date: Mon, 28 Dec 2020 23:23:00 GMT
< Connection: close
< X-Powered-By: PHP/8.0.0
< Content-Type: text/html; charset=UTF-8
<
{ [3 bytes data]
100 59152    0     3  100 59149      2  58796  0:00:01  0:00:01 --:--:-- 58740
* Closing connection 0
```

## Brisanje datoteke metodom DELETE

HTTP metoda DELETE koristi se za brisanje resursa navedenog u putanji ([više detalja o HTTP metodi DELETE na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/DELETE)).

Brisanje datoteke možemo u PHP-u izvesti funkcijom `unlink()` ([dokumentacija](https://www.php.net/manual/en/function.unlink.php)).

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "DELETE") {
    $webroot = getcwd();
    unlink($webroot . $_SERVER["REQUEST_URI"]);
    http_response_code(204);
}
```

Poslužitelj postavlja statusni kod u odgovoru na [204 No Content](https://http.cat/204) ([više detalja o HTTP statusnom kodu 204 No Content na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/204)) ako je brisanje bilo uspješno i nema nikakvih dodatnih informacija.

Ovaj program ima barem dva problema:

1. Poslužitelj će na odgovarajući zahtjev klijenta izbrisati bilo koju datoteku, pa specijalno i `index.php`.
1. Poslužitelj ne razlikuje situaciju kad je brisanje datoteke bilo uspješno od one kad datoteka čije se brisanje traži ne postoji, odnosno uvijek će odgovor klijentu imati statusni kod 204 No Content.

Riješimo redom te probleme. Kako bismo onemogućili klijenta u brisanju datoteke `index.php`, provjerimo je li putanja na koju je zahtjev izveden jednaka `/index.php` i vratimo klijentu odgovarajuću poruku:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "DELETE") {
    if ($_SERVER["REQUEST_URI"] == "/index.php") {
        http_response_code(403);
        echo "<p>Neće ići, hakeru!</p>\n";
    } else {
        $webroot = getcwd();
        unlink($webroot . $_SERVER["REQUEST_URI"]);
        http_response_code(204);
    }
}
```

!!! note
    Moguće je da ova varijanta programa još uvijek ima sigurnosnih propusta koji bi omogućili brisanje datoteke `index.php` uz korištenje naprednijih trikova, ali uzmimo da je za naše potrebe dovoljno dobra.

Funkcija `unlink()` vratit će `true` ako je brisanje bilo uspješno, a `false` ako nije, primjerice zato što datoteka ne postoji ili zato što proces PHP-ovog ugrađenog poslužitelja nema pravo brisanja datoteke jer njene dozvole pristupa to zabranjuju. Zanemarimo do daljnjega probleme s dozvolama pa imamo kod oblika:

``` php
<?php

if ($_SERVER["REQUEST_METHOD"] == "DELETE") {
    if ($_SERVER["REQUEST_URI"] == "/index.php") {
        http_response_code(403);
        echo "<p>Neće ići, hakeru!</p>\n";
    } else {
        $webroot = getcwd();
        $path = $_SERVER["REQUEST_URI"];
        if (unlink($webroot . $path)) {
            echo "<p>Datoteka $path uspješno obrisana.</p>\n";
        } else {
            http_response_code(404);
            echo "<p>Datoteka $path nije pronađena.</p>\n";
        }
    }
}
```

U slučaju da je brisanje uspješno, ranije korišteni statusni kod 204 No Content zamijenili smo za zadani statusni kod 200 OK jer odgovor sadrži poruku `<p>Datoteka $path uspješno obrisana.</p>`.

## Promjena datoteke metodom PATCH

HTTP metoda PATCH omogućuje djelomičnu promjenu postojećeg sadržaja na web poslužitelju ([više detalja o HTTP metodi PATCH na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/PATCH)). Metoda PATCH proširuje specifikaciju HTTP-a 1.1 ([RCF 2616 pod naslovom Hypertext Transfer Protocol -- HTTP/1.1](https://datatracker.ietf.org/doc/html/rfc2616)) i standardizirana je u [RFC-u 5789 pod naslovom PATCH Method for HTTP](https://datatracker.ietf.org/doc/html/rfc5789). Kod korištenja ove metode tijelo zahtjeva sadrži upute koje izmjene sadržaja treba napraviti. Pritom standard ne definira kako se te upute trebaju biti zapisane i prepušta taj odabir razvijateljima implementacije poslužitelja i klijenta. U praksi se najčešće koristi zapis izmjena temeljen na JSON-u iz [RFC-a 6902 naslovljenog JavaScript Object Notation (JSON) Patch](https://datatracker.ietf.org/doc/html/rfc6902).

Metodu PATCH spominjemo radi potpunosti i nećemo je ovdje detaljnije koristiti.
