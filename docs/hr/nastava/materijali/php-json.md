---
author: Vedran Miletić
---

# Obrada podataka zapisanih u obliku JavaScript Object Notation (JSON) u jeziku PHP

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

## Kodiranje i dekodiranje JSON-a

[JavaScript Object Notation (JSON)](https://www.json.org/) ([Wikipedia](https://en.wikipedia.org/wiki/JSON), [MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON)) je jednostavan format za razmjenu podataka koji se intenzivno koristi na suvremenom webu. Standardiziran je u okviru [RFC-a 7159 naslovljenog The JavaScript Object Notation (JSON) Data Interchange Format](https://tools.ietf.org/html/rfc7159). Primjer objekta zapisanog u JSON-u koji opisuje osobu (inspiriran [Wikipedijinim primjerom](https://en.wikipedia.org/wiki/JSON#Syntax)) je oblika:

``` json
{
  "firstName": "Ivan",
  "lastName": "Horvat",
  "isAlive": true,
  "age": 19,
  "address": {
    "streetAddress": "Radmile Matejčić 2a",
    "city": "Rijeka",
    "state": "Primorsko-goranska županija",
    "postalCode": "51000"
  },
  "jobTitle": "junior software engineer",
  "phoneNumbers": [
    {
      "type": "home",
      "number": "051/999-999"
    },
    {
      "type": "office",
      "number": "099/999-9999"
    }
  ],
  "twitterHandle": "@IvanNajjaciHorvat",
  "children": [],
  "spouse": null
}
```

Iz ovako zapisanih podataka lako je izdvojiti onaj koji nam treba te ga postaviti na odgovarajuće mjesto na web stranici. Za ilustraciju, možemo zamisliti kako bi ime, prezime, titula i korisničko ime na Twitteru bili iskorišteni kod stvaranja stranice [Team Sections od Tailwind CSS UI Components](https://tailwindui.com/components/marketing/sections/team-sections) koja bi navela Ivana Horvata kao dio tima, ali procesom izrade web stranica na taj način ćemo se detaljnije baviti nekom drugom prigodom.

Interpreter PHP-a [podržava kodiranje i dekodiranje JSON-a](https://www.php.net/manual/en/book.json.php) od [verzije 5.2.0](https://www.php.net/releases/5_2_0.php) nadalje.

### Kodiranje

Pretvobu PHP-ovih polja u oblik JSON, odnosno kodiranje JSON-a vršimo funkcijom `json_encode()` ([dokumentacija](https://www.php.net/manual/en/function.json-encode.php)) na način:

``` php
<?php

$arr1 = ["moja vrijednost", 1, 3.5, true];
$j1 = json_encode($arr1);
echo $j1;

$arr2 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true];
$j2 = json_encode($arr2);
echo $j2;
```

Pokretanjem ovog koda (u interaktivnom načinu rada, korištenjem sučelja naredbenog retka ili korištenjem ugrađenog web poslužitelja i klijenta po želji) dobivamo za prvo polje zapis podataka u obliku JSON:

``` json
["moja vrijednost",1,3.5,true]
```

te za drugo polje zapis:

``` json
{"moj kljuc":"moja vrijednost","broj":8,"drugi broj":3.5,"je li istina":true}
```

Uočimo da smo kod pretvorbe prvog PHP-ovog polja dobili polje u obliku JSON-u (znakovi `[` i `]`), a kod pretvorbe drugog objekt (znakovi `{` i `}` te znak `:` koji odvaja ključeve i njihove vrijednosti).

Dodatno, u JSON-u znakovi razmaka i novog retka između vrijednosti ne znače ništa pa funkcija `json_encode()` radi štednje prostora zapisuje podatke bez tih znakova. Želimo li elegantniji i za ljude čitljiviji ispis, možemo [iskoristiti zastavicu](https://www.php.net/manual/en/function.json-encode.php#refsect1-function.json-encode-parameters) `JSON_PRETTY_PRINT` ([dokumentacija](https://www.php.net/manual/en/json.constants.php)) na način:

``` php
<?php

$arr1 = ["moja vrijednost", 1, 3.5, true];
$j1 = json_encode($arr1, JSON_PRETTY_PRINT);
echo $j1;

$arr2 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true];
$j2 = json_encode($arr2, JSON_PRETTY_PRINT);
echo $j2;
```

Dobiveni zapisi podataka u obliku JSON su sada čitljiviji ljudima. Za prvo polje dobivamo:

``` json
[
    "moja vrijednost",
    1,
    3.5,
    true
]
```

Za drugo polje dobivamo:

``` json
{
    "moj kljuc": "moja vrijednost",
    "broj": 8,
    "drugi broj": 3.5,
    "je li istina": true
}
```

Ugnježđivanjem polja možemo dobiti složenije strukture:

``` php
<?php

$arr1 = ["moja vrijednost", 1, 3.5, true];
$arr2 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true];

$arr3 = $arr2;
$arr3["polje"] = $arr1;
// ekvivalentno $arr3 = ["moj kljuc" => "moja vrijednost", "broj" => 8, "drugi broj" => 3.5, "je li istina" => true, "polje" => ["moja vrijednost", 1, 3.5, true]];
$j3 = json_encode($arr3, JSON_PRETTY_PRINT);
echo $j3;
```

Pokretanjem programa dobivamo zapis podataka u obliku JSON:

``` json
{
    "moj kljuc": "moja vrijednost",
    "broj": 8,
    "drugi broj": 3.5,
    "je li istina": true,
    "polje": [
        "moja vrijednost",
        1,
        3.5,
        true
    ]
}
```

### Dekodiranje

Pretvorbu JSON-a u PHP-ova polja, odnosno dekodiranje JSON-a vršimo funkcijom `json_decode()` ([dokumentacija](https://www.php.net/manual/en/function.json-decode.php)). Uzmimo za primjer objekt koji opisuje osobu od ranije zapisan bez znakova razmaka i novog retka između vrijednosti:

``` php
<?php

$j = '{"firstName":"Ivan","lastName":"Horvat","isAlive":true,"age":19,"address":{"streetAddress":"Radmile Matejčić 2a","city":"Rijeka","state":"Primorsko-goranska županija","postalCode":"51000"},"jobTitle":"junior software engineer","phoneNumbers":[{"type":"home","number":"051/999-999"},{"type":"office","number":"099/999-9999"}],"twitterHandle":"@IvanNajjaciHorvat","children":[],"spouse":null}';
$person = json_decode($j, true);
print_r($person);
```

Prvi parametar funkcije `json_decode()` je znakovni niz koji sadrži zapis podataka u obliku JSON, a drugi postavljamo na vrijednost `true` jer želimo kao rezultat pretvorbe uvijek dobiti polje. Naime, zadana vrijednost drugog parametra je `null` i kod te vrijednosti možemo dobiti polje ili [objekt](https://www.php.net/manual/en/language.oop5.php) ovisno o postavljenim [vrijednostima zastavica](https://www.php.net/manual/en/json.constants.php). (Moguće je i kod ove funkcije [koristiti zastavice](https://www.php.net/manual/en/function.json-decode.php#refsect1-function.json-decode-parameters) na sličan način kao kod funkcije `json_encode()`, ali nam one neće trebati.) Uvjerimo se pokretanjem koda da smo dekodiranjem JSON-a dobili polje s danim podacima:

```
Array
(
    [firstName] => Ivan
    [lastName] => Horvat
    [isAlive] => 1
    [age] => 19
    [address] => Array
        (
            [streetAddress] => Radmile Matejčić 2a
            [city] => Rijeka
            [state] => Primorsko-goranska županija
            [postalCode] => 51000
        )

    [jobTitle] => junior software engineer
    [phoneNumbers] => Array
        (
            [0] => Array
                (
                    [type] => home
                    [number] => 051/999-999
                )

            [1] => Array
                (
                    [type] => office
                    [number] => 099/999-9999
                )

        )

    [twitterHandle] => @IvanNajjaciHorvat
    [children] => Array
        (
        )

    [spouse] =>
)
```

## Dodatak: oblikovanje JSON-a u sučelju naredbenog retka

Van interpretera PHP-a, odnosno u ljusci operacijskog sustava možemo iskoristiti programski alat jq ([službena stranica](https://stedolan.github.io/jq/), [dokumentacija](https://stedolan.github.io/jq/manual/)) za obradu JSON-a slično kao što za obradu običnog teksta koristimo `sed`. Željeni ispis dobivamo naredbom `jq` s filterom ulaza `.` koji kopira ulaz na izlaz bez da mijenja njegov sadržaj, ali ga pritom lijepo oblikuje:

``` shell
$ echo '{"moj kljuc":"moja vrijednost","broj":8,"drugi broj":3.5,"je li istina":true}' | jq .
{
  "moj kljuc": "moja vrijednost",
  "broj": 8,
  "drugi broj": 3.5,
  "je li istina": true
}
```
