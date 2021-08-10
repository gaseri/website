---
author: Vedran Miletić
---

# Mjerenje performansi web poslužitelja alatom Siege

[Siege](https://www.joedog.org/siege-home/) je alat za mjerenje performansi HTTP i HTTPS web poslužitelja pod opterećenjem. Opterećenje web poslužitelja možemo vizualizirati kao opsadu. Dostupan je pod [licencom GNU GPLv3](https://github.com/JoeDog/siege/blob/master/COPYING). Podržava protokole HTTP i HTTPS te specifične značajke protokola kao što su osnovna autentifikacija i kolačići. Dokumentacija je [dostupna putem službenih stranica](https://www.joedog.org/siege-manual/) i u man stranici `siege(1)` (naredba `man 1 siege`).

!!! tip
    Neki softveri, primjerice [OpenSSL](https://www.openssl.org/) i [MariaDB](https://mariadb.org/), imaju unikatna ili drugdje vrlo rijetko korištena imena pa je lako potražiti upute za njih na internetu. Kod Siegea [to](https://guardiansofgahoole.fandom.com/wiki/The_Siege) [nije](https://forgottenrealms.fandom.com/wiki/The_Siege) [slučaj](https://en.wikipedia.org/wiki/Siege_(disambiguation)), a istovremeno nije masovno korišten softver kao što su, primjerice, [Apache](https://httpd.apache.org/) ili [Word](https://products.office.com/word). Stoga je u eventualnoj potrazi za dodatnom dokumentacijom ili primjerima korištenja korisno pretraživati za pojmovima kao što su `siege web server tester`, `siege http benchmarking` ili sl.

## Osnovne mogućnosti i način korištenja

Uzmimo za testiranje [web aplikaciju za ispis IP adrese i korisničkog agenta](https://app.miletic.net/ip/). Dobra je ideja testirati vlastiti web poslužitelj jer drugi web poslužitelji mogu iznenadni velik broj zahtjeva shvatiti kao napad.

Pokrenemo li Siege bez parametara, radit će do prekida kombinacijom tipki Ctrl + C:

``` shell
$ siege https://app.miletic.net/ip/
^C
```

``` json
{
        "transactions":                         1279,
        "availability":                       100.00,
        "elapsed_time":                        52.61,
        "data_transferred":                   140.25,
        "response_time":                        1.00,
        "transaction_rate":                    24.31,
        "throughput":                           2.67,
        "concurrency":                         24.34,
        "successful_transactions":              1279,
        "failed_transactions":                     0,
        "longest_transaction":                 19.20,
        "shortest_transaction":                 0.02
}
```

Želimo li ograničiti vrijeme mjerenja performansi, primjerice na 10 sekundi, iskoristit ćemo parametar `--time`, odnosno `-t`:

``` shell
$ siege -t 10s https://app.miletic.net/ip/
```

``` json
{
        "transactions":                          253,
        "availability":                       100.00,
        "elapsed_time":                         9.35,
        "data_transferred":                    27.33,
        "response_time":                        0.74,
        "transaction_rate":                    27.06,
        "throughput":                           2.92,
        "concurrency":                         20.00,
        "successful_transactions":               255,
        "failed_transactions":                     0,
        "longest_transaction":                  8.10,
        "shortest_transaction":                 0.03
}
```

Parametrom `--concurrent`, odnosno `-c` možemo navesti broj istovremenih korisnika čije opterećenje će biti generirano. Postavimo taj broj na 100:

``` shell
$ siege -c 100 -t 10s https://app.miletic.net/ip/
```

``` json
{
        "transactions":                          166,
        "availability":                       100.00,
        "elapsed_time":                         9.03,
        "data_transferred":                    18.11,
        "response_time":                        1.22,
        "transaction_rate":                    18.38,
        "throughput":                           2.01,
        "concurrency":                         22.37,
        "successful_transactions":               167,
        "failed_transactions":                     0,
        "longest_transaction":                  6.31,
        "shortest_transaction":                 0.00
}
```

Ako ne navedemo ništa, Siege će generirati zahtjeve 10 istovremenih korisnika, što znači da u pretposljednjem i ovom primjeru možemo usporediti performanse poslužitelja kad mu pristupa istovremeno 10 i 100 korisnika kroz 10 sekundi. Uočimo:

- kako opada broj razmjena podataka između klijenta i poslužitelja (`"transactions"`) i broj tih razmjena po sekundi (`"transaction_rate"`) koji dobivamo dijeljenjem njihovog broja s vremenom mjerenja (`"elapsed_time"`)
- kako raste prosječno vrijeme odgovora (`"response_time"`); za razliku od tog vremena, druga izmjerena vremena, odnosno vremena trajanja najdulje i najkraće razmjene podataka (`"longest_transaction"` i `"shortest_transaction"`) su vremena trajanja po jedne razmjene pa nisu naročito mjerodavna
- kako opada propusnost (`"throughput"`) koju dobijemo dijeljenjem količine prenesenih podataka (`"data_transferred"`) s vremenom mjerenja (`"elapsed_time"`)

Po potrebi parametrom `--header`, odnosno `-H` možemo navesti zaglavlje koje će biti poslano u zahtjevu. Primjerice, možemo od web aplikacije zatražiti slanje odgovora u obliku JSON navođenjem zaglavlja Accept i pripadnog MIME tipa na način:

``` shell
$ siege -t 10s -H 'Accept: application/json' https://app.miletic.net/ip/
```

``` json
{
        "transactions":                           65,
        "availability":                       100.00,
        "elapsed_time":                         9.44,
        "data_transferred":                     0.00,
        "response_time":                        2.16,
        "transaction_rate":                     6.89,
        "throughput":                           0.00,
        "concurrency":                         14.88,
        "successful_transactions":                65,
        "failed_transactions":                     0,
        "longest_transaction":                  6.58,
        "shortest_transaction":                 0.20
}
```

## Mjerenje performansi poslužitelja pokrenutih na lokalnom računalu

Siege možemo na isti način koristiti i na web poslužiteljima pokrenutim na lokalnom računalu. Primjerice, ako imamo pokrenut PHP-ov ugrađeni web poslužitelj na vratima 8000, onda ćemo mu pristupiti naredbom:

``` shell
$ siege -c 1 http://localhost:8000/
```

Kako PHP-ov ugrađeni web poslužitelj u zadanim postavkama ne podržava istovremene zahtjeve više korisnika, broj istovremenih korisnika ćemo postaviti na 1. (Ako želite eksperimentirati s većim brojem istovremenim korisnika, od verzije PHP-a 7.4.0 nadalje možete postaviti varijablu okoline `PHP_CLI_SERVER_WORKERS` ([dokumentacija](https://www.php.net/manual/ro/features.commandline.webserver.php)) na broj veći od 1 kod pokretanja ugrađenog web poslužitelja pa taj isti broj iskoristiti u Siegeu.)

Apache HTTP Server može baratati s više korisnika pa možemo varirati broj korisnika ovisno o snazi računala na kojemu radimo. Ako Apache je pokrenut upotrebom Dockera, Siege s 50 istovremenih korisnika ćemo pokrenuti na način:

``` shell
$ siege -c 50 http://172.17.0.2/
```

## Složenije mogućnosti korištenja

Siege nudi i skriptu `bombardment` koja omogućuje pokretanje s postupnim povećanjem broja korisnika. Detalji o načinu korištenja se mogu promaći u man stranici `bombardment(1)` (naredba `man 1 bombardment`).
