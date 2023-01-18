---
marp: true
author: Vedran Miletić
title: Pisanje i provođenje automatiziranih testova programskog koda i web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Pisanje i provođenje automatiziranih testova programskog koda i web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Motivacija

U procesu razvoja aplikacije primijetite kvar (engl. *fault*, kolokvijalno pogreška, engl. *bug*), pronađete u kojem je dijelu koda i popravite ga. Pokrenete aplikaciju i nakon malo klikanja imate dojam da sve radi kako treba.

Na koji način ćete osigurati:

- da aplikacija *zaista* radi kako treba,
- da vaš popravak tog kvara nije izazvao druge kvarove i
- da se kvar ne vrati u kasnijoj verziji?

---

## Testiranje softvera (1/3)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Software_testing):

Testiranje softvera je proces koji se provodi kako bi se dionicima pružile informacije o kvaliteti testiranog softverskog proizvoda ili usluge. Može pružiti objektivan, neovisan pogled na softver kako bi omogućio tvrtki da uzme u obzir i razumije rizike implementacije softvera. Tehnike ispitivanja uključuju postupak izvršavanja programa ili aplikacije s namjerom pronalaženja kvarova i provjere je li softverski proizvod prikladan za upotrebu.

---

## Testiranje softvera (2/3)

Testiranje softvera uključuje izvršavanje softverske ili sustavske komponente za procjenu jednog ili više svojstava koja nas zanimaju. Općenito, ova svojstva pokazuju u kojoj mjeri komponenta ili sustav koji se ispituje:

- udovoljava zahtjevima koji su vodili njegov dizajn i razvoj,
- ispravno reagira na sve vrste ulaza,
- obavlja svoje funkcije u prihvatljivom roku,
- je dovoljno upotrebljiv,
- može se instalirati i pokretati u predviđenim okruženjima te
- ostvaruje ciljeve koje imaju njegovi dionici.

---

## Testiranje softvera (3/3)

Kako je broj mogućih testova čak i za jednostavne softverske komponente praktički beskonačan, sva testiranja softvera koriste neku strategiju za odabir testova koji su izvedivi za raspoloživo vrijeme i resurse. Kao rezultat toga, testiranje softvera obično izvršava aplikaciju ili neki njen dio s namjerom pronalaska kvarova. Testiranje je *iterativni postupak*: kada jedan je kvar otklonjen, može osvijetliti druge kvarove koji postoje zbog dubljih problema ili čak stvoriti nove.

---

![The ultimate inspiration is the deadline bg 95% left:55%](https://www.monkeyuser.com/assets/images/2020/198-feature-complete.png)

## Dovršen razvoj značajki softvera

Izvor: [Feature Complete](https://www.monkeyuser.com/2020/feature-complete/) (MonkeyUser, 20th November 2020)

---

## Jednostavan primjer testa u pytestu

``` python
# sadržaj datoteke test_sample.py
def inc(x):
    return x + 1

def test_inc_3():
    assert inc(3) == 4
```

``` shell
$ pytest
============================= test session starts =============================
platform linux -- Python 3.9.2, pytest-6.0.2, py-1.10.0, pluggy-0.13.0
rootdir: /home/vedranmiletic
collected 1 item

test_sample.py .                                                        [100%]

============================== 1 passed in 0.03s ==============================
```

---

## Primjer testa u Djangu

``` python
from django.test import TestCase
from myapp.models import Animal

class AnimalTestCase(TestCase):
    def setUp(self):
        Animal.objects.create(name="lion", sound="roar")
        Animal.objects.create(name="cat", sound="meow")

    def test_animals_can_speak(self):
        """Animals that can speak are correctly identified"""
        lion = Animal.objects.get(name="lion")
        cat = Animal.objects.get(name="cat")
        self.assertEqual(lion.speak(), 'The lion says "roar"')
        self.assertEqual(cat.speak(), 'The cat says "meow"')
```

---

## Testiranje metodom bijele kutije

Prema [Wikipediji](https://en.wikipedia.org/wiki/White-box_testing): Testiranjem metodom bijele (prozirne, staklene) kutije (engl. *white-box testing*) verificira se interna struktura programa. Testovi se implementiraju programiranjem ulaza i očekivanih izlaza.

![White Box Testing Diagram](https://upload.wikimedia.org/wikipedia/commons/e/e2/White_Box_Testing_Approach.png)

Uglavnom se primjenjuje na nivou jedinice izvornog koda (engl. *source code unit*), ali može se primijeniti i na nivoima integracije i sustava.

---

## Tehnike testiranja metodom bijele kutije

- **testiranje aplikacijskih programskih sučelja (API-ja)**: testiranje aplikacije pomoću javnih i privatnih API-ja
    - primjerice, kod korištenja API-ja može se raditi o pozivu funkcije programskog jezika u obliku `get_persons(5)`, korištenju objektno-orijentiranog pristupa ili slanju zahtjeva HTTP metodom GET na URI `/persons/5`
- **analiza pokrivenosti koda testovima** (engl. *code coverage*): izrada testova kako bi se udovoljilo nekim kriterijima pokrivenosti koda
    - primjerice, dizajner testa može stvoriti testove kako bi barem jednom izvršili sve ispise koje aplikacija vrši
- **metode ubrizgavanja kvarova**: namjerno uvode greške kako bi se procijenila učinkovitost strategija ispitivanja

---

![Jest Fast and Safe](https://d33wubrfki0l68.cloudfront.net/7ab37629fb8f2b135083d8301a67be7d3d37ca52/d6fe3/img/content/feature-fast.png)

---

![Jest Code Coverage](https://d33wubrfki0l68.cloudfront.net/e6a4c79760df80d72d39c289db1da75e012bca56/7df0d/img/content/feature-coverage.png)

---

## Alati za testiranje softvera

Primjeri alata za testiranje softvera na opisan način su:

- [pytest](https://docs.pytest.org/) (Python)
- [Mocha](https://mochajs.org/) i [Jest](https://jestjs.io/) (JavaScript)
    - [Mocha](https://convergentcoffee.com/what-is-mocha/) je inače fina kava, a [Jester](https://en.wikipedia.org/wiki/Jester) je dvorska luda i [smije se kao Joker](https://youtu.be/ocw-VGMiOmk)
- [NUnit](https://nunit.org/) (C#)
- [PHPUnit](https://phpunit.de/) (PHP)
- [GoogleTest](https://github.com/google/googletest) i [CppUnit](https://freedesktop.org/wiki/Software/cppunit/) (C++)
    - za C/C++ [Gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) vrši analizu pokrivenosti koda testovima
- [Test::More](https://metacpan.org/pod/Test::More) (Perl)
- [JUnit](https://junit.org/) (Java)
- ugrađeni sustavi za testiranje u [Ruby on Railsu](https://guides.rubyonrails.org/testing.html), [Elixiru](https://hexdocs.pm/ex_unit/1.12/ExUnit.html), [Rustu](https://doc.rust-lang.org/rust-by-example/testing/unit_testing.html) itd.

---

## Primjeri testova u web aplikacijama

🙋 **Pitanje:** Što testiraju ovi testovi?

- [WordPressovi (koristi PHPUnit)](https://github.com/WordPress/wordpress-develop/tree/trunk/tests/phpunit)
- [Collabora Online (koristi CppUnit)](https://github.com/CollaboraOnline/online/tree/master/test)
- [PeerTube (koristi Mochu)](https://github.com/Chocobozzz/PeerTube/tree/develop/server/tests)
- [Trac (koristi Pythonov modul unittest)](https://github.com/edgewall/trac/tree/trunk/trac/tests)
- [Redmine (koristi Railsov sustav za testiranje)](https://github.com/redmine/redmine/tree/master/test)
- [Mattermost (koristi Golangov modul testing)](https://github.com/mattermost/mattermost-server/tree/master/web)
- [pump.io (koristi vows)](https://github.com/pump-io/pump.io/tree/master/test)

---

## Testiranje metodom crne kutije

Prema [Wikipediji](https://en.wikipedia.org/wiki/Black-box_testing): Testiranje metodom crne kutije (engl. *black-box testing*) ili funkcionalno testiranje je proces testiranja bez poznavanja interne strukture softvera. Testeri samo znaju što softver treba raditi, ali ne i kako. Metode koje se koriste su:

- [raspodjela ekvivalencije](https://en.wikipedia.org/wiki/Equivalence_partitioning), [analiza graničnih vrijednosti](https://en.wikipedia.org/wiki/Boundary_value_analysis), [ispitivanje svih parova](https://en.wikipedia.org/wiki/All-pairs_testing),
- [tablice prijelaza stanja](https://en.wikipedia.org/wiki/State-transition_table), ispitivanje korištenjem [tablice odluka](https://en.wikipedia.org/wiki/Decision_table),
- [fuzz testiranje](https://en.wikipedia.org/wiki/Fuzzing) (**vrlo važna metoda**, najčešće služi za [pronalaženje sigurnosnih propusta u web aplikacijama](https://owasp.org/www-community/Fuzzing)),
- [testiranje na temelju modela](https://en.wikipedia.org/wiki/Model-based_testing), ispitivanje [slučajeva korištenja](https://en.wikipedia.org/wiki/Use_case),
- [ispitivanje istraživanjem](https://en.wikipedia.org/wiki/Exploratory_testing) i ispitivanje na temelju specifikacije.

---

## Fuzz testiranje (engl. *fuzz testing*, *fuzzing*)

> Fuzzing is a powerful strategy to find bugs in software. The idea is quite simple: **Generate a large number of randomly malformed inputs for a software to parse and see what happens.** If the program crashes then something is likely wrong. While fuzzing is a well-known strategy, **it is surprisingly easy to find bugs, often with security implications, in widely used software.**

-- [The Fuzzing Project](https://fuzzing-project.org/)

Primjeri alata za fuzz testiranje: [american fuzzy lop](https://en.wikipedia.org/wiki/American_fuzzy_lop_%28fuzzer%29), [Radamsa](https://gitlab.com/akihe/radamsa) i [APIFuzzer](https://github.com/KissPeter/APIFuzzer)

---

## Testiranje s kraja na kraj

Testiranje s kraja na kraj (engl. *end-to-end testing*) je suvremeni naziv za proces testiranja metodom crne kutije koji spaja više navedenih metoda. Primjeri alata koji se koriste za takvo testiranje su:

- [Cypress](https://www.cypress.io/)
- [Nightwatch.js](https://nightwatchjs.org/)
- [Selenium](https://www.selenium.dev/) (nije samo za testiranje, često se koristi i za struganje sadržaja s weba, engl. *web scraping*)

Kolokvijalno rečeno, ovi alati imitiraju korištenje web aplikacije od strane korisnika.

---

## *Dogfooding* (1/2)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Eating_your_own_dog_food):

- konzumiranje vlastite hrane za pse (engl. *eating your own dog food*, *dogfooding*) je praksa korištenja vlastitih proizvoda ili usluga
- može biti način na koji organizacija može testirati svoje proizvode u stvarnom svijetu koristeći tehnike upravljanja proizvodima, služi za kontrolu kvalitete i pokazuje povjerenje u vlastite softverske projekte ili proizvode
- naziv je dao [Paul Maritz u Microsoftu 1988. godine](https://www.centercode.com/blog/dogfooding-101) pokušavajući nagovoriti da interno više koriste vlastite softverske proizvode

---

## *Dogfooding* (2/2)

*Dogfooding* se može koristiti i kod projekata otvorenog koda, npr. developeri Mozille su kasnih 90-ih intenzivno koristili web preglednik, e-mail klijent i druge alate koje su razvijali pa prijavljivali probleme na koje su pritom naišli:

- [\[BETA\]\[DOGFOOD\]No proxy authentication](https://bugzilla.mozilla.org/show_bug.cgi?id=15927)
- [\[DOGFOOD\]Form submission with post data no longer works](https://bugzilla.mozilla.org/show_bug.cgi?id=16258)
- [\[DOGFOOD\]\[BLOCKER\]Start Seamonkey with problems for running multiple profiles.](https://bugzilla.mozilla.org/show_bug.cgi?id=20910)
- [\[DOGFOOD\]\[bugday\] mail news buttons disappear on click and mouse-over](https://bugzilla.mozilla.org/show_bug.cgi?id=21075)

---

![Or the illusion of choice bg 95% left:70%](https://www.monkeyuser.com/assets/images/2021/209-trolley-conundrum.png)

## Zagonetka trolejbusa

Izvor: [Trolley Condrum](https://www.monkeyuser.com/2021/trolley-conundrum/) (MonkeyUser, 16th March 2021)
