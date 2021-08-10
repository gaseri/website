---
author: Vedran Miletić
---

# Testiranje Python aplikacija

Imamo li datoteku `modul1.py` sadržaja

``` python
# -*- coding: utf-8 -*-
def funkcija1():
    """
    Funkcija uvijek vraća vrijednost 5.

    Vraća:
    Vrijednost 5, tipa int.
    """
    return 5
```

Jednostavan test možemo izvesti korištenjem naredbe `assert`. stvorimo datoteku `test_program.py` sadržaja:

``` python
import modul1

def test_funkcija1():
    assert modul1.funkcija() == 5
    print("Test funkcije funkcija1 uspješno prolazi.")

test_funkcija1()
```

Pokretanjem ovog programa vrši se testiranje modula `modul1`.

Ovo je ručni pristup testiranju u kojem sve radimo samostalno; Python u standardnoj biblioteci nudi modul `unittest` ([službena dokumentacija](https://docs.python.org/3/library/unittest.html)) koji nudi tipičnu funkcionalnost biblioteke za pisanje testova.

## Modul pytest

Korištenjem modula [pytest](https://www.pytest.org/) ranije prikazani kod može se pojednostaviti (naredba `pytest` ili ponekad `pytest-3`) na način:

``` python
import modul1

def test_funkcija():
    assert modul1.funkcija() == 5
```

Uočimo kako smo eliminirali korištenje funkcije `print()` za ispis rezultata testa i poziv funkcije. Kod većeg broja testova ovo pojednostavljenje je značajno.

Pokretanjem naredbe `pytest` u direktoriju u kojem se ova datoteka nalazi vrši se pokretanje svih funkcija čije ime počinje sa `test_` unutar svih datoteka čije ime počinje sa `test_` i na kraju se ispisuje statistika o tome koji testovi prolaze.

!!! admonition "Zadatak"
    - Promijenite konstantu `5` u testu u neku drugu konstantu. Prolazi li test?
    - Definirajte funkciju koja prima broj i vraća kvadrat tog broja. Definirajte test za nju s ulaznim podacima `3`, `15`, `24`.
