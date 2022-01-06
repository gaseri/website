---
author: Vedran Miletić, Domagoj Margan
---

# Rad s Python modulom numpy

!!! note
    Ovaj dio je sastavljen prema [NumPy tutorialu](https://numpy.org/doc/stable/user/quickstart.html).

Uključivanje modula `numpy` najčešće se vrši naredbom:

``` python
import numpy as np
```

## Rad s poljima

- `numpy.array(..., dtype=tip)` -- prima listu kao argument; `dtype` može biti bilo koji od standardnih Python tipova: `int`, `float`, `complex`, ...

    ``` python
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.array([[3.5, 4.2, 8.7, 11.3, 0.7]])
    ```

    - `np.array.shape` -- oblik
    - `np.array.ndim` -- dimenzija
    - `np.array.size` -- veličina (ukupan broj elemenata)
    - `np.array.dtype` -- tip podataka, moguće ga je specificirati prilikom stvaranja

        ``` python
        c = np.array([[3, 11], [7, 9]], dtype=complex)
        ```

    - `np.array.itemsize` -- veličina (u bajtovima) tipa podataka od kojih se polje sastoji

!!! admonition "Zadatak"
    - Stvorite polje s vrijednostima

        ```
        9 13 5
        1 11 7
        3 7 2
        6 0 7
        ```

    - Saznajte mu oblik, duljinu i tip elemenata i veličinu elementa u bajtovima.
    - Stvorite polje s istim vrijednostima, ali tako da su elementi tipa `float`.

- `numpy.zeros((n, m)), dtype=...)` stvara polje nula
- `numpy.ones((n, m), dtype=...)` stvara polje jedinica

Tipovi podataka definirani unutar modula `numpy`; potrebno koristiti kad radite Python kod koji integira s C/C++ kodom:

- `numpy.int8` -- pandan C/C++ tipu `char`
- `numpy.int16` -- pandan C/C++ tipu `short int`
- `numpy.int32` -- pandan C/C++ tipu `int` (ne uvijek!)
- `numpy.int64` -- pandan C/C++ tipu `long` (ne uvijek!)
- `numpy.float32` -- pandan C/C++ tipu `float`
- `numpy.float64` -- pandan C/C++ tipu `double`

!!! note
    Duljina tipova podataka `int` i `long` u C/C++-u varira ovisno o tome koristi li se 32-bitni ili 64-bitni operacijski sustav. Na većini platformi danas koriste se dva modela:

    - ILP32 -- `int`, `long` i pointer su duljine 32 bita,
    - LP64 -- `long` i pointer su duljine 64 bita (za `int` se implicitno pretpostavlja da je duljine 32 bita).

    Više informacija o tome možete naći u članku [64-Bit Programming Models: Why LP64?](https://unix.org/version2/whatsnew/lp64_wp.html).

!!! admonition "Zadatak"
    - Stvorite polje nula oblika `(5, 5)` u kojem su elementi tipa `numpy.float32`.
    - Stvorite polje jedinica oblika `(1000, 1000)`. Pokušajte ga ispisati naredbom `print`. Što se dogodi?

## Operacije na poljima

``` python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
# operacije koje rade element po element
c = a + b
d = b - a
e = a ** 2
f = a * b
g = np.sin(a)
h = np.exp(b)
# operacije koje mijenjaju početni array
a += b
b *= 3
```

!!! admonition "Zadatak"
    - Stvorite dva dvodimenzionalna polja `a` i `b` oblika `(3, 3)` s proizvoljnim vrijednostima, i to tako da prvo ima elemente tipa `numpy.float32`, a drugo elemente tipa `numpy.float64`.
    - Izračunajte `2 * a + b`, `cos(a)`, `sqrt(b)`. Uočite kojeg su tipa polja koja dobivate kao rezultate.
    - Množenje matrica izvodite funkcijom `numpy.dot()`; proučite njenu dokumentaciju i izvedite ju na svojim poljima.

## Pretvorba tipa

- `numpy.array.astype(type)` vraća polje s vrijednostima kao u početnom polju, ali tipa promijenjenog u `type`

    - korisno kod pretvorbe, primjerice, 64-bitnog `float` tipa u 32-bitni `float` tip
    - često ćemo koristiti kod Python koda koji prosljeđuje podatke u C/C++ kod

!!! admonition "Zadatak"
    Iskoristite dva dvodimenzionalna polja iz prethodnog zadatka da izračunajte `2 * a + b`, ali tako da pretvorite drugo u polje koje ima elementa tipa `numpy.float32`.

- `numpy.round(polje, broj_decimala)` vraća polje s vrijednostima zaokruženim na navedeni broj decimala

    - korisno kod usporedbe decimalnih brojeva, zbog nepreciznog spremanja brojeva u računalu (naročito se vidi kod konverzije podataka tipa `numpy.float64` u podatke tipa `numpy.float32`)
    - [strip sa Saturday Morning Breakfast Cereal na istu temu](https://www.smbc-comics.com/comic/2008-03-16)

!!! admonition "Zadatak"
    - Stvorite polje u kojem su sve vrijednosti jednake 9.45 tipa `float64` i pretvorite ga u polje tipa `float32` i rezultat spremite u novo polje. Uočavate li gubitak preciznosti?
    - Pretvorite dobiveno polje tipa `float32` u polje tipa `float64`. Je li rezultat jednak početnom polju?
    - Iskoristite round da na rezultirajućem polju tipa `float64` dobijete iste vrijednosti kao na početnom.

    **Napomena:** rezultat ovog zadatka uvelike ovisi o računalu na kojem radite.

!!! hint
    Na temu aritmetike brojeva s pomičnim zarezom napisani su brojni radovi od kojih svakako vrijedi pročitati [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).

## Čitanje polja iz datoteka

Imamo li datoteku `podaci.txt` sadržaja

```
2.4   5.6     29
2     824919  11
27.3  6.1429  9.1
```

Modul `numpy` nudi funkciju `numpy.loadtxt()` koja prima

``` python
import numpy as np

podaci = open("podaci.txt")
matrica = np.loadtxt(podaci)
podaci.close()
```

!!! admonition "Zadatak"
    - Stvorite dvije datoteke, nazovite ih `matrica_a.txt` i `matrica_b.txt`. Matrica u prvoj datoteci neka bude oblika `(3, 5)`, a u drugoj datoteci oblika `(5, 4)`.
    - Izvršite čitanje podataka, a zatim izračunajte produkt dvaju matrica. Možete li izračunati oba produkta ili samo jedan? Objasnite zašto.

## Posebne funkcije za generiranje polja

- `np.arange()`
- `np.linspace()`
- `np.ogrid()`
- `np.mgrid()`

!!! todo
    Ovdje nedostaje objašnjenje i zadatak.

## Indeksiranje, cijepanje i iteriranje polja

Jednodimenzionalna i višedimenzionalna polja možemo indeksirati, cijepati, iterirati i manipulirati baš kao i liste i znakovne nizove.

!!! admonition "Zadatak"
    Stvorite jednodimenzionalno polje `a` proizvoljnih cijelobrojnih vrijednosti veličine 10 te isprobajte iduće naredbe:

    - `a[3]`
    - `a[2:6]`
    - `a[:8:2] = 1337`
    - `a[[1,3,4]] = 0`
    - `a[::-2]`
    - `a[0] * a[2] -1`
    - `a[[0,0,2]] = [1,2,3]`

!!! admonition "Zadatak"
    Stvorite višedimenzionalno polje `a` proizvoljnih cijelobrojnih vrijednosti oblika (5,4) te isprobajte iduće naredbe:

    - `a[2]`
    - `a[-1]`
    - `a[2:3]`
    - `a[0:4, 2]`
    - `a[::-2]`
    - `b[1:3, : ]`

!!! admonition "Zadatak"
    Stvorite jednodimenzionalno polje `a` proizvoljnih cijelobrojnih vrijednosti veličine 10 i listu `i = [1,3,5,9]` te isprobajte iduće naredbe:

    - `a[i]`
    - `a[i][2]`
    - `a[i][1:3]`
    - `a[i][:-1]`
    - `a[i] * a[i][1]`
    - `a[i]**2 + i`
    - `a[i] / i[1]`
    - `np.sin(a[i]) * np.cos(i[2])`

## Manipulacija oblicima polja

Oblik polja možemo mijenjati idućim funkcijama:

- `np.array.ravel()` -- sravanavanje višedimenzionalnog polja
- `np.array.transpose()` -- transponiranje polja
- `np.array.reshape()` -- na mjestu vraća polje promjenjenog oblika
- `np.array.resize()` -- promjena oblika polja

!!! admonition "Zadatak"
    Stvorite višedimenzionalno polje `a` proizvoljnih cijelobrojnih vrijednosti oblika (5,4). Učinite iduće:

    - Trajno promijenite oblik polja u (2,10).
    - Transponirajte polje uz povećavanje svih vrijednosti polja za kosinus od 5.
    - Stvorite jednodimenzionalno polje `b` preoblikovanjem polja `a` tako da su vrijednosti elementa u `b` dvostruko veće od vrijednosti elemenata iz `a`.

Spajanje i razdvajanje polja možemo vršiti idućim funkcijama:

- `np.vstack()` -- vertikalno spajanje
- `np.hstack()` -- horizontalno spajanje
- `np.column_stack()` -- dodavanje jednodimenzionalnih polja kao stupce na dvodimenzionalna polja
- `np.row_stack()` -- dodavanje jednodimenzionalnih polja kao redove na dvodimenzionalna polja
- `np.concatenate()` -- spajanje polja po određenoj osi polja
- `np.hsplit()` -- horizontalno razdvajanje
- `np.vsplit()` -- vertikalno razdvajanje

!!! admonition "Zadatak"
    Stvorite polje `a` oblika (2,4) proizvoljnih cjelobrojnih vrijednosti.

    - Funkcijom za horizontalno razdvajanje razdvojite `a` na dva jednaka dijela te rezultat spremite u `b`. Kojeg tipa podataka je `b`. Što sadrži?
    - Nad `b` iskoristite funkcije za horizontalno te za vertikalno spajanje polja pa zatim usporedite dobivene rezultate. Kojeg su oblika dobivena polja?

## Kopije objekata

Pridruživanje polja određenoj varijabli ne stvara se kopija objekta polja pa ni podataka koji ga sačinjavaju. Izmjenom vrijednosti u polju putem te varijable mijenjamo vrijednosti samog polja, primjerice:

``` python
>>> a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> b = a
# Nije stvoren novi objekt
>>> b is a
True
>>> b.shape = 5,2
# Promjena objekta b mijenja i objekt a
>>> a.shape
(5, 2)
```

Kopiju možemo stvoriti funkcijom `np.array.copy()`.

!!! admonition "Zadatak"
    - Stvorite jednodimenzionalno polje `a` proizvoljnih cijelobrojnih vrijednosti veličine 10 te funkcijom stvorite objekt `a_kopija` koji je kopija objekta `a`.
    - Promjenite proizvoljnu vrijednost polja `a_copy` te nakon toga promijenite oblik proizvoljnom funkcijom za mijenjanje oblika.
    - Usporedite sadržaj i oblik polja `a` i `a_copy`. Što možete zaključiti? Je li promjena vrijednosti objekta `a_copy` utjecala na vrijednosti objekta `a`?

## Polinomski fit

Numpy može izvesti fitanje polinoma na zadane točke korištenjem metode najmanjih kvadrata funkcijom `numpy.polyfit()` ([službena dokumentacija](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)).
