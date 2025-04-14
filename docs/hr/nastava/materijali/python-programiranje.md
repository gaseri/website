---
author: Matea Turalija, Vedran Miletić
---

# Programiranje i programski jezik Python

Motivacija:

- [The Real Reason Why Everyone Should Learn to Code](https://blog.teamtreehouse.com/havent-started-programming-yet)
- [Computational Biologist Melissa Wilson on Sex Chromosomes, Gila Monsters, and Career Advice](https://biobeat.nigms.nih.gov/2019/06/computational-biologist-melissa-wilson-on-sex-chromosomes-gila-monsters-and-career-advice/)

[Python](https://en.wikipedia.org/wiki/Python_(programming_language)) je programski jezik visoke razine opće namjene. Odlikuje ga jednostavnost učenja i čitljiv kod. Može se primijeniti u raznim područjima poput podatkovne znanosti, velikih podatka, testiranja, automatizacije, strojnog učenja, računalnih i mobilnih aplikacija i drugo. Nudi brzu izradu prototipa, jasnu sintaksu, veliki ekosustav biblioteka i podršku za više platformi.

## Preduvjeti

- [Visual Studio Code](inf-biotech-instalacija-softvera-windows-ubuntu.md#visual-studio-code)
- [Python](inf-biotech-instalacija-softvera-windows-ubuntu.md#python)
- [Jupyter Notebooks u Visual Studio Codeu](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

[Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/) projekt je otvorenog koda koji omogućuje jednostavno kombiniranje Markdown teksta i izvršnog Python izvornog koda na jednom platnu koje se zove bilježnica. Visual Studio Code podržava rad s Jupyter Notebooks izvorno i putem Python kodnih datoteka.

Ukoliko nemate instaliranu jezgru IPythona, možete je instalirati naredbom:

``` shell
pip install -U ipykernel
```

## Osnovna sintaksa

U VS Codeu stvorimo novu praznu Jupyter bilježnicu kombinacijoom tipki (++ctrl+shift+p++) i u paleti naredbi odaberite `Jupyter Notebook: Create New Blank Notebook`. Ova opcija stvara novu bilježnicu ekstenzije `.ipynb`.

U izvršnjoj ćeliji (engl. *execute cell*) napišimo i pokrenimo sljedeće:

``` python
print("Hello, World!")
```

!!! example "Zadatak"
    Ispiši svoje ime [i](https://youtu.be/6LyLkcwIiqI) prezime na ekran.

## Komentari

Komentari se označavaju znakom `#` i protežu se do kraja linije. Oni imaju ulogu opisivanja dijelova koda radi lakšeg razumijevanja i ne izvršavaju se tijekom izvođenja programa.

``` python
# Program koji ispisuje Hello, World! na ekran
print("Hello, World!") # Još jedan komentar
# Treći komentar
print("# Ovo nije komentar jer se nalazi unutar navodnika!")
```

Python omogućava korištenje komentara i kroz nekoliko linija programskog koda. Takvi komentari počinju i završavaju s tri dvostruka navodnika`"""`:

``` python
"""Ovo je komentar
koji se proteže
kroz nekoliko
linija koda"""
```

## Varijable

Varijable u programu čuvaju vrijednosti i omogućuju pristup tim informacijama iz različitih dijelova koda. Možemo ih zamisliti kao "kutije" u koje pohranjujemo vrijednosti, koje zatim možemo koristiti, mijenjati ili pridruživati drugim varijablama.

Pridruživanje vrijednosti varijabli vrši se pomoću operatora `=`. Nazivi varijabli trebali bi logički odražavati pohranjenu vrijednost. Na primjer, varijablu koja sadrži zbroj brojeva prikladnije je nazvati `zbroj` nego `umnozak`.

``` python
varijabla = 5
print(varijabla)
```

!!! example "Zadatak"
    Spremi svoje ime [i](https://youtu.be/6LyLkcwIiqI) godine u varijable te ispiši podatke na ekran. Na primjer, `Iva ima 25 godina`.

    Napomena: kada u funkciji `print()` želimo ispisati više varijabli odvojit ćemo ih zarezom.

## Aritmetički operatori

Python podržava nekoliko aritmetičkih operatora za rad s brojevima. U nastavku je prikazana tablica koja sadrži popis dostupnih aritmetičkih operatora:

| Operator | Namjena | Primjer | Rezultat, `a = 10`; `b = 5` |
| :------: | ------- | ------- | --------------------------- |
| `+` | Zbrajanje (engl. *addition*) | `x = a + b` | `x = 15` |
| `-` | Oduzimanje (engl. *subtraction*) | `x = a – b` | `x = 5` |
| `*` | Množenje (engl. *multiplication*) | `x = a * b` | `x = 50` |
| `/` | Dijeljenje (engl. *division*) | `x = a / b` | `x = 2` |
| `%` | Ostatak dijeljenja (engl. *modulus*) | `x = a % b` | `x = 0` |
| `**` | Potenciranje (engl. *power*) | `x = a ** b` | `x = 100 000` |
| `//` | Cjelobrojno dijeljenje (engl. *flor division*) | `x = a // b` | `x = 2` |

Pored prethodno navedenih aritmetičkih postoje i skraćene verzije operatora pridruživanja:

| Operator | Primjer | Osnovni oblik | Rezultat, `a = 10`; `b = 5` |
| :------: | ------- | ------------- | --------------------------- |
| `+=` | `a += b` | `a = a + b` | `a = 15` |
| `-=` | `a -= b` | `a = a – b` | `a = 5` |
| `*=` | `a *= b` | `a = a * b` | `a = 50` |
| `/=` | `a /= b` | `a = a / b` | `a = 2` |
| `%=` | `a %= b` | `a = a % b` | `a = 0` |
| `**=` | `a **=b` | `a = a ** b` | `a = 100000` |
| `//=` | `a //= b` | `a = a // b` | `a = 2` |
| `=` | `a = b` | `a = b` | `a = 5` |

!!! example "Zadatak"
    1. U varijable imena po izboru spremi dva broja te na ekran ispiši njihov zbroj i umnožak.
    2. Napiši zbroj i umnožak iz prethodnog zadatka koristeći se skraćenim verzijama operatora pridruživanja.

## Operatori usporedbe

Operatori usporedbe uspoređuju vrijednosti s obje strane operatora i određuju rezultat usporedbe. Oni vraćaju vrijednosti 0 (netočno, engl. *false*) ili 1 (točno, engl. *true*). Često se koriste za kontrolu toka programa, posebno u uvjetima koji provjeravaju točnost rezultata logičkih operatora.

| Operator | Opis | Primjer | Rezultat, `a = 5`; `b = 10` |
| :------: | ---- | ------- | --------------------------- |
| `==` | Usporedba jednakosti | `a == b` | False |
| `!=` | Usporedba nejednakosti | `a != b` | True |
| `>` | Strogo veće od | `a > b` | False |
| `<` | Strogo manje od | `a < b` | True |
| `>=` | Veće ili jednako od | `a >= b` | False |
| `<=` | Manje ili jednako od | `a <= b` | True |

## Naredba `if`

Naredba `if` pripada kontroli toka za odluke. U najjednostavnijem obliku ova struktura sastoji se od zaglavlja i tijela odluke. Ako je uvjet zadovoljen, izvršava se tijelo odluke, u suprotnom, tijelo odluke se neće izvršiti.

Proučite primjere ispod i razmislite što će se ispisati na ekran:

``` python hl_lines="3"
a = 15
b = 200
if b > a:
    print("b je veće od a")
```

``` python hl_lines="3 5"
a = 15
b = 15
if b > a:
    print("b je veće od a")
elif a == b:
    print("a i b su jednaki")
```

``` python hl_lines="3 5 7"
a = 200
b = 15
if b > a:
    print("b je veće od a")
elif a == b:
    print("a i b su jednaki")
else:
    print("a je veće od b")
```

!!! example "Zadatak"
    1. Ispišite "1" ako je `a` jednako `b`, "2" ako je `a` veće od `b`, inače ispišite "3". Dodijelite vrijednosti varijblama po želji.
    2. Ispišite "Uvjet je zadovoljen" ako je `a` jednako `b` ili ako je `c` jednako `d`. Dodijelite vrijednosti varijblama po želji.

## Biblioteka NumPy

[NumPy](https://numpy.org/doc/stable/user/) je biblioteka za programski jezik Python koja pruža podršku za rad s višedimenzionalnim nizovima i matricama, zajedno s kolekcijom matematičkih funkcija za rad s tim strukturama podataka. Iako u Pythonu imamo liste koje funkcioniraju kao nizovi, rad s njima je spor u odnosu na druge programske jezike. NumPy koristi objekt niza koji se naziva [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) i nudi mnogo funkcionalnosti koje olakšavaju rad s nizovima.

Ako već imate instaliran Python i [pip](https://pip.pypa.io/en/stable/getting-started/), tada je instalacija NumPyja vrlo jednostavna:

``` shell
pip install numpy
```

Da biste pristupili NumPy-u i njegovim funkcijama, potrebno ga je uvesti u svoj Python kod:

``` python
import numpy as np
```

Skraćujemo uvođenje imena na `np` kako bismo poboljšali čitljivost koda koji koristi NumPy. Ova široko prihvaćena konvencija čini kod čitljivijim i jasnijim.

### Nula-dimenzionalni niz

Nula-dimenzionalni (0D) nizovi predstavljaju pojedinačne brojeve ili skalare. Ovi nizovi nemaju dimenzije, što znači da su jednostavno pojedinačni elementi, a ne vektori, matrice ili višedimenzionalni nizovi.

``` python
import numpy as np

niz = np.array(1)

print(niz)
```

### Jednodimenzionalni niz

Jednodimenzionalni (1D) niz je niz koji sadrži elemente poput uobičajenog niza ili vektora.

``` python
import numpy as np

niz = np.array([1, 2, 3, 4, 5, 6])

print(niz)
```

### Dvodimenzionalni niz

Dvodimenzionalni (2D) niz predstavlja matricu koja se sastoji od redova i stupaca elemenata.

``` python
import numpy as np

niz = np.array([[1, 2, 3], [4, 5, 6]])

print(niz)
```

Ovaj niz ima dva retka i tri stupca. Svaki redak predstavlja jednodimenzionalni niz, a skup redaka tvori dvodimenzionalni niz.

### Trodimenzionalni niz

Trodimenzionalni (3D) niz je skup elemenata organiziranih u tri dimenzije. To se može zamisliti kao skup dvodimenzionalnih nizova, stvarajući 3D strukturu podataka.

``` python
import numpy as np

niz = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(niz)
```

Broj dimenzije niza možemo saznati pomoću atributa `ndim`, a ukupan broj elemenata u nizu pomoću `size`.

``` python
import numpy as np

a = np.array(1)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

print(a.size)
print(b.size)
print(c.size)
print(d.size)
```

### Jednostavno kreiranje niza

Osim stvaranja niza od već definiranog niza elemenata, možete jednostavno stvoriti niz ispunjen nulama ili jedinicama pomoću `np.zeros(n)`, `np.ones(n)`, gdje je `n` broj elemenata niza.

!!! example "Zadatak"
    Kreirajte nula niz od 5 elemenata i niz ispunjen jedinicama od 3 elementa te ih ispišite na ekran.

Možete stvoriti niz s intervalom elemenata `np.arange(n)` ili niz s ravnomjerno raspoređenih intervala pri čemu ćete navesti prvi `n` i posljednji broj `m` te veličinu koraka `k`, `np.arange(n, m, k)`.

!!! example "Zadatak"
    1. Kreirajte niz koji sadrži elemente u intervalu od 10 do 15 i ispišite ga na ekran.
    2. Kreirajte niz koji sadrži elemente u intervalu od 2 do 8 s veličinom koraka 2 i ispišite ga na ekran.
    3. Stvorite niz od 5 elemenata. Neka se ti elementi generiraju nasumično u rasponu od 50 do 100 ([`np.random.randint`](https://www.programiz.com/python-programming/numpy/random)). Zatim izračunajte zbroj svih elemenata niza ([`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum)), srednju vrijednost ([`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html#numpy.ndarray.mean)) te razliku između najvećeg ([`np.max`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html#numpy.ndarray.max)) i najmanjeg ([`np.min`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html#numpy.ndarray.min)) elementa niza. Ispišite rezultate na ekran.

### Pristup elementima niza

Za dohvaćanje određenog elementa niza koristi se [indeksiranje](https://numpy.org/doc/stable/user/basics.indexing.html). Indeksi počinju s 0, što znači da prvi element niza ima indeks 0 (`niz[0]`), drugi indeks 1 (`niz[1]`) i tako dalje.

!!! example "Zadatak"
    1. Kreirajte niz koji sadrži elemente u intervalu od 1 do 10. Pristupite prvom i drugom elementu niza te ih zbrojite.
    2. Pomoću negativnog indeksiranja pristupite zadnjem elementu niza.

Elementima niza možemo pristupiti i momoću definiranog intervala od interesa:

- `niz[1:5]`
- `niz[3:]`
- `niz[:5]`
- `niz[1:9:2]`
- `niz[-3:3:-1]`

Za pristup elementima iz 2D nizova možemo koristiti cijele brojeve odvojene zarezima koji predstavljaju dimenziju i indeks elementa. Zamislite 2D nizove poput tablice s redovima i stupcima, gdje dimenzija predstavlja red, a indeks predstavlja stupac.

!!! example "Zadatak"
    Kreirajte 2D niz oblika `[[1,2,3,4,5], [6,7,8,9,10]]` i pristupite drugom elementu prvog niza.

Elementima iz 3D nizova također pristupamo pomoću cijelih brojeva odvojenih zarezima koji predstavljaju dimenzije i indeks elementa.

!!! example "Zadatak"
    Kreirajte 3D niz oblika `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]` i pristupite trećem elementu drugog podniza u prvom nizu.

## Biblioteka Matplotlib

[Matplotlib](https://matplotlib.org/stable/) je biblioteka za stvaranje statičnih, animiranih i interaktivnih vizualizacija. Većina Matplotlib pomoćnih programa nalazi se pod podmodulom `pyplot` i obično se uvoze pod aliasom `plt`.

``` python
import matplotlib.pyplot as plt
```

Koristit ćemo funkciju [`plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) za crtanje točaka u dijagramu. Prema zadanim postavkama, funkcija crta crtu od točke do točke. Primjerice, želimo li iscrtati liniju od točke `(1, 2)` do točke `(6, 8)`, moramo proslijediti dva niza `[1, 6]` i `[2, 8]` funkciji iscrtavanja:

``` python
import matplotlib.pyplot as plt
import numpy as np

xtocke = np.array([1, 6])
ytocke = np.array([2, 8])

plt.plot(xtocke, ytocke)
plt.show()
```

!!! example "Zadatak"
    1. Stvorite niz `godine` u intervalu od 1990. do tekuće godine i drugi niz `broj_radova` kojeg cete generirati slučajnim odabirom od 50 do 5 000. Neka je niz `broj_radova` iste veličine kao i niz `godine`. Nacrtajte grafički prikaz broja objavljenih radova po godinama.
    2. Izmjenite prethodini zadatak tako da grafičkom prikazu dodate odgovarajući [naslov](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html) i [oznake](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html).
    3. Nacrtajte grafički prikaz broja objavljenih radova od 1980 do 2020 koji uključuju izraz "molecular dynamics" u naslovu, sažetku ili ključnim riječima. Analizu provedite korištenjem baze [Web of Science](https://www.webofknowledge.com/). Dodajte odgovarajući naslov i godine.
    4. Nacrtajte grafički prikaz pojednostavljenog [Lennard-Jonesovog potencijala](https://en.wikipedia.org/wiki/Lennard-Jones_potential):

        $$U_{LJ}(r) = 4 \epsilon [(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6]$$

        zajedno s njegovim pojedinačnim članovima. Za parametre $\epsilon$ i $\sigma$ uzmite vrijednost `1`, a raspon udaljensoti na kojem će se prikazivati potencijal definirajte pomoću funkcije [`np.linspace()`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) i uzmite vrijednosti od `1` do `3`.
