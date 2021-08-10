---
author: Vedran Miletić
---

# Python: moduli

- moduli omogućavaju odvajanje neovisne funkcionalnosti u zasebne cjeline

## Korištenje postojećih modula

- `import ime_modula` učitava modul `ime_modula`

    - sve konstante i funkcije postaju dostupne u obliku `ime_modula.ime_funkcije()`, odnosno  `ime_modula.ime_konstante`
    - popis konstanti i funkcija dobijemo s `dir(ime_modula)`

    ``` python
    import math
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    #include <cmath>
    using namespace std;
    ```

- kompletiranje imena slično `bash`-u

    ``` python
    import readline
    import rlcompleter
    readline.parse_and_bind("tab: complete")
    ```

!!! admonition "Zadatak"
    Proučite dokumentaciju modula `random`, a zatim iskoristite funkcije koje nudi da bi generirali:

    - slučajan cijeli broj u rasponu $[1, 10]$,
    - slučajan realan broj u intervalu $[1, 5 \rangle$,
    - deset slučajnih cijelih brojeva u rasponu $[1, 50]$, i to tako da nema ponavljanja.

## Stvaranje vlastitih modula

Python modul može biti jedna datoteka ili jedan direktorij.

!!! note
    Mi ćemo se ovdje baviti samo slučajem kada se radi o modulu koji je jedna datoteka; studenti koje zanima situacija kada je u pitanju direktorij više informacija o tome mogu pronaći u [službenoj dokumentaciji](https://docs.python.org/3/tutorial/modules.html).

Stvorimo u direktoriju dvije datoteke, datoteku `modul1.py` sadržaja

``` python
def funkcija():
   return 42

varijabla = "paralelno i distribuirano programiranje"
```

i datoteku `program1.py` sadržaja

``` python
import modul1

if __name__ == "__main__":
    print(modul1.funkcija())
    print(modul1.varijabla)
```

a zatim pokrenimo datoteku `program1.py`. Konstrukt `if __name__ == '__main__'` omogućuje nam uključivanje koda u datoteci `program1.py` kao modula; to općenito ima smisla kada se unutar samog programa također definiraju neke funkcije. Naime,

- u slučaju da pokrenemo kod naredbom ljuske `python program1.py`, `__name__` će biti jednako `"__main__"`, dok
- u slučaju da pokrenemo kod uključivanjem u novoj datoteci `program2.py` Python naredbom `import program1`, `__name__` će biti jednako `"program1"`.

Da zaključimo, Python nam ovime omogućuje korištenje svakog našeg programa kao modula.

!!! admonition "Zadatak"
    - U datoteci `modul1.py` dodajte još jednu funkciju, nazovite ju `moja_funkcija()` s argumentima `arg1` i `arg2` koja vraća `42 * arg1 + 24 * arg2`.
    - Stvorite datoteku `modul2.py`, u njoj definirajte funkciju `say_hello()` koja vraća niz znakova `"I just came to say hello, o-o-o-o-o"`.
    - U datoteci `program1.py` uključite drugi modul i pozovite obje funkcije.

## Primjer modularizacije za PyCUDA aplikaciju

Kod programiranja većih aplikacija potrebno je kod modularizirati, dokumentirati i testirati. Pokažimo to na primjeru koda za računanje zbroja matrica u jednom bloku. Vidjet ćemo da je sasvim prirodno u tako podijeljen kod dodati još funkcionalnosti, primjerice množenje matrica ili nešto drugo.

Datoteka `matrix_gpu_ops.cu` je sadržaja

``` c
__global__ void matrix_sum (float *dest, float *a, float *b)
{
  const int i = threadIdx.y * blockDim.x + threadIdx.x;
  dest[i] = a[i] + b[i];
}
```

Datoteka `matrix_gpu_ops.py` je sadržaja

``` python
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as ga
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule(open("matrix_gpu_ops.cu").read())

matrix_sum = mod.get_function("matrix_sum")

def zbroj_matrica(a, b):
    """
    Funkcija vrši zbrajanje matrica.

    Argumenti:
    a -- matrica tipa np.array
    b -- matrica tipa np.array

    Vraća:
    Zbroj matrica a i b.
    """
    # provjera jesu li prikladnog formata
    assert a.shape == b.shape

    result_gpu = np.empty_like(a)
    matrix_sum(drv.Out(result_gpu), drv.In(a), drv.In(b),
               block=(20,20,1), grid=(1,1))

    return result_gpu

# TODO matrix_product = mod.get_function("matrix_product")

def produkt_matrica(a, b):
    """
    Nije implementiran.
    """
    return None
```

Datoteka `test_matrix_gpu_ops.py` je oblika

``` python
import matrix_gpu_ops
import numpy as np

def test_zbroj_matrica():
    a = np.ones((20, 20), dtype=np.float32)
    b = np.ones((20, 20), dtype=np.float32)
    result_cpu = a + b
    assert (matrix_gpu_ops.zbroj_matrica(a, b) == result_cpu).all()
```

Naposlijetku, datoteka `program.py` u kojoj koristimo napisanu funkciju je oblika

``` python
import matrix_gpu_ops
import numpy as np

a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)
print("Zbroj matrica a i b iznosi", matrix_gpu_ops.zbroj_matrica(a, b))
```

Pri čemu možemo imati i neke druge vrijednosti.

!!! admonition "Zadatak"
    - Implementirajte množenje matrica, i pritom dodajte pripadnu dokumentaciju i testove.
    - Modificirajte odgovarajući dio koda tako da radi za druge formate matrica osim `(20, 20)`, odnosno da u ovisnosti o obliku matrice poziva odgovarajući broj niti po `x` i `y` koordinatama, a zatim prilagodite dokumentaciju i testove.
