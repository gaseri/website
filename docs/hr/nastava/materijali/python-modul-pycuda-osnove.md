---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: osnove rada s GPU-om

## Podešenja za rad u laboratoriju

Emacs je potrebno naučiti da prepoznaje CUDA/OpenCL datoteke kao C. U datoteku `.emacs` dodajte

``` common-lisp
(setq auto-mode-alist (cons '("\\.cu$" . c-mode) auto-mode-alist))
(setq auto-mode-alist (cons '("\\.cl$" . c-mode) auto-mode-alist))
```

Prva linija postavlja CUDA C/C++ datoteke s izvornim kodom da koriste Emacsov način rada s C-om, a druga radi to isto za datoteke s OpenCL izvornim kodom.

## Hello world primjer

Hello world primjer izgradit ćemo u nekoliko koraka:

- uvoz `pycuda.autoinit`, `pycuda.driver` i `numpy` modula
- uvoz funkcije `SourceModule` iz modula `pycuda.compiler`

    ``` python
    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy as np
    from pycuda.compiler import SourceModule
    ```

- definicija tzv. **zrna** (engl. *kernel*), funkcija koja se izvodi na uređaju, odnosno funkcije koja će se izvoditi na GPU-u (u C sintaksi) i dohvaćanje funkcije u Python

    - `__global__` i `void` su nužni da bi funkcija bila kernel
    - koristimo `printf` umjesto C++-ovog `std::cout`

    ``` python
    mod = SourceModule("""
    #include <stdio.h>

    __global__ void hello()
    {
      printf ("Pozdrav s GPU-a!\\n");
    }
    """)

    hello = mod.get_function("hello")
    ```

!!! note
    Uočite dvostruki znak `\` unutar funkcije `hello()` koji je potreban obzirom da se kod unosi kao znakovni niz. U složenijim primjerima odvajat ćemo kod za CPU i kod za GPU i neće biti potrebe za dupliranjem tog znaka.

- izvođenje koda na GPU-u ([dokumentacija koja objašnjava pozivanje funkcija](https://documen.tician.de/pycuda/driver.html#pycuda.driver.Function))

    - prvo se navode argumenti kernela kojih ovdje nema
    - `block` i `grid` su stvari kojima se bavimo kasnije

    ``` python
    hello(block=(1,1,1), grid=(1,1))
    ```

- izvođenje koda na CPU-u

    ``` python
    print("Pozdrav s CPU-a!")
    ```

!!! admonition "Zadatak"
    - Unutar istog modula s izvornim kodom definirajte još jednu funkciju i nazovite je `my_hello()` koja pozdravlja na francuskom (`"Salutations avec le GPU!"`). Dohvatite je u Pythonu i pozovite na isti način kao `hello()`.
    - Unutar funkcije `my_hello()` inicijalizirajte dvije varijable, `var1` tipa `int` vrijednosti `480` i `var2` tipa `float` vrijednosti `2.075`. Proučite dio koji se odnosi na C unutar [Wikipedijine stranice za printf](https://en.wikipedia.org/wiki/Printf_format_string) i učinite da ispisuje i `Vrijednost varijable var1 je <vrijednost var1>, a vrijednost varijable var2 <vrijednost var2>.`
    - Varirajte brojeve u uređenoj trojci `block` i uređenom paru `grid` za vašu funkciju, stavite primjerice 2 ili 3 umjesto 1 na nekim mjestima. Što uočavate?

## Odvajanje CUDA C/C++ koda u posebnu datoteku

Baratanje sa kodom koji je znakovni niz često je dosta nezgrapno (neadekvatan je highlighting, nedostaje kompletiranje koda, itd.). Zbog toga želimo odvojiti CUDA C/C++ kod od Python koda PyCUDA aplikacije.

U prethodnom hello world primjeru cjelokupni kod je oblika

``` python
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <stdio.h>

__global__ void hello()
{
  printf ("Pozdrav s GPU-a!\\n");
}
""")

hello = mod.get_function("hello")

hello(block=(1,1,1), grid=(1,1))

print("Pozdrav s CPU-a!")
```

Kod možemo podijeliti u dvije datoteke, od kojih je prva naziva `hello.cu` i sadržaja

``` c
#include <stdio.h>

__global__ void hello()
{
  printf ("Pozdrav s GPU-a!\n");
}
```

Uočite kako sada nema potrebe za dodatnim znakom `\` kod `\n`, što dodatno povećava eleganciju ovog pristupa. Python kod sada na odgovarajućem mjestu učitava `hello.cu` i on je oblika

``` python
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule(open("hello.cu").read())

hello = mod.get_function("hello")

hello(block=(1,1,1), grid=(1,1))

print("Pozdrav s CPU-a!")
```

Pokretanjem uočavamo da kod radi identično kao onaj kod kojeg su ova dva dijela spojeni. Uočimo također da je linija

``` python
mod = SourceModule(open("hello.cu").read())
```

zapravo skraćeni zapis za

``` python
datoteka = open("hello.cu")
datoteka_sadrzaj = datoteka.read()
mod = SourceModule(datoteka_sadrzaj)
```

i taj skraćeni zapis ćemo zbog praktičnosti i dalje koristiti. U narednim primjerima ime datoteke s ekstenzijom `.cu` će varirati, stoga pripazite da ju adekvatno nazovete.
