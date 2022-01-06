---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: otklanjanje grešaka i curenja memorije

Jednostavno otklanjanje grešaka ("debuggiranje") smo već radili ispisivanjem vrijednosti varijabli korištenjem funkcije `printf()`. Sofisticiraniji način za istu stvar je korištenjeme makro naredbe, primjerice `INFUNIRI_DEBUG` na način

``` c
#include <stdio.h>
#define INFUNIRI_DEBUG

__global__ void funkcija (...)
{
  ...
  #ifdef INFUNIRI_DEBUG
  printf ("Varijabla 1 ima vrijednost %d, varijabla 2 ima vrijednost %2.3f\n", var1, var2);
  #endif
  ...
}
```

Prednost ovog pristupa je što kod završne verzije alata nije potrebno brisati čitav niz poziva funkcije `printf()`, već je dovoljno maknuti ili zakomentirati liniju `#define INFUNIRI_DEBUG`. Naime, to će učiniti da sve makro naredbe `#ifdef INFUNIRI_DEBUG` vrate `false`, i do prevođenja koda koji sadrži `printf()` neće ni doći.

## Program za otklanjanje grešaka

### Podsjetnik: prevođenje C++ programa

Primjerice, uzmemo li jednostavan C++ program nazvan `program1.cpp` čiji je kod oblika:

``` c++
#include <iostream>

using namespace std;

void ispis (int &var1, float &var2)
{
  var1 = 3;
  var2 = 8.3;
  cout << "Funkcija ispis" << endl;
}

int main ()
{
  int a = 5;
  float b = 2.7;
  cout << "Pocetak" << endl;
  ispis (a, b);
  cout << "Kraj" << endl;
  return 0;
}
```

Prevođenje i pokretanje izvodimo na način

``` bash
$ g++ program1.cpp -o program1
$ ./program1
```

### Rad s alatom gdb

Program za pronalaženje pogrešaka (engl. *debugger*) je alat koji pomaže programeru u pronalaženju semantičkih grešaka u kodu. Sam po sebi, on ne ispravlja kod.

Radi tako da pokreće instancu programa u kontroliranom okruženju i time omogućuje programeru da:

- pokreće program u koracima na razini programskog jezika,
- ispiše vrijednosti varijabli i izraza za vrijeme pokretanja,
- promijeni tok programa kod pokretanja,
- (neki debuggeri) obrnuto pronalaženje grešaka, odnosno odlazak unatrag i poništavanje destruktivnih operacija spremanjem serije stanja.

Koristit ćemo [GNU Debugger](https://en.wikipedia.org/wiki/GNU_Debugger) (`gdb`), koji radi na operacijskim sustavima sličnim Unixu i Windowsima. Nema vlastito grafičko sučelje, ali postoji niz alata koji nude korisniku prijateljsko sučelje (primjerice, `ddd` te IDE-i kao što su [Eclipse](https://www.eclipse.org/) i [NetBeans](https://netbeans.org/)).

`gdb` je simbolički debugger, odnosno radi na razini izvornog koda, pa je sposoban analizirati program na razini programskog jezika (ne samo asemblerskoj). Simbolički debuggeri su specifični za programski jezik s kojim rade i zahtijevaju dodatne informacije (debug simbole) kako bi preslikali asemblerske instrukcije na izvorni kod.

Debug simboli proizvode se kod prevođenja programa (primjerice, parametar `-g` kod `gcc`-a) i:

- integrirani u izvršnu datoteku (u tom slučaju su izvršne datoteke mnogo veće), ili
- odvojeni od izvršne datoteke (primjerice, kod .deb/.rpm paketa u -dbg/-debug paketima).

Debug simboli sadrže informacije o:

- koje linije izvornog koda stvaraju koje asemblerske instrukcije,
- imenima varijabli.

Dakle, pokretanje pomoću alata `gdb` vršimo na način

``` bash
$ g++ -g program1.cpp -o program1_debug
$ gdb program1_debug
```

Koristit ćemo iduće naredbe

- `quit` -- izlaz iz alata
- `break <broj linije ili ime funkcije>` -- postavljanje točke prekida izvođenja programa; dvije mogućnosti

    - broj linije na kojoj postavljamo breakpoint
    - ime funkcije na kojoj postavljamo breakpoint

- `run` -- pokretanje programa do prvog breakpointa
- `continue` -- nastavak izvođenja nakon stajanja na breakpointu
- `print <ime varijable>` -- ispis podataka o varijabli (trenutna adresa i vrijednost)
- `info locals` -- ispis podataka o lokalnim varijablama unutar trenutnog dosega
- `info args` -- ispis podataka o argumentima trenutne funkcije
- `help <naredba>` -- pomoć o naredbi
- `next`, `step`, `finish`, [...](https://beej.us/guide/bggdb/)

!!! admonition "Zadatak"
    - Unutar funkcije ispis dodajte još jednu promjenu varijabli, a zatim još jedan ispis teksta po vlastitom izboru. Izvršite prevođenje koda s debug simbolima i pokrenite ga u alatu `gdb`.

        - Postavite tri breakpointa: funkcija ispis, postojeća linija koja sadrži `cout` unutar funkcije i linija koja sadrži `cout` koju ste sami dodali.
        - Izvršite pokretanje. Kod svakog breakpointa ispišite stanje obje varijable.

### Specifičnosti alata cuda-gdb

Specifične naredbe alata `cuda-gdb` su

- `cuda kernel` -- ispis podataka o trenutnom CUDA zrnu ili odabir zrna
- `cuda grid` -- ispis podataka o trenutnoj CUDA rešetci ili odabir rešetke
- `cuda block` -- ispis podataka o trenutnom CUDA bloku ili odabir bloka
- `cuda thread` -- ispis podataka o trenutnoj CUDA niti ili odabir niti
- `cuda device` -- ispis podataka o trenutnom CUDA uređaju ili odabir uređaja
- `cuda sm` -- ispis podataka o trenutnom CUDA streaming multiprocesoru ili odabir streaming multiprocesora
- `cuda warp` -- ispis podataka o trenutnoj CUDA osnovi ili odabir osnove
- `cuda lane` -- ispis podataka o trenutnoj CUDA stazi ili odabir staze

Modul PyCUDA se može koristiti u kombinaciji s alatom `cuda-gdb` prema [službenim uputama](https://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions/#is-it-possible-to-use-cuda-gdb-with-pycuda). Iskoristimo primjer koji vrši zbrajanje vektora; alat pokrećemo naredbom

``` bash
$ cuda-gdb --args python -m pycuda.debug zbrajanje_vektora.py
...
(cuda-gdb) break vector_sum
... (zanemarite grešku da funkcija nije definirana) ...
(cuda-gdb) run
```

!!! admonition "Zadatak"
    Na primjeru za zbroj vektora isprobajte ovdje navedene specifične naredbe alata `cuda-gdb`.

## Otklanjanje curenja memorije

!!! todo
    Ovaj dio treba napisati u cijelosti.
