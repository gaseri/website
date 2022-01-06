---
author: Vedran Miletić
---

# Prevođenje C, C++ i Fortran programa u izvršni kod

[GNU Compiler Collection](https://en.wikipedia.org/wiki/GNU_Compiler_Collection) (GCC) je najkorišteniji otvoreni skup program-prevoditelja.

- inicijalno je podržavao samo C, pa je GCC bila kratica za *GNU C Compiler*
- podržava jezike: C (`gcc`), C++ (`g++`), Objective-C (`gobjc`), Objective-C++ (`gobjc++`), Fortran (`gfortran`), Java (`gcj`), Ada (`gnat`) i Go (`gccgo`)

[Clang](https://en.wikipedia.org/wiki/Clang) i [LLVM](https://en.wikipedia.org/wiki/LLVM) čine alternativu GCC-u.

- razvoj je započeo na [University of Illinois at Urbana-Champaign](https://illinois.edu/), danas ga dobrim dijelom sponzorira Apple
- Clang podržava C, Objective-C (`clang`), C++ i Objective-C++ (`clang++`), a LLVM ima dodatke koji podržavaju i druge programske jezike

!!! admonition "Zadatak"
    - Napišite u `emacs`-u "hello world" C++ program s pozdravom po vašem izboru i spremite ga kao `prog.cpp`.
    - Iskoristite `g++` ili `clang++` za prevođenje u izvršni kod. Pokrenite rezultirajući program.
    - Razmotrite rezultat s `hexdump`-om. Pronađite način da ga ispišete u kanonskom obliku.

!!! admonition "Dodatni zadatak"
    - Prema uputama [Wikibooksove knjige o Fortranu](https://en.wikibooks.org/wiki/Fortran/Hello_world) složite program s istim pozdravom kao u prethodnom zadatku i spremite ga kao `prog.f`. Pripazite da imate uvlaku od 6 znakova u svakom retku.
    - Iskoristite `gfortran` za prevođenje u izvršni kod. Pokrenite rezultirajući program.
    - Razmotrite rezultat s `hexdump`-om i usporedite ga s programom iz prethodnog zadatka.

## Izrada biblioteka i povezivanje

[Biblioteka](https://en.wikipedia.org/wiki/Library_(computing)) (engl. *library*) je kolekcija resursa (funkcija, klasa, konstanti, ...) koju programi mogu koristiti. Biblioteke omogućuju višestruko iskorištenje koda.

Biblioteke se razlikuju od uobičajenih programa po tome što nemaju funkciju `main()`. Primjerice, od dvije datoteke `bibl1.cpp` sadržaja

``` c++
void set_value1 (int *num)
{
  *num = 10;
}
```

i `bibl2.cpp` sadržaja

``` c++
void set_value2 (int *num)
{
  *num = 20;
}
```

datoteka zaglavlja `bibl.h` je oblika

``` c++
void set_value1 (int *num);
void set_value2 (int *num);
```

dok datoteka programa `mainprog.cpp` koji koristi te dvije funkcije može biti oblika

``` c++
#include <iostream>
#include "bibl.h"

int main ()
{
  int a, b;
  set_value1 (&a);
  set_value2 (&b);
  std::cout << "a=" << a << " b=" << b << std::endl;
  return 0;
}
```

Povezivanje (engl. *linking*) podrazumijeva povezivanje s bibliotečnim datotekam.

- statičko povezivanje (engl. *static linking*) u izvršnu datoteku uključuje sve potrebne dijelove bibliotečnih datoteka
- dinamičko povezivanje (engl. *dynamic linking*) u izvršnu datoteku stavlja samo poveznice na bibliotečne datoteke

Statičko povezivanje u gornjem primjeru izveli bismo na način:

``` shell
$ g++ -c bibl1.cpp bibl2.cpp
$ ar rcs libbibl.a bibl1.o bibl2.o
$ g++ -static -o mainprog mainprog.cpp -L. -lbibl
$ ./mainprog
```

Program `ar` je arhiver, donekle sličan `tar`-u koji već poznajemo. On ovdje služi za stvaranje statičke biblioteke, datoteke s ekstenzijom `.a`.

Parametar `-L` kod `g++`-a navodi dodatnu putanju u kojoj treba tražiti biblioteke (pored predefiniranih na sustavu), a parametar `-l` navodi se jednom ili više puta zajedno s imenom biblioteke na koju je potrebno povezati program. Poredak parametara je značajan; više detalja ima [u komentaru Paula Pluzhnikova](https://groups.google.com/g/gnu.gcc.help/c/muvgXVAU6l0/m/fVpqbXYp7cEJ) na temu [linking problems and order](https://groups.google.com/g/gnu.gcc.help/c/muvgXVAU6l0/m/soh5AV77U3gJ) ([Google grupa gnu.gcc.help](https://groups.google.com/g/gnu.gcc.help/)).

!!! admonition "Zadatak"
    Modificirajte gornji primjer tako da biblioteka uključuje i datoteku `bibl3.cpp` sa dvjema funkcijama po vašem izboru čije se deklaracije navode u datoteci zaglavlja i koje se pozivaju u `mainprog.cpp`. Izvedite prevođenje programa sa statičkim povezivanjem.

Dinamičko povezivanje u datotekama primjera izveli bismo na način:

``` shell
$ g++ -fPIC -c bibl1.cpp bibl2.cpp
$ g++ -shared -Wl,-soname,libbibl.so.1 -o libbibl.so.1.0 bibl1.o bibl2.o
$ ln -sf libbibl.so.1.0 libbibl.so
$ ln -sf libbibl.so.1.0 libbibl.so.1
$ g++ -o mainprog -L. mainprog.cpp -lbibl
$ export LD_LIBRARY_PATH=.
$ ./mainprog
```

Parametar `-fPIC` osigurava da je strojni kod koji GCC stvara prevođenjem [neovisan o o položaju u memoriji](https://en.wikipedia.org/wiki/Position-independent_code), što omogućava dijeljenje istog od većeg broja programa.

Biblioteka u primjeru ima verziju `1.0`, što se vidi iz njezinog imena. Simboličke poveznice se stvaraju kako bismo joj mogli pristupiti i programi koji traže istoimenu biblioteku navođenjem samo verzije `1`, te programi koji traže istoimenu biblioteku bez da navode verziju.

Varijabla okoline `LD_LIBRARY_PATH` navodi dodatne putanje (pored onih predefiniranih na sustavu) u kojima je potrebno tražiti biblioteke.

!!! admonition "Zadatak"
    Modificirajte gornji slučaj da biblioteka uključuje i datoteku `bibl3.cpp` sa dvjema funkcijama po vašem izboru čije se deklaracije navode u datoteci zaglavlja i koje se pozivaju u `mainprog.cpp`. Izvedite prevođenje programa s dinamičkim povezivanjem.

!!! todo
    Ovdje treba opisati parametre za GCC koji se odnose na linkanje, alate `ldd` i `readelf`, a primjer može biti s [Boost](https://en.wikipedia.org/wiki/Boost_(C++_libraries)) bibliotekama `filesystem` i `datetime`.

!!! admonition "Zadatak"
    Usporedite međusobno dva programa i dvije biblioteke stvorene u prethodna dva zadatka.

    - Razmotrite veličinu datoteke i objasnite uzrok razlikama koje vidite.
    - Usporedite izlaz naredbe `file`.

## Alat `make`

Alat `make` "automatski" izgrađuje izvršne programe i bibliotečne datoteke iz izvornog koda.

- može raditi pomoću instrukcija zadanih u datoteci [Makefile](https://en.wikipedia.org/wiki/Make_(software)#Makefiles)
- može raditi na temelju predefiniranih ciljeva na temelju imena datoteka, u slučaju da se Makefile ne koristi
- postoje dvije inačice koje nisu 100% kompatibilne: [BSD Make i GNU Make](https://en.wikipedia.org/wiki/Make_(software)#Modern_versions)

!!! admonition "Zadatak"
    - Prevedite prethodno napisani program uz pomoć `make`-a. Koju naredbu `make` pokreće? Koji parametar čini da je ime izvršne datoteke isto kao ime datoteke izvornog koda bez ekstenzije?
    - Objasnite što se dogodi kada ponovno pokušate napraviti prevođenje. Koji je razlog tome?
    - Što bi se dogodilo da preimenujete datoteku koja sadrži program u datoteku s ekstenzijom `.c`?

`make` cilj (engl. *target*) je imenovani niz naredbi koje su potrebne za izgradnju određene ciljne datoteke (najčešće neke vrste izvršne datoteke)

- navedeni u Makefile datoteci, naziv `Makefile` ili `GNUmakefile`
- brojni primjeri Makefile datoteka mogu se pronaći [na Wikipedijinoj stranici](https://en.wikipedia.org/wiki/Make_(software)#Example_makefiles) i drugdje

Makefile se sastoji od jednog ili više ciljeva, koji mogu imati druge ciljeve kao svoje komponente.

``` makefile
target [target ...]: [component ...]
  [command 1]
      .
      .
      .
  [command n]
```

!!! admonition "Zadatak"
    Napišite jednostavan `Makefile` za prethodno napisani hello world program, koji sadrži:

    - cilj `doc` koji u datoteku README sprema tekst `Ovo je dokumentacija mojeg hello world programa`,
    - cilj `compile` koji prevodi program u izvršni kod,
    - cilj `static-compile` koji prevodi program u izvršni kod i pritom koristi statičko povezivanje,
    - cilj `pack` koji radi arhivu koja sadrži izvršni kod i README datoteku. Obavezno iskoristite komponente.

`Makefile` varijable slične su varijablama ljuske; **primjer:**

- zadavanje `NAME = mojprogram`, pozivanje sa `$(NAME)`
- zadavanje `VERSION = 3.2`, pozivanje sa `$(VERSION)`
- nazivaju se i makroi, zbog mogućnosti ekspanzije: `WHATSNEW_FILE_NAME = NEWS-$(VERSION)`

!!! admonition "Zadatak"
    Modificirajte prethodno napisani `Makefile`, tako da se

    - izvršna datoteka programa generira s imenom oblika `ime-verzija`,
    - arhiva generira s imenom oblika `ime-verzija.tar.bz2`.
