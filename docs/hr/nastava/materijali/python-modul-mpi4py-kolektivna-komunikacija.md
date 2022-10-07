---
author: Vedran Miletić
---

# Python modul mpi4py: kolektivna komunikacija

Ova značajka je dio [standarda MPI-1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/mpi-report.html).

- operacije broadcast, scatter, gather, allgather, alltoall -- jednostavno objašnjenje dano je na slajdovima 30 i 31 prezentacije [MPI for Python od Lisandra Dalcina](https://www.bu.edu/pasi/files/2011/01/Lisandro-Dalcin-mpi4py.pdf)
- drastično pojednostavljuje način rada u situaciji više procesa izvodi vrlo sličan ili identičan niz naredbi

    - komunikacija točka-do-točke: u pravilu se slanja i primanja kodiraju posebno za svaki proces
    - kolektivna komunikacija: komunikacijske operacije pišu se jednom za sve procese koji sudjeluju u komunikaciji

## Operacija barrier

- `comm.Barrier()` -- sinkronizacija korištenjem barijere; čeka se da svi procesi dođu do barijere

``` python
import numpy as np
# import random
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sleep_time = 10 * np.random.random()
# sleep_time = 10 * random.random()
print("Proces ranga", rank, "spavat će", sleep_time, "sekundi")
time.sleep(sleep_time)

comm.Barrier()

print("Proces ranga", rank, "prošao je barijeru")
```

## Operacija broadcast

- `comm.bcast(obj, root)` vrijednost varijable sa korijenskog procesa šalje svima

    - `obj` je Python objekt koje si šalje
    - `root` je rang korijenskog procesa; podatak s njega bit će poslan svim ostalima i **prepisati preko podataka na ostalim procesima**

``` python
# varijanta s Python objektima
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    sendmsg = (7, "abc", [1.0, 2+3j], {3: 4})
else:
    sendmsg = None

recvmsg = comm.bcast(sendmsg, root=0)
```

- `comm.Bcast(array, root)` -- vrijednost varijable sa korijenskog procesa šalje svima

    - `array` je numpy polje koje se šalje; mora biti istog oblika na svim procesima
    - `root` je rang korijenskog procesa; podatak s njega bit će poslan svim ostalima i **prepisati preko podataka na ostalim procesima**

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    msg = np.array([1.5, 2.5, 3.5, 4.5])
else:
    msg = np.empty(4)

print("Prije broadcasta proces ranga", rank, "ima vrijednost varijable", msg)
comm.Bcast(msg, root=0)
print("Nakon broadcasta proces ranga", rank, "ima vrijednost varijable", msg)
```

!!! admonition "Zadatak"
    - Prilagodite kod primjera tako da se broadcastom šalje lista s 4 podliste s po 4 elementa, odnosno polje oblika (4, 4), u oba slučaja s proizvoljnim vrijednostima.
    - Učinite da nakon broadcasta proces ranga 2 promijeni vrijednost elementa na poziciji (1, 1) tako da je uveća za 5, i zatim napravite broadcast s njega svim ostalima.
    - Inicijalizirajte na procesu ranga 1 polje formata (5, 15) kodom `np.array([list(range(x, x + 15)) for x in range(5)])` i napravite broadcast s njega svim ostalima (pripazite na tip polja u koje primate). Učinite da svaki od procesa po primitku polja ispisuje na ekran vrijednost na koordinati (rang, rang).

    Na svim procesima ispišite primljene podatke na ekran da provjerite je li operacija bila uspješna.

## Operacija scatter

- `comm.scatter(sendmsg, root)` listu od `n` elemenata raspršuje na `n` procesa, tako da svaki proces dobiva po jedan element i to točno onaj s indeksom koliki je njegov rang

    - `sendmsg` je Python lista koja se raspršuje; **mora biti iste veličine kao broj procesa** na korijenskom procesu, i `None` na ostalima
    - `root` je rang korijenskog procesa

``` python
# varijanta s Python objektima
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    sendmsg = [i**2 for i in range(size)]
else:
    sendmsg = None

recvmsg = comm.scatter(sendmsg, root=0)
```

- `comm.Scatter(sendmsg, recvmsg, root)` -- listu od `n` elemenata raspršuje na `n` procesa, tako da svaki proces dobiva po jedan element i to točno onaj s indeksom koliki je njegov rang

    - `sendmsg` je numpy polje koje se raspršuje; **mora biti iste veličine kao broj procesa** na korijenskom procesu, i `None` na ostalima
    - `recvmsg` je numpy polje veličine jednog elementa u koje se sprema rezultat raspršenja
    - `root` je rang korijenskog procesa

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    sendmsg = np.arange(size, dtype=np.int32)
else:
    sendmsg = None

recvmsg = np.empty(1, dtype=np.int32)
comm.Scatter(sendmsg, recvmsg, root=0)
print("Proces ranga", rank, "ima vrijednost poruke za slanje", sendmsg, "primljena poruka", recvmsg)
```

!!! admonition "Zadatak"
    - Modificirajte kod tako da se izvodi u 5 procesa, a scatter se vrši nad matricom proizvoljnih vrijednosti koja ima 5 redaka i 4 stupca zapisanom u obliku liste, odnosno numpy polja.
    - Učinite da svaki od procesa posljednji element u primljenom retku matrice postavlja na vrijednost 0, a zatim ispisuje taj redak na ekran.

## Operacije gather i allgather

- `comm.gather(sendmsg, root)` radi obrnuto od `scatter()`, po jednu varijablu sa svakog procesa skuplja u listu na korijenskom procesu i to tako da vrijednost postaje element s indeksom koliki je rang pripadnog procesa

    - `sendmsg` je Python objekt koji se šalje korijenskom procesu
    - `root` je rang korijenskog procesa

- `comm.allgather(sendmsg)` vrši istu operaciju kao `gather()`, osim što sada rezultirajuću listu dobivaju svi procesi, a ne samo korijenski

    - `sendmsg` je Python objekt koji se šalje

``` python
# varijanta s Python objektima
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendmsg = rank**2

recvmsg1 = comm.gather(sendmsg, root=0)

recvmsg2 = comm.allgather(sendmsg)
```

- `comm.Gather(sendmsg, recvmsg, root)` -- vrši obrnut proces od `Scatter()`, po jednu varijablu sa svakog procesa skuplja u polje na korijenskom procesu i to tako da se vrijednost postavlja na indeks koliki je rang pripadnog procesa

    - `sendmsg` je numpy polje koje se šalje korijenskom procesu
    - `recvmsg` je numpy polje u koje se sakupljaju primljene vrijednosti; ono mora biti veličine koliki je broj procesa na korijenskom procesu, i `None` na ostalima
    - `root` je rang korijenskog procesa

- `comm.Allgather(sendmsg, recvmsg)` -- vrši operaciju sakupljanja kao i `Gather()`, osim što sada rezultirajuću listu dobivaju svi procesi, a ne samo korijenski

    - `sendmsg` je numpy polje koje se šalje
    - `recvmsg` je numpy polje u koje se sakupljaju primljene vrijednosti

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendmsg = np.array([rank**2], dtype=np.int32)

print("Proces ranga", rank, "ima vrijednost poruke za slanje", sendmsg)

if rank == 0:
    recvmsg1 = np.empty(size, dtype=np.int32)
else:
    recvmsg1 = None

comm.Gather(sendmsg, recvmsg1, root=0)

recvmsg2 = np.empty(size, dtype=np.int32)

comm.Allgather(sendmsg, recvmsg2)

print("Proces ranga", rank, "ima vrijednost prve primljene poruke", recvmsg1, "i druge primljene poruke", recvmsg2)
```

!!! admonition "Zadatak"
    - Promijenite program da se izvodi u 5 procesa, i to tako da svaki od procesa inicijalizira listu, odnosno polje, duljine 4 u kojem je prvi element njegov rang, a ostali elementi su slučajne vrijednosti u rasponu od 0.0 do 1.0.
    - Napravite operaciju gather sa korijenskim procesom ranga 3, te operaciju allgather. Svi procesi neka oba rezultata ispišu na ekran.

## Operacije reduce i allreduce

- `comm.reduce(sendmsg, op, root)` vrši proces redukcije (suma, produkt, maksimum, minimum) na danom skupu elemenata koji čini po jedna vrijednost sa svakog od procesa i rezultat redukcije sprema u varijablu na korijenskom procesu

    - `sendmsg` je Python objekt na kojem se vrši redukcija
    - `op` je operator redukcije: `MAX`, `MIN`, `SUM`, `PROD`, ... ([čitav popis predefiniranih operatora moguće je pronaći u standardu MPI 1.1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node78.html))

- `comm.allreduce(sendmsg, op)` vrši istu operaciju kao `reduce()`, osim što sada rezultat dobivaju svi procesi, a ne samo korijenski

    - `sendmsg` je Python objekt na temelju kojeg se vrši redukcija
    - `op` je operator redukcije

``` python
# varijanta s Python objektima
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendmsg = rank

recvmsg1 = comm.reduce(sendmsg, op=MPI.SUM, root=0)

recvmsg2 = comm.allreduce(sendmsg)
```

- `comm.Reduce(sendmsg, recvmsg, op, root)` -- vrši proces redukcije (suma, produkt, maksimum, minimum) na danom skupu elemenata koji čini po jedna vrijednost sa svakog od procesa i rezultat redukcije sprema u varijablu na korijenskom procesu

    - `sendmsg` je numpy polje na kojem se vrši redukcija
    - `recvmsg` je numpy polje u koje se sprema rezultat; ono mora biti odgovarajuće veličine na korijenskom procesu, i `None` na ostalima
    - `op` je operator redukcije: `MAX`, `MIN`, `SUM`, `PROD`, ... ([čitav popis predefiniranih operatora moguće je pronaći u standardu MPI 1.1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node78.html))
    - `root` je rang korijenskog procesa

- `comm.Allreduce(sendmsg, recvmsg, op)` -- vrši istu operaciju kao `Reduce()`, osim što sada rezultat dobivaju svi procesi, a ne samo korijenski

    - `sendmsg` je numpy polje na temelju kojeg se vrši redukcija
    - `recvmsg` je numpy polje u koje se sprema rezultat
    - `op` je operator redukcije

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendmsg = np.array([rank**2], dtype=np.int32)

if rank == 0:
    recvmsg1 = np.empty(1, dtype=np.int32)
else:
    recvmsg1 = None

comm.Reduce(sendmsg, recvmsg1, op=MPI.SUM, root=0)

recvmsg2 = np.empty(1, dtype=np.int32)

comm.Allreduce(sendmsg, recvmsg2, op=MPI.SUM)

print("Proces ranga", rank, "ima vrijednost prve primljene poruke", recvmsg1, "i druge primljene poruke", recvmsg2)
```

!!! admonition "Zadatak"
    Promijenite kod tako da:

    - na glavnom procesu se inicijalizira lista ili polje slučajnih vrijednosti, koje se raspršuju (scatter) na sve procese,
    - svaki od procesa kvadrira vrijednost koju primi,
    - zatim se vrši redukcija na proces ranga 0 operacijom MIN i redukcija na sve procese operacijom MAX.

    Neka se takav program izvodi u 8 procesa. Alternativno, napravite da se može izvoditi u proizvoljnom broju procesa.

## Operacija alltoall

- `comm.alltoall(sendmsg)` -- element vektora sa indeksom `j` iz `sendmsg` na procesu `i` postaje element vektora sa indeksom `i` u `recvmsg` na procesu `j`

    - `sendmsg` je lista koja se šalje, veličine jednake ukupnom broju procesa

``` python
# varijanta s Python objektima
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendmsg = [rank * size + i for i in range(size)]

recvmsg = comm.alltoall(sendmsg)

print("Proces ranga", rank, "ima vrijednost poslane poruke", sendmsg, "vrijednost primljene poruke", recvmsg)
```

- `comm.Alltoall(sendmsg, recvmsg)` -- element vektora sa indeksom `j` iz `sendmsg` na procesu `i` postaje element vektora sa indeksom `i` u `recvmsg` na procesu `j`

    - `sendmsg` je numpy polje koje se šalje, veličine jednake ukupnom broju procesa
    - `recvmsg` je numpy polje koje se prima, veličine jednake ukupnom broju procesa

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendmsg = np.array([rank * size + i for i in range(size)], dtype=np.int32)

recvmsg = np.empty(size, dtype=np.int32)

comm.Alltoall(sendmsg, recvmsg)

print("Proces ranga", rank, "ima vrijednost poslane poruke", sendmsg, "vrijednost primljene poruke", recvmsg)
```

!!! todo
    Ovdje nedostaje zadatak.

## Primjeri korištenja kolektivne komunikacije

### Zbroj vektora

- sekvencijalni kod je vrlo jednostavan smo, kao što smo već vidjeli
- korištenje kolektivne komunikacije tipa scatter-gather značajno pojednostavljuje kod u odnosu na komunikaciju tipa točka-do-točke; pretpostavka je da se kod izvodi u 4 procesa, inače `scatter()/Scatter()` i `gather()/Gather()` javljaju grešku

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = [1, 2, 3, 9]
        b = [4, 5, 6, 7]
    else:
        a = None
        b = None

    komponenta_od_a = comm.scatter(a, root=0)
    komponenta_od_b = comm.scatter(b, root=0)
    zbroj = komponenta_od_a + komponenta_od_b
    print("Proces ranga", rank, "izračunao je zbroj", zbroj)

    zbroj_vektor = comm.gather(zbroj, root=0)
    if rank==0: # ostali procesi će ispisati None, slično kao kod vektora za scatter()
        print("Zbroj vektora a, b i c iznosi", zbroj_vektor)
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1, 2, 3, 9], dtype=np.int32)
        b = np.array([4, 5, 6, 7], dtype=np.int32)
        zbroj_vektor = np.empty(4, dtype=np.int32)
    else:
        a = None
        b = None
        zbroj_vektor = None

    element_a = np.empty(1, dtype=np.int32)
    comm.Scatter(a, element_a, root=0)
    element_b = np.empty(1, dtype=np.int32)
    comm.Scatter(b, element_b, root=0)

    element_zbroj = element_a + element_b
    print("Proces ranga", rank, "izračunao je zbroj", element_zbroj)

    comm.Gather(element_zbroj, zbroj_vektor, root=0)
    if rank == 0:
        print("Zbroj vektora a i b iznosi", zbroj_vektor)
    ```

!!! admonition "Zadatak"
    - Promijenite kod da su vektori veličine 6 elemenata umjesto 4, i izvedite kod u 6 procesa.
    - Dodajte još jedan vektor veličine 6 elemenata i izračunajte zbroj tri vektora umjesto zbroja dva vektora.

!!! admonition "Zadatak"
    Učinite da program umjesto zbroja vektora računa produkt vektora po elementima, odnosno vektor u kojem je svaki element produkt elemenata u vektorima s pripadnim indeksom.

!!! admonition "Zadatak"
    Za matrice `a` i `b` moguće je odrediti njihov zbroj na način:

    ``` python
    # varijanta s Python objektima
    a = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    b = [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]]
    zbroj = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]

    for i in range(3):
        for j in range(3):
            zbroj[i][j] = a[i][j] + b[i][j]
    ```

    ``` python
    # varijanta s NumPy poljima
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    b = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=int32)
    zbroj = np.zeros((3, 3), dtype=np.int32)

    for i in range(3):
        for j in range(3):
            zbroj[i][j] = a[i][j] + b[i][j]

    # zbroj = a + b
    ```

    Primijenite kolektivnu komunikaciju tipa scatter-gather da zbroj matrica računate u 3 procesa.

### Računanje skalarnog produkta vektora

Skalarni produkt vektora `a` i `b` veličine `n` elemenata je izraz oblika

``` python
skalarni_produkt = a[0] * b[0] + a[1] * b[1] + ... a[n] * b[n]
```

- serijski kod za skalarni produkt je oblika

    ``` python
    # varijanta s Python objektima
    a = [1, 2, 3, 9, 10, 11]
    b = [4, 5, 6, 7, 8, 24]

    skalarni_produkt = 0
    for i in range(6):
        skalarni_produkt += a[i] + b[i]

    # varijanta s NumPy poljima
    a = np.array([1, 2, 3, 9, 10, 11], dtype=np.float32)
    b = np.array([4, 5, 6, 7, 8, 24], dtype=np.float32)

    skalarni_produkt = 0.0
    for i in range(6):
        skalarni_produkt += a[i] * b[i]

    # skalarni_produkt = np.dot(a, b)
    ```

- paralelni kod za koji koristi kolektivnu komunikaciju tipa scatter-gather je oblika

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = [1, 2, 3, 9, 10, 11]
        b = [4, 5, 6, 7, 8, 24]
    else:
        a = None
        b = None

    komponenta_od_a = comm.scatter(a, root=0)
    komponenta_od_b = comm.scatter(b, root=0)
    produkt = komponenta_od_a * komponenta_od_b

    produkt_vektora = comm.gather(produkt, root=0)
    if rank==0:
        suma = 0
        for x in produkt_vektora:
            suma += x
        print("Skalarni produkt iznosi", suma)
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1, 2, 3, 9, 10, 11], dtype=np.float32)
        b = np.array([4, 5, 6, 7, 8, 24], dtype=np.float32)
    else:
        a = None
        b = None

    element_a = np.empty(1, dtype=np.float32)
    comm.Scatter(a, element_a, root=0)
    element_b = np.empty(1, dtype=np.float32)
    comm.Scatter(b, element_b, root=0)
    produkt_elemenata = element_a * element_b

    vektor_produkata_elemenata = np.empty(6, dtype=np.float32)
    comm.Gather(produkt_elemenata, vektor_produkata_elemenata, root=0)

    if rank == 0:
        print("Skalarni produkt je", vektor_produkata_elemenata.sum())
    ```

- paralelni kod za koji koristi kolektivnu komunikaciju tipa scatter-reduce je oblika

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = [1, 2, 3, 9, 10, 11]
        b = [4, 5, 6, 7, 8, 24]
    else:
        a = None
        b = None

    komponenta_od_a = comm.scatter(a, root=0)
    komponenta_od_b = comm.scatter(b, root=0)
    produkt = komponenta_od_a * komponenta_od_b

    skalarni_produkt_vektora = comm.reduce(produkt, op=MPI.SUM, root=0)
    if rank == 0:
        print("Skalarni produkt je", skalarni_produkt_vektora)
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1, 2, 3, 9, 10, 11], dtype=np.float32)
        b = np.array([4, 5, 6, 7, 8, 24], dtype=np.float32)
    else:
        a = None
        b = None

    element_a = np.empty(1, dtype=np.float32)
    comm.Scatter(a, element_a, root=0)
    element_b = np.empty(1, dtype=np.float32)
    comm.Scatter(b, element_b, root=0)
    produkt_elemenata = element_a * element_b

    skalarni_produkt = np.empty(1, dtype=np.float32)
    comm.Reduce(produkt_elemenata, skalarni_produkt, op=MPI.SUM, root=0)
    if rank == 0:
        print("Skalarni produkt je", skalarni_produkt)
    ```

!!! admonition "Zadatak"
    Dodajte kod za još dvije redukcije:

    - prva neka nalazi najmanju vrijednost među dobivenim produktima,
    - druga neka nalazi najveću vrijednost među dobivenim produktima.

    (**Uputa:** `op=MPI.MIN` i `op=MPI.MAX`)

### Produkt matrice i vektora

- sekvencijalni produkt matrice i vektora

    ``` python
    # varijanta s NumPy poljima
    import numpy as np

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    x = np.array([10, 11, 12], dtype=np.float32)

    y = np.empty(3, dtype=np.float32)

    for i in range(3):
        y[i] = np.dot(A[i], x)

    print("Produkt matrice\n", A)
    print("i vektora\n", x)
    print("iznosi\n", y)

    # y = np.dot(A, x)
    ```

- paralelni kod korištenjem `Scatter()` i `Reduce()`

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 22, 33]], dtype=np.float32)
        x = np.array([10, 11, 12], dtype=np.float32)
        y = np.empty(4, dtype=np.float32)
    else:
        A = None
        x = np.empty(3, dtype=np.float32)
        y = None

    redak_A = np.empty(3, dtype=np.float32)
    comm.Scatter(A, redak_A, root=0)
    comm.Bcast(x, root=0)

    element_y = np.dot(redak_A, x)
    comm.Gather(element_y, y, root=0)

    if rank == 0:
        print("Produkt matrice\n", A)
        print("i vektora\n", x)
        print("iznosi\n", y)
    ```

!!! admonition "Zadatak"
    Proširite navedeni kod da umjesto produkta matrice i vektora računa produkt dvaju matrica.

### Aproksimativno računanje broja $\pi$ kao sume niza

Broj $\pi$ može se aproksimirati formulom

$$
\pi = \int_0^1 \frac{4}{1 + x^2} dx \approx \frac{1}{n} \sum_{i = 0}^{n - 1} \frac{4}{1 + (\frac{i + 0.5}{n})^2}.
$$

- sekvencijalni kod je oblika

    ``` python
    # varijanta s Python objektima
    import math

    n_iterations = 10

    def compute_pi(n):
        h = 1.0 / n
        s = 0.0
        for i in range(n):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        return s * h

    pi = compute_pi(n_iterations)
    error = abs(pi - math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np

    n_iterations = 10

    def compute_pi(n):
        h = 1.0 / n
        s = 0.0
        for i in range(n):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        return s * h

    pi = compute_pi(n_iterations)
    error = abs(pi - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

- paralelni kod je oblika

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI
    import math

    n_iterations = 10

    def compute_pi(n, start=0, step=1):
        h = 1.0 / n
        s = 0.0
        for i in range(start, n, step):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        return s * h

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        n = 10
    else:
        n = None

    n = comm.bcast(n, root=0)

    pi_part = compute_pi(n, rank, size)

    pi = comm.reduce(pi_part, op=MPI.SUM, root=0)

    if rank == 0:
       error = abs(pi - math.pi)
       print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    n_iterations = 1000

    def compute_pi(n, start=0, step=1):
        h = 1.0 / n
        s = 0.0
        for i in range(start, n, step):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        return s * h

    pi_part = np.empty(1)
    pi_part[0] = compute_pi(n_iterations, start=rank, step=size)

    if rank == 0:
        pi = np.empty(1)
    else:
        pi = None

    comm.Reduce(pi_part, pi, op=MPI.SUM, root=0)

    if rank == 0:
        error = abs(pi - np.math.pi)
        print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

!!! admonition "Zadatak"
    - Modificirajte gornji primjer tako da dodate kod koji mjeri vrijeme izvođenja za svaki od procesa.
    - Usporedite vrijeme izvođenja algoritma za 2, 3, 4 procesa kad svaki proces izvodi 10^4^, 10^5^, 10^6^ iteracija. Opišite svoje zaključke.

### Aproksimativno računanje broja $\pi$ korištenjem Monte Carlo metode

[Metode Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) su skupina metoda koja na temelju slučajnih brojeva i velikog broja pokusa daju aproksimaciju određene vrijednosti.

Broj $\pi$ može se aproksimativno računati i korištenjem Monte Carlo metode. Da bi to vidjeli, uzmimo da je u koordinatnom sustavu ucrtan jedinični krug unutar jediničnog kvadrata. Promotrimo njegovu četvrtinu u prvom kvadrantu:

- četvrtina kvadrata je površine 1 (obzirom da su mu obje stranice duljine 1),
- čevrtina kruga je površine $\frac{\pi}{4}$ (obzirom da mu je radijus 1).

Slučajno odabrana točka unutar četvrtine kvadrata ima vjerojatnost $\frac{\pi}{4}$ da upadne unutar četvrtine kruga. Dakle, ako s $n$ označimo broj slučajno odabranih točaka, a s $h$ broj točaka koje se od $n$ slučajno odabranih nalaze unutar četvrtine kruga, aproksimativno možemo odrediti broj $\pi$ kao

$$
\pi = 4 \times \frac{h}{n}.
$$

Povećavajući $n$ dobivamo točniju aproksimaciju, a navedena metoda naziva se Monte Carlo metoda. Programski kod koji aproksimira broj $\pi$ korištenjem Monte Carlo metode je oblika:

- sekvencijalni kod

    ``` python
    import numpy as np
    # import random
    # import math

    n_random_choices = 10000
    hits = 0
    throws = 0

    for i in range (0, n_random_choices):
        throws += 1
        x = np.random.random() # x = random.random()
        y = np.random.random() # y = random.random()
        dist = np.math.sqrt(x * x + y * y) # dist = math.sqrt(x * x + y * y)
        if dist <= 1.0:
            hits += 1

    pi = 4 * (hits / throws)

    error = abs(pi - np.math.pi) # error = abs(pi - math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

- paralelni kod korištenjem `Reduce()`

    ``` python
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_random_choices = 100000
    hits = 0
    throws = 0

    for i in range (0, n_random_choices):
        throws += 1
        x = np.random.random()
        y = np.random.random()
        dist = np.math.sqrt(x * x + y * y)
        if dist <= 1.0:
            hits += 1

    pi_part = np.empty(1)
    pi_part[0] = 4 * (hits / throws)

    if rank == 0:
        pi_reduced = np.empty(1)
    else:
        pi_reduced = None

    comm.Reduce(pi_part, pi_reduced, op=MPI.SUM, root=0)

    if rank == 0:
        pi = pi_reduced[0] / size
        error = abs(pi - np.math.pi)
        print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

!!! admonition "Zadatak"
    - Modificirajte primjer dan za Monte Carlo simulaciju da dodate kod koji mjeri vrijeme izvođenja za svaki od procesa.
    - Usporedite vrijeme izvođenja simulacije za 2, 3, 4 procesa kad svaki proces izvodi 10^4^, 10^5^, 10^6^ iteracija. Opišite svoje zaključke.

!!! admonition "Zadatak"
    Usporedite vrijeme izvođenja programa i točnost aproksimacije pi korištenjem:

    - sumiranja elemenata niza,
    - Monte Carlo simulacije,

    za 2, 3, 4 procesa kad svaki proces izvodi 10^4^, 10^5^, 10^6^ iteracija. Opišite svoje zaključke.

!!! admonition "Dodatni zadatak"
    Napišite program koji koristi MPI za izračun zbroja kvadrata brojeva u rasponu od 1 do 500000 u 4 procesa korištenjem kolektivne komunikacije tipa scatter-reduce. Raspodijelite po želji; za to morate napraviti listu oblika

    ``` python
    brojevi = [[1, 2, 3, ..., 125000],
               [125001, 125002, ..., 250000],
               [250001, 250002, ..., 375000],
               [375001, 375002, ..., 500000]]
    ```

    odnosno potrebno je da ima 4 podliste od kojih će savka biti dana odgovarajućem procesu. Svi procesi računaju zbroj kvadrata brojeva koje su dobili. Nakon završetka obrade na procesu ranga 0 sakupite rezultate i to tako da izvršite redukciju korištenjem sumiranja.

!!! admonition "Dodatni zadatak"
    Napišite program koji koristi MPI za izračun zbroja kubova brojeva u rasponu od 1 do 300000 u 6 procesa korištenjem kolektivne komunikacije tipa scatter-reduce. Na procesu ranga 0 inicijalizirajte listu pojedinih raspona i raspodijelite je procesima koji kubiraju dobivene brojeve i zbrajaju ih. Nakon završetka obrade na procesu ranga 0 izvedite redukciju sumiranjem i na procesu ranga 0 ispišite rezultat na ekran. Ostali procesi neka ne ispisuju ništa.
