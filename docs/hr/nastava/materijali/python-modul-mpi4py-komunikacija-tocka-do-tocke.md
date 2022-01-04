---
author: Vedran Miletić
---

# Python modul mpi4py: komunikacija točka-do-točke

Ova značajka je dio [standarda MPI-1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/mpi-report.html).

U nastavku donekle slijedimo [službeni mpi4py-ev tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html).

## Blokirajuća komunikacija

- čeka na razmjenu poruka
- `comm.send(obj, dest, tag)`

    - `dest` je rang procesa koji prima poruku
    - `tag` je **opcionalan** broj u rasponu 0--32767 koji je oznaka poruke; služi za razdvajanje poruka različitog tipa

- `obj = comm.recv(src, tag)`

    - `src` je rang procesa koji je poslao poruku

``` python
# varijanta s Python objektima
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("Potrebna su barem dva procesa za izvođenje")
    exit()

if rank == 0:
    sendmsg = 777
    comm.send(sendmsg, dest=1)
    recvmsg = comm.recv(source=1)
elif rank == 1:
    recvmsg = comm.recv(source=0)
    sendmsg = ["abc", 3.14]
    comm.send(sendmsg, dest=0)
else:
    print("Proces ranga", rank, "ne razmjenjuje poruke")
    exit()

print("Proces ranga", rank, "poslao je poruku", sendmsg)
print("Proces ranga", rank, "primio je poruku", recvmsg)
```

- `comm.Send(array, dest, tag)`

    - `array` je numpy polje koje šaljemo; mpi4py automatski prepoznaje tip podataka (`np.float32`, `np.float64`, `np.int32`, `np.int64`, ...), ali moguće ga je i ručno specificirati
    - `dest` je rang procesa koji prima poruku
    - `tag` je **opcionalan** broj u rasponu 0--32767 koji je oznaka poruke; služi za razdvajanje poruka različitog tipa, što je korisno kod složenijih aplikacija

- `comm.Recv(array, source, tag)`

    - `array` je numpy polje u koje se sprema rezultat; **treba ga stvoriti unaprijed**
    - `source` je rang procesa koji je poslao poruku
    - `tag` je isto kao i kod funkcije `Send()`

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("Potrebna su barem dva procesa za izvođenje")
    exit()

if rank == 0:
    sendmsg = np.array([366], dtype=np.int32)
    comm.Send(sendmsg, dest=1)

    recvmsg = np.empty(1, dtype=np.float32)
    comm.Recv(recvmsg, source=1)

elif rank == 1:
    recvmsg = np.empty(1, dtype=np.int32)
    comm.Recv(recvmsg, source=0)

    sendmsg = np.array([10.0], dtype=np.float32)
    comm.Send(sendmsg, dest=0)

else:
    print("Proces ranga", rank, "ne razmjenjuje poruke")
    exit()

print("Proces ranga", rank, "poslao je poruku", sendmsg)
print("Proces ranga", rank, "primio je poruku", recvmsg)
```

!!! admonition "Zadatak"
    Modficirajte program tako da poruke razmjenjuju tri procesa, i to tako da proces 0 šalje podatak tipa po vašoj želji i vrijednosti po vašoj želji procesu 1, proces 1 šalje isti podatak procesu 2, a proces 2 podatak šalje procesu 0.

!!! admonition "Zadatak"
    Napišite program koji se izvodi u tri procesa i koristi blokirajuću komunikaciju:

    - proces ranga 1 računa zbroj i produkt parnih prirodnih brojeva manjih ili jednakih 10 i rezultat šalje procesu ranga 0 kao listu koja sadrži dva elementa tipa `int` ili numpy polje od dva elementa tipa `numpy.int32`,
    - proces ranga 2 računa zbroj i produkt prirodnih brojeva manjih ili jednakih 20 i rezultat šalje procesu ranga 0 kao listu koja sadrži dva elementa tipa `int` ili kao numpy polje od dva elementa tipa `numpy.int64`,
    - proces ranga 0 prima rezultate i ispisuje ih na ekran.

## Navođenje tipova podataka

Kod funkcija `Send()`, `Recv()` i ostalih u nastavku koje primaju numpy polja **može se navesti** i tip podatka, ali ne mora; tada se umjesto `array` kao argument funkciji daje lista oblika `[array, mpi_type]`, pri čemu je `mpi_type` MPI tip podataka elemenata u polju i može imati vrijednosti

- `MPI.INT` -- pandan `numpy.int32`
- `MPI.LONG` -- pandan `numpy.int64`
- `MPI.FLOAT` -- pandan `numpy.float32`
- `MPI.DOUBLE` -- pandan `numpy.float64`

U većini slučajeva mpi4py automatski prepoznaje tip i nema potrebe za eksplicitnim navođenjem.

``` python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("Potrebna su barem dva procesa za izvođenje")
    exit()

if rank == 0:
    sendmsg = np.array([366], dtype=np.int32)
    comm.Send([sendmsg, MPI.INT], dest=1)

    recvmsg = np.empty(1, dtype=np.float32)
    comm.Recv([recvmsg, MPI.FLOAT], source=1)

elif rank == 1:
    recvmsg = np.empty(1, dtype=np.int32)
    comm.Recv([recvmsg, MPI.INT], source=0)

    sendmsg = np.array([10.0], dtype=np.float32)
    comm.Send([sendmsg, MPI.FLOAT], dest=0)

else:
    print("Proces ranga", rank, "ne razmjenjuje poruke")
    exit()

print("Proces ranga", rank, "poslao je poruku", sendmsg)
print("Proces ranga", rank, "primio je poruku", recvmsg)

```

## Neblokirajuća komunikacija

- ne čeka na razmjenu poruka, izvođenje se nastavlja
- `request = comm.isend(obj, dest, tag)` -- `obj`, `dest` i `tag` imaju isto značenje kao kod `comm.send()`
- `request.Wait()` čeka na završetak izvršenja zahtjeva za slanjem
- (`comm.irecv()` *za sada* nije implementiran)

``` python
# varijanta s Python objektima
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("Potrebna su barem dva procesa za izvođenje")
    exit()

if rank == 0:
    sendmsg = 777
    target = 1
elif rank == 1:
    sendmsg = "abc"
    target = 0
else:
    exit()

# procesi ranga 0 i 1 izvode ovaj dio koda
# uočite različite vrijednosti sendmsg i target
request = comm.isend(sendmsg, dest=target)
recvmsg = comm.recv(source=target)
request.Wait()
```

- `MPI.Request.Waitall()` prima listu zahtjeva za slanjem na koje mora čekati prije nastavka izvođenja

``` python
# varijanta s Python objektima
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendmsg = [rank] * 3
right = (rank + 1) % size
left = (rank - 1) % size

req1 = comm.isend(sendmsg, dest=right)
req2 = comm.isend(sendmsg, dest=left)
lmsg = comm.recv(source=left)
rmsg = comm.recv(source=right)

MPI.Request.Waitall([req1, req2])
print("'Lijeve' poruke su jednake", lmsg == [left] * 3)
print("'Desne' poruke su jednake", assert rmsg == [right] * 3)
```

- `request = comm.Isend(array, dest, tag)` -- `array`, `dest`, `tag` su isti kao i kod funkcije `Send()`
- `request = comm.Irecv(array, source, tag)` -- `array`, `source`, `tag` su isti kao i kod funkcije `Recv()`
- `request.Wait()` čeka na izvršenje zahtjeva za slanjem ili primanjem; kada ova funkcija napravi `return` polje u varijabli `array` je poslano, ili je u varijablu `array` spremljeno primljeno polje
- `MPI.Request.Waitall([request1, request2])` čeka na izvršenje zahtjeva `request1` i `request2`; praktično za kraći zapis kod velikog broja zahtjeva

``` python
# varijanta s NumPy poljima
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("Potrebna su barem dva procesa za izvođenje")
    exit()

if rank == 0:
    sendmsg = np.array([10.0], dtype=np.float64)
    request1 = comm.Isend(sendmsg, dest=1)

    recvmsg = np.empty(1, dtype=np.int64)
    request2 = comm.Irecv(recvmsg, source=1)

    request1.Wait()
    request2.Wait()

elif rank == 1:
    recvmsg = np.empty(1, dtype=np.float64)
    request2 = comm.Irecv(recvmsg, source=0)

    sendmsg = np.array([366], dtype=np.int64)
    request1 = comm.Isend(sendmsg, dest=0)

    MPI.Request.Waitall([request1, request2])

else:
    exit()

print("Proces ranga", rank, "poslao je poruku", sendmsg)
print("Proces ranga", rank, "primio je poruku", recvmsg)
```

!!! admonition "Zadatak"
    Napišite program koji se izvodi u tri procesa i koristi neblokirajuću komunikaciju:

    - proces ranga 1 računa zbroj i produkt parnih prirodnih brojeva manjih ili jednakih 10 i rezultat šalje procesu ranga 0

        - kao listu koja sadrži dva elementa tipa `int`, ili
        - kao numpy polje od dva elementa tipa `numpy.int32`;

    - proces ranga 2 računa zbroj i produkt prirodnih brojeva manjih ili jednakih 20 i rezultat šalje procesu ranga 0

        - kao listu koja sadrži dva elementa tipa `int`, ili
        - kao numpy polje od dva elementa tipa `numpy.int64`;

    - proces ranga 0 prima rezultate i ispisuje ih na ekran.

## Primjeri primjene komunikacije točka-do-točke

### Zbroj vektora

- sekvencijalni kod je vrlo jednostavan

    ``` python
    # varijanta s Python objektima
    a = [1, 2, 3, 9]
    b = [4, 5, 6, 7]
    zbroj = [0, 0, 0, 0]

    for i in range(4):
        zbroj[i] = a[i] + b[i]
    ```

    ``` python
    # varijanta s NumPy poljima
    a = np.array([1.0, 2.0, 3.0, 9.0])
    b = np.array([4.0, 5,0, 6.0, 7.0])
    zbroj = np.empty(4)

    for i in range(4):
        zbroj[i] = a[i] + b[i]

    # zbroj = a + b
    ```

- kod postaje izrazito složen, s puno ponavljanja, kada se koristi komunikacija točka-do-točke; primjer za 4 procesa u kojem proces ranga 0 inicijalizira vektore, računa zbroj elemenata vektora s indeksom 0 i šalje ostale elemente vektorima odgovarajućim procesima

    - koristimo rane spomenute `tag`-ove kod slanja i primanja da razdvojimo elemente vektora `a` od komponenata vektora `b`

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = [1, 2, 3, 9] # Zadatak: povećajte vektore na 4 komponente
        b = [4, 5, 6, 7]
        zbroj = [0, 0, 0, 0]
        zbroj[0] = a[0] + b[0]
        comm.send(a[1], dest=1, tag=0)
        comm.send(b[1], dest=1, tag=1)
        zbroj[1] = comm.recv(source=1)
        comm.send(a[2], dest=2, tag=0)
        comm.send(b[2], dest=2, tag=1)
        zbroj[2] = comm.recv(source=2)
        comm.send(a[3], dest=3, tag=0)
        comm.send(b[3], dest=3, tag=1)
        zbroj[3] = comm.recv(source=3)
        print("Zbroj je", zbroj)
    elif rank == 1:
        komponenta_od_a = comm.recv(source=0, tag=0)
        komponenta_od_b = comm.recv(source=0, tag=1)
        zbroj = komponenta_od_a + komponenta_od_b
        comm.send(zbroj, dest=0)
    elif rank == 2:
        komponenta_od_a = comm.recv(source=0, tag=0)
        komponenta_od_b = comm.recv(source=0, tag=1)
        zbroj = komponenta_od_a + komponenta_od_b
        comm.send(zbroj, dest=0)
    elif rank == 3:
        komponenta_od_a = comm.recv(source=0, tag=0)
        komponenta_od_b = comm.recv(source=0, tag=1)
        zbroj = komponenta_od_a + komponenta_od_b
        comm.send(zbroj, dest=0)
    else:
        exit()
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1, 2, 3, 9], dtype=np.float32)
        b = np.array([4, 5, 6, 7], dtype=np.float32)
        zbroj = np.zeros(4, dtype=np.float32)
        zbroj[0:1] = a[0:1] + b[0:1]
        comm.Send(a[1:2], dest=1, tag=0)
        comm.Send(b[1:2], dest=1, tag=1)
        comm.Send(a[2:3], dest=2, tag=0)
        comm.Send(b[2:3], dest=2, tag=1)
        comm.Send(a[3:4], dest=3, tag=0)
        comm.Send(b[3:4], dest=3, tag=1)

        comm.Recv(zbroj[1:2], source=1)
        comm.Recv(zbroj[2:3], source=2)
        comm.Recv(zbroj[3:4], source=3)

        print("Zbroj je", zbroj)

    elif rank == 1:
        element_a = np.zeros(1, dtype=np.float32)
        comm.Recv(element_a, source=0, tag=0)
        element_b = np.zeros(1, dtype=np.float32)
        comm.Recv(element_b, source=0, tag=1)
        element_zbroj = element_a + element_b
        comm.Send(element_zbroj, dest=0)

    elif rank == 2:
        element_a = np.zeros(1, dtype=np.float32)
        comm.Recv(element_a, source=0, tag=0)
        element_b = np.zeros(1, dtype=np.float32)
        comm.Recv(element_b, source=0, tag=1)
        element_zbroj = element_a + element_b
        comm.Send(element_zbroj, dest=0)

    elif rank == 3:
        element_a = np.zeros(1, dtype=np.float32)
        comm.Recv(element_a, source=0, tag=0)
        element_b = np.zeros(1, dtype=np.float32)
        comm.Recv(element_b, source=0, tag=1)
        element_zbroj = element_a + element_b
        comm.Send(element_zbroj, dest=0)

    else:
        exit()
    ```

!!! admonition "Zadatak"
    - Promijenite kod da su vektori veličine 6 elemenata umjesto 4, i izvedite kod u 6 procesa.
    - Dodajte još jedan vektor veličine 6 elemenata i izračunajte zbroj tri vektora umjesto zbroja dva vektora.

- uočimo li ponavljanje koda za rangove 1, 2 i 3 kod možemo pojednostaviti na način

    ``` python
    # varijanta s Python objektima
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = [1, 2, 3, 9] # Zadatak: povećajte vektore na 4 komponente
        b = [4, 5, 6, 7]
        zbroj = [0, 0, 0, 0]
        zbroj[0] = a[0] + b[0]
        comm.send(a[1], dest=1, tag=0)
        comm.send(b[1], dest=1, tag=1)
        zbroj[1] = comm.recv(source=1)
        comm.send(a[2], dest=2, tag=0)
        comm.send(b[2], dest=2, tag=1)
        zbroj[2] = comm.recv(source=2)
        comm.send(a[3], dest=3, tag=0)
        comm.send(b[3], dest=3, tag=1)
        zbroj[3] = comm.recv(source=3)
        print("Zbroj je", zbroj)
    elif rank == 1 or rank == 2 or rank == 3:
        komponenta_od_a = comm.recv(source=0, tag=0)
        komponenta_od_b = comm.recv(source=0, tag=1)
        zbroj = komponenta_od_a + komponenta_od_b
        comm.send(zbroj, dest=0)
    else:
        exit()
    ```

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1, 2, 3, 9], dtype=np.float32)
        b = np.array([4, 5, 6, 7], dtype=np.float32)
        zbroj = np.zeros(4, dtype=np.float32)
        zbroj[0:1] = a[0:1] + b[0:1]
        comm.Send(a[1:2], dest=1, tag=0)
        comm.Send(b[1:2], dest=1, tag=1)
        comm.Send(a[2:3], dest=2, tag=0)
        comm.Send(b[2:3], dest=2, tag=1)
        comm.Send(a[3:4], dest=3, tag=0)
        comm.Send(b[3:4], dest=3, tag=1)

        comm.Recv(zbroj[1:2], source=1)
        comm.Recv(zbroj[2:3], source=2)
        comm.Recv(zbroj[3:4], source=3)

        print("Zbroj je", zbroj)

    elif rank == 1 or rank == 2 or rank == 3:
        element_a = np.zeros(1, dtype=np.float32)
        comm.Recv(element_a, source=0, tag=0)
        element_b = np.zeros(1, dtype=np.float32)
        comm.Recv(element_b, source=0, tag=1)
        element_zbroj = element_a + element_b
        comm.Send(element_zbroj, dest=0)

    else:
        exit()
    ```

!!! admonition "Zadatak"
    - Ponovno promijenite kod da su vektori veličine 6 elemenata umjesto 4, i izvedite kod u 6 procesa.
    - Dodajte još jedan vektor veličine 6 elemenata i izračunajte zbroj tri vektora umjesto zbroja dva vektora.

- paralelizacija se može izvesti i dijeljenjem na komponente koje se sastoje od više elemenata

    - primjerice, proces ranga 0 može podijeliti vektore tako da od svakog po dva elementa pošalje procesima ranga 1 i 2

    ``` python
    # varijanta s NumPy poljima
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        a = np.array([1.0, 2.0, 3.0, 9.0])
        b = np.array([4.0, 5.0, 6.0, 7.0])
        zbroj = np.empty(4)

        comm.Send(a[0:2], dest=1, tag=0)
        comm.Send(b[0:2], dest=1, tag=1)

        comm.Send(a[2:4], dest=2, tag=0)
        comm.Send(b[2:4], dest=2, tag=1)

        comm.Recv(zbroj[0:2], source=1)
        comm.Recv(zbroj[2:4], source=2)

        print("Zbroj je", zbroj)

        zbroj_provjera = np.empty(4)

        for i in range(4):
            zbroj_provjera[i] = a[i] + b[i]

        # zbroj_provjera = a + b

        print("Provjera prolazi:", zbroj == zbroj_provjera)

    elif rank == 1:
        dio_a = np.empty(2)
        dio_b = np.empty(2)
        comm.Recv(dio_a, source=0, tag=0)
        comm.Recv(dio_b, source=0, tag=1)

        zbroj = dio_a + dio_b
        comm.Send(zbroj, dest=0)

    elif rank == 2:
        dio_a = np.empty(2)
        dio_b = np.empty(2)
        comm.Recv(dio_a, source=0, tag=0)
        comm.Recv(dio_b, source=0, tag=1)

        zbroj = dio_a + dio_b
        comm.Send(zbroj, dest=0)

    else:
        exit()
    ```

!!! admonition "Zadatak"
    Napišite dvije varijante ovog programa. Prva varijanta neka koristi blokirajuću komunikaciju, a druga neblokirajuću. Obje varijante izvode se u četiri procesa.

    - Na procesu ranga 0 incijaliziraju se dva vektora veličine 6 elemenata. Po dva elementa svakog vektora šalju se procesima ranga 1, 2 i 3.
    - Procesi ranga 1, 2 i 3 vrše zbroj odgovarajućih elemenata vektora i rezultat šalju procesu ranga 0.
    - Proces ranga 0 prima rezultate od procesa ranga 1, 2 i 3 i vrši provjeru točnosti rješenja.

!!! admonition "Dodatni zadatak"
    Napišite program koji koristi MPI za izračun zbroja kvadrata brojeva u rasponu od 1 do 500000 u 3 procesa korištenjem komunikacije točka-do točke, i to tako da proces ranga 0 šalje procesu ranga 1 i procesu ranga 2 liste koje sadrže brojeve od 1 do 250000 i od 250000 do 500000 (respektivno). Iskoristite blokirajuću komunikaciju. Procesi ranga 1 i 2 računaju zbroj kvadrata brojeva koje su dobili. Procesu ranga 0 procesi ranga 1 i 2 javljaju rezultate koje su dobili. Proces ranga 0 prima oba rezultata i njihov zbroj ispisuje na ekran.

!!! admonition "Dodatni zadatak"
    Napišite program koji koristi MPI za izračun zbroja kubova brojeva u rasponu od 1 do 300000 u 6 procesa korištenjem komunikacije točka-do točke. Kod slanja iskoristite neblokirajuću komunikaciju. Raspon raspodijelite po procesima po želji. Proces ranga 0 je onaj kojem će preostalih pet procesa javiti svoje rezultate i koji će rezultate sumirati te ispisati na ekran. Ostali procesi neka ne ispisuju ništa.
