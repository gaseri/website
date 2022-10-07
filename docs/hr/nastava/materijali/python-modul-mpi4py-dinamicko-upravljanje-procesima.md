---
author: Vedran Miletić
---

# Python modul mpi4py: dinamičko upravljanje procesima

Ova značajka je dio [standarda MPI-2](https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/mpi2-report.html).

Dokumentacija svih funkcija koje koristimo u nastavku dana je u sklopu [pregleda dostupne funkcionalnosti u mpi4py-u](https://mpi4py.readthedocs.io/en/stable/overview.html).

- korisno kod stvaranja složenih distribuiranih aplikacija
- po potrebi može spajati neovisne paralelne programe napisane u različitim jezicima

    - za povezivanje dvaju aplikacija koriste se metode `comm.Connect()` i `comm.Accept()`

## Stvaranje novog procesa iz pokrenutog programa

- kolektivna operacija koja stvara komunikator
- **lokalna grupa** je grupa procesa koji stvaraju nove procese (roditelji)
- **udaljena grupa** je grupa novih procesa (djeca)
- `comm.Spawn()` -- procesi roditelji stvaraju procese djecu; metoda vraća novi interkomunikator
- `comm.Get_parent()` -- metoda kojom procesi djeca mogu dohvatiti komunikator roditelja, obzirom da imaju vlastiti COMM_WORLD
- `comm.Disconnect()` -- prekida vezu roditelja i djeteta; nakon toga obje grupe mogu nastaviti s izvođenjem

- roditeljski proces stvara 3 djece i vrši razmjenu poruka s djetetom ranga 1

    ``` python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_SELF.Spawn('/usr/bin/python3', args=['hello-child.py'], maxprocs=3)

    x = np.array([1.0, 2.5, 4.0, 5.5], dtype=np.float32)
    comm.Send(x, dest=1)
    print("Proces roditelj je poslao", x)

    y = np.empty(4, dtype=np.int32)
    comm.Recv(y, source=1)
    print("Proces roditelj je primio", y)

    comm.Disconnect()
    ```

- dijete ranga 1 vrši razmjenu poruka s roditeljem, ostala djeca ne rade ništa; **datoteku je potrebno nazvati** `hello-child.py`

    ``` python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 1:
        x = np.empty(4, dtype=np.float32)
        comm.Recv(x, source=0)
        print("Proces dijete ranga", rank, "je primio", x)

        y = np.array([2, 3, 5, 9], dtype=np.int32)
        comm.Send(y, dest=0)
        print("Proces dijete ranga", rank, "je poslao", y)

    comm.Disconnect()
    ```

!!! admonition "Zadatak"
    - Učinite da i dijete ranga 2 vrši razmjenu poruka s roditeljem. Dijete neka šalje array vrijednosti po vašem izboru veličine 6 tipa `np.int64`, a roditelj neka šalje array vrijednosti po vašem izboru veličine 8 tipa `np.float64`.
    - Povećajte broj djece sa 3 na 5.

## Primjeri primjene dinamičkog upravljanja procesima

### Paralelno računanje broja $\pi$ uz dinamičko upravljanje procesima

- roditelj, datoteka `pi-parent.py`

    ``` python
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_SELF.Spawn('/usr/bin/python3', args=['pi-child.py'], maxprocs=5)

    n_iterations = np.array(10, dtype=np.int32)
    comm.Bcast(n_iterations, root=MPI.ROOT)

    PI = np.array(0.0, dtype=np.float64)
    comm.Reduce(None, PI, op=MPI.SUM, root=MPI.ROOT)

    comm.Disconnect()

    error = abs(PI - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (PI, error))
    ```

- dijete, datoteka `pi-child.py`

    ``` python
    import numpy as np
    from mpi4py import MPI

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_iterations = np.array(0, dtype=np.int32)
    comm.Bcast(n_iterations, root=0)

    h = 1.0 / n_iterations
    s = 0.0
    for i in range(rank, n_iterations, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)

    PI = np.array(s * h, dtype=np.float64)
    comm.Reduce(PI, None, op=MPI.SUM, root=0)

    comm.Disconnect()
    ```

!!! admonition "Zadatak"
    Pretvorite ranije obrađeno računanje broja pi korištenjem Monte Carlo metode u varijantu koja koristi dinamičko upravljanje procesima.

### Primjeri koji uključuju korištenje modula scipy

!!! todo
    Ovaj dio treba napisati u cijelosti.
