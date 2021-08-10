---
author: Vedran Miletić
---

# Python modul mpi4py: osnove

- `module mpi4py` nudi MPI sučelje za Python, dizajnirano po uzoru na već spomenuti [Boost.MPI](https://www.boost.org/doc/libs/release/libs/mpi/) za C++

    - uključuje gotovo sve funkcije iz MPI standarda; radi na OpenMPI i MPICH2 platformama
    - napisan u kombinaciji Pythona i C-a; sučelje visoke razine koje nudi brzinu izvođenja blisku C-u

- podržava značajke [standarda MPI-1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/mpi-report.html)

    - komunikaciju točka-to-točke: `send()/recv()`, `isend()/irecv()`, `test()/wait()`
    - kolektivnu komunikaciju: barijere, `broadcast()`, `scatter()`, `gather()`, `reduce()`, `scan()`
    - grupe procesa i komunikacijske domene

- podržava značajke [standarda MPI-2](https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/mpi2-report.html)

    - dinamičko upravljanje procesima: `spawn()`, `connect()/accept()`
    - pristup udaljenoj memoriji: `put()/get()/accumulate()`
    - paralelni ulaz/izlaz: `read()/write()`

- za objekte u komunikaciji nude se dvije mogućnosti

    - Python objekti -- serijalizira i deserijalizira ih modul `pickle` -- može biti zahtjevno u terminima CPU i memorijskih resursa za veće skupove podataka (time se bavimo na Operacijskim sustavima 2)
    - NumPy polja -- nema potrebe za serijalizacijom i deserijalizacijom -- puno brže, brzina bliska C-u (time se bavimo na Distribuiranim sustavima)

- hello world primjer

    - uočite da nema dijeljene memorije, ali da određene varijable imaju različite vrijednosti, slično kao `os.fork()`

    ``` python
    from mpi4py import MPI

    name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("Pozdrav sa domaćina", name, "od procesa ranga", rank, "od ukupno", size, "procesa")

    if rank == 0:
        vendor = MPI.get_vendor()
        print("Podaci o implementaciji MPI-a koja se koristi:", vendor[0], vendor[1])
    ```

- spremimo li navedeni kod u datoteku `hello.py` pokretanje u 2 procesa se izvodi naredbom

    ``` shell
    $ mpirun -np 2 python3 hello.py
    ```

!!! admonition "Zadatak"
    - Pokrenite kod tako da se izvodi u 8 procesa.
    - Učinite da procesi s parnim identifikatorom ispisuju i `Ja sam proces sa parnim identifikatorom`, a procesi s neparnim identifikatorom ispisuju `Ja sam proces s neparnim identifikatorom`.
    - Dodajte da procesi s neparnim identifikatorom pored toga ispisuju i `Kvadrat mog identifikatora iznosi` te kvadrat svog identifikatora.
