---
author: Vedran Miletić
---

# Python modul mpi4py: komunikatori i grupe procesa

Ova značajka je dio [standarda MPI-1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/mpi-report.html).

Dokumentacija svih funkcija koje koristimo u nastavku dana je u sklopu [pregleda dostupne funkcionalnosti u mpi4py-u](https://mpi4py.readthedocs.io/en/stable/overview.html).

- komunikator = grupa procesa + komunikacijski kontekst

    - grupa procesa -- skup procesa koji čine sudjeluju u izvođenju MPI posla; neovisna o grupi procesa na razini OS-a
    - komunikacijski kontekst -- izolira poruke u različitim dijelovima programa; **skriven od korisnika, nećemo se njime dalje baviti**

- `MPI.COMM_WORLD` -- komunikator koji uključuje sve pokrenute procese tog posla
- `MPI.COMM_SELF` -- komunikator koji uključuje samo proces sam (rijetko se koristi)
- `MPI.COMM_NULL` -- prazan komunikator

- metode za pristup podacima

    - `comm.Get_group()` -- metoda komunikatora koja dohvaća grupu

- objekti tipa `Group`

    - `group.Excl(ranks)` -- iz grupe isključuje procese s rangovima danim u listi `ranks`
    - `group.Incl(ranks)` -- u grupi ostavlja samo one procese navedene u listi `ranks`, i to u poretku navedenom u listi `ranks`
    - `group.Difference(group1, group2)`, `group.Intersect(group1, group2)`, `group.Union(group1, group2)` -- stvaraju grupu koja je rezultat skupovne razlike, presjeka, odnosno unije grupa `group1` i `group2`
    - `group.Free()` -- prazni grupu, odnosno miče sve procese iz grupe

## Konstruktori komunikatora

- `comm.Dup()` -- stvara duplikat komunikatora
- `comm.Create(group)` -- stvara komunikator zasnovan na grupi `group`

    ``` python
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    group = comm.Get_group()

    newgroup = group.Excl([0])
    newcomm = comm.Create(newgroup)

    if comm.Get_rank() == 0:
        assert newcomm == MPI.COMM_NULL
    else:
        assert newcomm.Get_size() == comm.Get_size() - 1
        assert newcomm.Get_rank() == comm.Get_rank() - 1

    group.Free()
    newgroup.Free()

    if newcomm:
        newcomm.Free()
    ```

!!! admonition "Zadatak"
    Stvorite grupu procesa koja sadrži samo procese iz grupe COMM_WORLD koji imaju parni rang. Iskoristite novu grupu da bi stvorili novi komunikator. (**Uputa:** iskoristite `Group.Incl()`, `Group.Range_incl()`  ili `Group.Excl()`.)

- `comm.Split(color, key)` -- dijeli komunikator prema boji i ključu

    - **boja procesa** je cijeli broj koji određuje raspored procesa po podgrupama
    - **ključ procesa** je cijeli broj koji određuje kojim redom će se dodijeliti rang procesima u novonastalim podgrupama

    ``` python
    from mpi4py import MPI

    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    if world_rank < world_size // 2:
        color = 55
        key = -world_rank
    else:
        color = 77
        key = +world_rank

    newcomm = MPI.COMM_WORLD.Split(color, key)
    # dobivamo dvije podgrupe procesa koje ne komuniciraju međusobno
    newcomm.Free()
    ```

!!! admonition "Zadatak"
    - Iskoristite `Comm.Split()` da bi razdijelili `COMM_WORLD` u dva dijela.
    - Prvi dio neka sadrži procese s parnim rangom iz `COMM_WORLD` u uzlaznom poretku.
    - Drugi dio neka sadrži procese s neparnim rangom iz `COMM_WORLD` u silaznom poretku.

MPI terminologija razlikuje intrakomunikator od interkomunikatora:

- **Intrakomunikator** (engl. *intracommunicator*) je komunikator koji se koristi za komunikaciju unutar jedne grupe procesa.
- **Interkomunikator** (engl. *intercommunicator*) je komunikator koji se koristi za komunikaciju između dvaju ili više grupa procesa, primjerice kod dinamičkog upravljanja procesima.

    - U standardu MPI-1, interkomunikator se koristi za komunikaciju točka-do-točke između dvije disjunktne grupe procesa.
    - U standardu MPI-2, interkomunikator se može koristiti i za kolektivnu komunikaciju između dvije ili više grupa procesa.
