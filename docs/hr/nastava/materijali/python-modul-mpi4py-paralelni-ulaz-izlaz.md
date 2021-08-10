---
author: Vedran Miletić
---

# Python modul mpi4py: paralelni ulaz/izlaz

Ova značajka je dio [standarda MPI-2](https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/mpi2-report.html).

Dokumentacija svih funkcija koje koristimo u nastavku dana je u sklopu [pregleda dostupne funkcionalnosti u mpi4py-u](https://mpi4py.readthedocs.io/en/stable/overview.html).

## Razlike POSIX-ovog i MPI-evog sučelja za ulaz i izlaz

- POSIX-ovo sučelje za korištenje datotečnog sustava (`open()`, `read()`, `write()`, ...) nije prilagođeno za paralelni ulaz/izlaz

    - nedostaje mu particioniranje podataka u datotekama između procesa
    - nedostaje mu koračni (engl. *strided*) pristup podacima
    - nema mogućnost kontrole na koji način će datoteka biti zapisana na uređaje za pohranu podataka

- MPI-2 uključuje vlastito sučelje za paralelni ulaz/izlaz kojemu je osnovni objekt datoteka

    - datoteka u ovom slučaju nije niz bajtova, već je skup podataka koji imaju tipove

        - podržan je sekvencijalni i slučajni pristup bilo kojem broju članova tog skupa

    - datoteku otvara istovremeno grupa procesa

## MPI datoteka

- klasa `MPI.File` služi za izvođenje ulazno-izlaznih operacija

    - metoda `MPI.File.Open()` otvara datoteku; poziva se na svim procesima unutar komunikatora s imenom datoteke i načinom pristupa datoteci: `MPI_MODE_RDONLY`, `MPI_MODE_RDWR`, `MPI_MODE_WRONLY`, ... (popis svih načina moguće je pronaći u [odgovarajućem dijelu standarda MPI 2.0](https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node175.htm))
    - metoda `MPI.File.Close()` zatvara datoteku
    - funkcije za čitanje i pisanje moguće je pronaći u [odgovarajućem dijelu standarda MPI 2.0](https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node186.htm); pritom u mpi4py npr. funkcija `MPI_FILE_WRITE_AT_ALL` postaje `MPI.File.Write_at_all()`

    ``` python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    fh = MPI.File.Open(comm, "./datafile.bin", MPI.MODE_WRONLY|MPI.MODE_CREATE)

    buffer = np.ones(10, dtype=np.int32)
    buffer *= rank

    print("Na procesu ranga", rank, "bit će zapisana vrijednost", buffer)

    offset = rank * buffer.nbytes
    fh.Write_at_all(offset, buffer)

    fh.Close()
    ```

    ``` python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    fh = MPI.File.Open(comm, "./datafile.bin", MPI.MODE_RDONLY)

    buffer = np.empty(10, dtype=np.int32)

    offset = rank * buffer.nbytes
    fh.Read_at_all(offset, buffer)

    print("Na procesu ranga", rank, "učitana je vrijednost", buffer)

    fh.Close()
    ```

!!! admonition "Zadatak"
    - Promijenite veličinu polja na 100 umjesto 10 i usporedite dobivene datoteke naredbom `hexdump`.
    - Pokrenite program na 6 procesa umjesto 4 i usporedite dobivene datoteke naredbom `hexdump`.
    - MPI nudi funkcije za neblokirajuće zapisivanje datoteka: `MPI_FILE_WRITE_AT_ALL_BEGIN` (`MPI.File.Write_at_all_begin()` u mpi4py) započinje, a `MPI_FILE_WRITE_AT_ALL_END` (`MPI.File.Write_at_all_end()` u mpi4py) završava zapisivanje. Iskoristite te funkcije umjesto blokirajućih, a zatim pronađite i iskoristite funkcije za neblokirajuće čitanje datoteka.

## Korištenje pogleda na MPI datoteku

- klasa `MPI.File`

    - metoda `MPI.File.Delete()` briše datoteku
    - metode `MPI.File.Set_view()` i `MPI.File.Get_view()` mijenjaju pogled na datoteku koji definira skup podataka koji je vidljiv i dostupan iz otvorene datoteke kao uređen skup osnovnih tipova podataka

    ``` python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
    fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

    item_count = 10

    buffer = np.empty(item_count, dtype='i')
    buffer[:] = rank

    filetype = MPI.INT.Create_vector(item_count, 1, size)
    filetype.Commit()

    displacement = MPI.INT.Get_size()*rank
    fh.Set_view(displacement, filetype=filetype)

    fh.Write_all(buffer)
    filetype.Free()
    fh.Close()
    ```
