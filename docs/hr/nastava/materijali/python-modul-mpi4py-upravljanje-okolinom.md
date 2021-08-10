---
author: Vedran Miletić
---

# Python modul mpi4py: upravljanje okolinom

Ova značajka je dio [standarda MPI-1](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/mpi-report.html).

Dokumentacija svih funkcija koje koristimo u nastavku dana je u sklopu [pregleda dostupne funkcionalnosti u mpi4py-u](https://mpi4py.readthedocs.io/en/stable/overview.html).

## Inicijalizacija i finalizacija

- `Init()` -- incijalizacija okoline za izvođenje MPI aplikacije
- `Finalize()` -- uništavanje okoline izvođenja MPI aplikacije

## Informacije o implementaciji

- `Get_version()` -- vraća verziju MPI standarda koja se koristi kao uređeni par oblika `(major, minor)`
- `Get_library_version()` -- vraća verziju MPI standarda koja se koristi kao niz znakova
- `Get_processor_name()` -- vraća ime domaćina koji izvodi proces

!!! todo
    Ovdje nedostaje zadatak.

## Mjerači vremena

- `Wtime()` -- vraća vrijeme operacijskog sustava kao vrijednost tipa `float`, ekvivalent funkciji `time()` unutar Python modula `time`
- `Wtick()` -- vraća informacije o preciznosti mjerača vremena

!!! todo
    Ovdje nedostaje zadatak sa mjerenjem vremena izvođenja nekog algoritma.

## Baratanje pogreškama

Baratanje pogreškama izvodi se hvatanjem iznimki. Pozivi MPI funkcija u slučaju greške podižu iznimku, instancu klase `Exception`.
