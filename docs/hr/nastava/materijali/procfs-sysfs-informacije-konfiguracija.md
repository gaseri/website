---
author: Vedran Miletić
---

# Datotečni sustavi Procfs i Sysfs

## Procfs: informacije o procesima

- direktorij `/proc` je [procfs](https://en.wikipedia.org/wiki/Procfs)

    - virtualni datotečni sustav, nije fizički zapisan na disku
    - daje informacije o procesima i informacije o hardveru

- informacije o procesima

    - direktoriji oblika `/proc/$PID/`
    - sadrže: `attr/`, `auxv`, `cgroup`, `clear_refs`, `cmdline`, `comm`, `coredump_filter`, `cpuset`, `cwd`, `environ`, `exe`, `fd/`, `fdinfo/`, `io`, `latency`, `limits`, `loginuid`, `maps`, `mem`, `mountinfo`, `mounts`, `mountstats`, `net/`, `oom_adj`, `oom_score`, `pagemap`, `personality`, `root`, `sched`, `schedstat`, `sessionid`, `smaps`, `stack`, `stat`, `statm`, `status`, `syscall`, `task/`, `wchan`
    - `sed "s/\x00/\n/g"` zamjenjuje ASCII znak null s novim retkom, vrlo korisno za pregledavanje ovih datoteka

!!! admonition "Zadatak"
    - Saznajte PID svoje instance `bash` ljuske i PID instance koju je pokrenuo netko od preostalih studenata. Pronađite njihove direktorije unutar `procfs`-a.
    - Razmotrite dozvole. Objasnite kojem korisniku i kojoj grupi su dodijeljene datoteke.
    - Usporedite što informacije koje možete saznati o svakom od tih procesa. Jesu li vam sve informacije dostupne?
    - Objasnite sadržaj datoteka i simboličkih poveznica `cmdline`, `cwd`, `environ`, `exe`, `io`, `status`, `task`. Upotrijebite `sed` po potrebi.

## Procfs: informacije o hardveru i postavkama sustava

- informacije o hardveru i postavkama sustava

    - sve datoteke, direktoriji i simboličke poveznice kojima imena oblika `/proc/[a-z]*`

!!! admonition "Zadatak"
    Pronađite informacije o ACPI podršci hardvera na kojem radite, specifično:

    - ima li procesor podršku za baratanje energijom,
    - informacije o trenutnoj i kritičnoj temperaturi.

Procfs: sučelje sysctl

- direktorij `/proc/sys` osim informacija, omogućuje i dinamičko mijenanje parametara sustava
- [Sysctl sučelje](https://en.wikipedia.org/wiki/Sysctl) (naredba `sysctl`) omogućuje čitanje i promjenu parametara sustava

!!! todo
    Ovdje nedostaje zadatak.

## Sysfs

- direktorij  `/sys` je `Sysfs`, modernija zamjena za `procfs`

    - `/sys` != `/proc/sys`

!!! admonition "Zadatak"
    Pronađite informacije koje ste ranije očitali pomoću alata `lspci` unutar `/sys/devices/`. (**Uputa:**  pronađite poddirektorij koji se odnosi na mrežnu karticu.)
