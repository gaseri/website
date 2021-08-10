---
author: Vedran Miletić
---

# Python: međuprocesna komunikacija: čekanje na završetak ulazno/izlaznih operacija

- `module select` ([službena dokumentacija](https://docs.python.org/3/library/select.html)) omogućuje pristup sustavskim pozivima `select()` i `poll()` (svi operacijski sustavi slični Unixu), `epoll()` (samo Linux), `devpoll()` (samo Solaris i derivati) te `kqueue()` i `kevent()` (samo BSD i derivati)

    - mi ćemo se ovdje ograničiti na `select()`; pozivi `kqueue()` i `kevent()` rade slične stvari i imaju svojih prednosti koje nam ovdje nisu značajne, a njima govori [Benno Rice u predavanju What UNIX Cost Us od 10:40 do 17:20](https://youtu.be/9-IWMbJXoLM?t=640)

- `select.select(rlist, wlist, xlist[, timeout])` čeka do trenutka kad su jedan ili više opisnika datoteke navedeni u listama `rlist`, `wlist` i `xlist` spremni za ulazno-izlazne operacije čitanje, zapisivanje i "iznimno stanje" (respektivno)

    - liste `rlist`, `wlist` i `xlist` specijalno mogu biti i prazne; primjerice, kod tokovnih mrežnih utičnica baratanje s više klijenata može koristiti samo `rlist` pa ostaviti `wlist` i `xlist` praznima
    - parametar `timeout` specificira maksimalno vrijeme čekanja u sekundama; ukoliko nije naveden ili ima vrijednost `None`, sustavski poziv će čekati koliko god treba da barem jedan opisnik postane spreman za ulazno-izlazne operacije
    - vraća uređenu trojku `(rsublist, wsublist, xsublist)` gdje su `rsublist`, `wsublist` i `xsublist` svaki od elemenata podliste `rlist`, `wlist` i `xlist` (respektivno) danih u argumentima funkcije

    ``` python
    import select

    dat1 = open('/etc/os-release')
    dat2 = open('/etc/passwd')
    readready, writeready, exceptionready = select.select([dat1, dat2], [], []) # vraća ([dat1, dat2], [], [])
    for datoteka in readready:
        datoteka.read()
    dat1.close()
    dat2.close()
    ```

- `select.poll()` vraća objekt koji na koji se mogu registrirati opisnici datoteka, a zatim se može koristiti za ispitivanje spremnosti na određenu operaciju

!!! admonition "Zadatak"
    Prisjetite se da `mkfifo` stvara imenovane cijevi. Iskoristite ga da stvorite imenovanu cijev `mojacijev1`, zatim zapisujte u nju `cat`-om proizvoljan sadržaj (`cat > mojacijev1`).

    - Otvorite istu imenovanu cijev kao datoteku za čitanje u Pythonu i iskoristite `read()` za čitanje sadržaja. Uočite kako `read()` čeka ako nema sadržaja za pročitati.
    - Napišite kod koji koristi `select()` da otkrije kada može čitati sadržaj datoteke kako biste izbjegli čekanje.
    - Provjerite što se dogodi kada prekinete izvođenje `cat`-a.
