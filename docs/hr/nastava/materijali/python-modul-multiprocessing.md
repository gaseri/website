---
author: Vedran Miletić
---

# Python: višeprocesnost na jednom računalu

- `module multiprocessing` ([službena dokumentacija](https://docs.python.org/3/library/multiprocessing.html)) nudi paralelizam izvođenja zasnovan na procesima i komunikaciji porukama

    - prilikom korištenja na operacijskom sustavu Windows postoje [određena ograničenja](https://docs.python.org/3/library/multiprocessing.html#windows)

- `multiprocessing.cpu_count()` vraća broj procesora na sustavu
- `multiprocessing.Process(target=funkcija, args=(arg1, arg2))`

    - `p.start()` započinje izvođenje procesa
    - `p.is_alive()` vraća `True` ako se proces i dalje izvodi
    - `p.terminate()` "nasilno" prekida izvođenje procesa
    - `p.join()` čeka da proces da završi s izvođenjem ("pridružuje" mu se; donekle slično `os.wait()`)
    - `p.run()` nije isto što i `p.start()`; pokreće funkciju koja je dana kao `target` unutar trenutnog procesa, **ne stvara novi proces**
    - `p.name`
    - `p.daemon`
    - `p.exitcode`

- `multiprocessing.active_children()` vraća broj aktivnih procesa

!!! admonition "Zadatak"
    - Definirajte funkciju `najveci_djelitelj(n)` koja prima kao argument prirodni broj `n` i traži najveći prirodni broj manji od `n` koji je njegov djelitelj. Funkcija redom isprobava brojeve manje od `n` i ispisuje na ekran poruku oblika `broj n, djelitelj d, ostatak r`; u trenutku kada je ostatak 0, funkcija završava sa izvođenjem (i ne vraća ništa).
    - Pokrenite tri procesa `p1`, `p2`, `p3` redom za brojeve 651929250, 421858921, 2188312. Pripazite da `(arg1,)` nije isto što i `(arg1)`.
    - U glavnom procesu "odspavajte" 5 sekundi, a zatim prekinite izvođenje `p2`, provjerite jesu li živi `p1` i `p3`, i one koji jesu pridružite na glavni proces.

- `multiprocessing.Queue()`, međuprocesna komunikacija čije je sučelje slično redu čekanja

    - `q.get()`
    - `q.put()`

- red može koristiti više procesa, i zbog mogućnosti različitog redanja za izvođenje na procesoru moguće je dobiti vrlo različite rezultate po pitanju poretka elemenata u konačnom rezultatu

!!! admonition "Zadatak"
    Modificirajte prethodni zadatak tako da u glavnom procesu inicijalizirate red čekanja i proslijedite ga kao argument funkcije svakom od procesa koji pokrećete. Svaki proces neka u red čekanja stavi uređeni par `(n, rj)`, pri čemu je `n` broj koji je prethodno dan kao prvi argument funkciji, a `rj` zadnji isprobani broj prije završetka izvođenja algoritma (drugim rječima, rješenje algoritma). U glavnom procesu izvedite primite rješenja i ispišite ih na ekran.

- `multiprocessing.Pipe()`, međuprocesna komunikacija čije je sučelje slično dvosmjernoj cijevi, vraća uređeni par `conn1, conn2` koji predstavlja konekcije na obje strane cijevi

    - `conn.send()` šalje podatke kroz cijev

        - za razliku od socketa može slati proizvoljne strukture (znakovne nizove, uređene n-torke, liste, rječnike, ...)

    - `conn.recv()` prima podatke iz cijevi

- može ga koristiti po jedan proces sa svake strane, u protivnom postoji rizik od iskrivljenja podataka

!!! admonition "Zadatak"
    Modificirajte prethodni zadatak tako da u glavnom procesu incijalizirate tri cijevi i proslijedite po jedan kraj jedne cijevi kao argument svakom od procesa koji pokrećete. Svaki proces neka u svoju cijev pošalje uređeni par `(n, rj)`, pri čemu je `n` broj koji je prethodno dan kao prvi argument funkciji, a `rj` zadnji isprobani broj prije završetka izvođenja algoritma (drugim rječima, rješenje algoritma). U glavnom procesu izvedite primite rješenja i ispišite ih na ekran.

- `multiprocessing.Pool(n)`

    - `pool.apply()` ima sličnu sintaksu kao inicijalizacija niti i procesa
    - `pool.map()` je paralelna verzija funkcije `map()`
    - `result = pool.apply_async()`
    - `result = pool.map_async()`

        - `result.get(timeout=n)`

    - `it = pool.imap()`
    - `it = pool.imap_unordered()`

!!! admonition "Zadatak"
    Riješite prvi zadatak u ovom dijelu korištenjem objekta `multiprocessing.Pool`. Riješite problem na tri načina.

    - Iskoristite sinkroni poziv korištenjem tri poziva funkcije `apply()`.
    - Iskoristite asinkroni poziv korištenjem jednog poziva funkcije `map()`.
    - Iskoristite funkciju `imap_unordered()`.

- `multiprocessing.Value()`
- `multiprocessing.RawValue()`
- `multiprocessing.Array()`
- `multiprocessing.RawArray()`
- `multiprocessing.Manager()`

    - `m.BoundedSemaphore()`
    - `m.Condition()`
    - `m.Event()`
    - `m.Lock()`
    - `m.RLock()`
    - `m.Semaphore()`

!!! todo
    Ovdje nedostaje objašnjenje i zadatak.
