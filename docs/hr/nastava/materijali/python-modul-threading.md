---
author: Vedran Miletić
---

# Python: dodatne usluge operacijskog sustava: višenitnost

> Why did the multithreaded chicken cross the road?
>
> to To other side. get the

## Pojam procesne niti

- [POSIX Threads](https://en.wikipedia.org/wiki/POSIX_Threads), poznati i pod nazivom *Pthreads*, omogućavaju procesima paralelizam izvođenja zasnovan na procesnim nitima i dijeljenoj memoriji

    - POSIX standard za rad s procesnim nitima koji ima više različitih implementacija

        - Linux koristi [Native POSIX Thread Library (NPTL)](https://en.wikipedia.org/wiki/Native_POSIX_Thread_Library), dokumentacija je dostupna putem `man phtreads`, `man pthread.h` i `man pthread_*`
        - FreeBSD koristi [libthr](https://www.freebsd.org/cgi/man.cgi?query=libthr&sektion=3), dokumentacija je dostupna putem u `pthread(3)` (`man 3 phtread`)

- `module threading` ([službena dokumentacija](https://docs.python.org/3/library/threading.html)) nudi pristup POSIX threads sučelju
- `threading.enumerate()` vraća listu svih trenutno aktivnih niti (uvijek je unutra barem glavna nit u kojoj se program izvodi)
- `threading.Thread(target=funkcija, args=(arg1, arg2))` stvara objekt tipa `Thread` koji će pokrenuti funkciju `funkcija` s argumentima `arg1` i `arg2`

    - `thread.start()` započinje izvođenje procesne niti
    - `thread.run()` nije isto što i `thread.start()`; pokreće funkciju koja je dana kao `target` unutar trenutne procesne niti, **ne stvara novu procesnu nit**
    - `thread.join()` čeka da procesna nit da završi s izvođenjem ("pridružuje" joj se; donekle slično `os.wait()`)
    - `thread.is_alive()` vraća `True` ako se procesna nit i dalje izvodi

    ``` python
    import threading
    import time

    def f1():
        time.sleep(1)
        print("Pokrenuta je funkcija f1")

    def f2(arg):
        time.sleep(1)
        print("Pokrenuta je funkcija f2 s argumentom", arg)

    def f3(arg1, arg2):
        time.sleep(1)
        print("Pokrenuta je funkcija f2 s argumentima", arg1, "i", arg2)

    t1 = threading.Thread(target=f1)
    t2 = threading.Thread(target=f2, args=(5,)) # (5,) je tipa tuple i duljine 1, a (5) == 5
    t3 = threading.Thread(target=f2, args=("ovo je thread t3",))
    t4 = threading.Thread(target=f3, args=(2.7, [1, 2]))

    print(threading.enumerate()
    print("Je li nit t1 je pokrenuta?", t1.start())
    print("Je li nit t2 je pokrenuta?", t2.start())
    print("Je li nit t3 je pokrenuta?", t3.start())
    print("Je li nit t4 je pokrenuta?", t4.start())

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    print(threading.enumerate()
    print("Je li nit t1 je pokrenuta?", t1.start())
    print("Je li nit t2 je pokrenuta?", t2.start())
    print("Je li nit t3 je pokrenuta?", t3.start())
    print("Je li nit t4 je pokrenuta?", t4.start())

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    print(threading.enumerate()
    print("Je li nit t1 je pokrenuta?", t1.start())
    print("Je li nit t2 je pokrenuta?", t2.start())
    print("Je li nit t3 je pokrenuta?", t3.start())
    print("Je li nit t4 je pokrenuta?", t4.start())
    ```

!!! admonition "Zadatak"
    - Definirajte funkciju `countdown_to_zero(n)` koja prima kao argument prirodni broj `n` i radi odbrojavanje, odnosno u `while` petlji smanjue broj za 1 sve dok ne dođe do nule.
    - Pokrenite tri niti `t1`, `t2`, `t3` redom za brojeve 651929250, 421858921, 2188312. Pripazite da i ovdje `(arg1,)` nije isto što i `(arg1)`.
    - U glavnom procesu "odspavajte" 5 sekundi, a zatim provjerite jesu li živi `t1`, `t2` i `t3`, i one koji jesu pridružite na glavni proces.

## Zaključavanje

- stanje utrke (engl. *race condition*) javlja se kod zapisivanja vrijednosti dijeljene varijable od strane više procesnih niti istovremeno, primjerice dvije procesne niti imaju zadatak povećati vrijednost dijeljene varijable koja trenutno ima vrijednost 2 za 1:

    - prva nit očita trenutnu vrijednost 2 i poveća je za 1, dobije rezultat 3
    - druga nit očita trenutnu vrijednost 2 i poveća je za 1, dobije rezultat 3
    - prva nit zapiše dobivenu vrijednost 3
    - druga nit zapiše dobivenu vrijednost 3
    - očekivan rezultat nakon 2 uvećanja je 4, a imamo 3

    ``` python
    # u ovako jednostavnom slučaju ćemo vrlo teško izvesti iznad opisani poredak operacija
    # zadatak niže će ilustrirati kakve probleme stvara stanje utrke
    import threading

    x = 2

    def uvecaj():
        global x
        x = x + 1

    t1 = threading.Thread(target=uvecaj)
    t2 = threading.Thread(target=uvecaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Vrijednost je", x)
    ```

- `threading.Lock()` -- jednostavan objekt zaključavanja koji se može dohvatiti samo jednom; svi ostali pokušaji dohvaćanja će blokirati dok se ne dogodi otpuštanje zaključavanja od strane koja je napravila dohvaćanje

    - `lock.acquire([blocking])` -- dohvaćanje zaključavanja
    - `lock.release()` -- otpuštanje zaključavanja
    - `lock.locked()` -- vraća `True` ako je zaključavanje dohvaćeno

    ``` python
    import threading

    x = 2
    lock_x = threading.Lock()

    def uvecaj():
        global x
        lock_x.acquire()
        x = x + 1
        lock_x.release()

    t1 = threading.Thread(target=uvecaj)
    t2 = threading.Thread(target=uvecaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Vrijednost je", x)
    ```

- `threading.RLock()` -- jednostavan objekt zaključavanja koji se može dohvatiti više puta, ali samo od strane niti koja ga je već dohvatila; svi ostali pokušaji dohvaćanja će blokirati dok se ne dogodi onoliko otpuštanja zaključavanja od strane koja je napravila dohvaćanje koliko je napravljeno dohvaćanja

    - `rlock.acquire([blocking])` -- dohvaćanje zaključavanja
    - `rlock.release()` -- otpuštanje zaključavanja
    - `lock.locked()` -- vraća `True` ako je zaključavanje dohvaćeno

    ``` python
    import threading

    x = 2
    rlock_x = threading.RLock()

    def uvecaj():
        global x
        rlock_x.acquire()
        # može i više puta što ima smisla kod složenijih programa
        x = x + 1
        # svaki acquire() mora imati pripadni release()
        rlock_x.release()

    t1 = threading.Thread(target=uvecaj)
    t2 = threading.Thread(target=uvecaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Vrijednost je", x)
    ```

!!! admonition "Zadatak"
    Napišite program koji računa zbroj prvih 500000 prirodnih brojeva u dvije procesne niti; definirajte globalnu varijablu zbroj, i učinite jedna procesna nit u globalnu varijablu zbraja brojeve od 1 do 250000, a druga od 250001 do 500000.

    Izvedite program bez zaključavanja, s običnim zaključavanjem i s višestrukim zaključavanjem. Ima li ovdje potrebe za višestrukim zaključavanjem? Objasnite zašto.

!!! admonition "Zadatak"
    Analizirajte vrijeme potrebno za izvođenje programa za sljedeće varijante.

    - Promijenite program iz prethodnog zadatka da računa produkt brojeva u danom rasponu umjesto zbroja. Usporedite vrijeme izvođenja tog programa i programa iz prethodnog zadatka.
    - Pored toga, usporedite performanse kod varijante koja dohvaća lock u svakoj iteraciji for petlje s vremenom izvođenja varijante koja to dohvaća lock prije for petlje. Koja je brža? Objasnite zašto. (Raspon smanjite po potrebi.)
    - Naposlijetku, isprobajte korištenje višestrukog zaključavanja umjesto običnog. Ima li razlike u vremenu izvođenja? Objasnite zašto.

- `threading.Condition([lock])` je proširenje pojma zaključavanja kojim se ovdje nećemo detaljnije baviti

## Semafori

- `threading.Semaphore([value])` proširuje zaključavanje i ima brojač početne vrijednosti `value` koji omogućuje višestruko dohvaćanje (najviše `value` puta) i otpuštanje; izmislio ih je [Edsger W. Dijkstra](https://en.wikipedia.org/wiki/Edsger_W._Dijkstra)

    - `s.acquire([blocking])` dohvaća zaključavanje
    - `s.release()` otpušta zaključavanje

    ``` python
    import threading

    x = 2
    semafor_x = threading.Semaphore(1) # semafor s vrijednošću 1 ima istovjetno ponašanje kao zaključavanje

    def uvecaj():
        global x
        semafor_x.acquire()
        x = x + 1
        semafor_x.release()

    t1 = threading.Thread(target=uvecaj)
    t2 = threading.Thread(target=uvecaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Vrijednost je", x)
    ```

- `threading.BoundedSemaphore([value])` je semafor koji ograničava broj otpuštanja koje je moguće pozvati da ne prijeđu početno postavljenu vrijednost `value`

    - `bs.acquire([blocking])` dohvaća zaključavanje
    - `bs.release()` otpušta zaključavanje

    ``` python
    import threading

    x = 2
    semafor_x = threading.BoundedSemaphore(1) # semafor s vrijednošću 1 ima istovjetno ponašanje kao zaključavanje

    def uvecaj():
        global x
        semafor_x.acquire()
        x = x + 1
        semafor_x.release()
        # dodatni poziv semafor_x.release() bacio bi iznimku tipa ValueError

    t1 = threading.Thread(target=uvecaj)
    t2 = threading.Thread(target=uvecaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Vrijednost je", x)
    ```

!!! admonition "Zadatak"
    Knjižnica nudi tri knjige:

    - Marko Marulić: Judita, 5 komada
    - Fjodor Mihajlovič Dostojevski: Zločin i kazna, 3 komada
    - Eugen Kumičić: Urota zrinsko-frankopanska, 4 komada

    Reprezentirajte te tri knjige u programu kao tri niza znakova proizvoljnog sadržaja, i pridružite im semafore s odgovarajućom vrijednosti. (Razmislite hoćete li koristiti ograničene semafore.)

    Napišite funkciju `lektira()` koja prima jedan argument tipa znakovni niz, a to je ime učenika, koje se zatim ispisuje na ekran (npr. `"Ja sam Domagoj i posudit ću tri knjige"`). Učenik zatim "posuđuje te tri knjige", odnosno dohvaća njihove semafore i radi na njima `acquire()`, pa "čita te knjige", odnosno prvo ispisuje ime knjige na ekran (npr. `"Ja sam Sonja i čitam Fjodor Mihajlovič Dostojevski: Zločin i kazna"`). Naposlijetku "vraća te tri knjige", odnosno ispisuje da vraća knjige (npr. `"Ja sam Ivan i vraćam tri knjige"`), a zatim radi `release()`.

    Učinite da sedam učenika posuđuje knjige, odnosno pokrenite sedam procesnih niti za učenike imena redom Domagoj, Ivan, Luka, Snežana, Romana, Sonja, Marta.

## Događaji

- `threading.Event()` je jednostavan mehanizam sinkronizacije između procesnih niti kod kojeg jedna nit signalizira da se događaj dogodio, a druga čeka na signalizaciju

    - `event.is_set()` vraća `True` ako se događaj dogodio
    - `event.set()` postavlja da se događaj dogodio
    - `event.clear()` vraća na početno stanje (događaj se nije dogodio)
    - `event.wait([timeout])` čeka na signalizaciju da se događaj dogodio

    ``` python
    import threading

    event = threading.Event()

    def postavi():
        global event
        print("Događaj će biti postavljen")
        event.set()

    def cekaj():
        global event
        event.wait()
        print("Dočekano je postavljanje događaja")

    t1 = threading.Thread(target=postavi)
    t2 = threading.Thread(target=cekaj)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    ```

!!! admonition "Zadatak"
    Pečete [Ledolette s nadjevom od marelica](https://www.ledo.hr/hr/proizvodi/tijesta/slatka-tijesta/ledolette-s-nadjevom-od-marelica) koji zahtijevaju dvije minute za vađenje iz Ledo škrinje i otpakiravanje i 20 minuta u pećnici. Napravite dvije procesne niti, od kojih jedna pokreće funkciju `odmrzavanje(n)`, gdje `n` broj minuta koje se Ledolette pripremaju za stavljanje u pećnicu, i drugu koja pokreće funkciju `pecnica(n)`, gdje je `n` broj minuta koje se kroasani peku u pećnici. Iskoristite `time.sleep()` s praktično upotrebljivim vrijednostima u sekundama da simulirate čekanje. Pokrenite istovremeno obje niti, ali učinite da druga nit čeka na događaj odmrzavanja kroasana koji prva nit postavlja.

!!! admonition "Dodatni zadatak"
    Napišite program koji vrši zbroj kubova brojeva u rasponu od 1 do 300000 u 3 niti i raspodijelite tako da 1. nit računa raspon od 1 do 100000, 2. nit od 100001 do 200000, 3. nit od 200001 do 300000. Iskoristite višenitnosti i varijablu u koju ćete spremiti zbroj učinite dijeljenom; iskoristite zaključavanje kod promjene varijable u svakoj od niti; napravite događaj koji postavlja 3. nit u trenutku kad završi s izvođenjem, i učinite da na njega čekaju preostale dvije niti.

## Brojači

- `threading.Timer(interval, funkcija)` pokreće funkciju u posebnoj procesnoj niti nakon što prođe vremenski interval trajanja danog u sekundama

    - `timer.start()` započinje čekanje vrijednosti intervala
    - `timer.cancel()` otkazuje čekanje

    ``` python
    import threading

    def doceka():
        print("Tko čeka, dočeka")

    t = threading.Timer(10, doceka)
    t.start() # ispisuje "Tko čeka, dočeka" nakon 10 sekundi
    ```

!!! admonition "Zadatak"
    Kuhate čaj i treba vam 1 minuta da zavrije voda, a istovremeno u pećnici koja ima pokvaren timer pečete kolačiće madelaine koji će biti taman kako treba za 2 minute. Napravite program sa dva timera, svaki sa pripadnom funkcijom, od kojih će jedna ispisati na ekran `"Voda spremna za čaj!"`, a druga `"Kolačići madelaine taman kako treba!"` u odgovarajućim trenucima.

## Barijere

- `threading.Barrier([parties])` omogućuje zadanom broju niti koji iznosi vrijednosti parametra `parties` da čekaju jedna na drugu prije nastavka izvođenja; izvođenje se nastavlja kad sve pozovu funkciju `wait()`

    - `barrier.wait()` pokreće čekanje na ostale niti da dođu do barijere
    - `barrier.reset()` vraća barijeru u prvobitno stanje, bez niti koje čekaju
    - `barrier.abort()` postavlja barijeru u polomljeno stanje, u kojem će svi pozivi funkciji `wait()` rezultirati iznimkom

    ``` python
    import threading

    b = threading.Barrier(2)

    def f1():
        global b
        print("Prije barijere 1")
        b.wait()
        print("Nakon barijere 1")

    def f2():
        global b
        print("Prije barijere 1")
        b.wait()
        print("Nakon barijere 1")

    t1 = threading.Thread(target=f1)
    t2 = threading.Thread(target=f2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    ```

!!! admonition "Zadatak"
    Dodajte dijeljenu barijeru u obje funkcije prethodnog zadatka tako da se po završetku izrade čaja, odnosno kolačića, čeka na onu drugu stranu, a zatim na ekran ispiše:

    > I čim sam prepoznao okus u lipov čaj namočena komada madelaine, koji mi je svake nedjelje davala tetka Leonie (tada još doduše nisam znao razlog, zbog koga me ta uspomena tako usrećivala, nego sam to otkriće morao odgoditi za mnogo poslije), odmah se pojavi i stara, siva kuća na ulicu, u kojoj je bila njena soba, pa se kao pozorišni dekor pridruži malom paviljonu, koji je gledao na vrt, a koji su na njenoj stražnjoj strani nadogradili za moje roditelje (i baš to je bio onaj krnji komad, koji sam do sad jedini vidio); a s kućom se pojavi i grad. Trg, kamo su me slali prije ručka, ulice, kojima sam trčao od jutra do večeri, po svakom vremenu, i šetnje, na koje smo odlazili, kad je bilo lijepo vrijeme. I kao što se dešava u onoj igri, kojom se Japanci zabavljaju uranjajući u porculansku zdjelu punu vode komadiće do tad bezlična papira, koji se tek što je umočen, isteže, savija, bojadiše, diferencira, pretvara u cvijeće, kuće i određene osobe, koje je moguće prepoznati, tako je i sad sve cvijeće iz našeg vrta, iz Swannova perivoja, tako su svi lopoči s Vivonne, oni dobri seoski ljudi, njihovi mali domovi, crkva i cio Combray sa svojom okolinom, tako je sve što ima oblik i čvrstoću, i grad i vrtovi, izašlo iz moje čaše čaja.

!!! admonition "Dodatni zadatak"
    Napišite program koji vrši zbroj kvadrata brojeva u rasponu od 1 do 500000 u 3 niti (raspodijelite po želji). Iskoristite višenitnosti i varijablu u koju ćete spremiti zbroj učinite dijeljenom; iskoristite zaključavanje kod promjene varijable u svakoj od niti; napravite dijeljenu barijeru između tri niti tako da niti čekaju jedna na drugu i istovremeno završavaju izvođenje.
