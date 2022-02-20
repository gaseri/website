---
author: Vanja Slavuj, Vedran Miletić
---

# Python: međuprocesna komunikacija: baratanje signalima

- signali predstavljaju način slanja obavijesti programima OS-a o određenom događaju u sustavu

    - najčešće se obrađuju asinkrono (obzirom na uobičajeno izvođenje programa)
    - često se nazivaju i programskim prekidima jer prekidaju uobičajeno izvođenje programa
    - može ih generirati sam sustav (npr. kod reboota se šalje `TERM` pa `KILL` svim procesima) ili pojedini program (npr. korisnik pokrene naredbu `kill` ili koristi funkciju `kill()`)

- svaki je signal definiran jedinstvenim cjelobrojnim identifikatorom (npr. 2) i nazivom (npr. `INTERRUPT`); popis svih signala dostupan naredbom `kill -l`:

    ``` shell
    $ kill -l
     1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL       5) SIGTRAP
     6) SIGABRT      7) SIGBUS       8) SIGFPE       9) SIGKILL     10) SIGUSR1
    11) SIGSEGV     12) SIGUSR2     13) SIGPIPE     14) SIGALRM     15) SIGTERM
    16) SIGSTKFLT   17) SIGCHLD     18) SIGCONT     19) SIGSTOP     20) SIGTSTP
    21) SIGTTIN     22) SIGTTOU     23) SIGURG      24) SIGXCPU     25) SIGXFSZ
    26) SIGVTALRM   27) SIGPROF     28) SIGWINCH    29) SIGIO       30) SIGPWR
    31) SIGSYS      34) SIGRTMIN    35) SIGRTMIN+1  36) SIGRTMIN+2  37) SIGRTMIN+3
    38) SIGRTMIN+4  39) SIGRTMIN+5  40) SIGRTMIN+6  41) SIGRTMIN+7  42) SIGRTMIN+8
    43) SIGRTMIN+9  44) SIGRTMIN+10 45) SIGRTMIN+11 46) SIGRTMIN+12 47) SIGRTMIN+13
    48) SIGRTMIN+14 49) SIGRTMIN+15 50) SIGRTMAX-14 51) SIGRTMAX-13 52) SIGRTMAX-12
    53) SIGRTMAX-11 54) SIGRTMAX-10 55) SIGRTMAX-9  56) SIGRTMAX-8  57) SIGRTMAX-7
    58) SIGRTMAX-6  59) SIGRTMAX-5  60) SIGRTMAX-4  61) SIGRTMAX-3  62) SIGRTMAX-2
    63) SIGRTMAX-1  64) SIGRTMAX
    ```

- nakon primitka signala, proces može:

    - zaustaviti izvođenje (tako da ga kasnije može nastaviti)
    - nastaviti izvođenje
    - prekinuti izvođenje
    - prekinuti izvođenje uz stvaranje tzv. deponiranje jezgre (engl. *core dump*) datoteke sa stanjem memorije u trenutku prekida
    - ignorirati (zanemariti) signal

- obrada signala po pristizanju zapravo ovisi o načinu na koji je programer napisao program koji implementira obradu signala (sučelje u C/C++-u detaljnije opisano u `signal(3)`)

- `module signal` ([službena dokumentacija](https://docs.python.org/3/library/signal.html)) nudi mehanizme za primjenu obrade signala; sastoji se uglavnom od:

    - varijabli za određivanje načina obrade signala tj. upravljača (engl. *handler*)
    - varijabli za definiranje signala
    - funkcija za registriranje upravljača koji određuje obradu (izvođenje) signala po njegovu pristizanju
    - funkcija za višedretveni rad sa signalima

- hvatanje signala i njegova obrada moguća je definiranjem upravljača

    - upravljač je povratni poziv funkcije (engl. *callback*), odnosno funkcija koja se prosljeđuje nekoj drugoj funkciji kao argument i koja se zatim poziva u toj vanjskoj funkciji kako bi izvršila neku zadaću
    - upravljač je funkcija koja se poziva kada procesu (koji pokreće Python program) pristigne određeni signal
    - upravljač je potrebno najprije registrirati pozivom funkcije `signal.signal(signalnum, handler)`

        - `signalnum` je broj signala; ne moramo pamtiti brojeve jer postoje konstante `signal.SIGABRT`, `signal.SIGBREAK`, `signal.SIGCONT`, `signal.SIGKILL`, `signal.SIGSEGV`, `signal.SIGTERM` i dr.
        - `handler` je upravljač, odnosno funkcija koja prima dva argumenta

            - cjelobrojni identifikator signala
            - stanje stoga od trenutka kada je izvođenje programa prekinuto signalom

- po pristizanju signala procesu, signal je moguće:

    - obraditi upravljačem (opisano iznad)
    - ignorirati -- koristi se predefinirani upravljač `signal.SIG_IGN` koji ignorira primljeni signal
    - obraditi tako da on izvršava zadanu (engl. *default*) zadaću -- koristi se predefinirani upravljač `signal.SIG_DFL` koji provodi zadanu funkciju primanja signala

    ```
    import os
    import signal

    def upravljacHUP(broj_signala, stog):
        print('Pristigao je signal broj {}.'.format(broj_signala))

    signal.signal(signal.SIGHUP, upravljacHUP)
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignorirat će primljeni signal 2, SIGINT
    signal.signal(signal.SIGTERM, signal.SIG_DFL) # primljeni signal 15, SIGTERM će izvesti zadanu operaciju

    while True:
        pass
    ```

!!! admonition "Zadatak"
    Proširite kod primjera tako da ignorira SIGCONT, a da nakon primanja SIGINT-a ispiše `Primljen SIGINT, nastavljam izvođenje`. Uvjerite se da se zaista program ne može zaustaviti kombinacijom tipki ++control+c++ (`^C`) koja šalje SIGINT.

!!! admonition "Dodatni zadatak"
    Napišite program koji hvata signal SIGKILL, čeka na pristizanje signala, te po pristizanju signala KILL na zaslon ispisuje poruku o identifikatoru signala i završava izvođenje. Uočite što se događa i pokušajte objasniti zbog čega.

- modul `signal` nudi funkcije za rad s upravljačem, među kojima su i:

    - `signal.pause()` -- proces čeka na dolazak signala
    - `signal.getsignal(sig)` -- vraća naziv upravljača povezanog sa signalom čiji je identifikator sig
    - `signal.strsignal(sig)` -- vraća puni naziv signala čiji je identifikator sig (koristi se od verzije Pythona 3.8)
    - `signal.valid_signals()` -- vraća popis signala koji su definirani u OS-u (koristi se od verzije Pythona 3.8)

!!! admonition "Zadatak"
    Napišite program koji uz odgovarajuću poruku korisniku čeka pristizanje jednog od prvih 5 definiranih signala sustava (kojeg ćete poslati naredbom `kill` iz drugog terminala). Program nudi korisniku mogućnost provjere je li za uneseni cjelobrojni identifikator signala definiran upravljač i o tome obavještava korisnika.

!!! admonition "Zadatak"
    Napišite program koji na zaslon ispisuje vlastiti PID i čeka pristizanje signala SIGHUP, SIGINT ili SIGTERM (kojeg ćete poslati naredbom `kill` iz drugog terminala). Za pristigli signal HUP, program na zaslon ispisuje poruku o pristiglom broju signala i prekida svoje izvršavanje. Za pristigli signal INT, program ga ignorira i nastavlja čekati novi signal. Za pristigli signal TERM, program se ponaša onako kako je zadano tim signalom.

- rad sa signalima trebao bi slijediti pravila standarda POSIX

    - svaki pristigli signal program mora obraditi na primjeren način
    - za svaki signal potrebno je definirati odgovarajući upravljač ovisno o tome koja je zadaća (učinak) signala
    - programi koji zaprime signale `SIGHUP`, `SIGINT`, `SIGKILL` ili `SIGTERM`, trebali bi završiti svoje izvođenje, a u svakom drugom slučaju, uz završavanje, trebalo bi stvoriti datoteku deponija jezgre (engl. *core dump*) sa zapisom trenutnog stanja memorije (drugi argument koji se prosljeđuje upravljaču)

- alarmi su posebna vrsta signala

    - program traži od OS-a obavijest da je prošao određeni vremenski period
    - `signal.alarm(time)` -- funkcija koja zahtjeva slanje signala `SIGALRM` nekom procesu nakon `time` sekundi
    - `signal.SIGALRM` -- varijabla koja definira signal za mjerenje vremena (engl. *timer*)

!!! admonition "Zadatak"
    Izmijenite programski kod rješenja prethodnog zadatka na sljedeći način: ako korisnik ne pošalje signal unutar 15 sekundi, programu se automatski šalje alarm; po aktivaciji alarma, program na zaslon ispusuje trenutno vrijeme i poruku o neaktivnosti te završava izvođenje.

!!! admonition "Dodatni zadatak"
    Napiši program koji nakon pokretanja na zaslon ispisuje poruku `Proces-roditelj započinje`, stvara proces-dijete i čeka na njegov dovršetak. Proces-dijete započinje izvođenje uz odgovarajući poruku i PID na zaslonu te nudi korisniku unos opcije `A` za zaustavljanje njegova izvođenja (signal `TSTP`) ili `B` za završavanje izvođenja (signal `TERM`) uz zapis stoga u datoteku `backup_stog.txt`. U slučaju zaustavljanja procesa-djeteta, postavlja se alarm od 15 sekundi (koji se pokreće jednom kada se iz drugog terminala signalom nastavi njegovo izvođenje), a proces-dijete osluškuje signale. Kada alarm stigne, program radi isto što i opcijom `B` ranijeg izbornika. Kada proces-dijete završi svoje izvođenje, proces roditelj završava izvođenje (uz odgovarajuću poruku).
