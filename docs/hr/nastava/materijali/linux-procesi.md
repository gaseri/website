---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Upravljanje procesima

## Upravljanje poslovima ljuske

- dvije vrste naredbi, obzirom na vrijeme izvođenja

    - naredbe koje se izvode ograničeno vrijeme i same prekidaju izvođenje (`ls`, `pwd`, `rm`, `cp`, `mv`, ...)
    - naredbe koje se izvode do prekida izvođenja od strane korisnika (`man`, `less`, `tail -f`; `cat` bez argumenata, `grep` s jednim argumentom, ...)

- `^Z` posao koji se trenutno izvodi zaustavlja i baca u background

    - omogućuje baratanje sa više poslova u jednom terminalu

- `jobs` ispisuje popis pokrenutih poslova
- `%n` oznaka posao s brojem `n`
- `%+` označava posao koji se prethodno izvodio
- `%-` označava posao koji se pretprethodno izvodio
- `fg` vraća posao s danim brojem u foreground
- `bg` nastavlja izvođenje posla s danim brojem u backgroundu

!!! admonition "Zadatak"
    - Pokrenite, a zatim zaustavite u izvođenju i stavite u background, redom `cat` bez argumenata, `grep` s jednim argumentom i `tail -f .bashrc`.
    - Vratite u foreground posao s rednim brojem 2.
    - Prekinite izvođenje posla broj 1.
    - Saznajte može li naredba `jobs` izlistati i broj procesa. (**Uputa:** `man jobs` ne postoji; vidite možete li na neki način pogrešno upotrijebiti naredbu da vam baci grešku i na ekran ispiše pomoć kod korištenja.)

!!! admonition "Dodatni zadatak"
    - Pokrenite sljedeću naredbu: `emacs -nw dat1.txt &`. Što se dogodilo?
    - Probajte tipkati neki tekst. Što se događa?
    - Vratite pokrenuti Emacs u foreground. Što uočavate?

## Upravljanje procesima operacijskog sustava

- proces = program u izvođenju
- proces != posao

    - svaki posao je proces, ali obrat ne vrijedi

- identifikator procesa (engl. *process ID*, PID) je broj koji jednoznačno određuje proces na sustavu
- `ps` prikazuje procese trenutnog korisnika koji se izvode na trenutnom terminalu

    - podržava SysV, BSD i GNU stil naredbi; mi ćemo koristiti BSD stil naredbi

- `ps a` prikazuje procese trenutnog korisnika *i drugih korisnika*, uključujući korisnika root
- `ps x` prikazuje procese sa *i bez terminala*

    - proces bez terminala ima vrijednost `?` u odgovarajućem stupcu
    - zanimljiv primjer procesa bez terminala su procesi koji nastaju pokretanjem aplikacija iz grafičkog sučelja

- `ps u` je oblik ispisa prilagođen za korisnike

    - `u` dolazi od *user-oriented*; pored PID-a, vremena izvođenja i naredbe prikazuje i vrijeme početka izvođenja, stanje procesa, zauzeće memorije i procesora

- `ps axu` daje ispis svih procesa svih korisnika uz nešto više detalja
- `ps f` ispisuje šumu procesa

    - `f` dolazi od *forest*
    - šuma procesa podrazumijeva više stabala procesa

- `pstree` radi na sličan način kao `ps f`

    - svim procesima koji nemaju neposrednog roditelja prikazuje se `init` kao roditelj (PID roditelja je 1), pa postoji jedan korijen -- sada se ne radi se o šumi, već o stablu

- `pstree -p` jedan roditelj za sve procese
- [init](https://en.wikipedia.org/wiki/Init) (naredba `init`) pokreće određene servise kod pokretanja operacijskog sustava, o čemu više govorimo kasnije

    - PID procesa je 1

- `[kthreadd]` pokreće procesne niti jezgre operacijskog sustava

    - PID procesa je 2
    - više informacija ima u man stranicama `kthread_bind(9)`, `kthread_run(9)`, `kthread_stop(9)`, `kthread_create(9)`, `kthread_should_stop(9)`

!!! admonition "Zadatak"
    Pronađite u popisu procesa `python`. (Naravno, to će biti moguće samo ako je sustav prethodno pripremljen za rješavanje zadatka.)

    - Otkrijte ime ili user ID korisnika kojem proces pripada.
    - Otkrijte PID roditelja tog procesa.

## Signali

- [signal](https://en.wikipedia.org/wiki/Unix_signal) se koristi za obavještavanje procesa ili procesne niti o nekom događaju

    - svaki signal ima svoj jedinstveni naziv tj. kraticu koja počinje sa `SIG` (npr. `SIGINT`), te odgovarajući broj
    - po primitku signala proces reagira na određeni način

- `kill -l` daje popis signala

    - dva smo već koristili: ++control+c++ šalje signal `2) SIGINT`, a ++control+z++ signal `20) SIGTSTP`
    - dva signala koja proces ne može uhvatiti su:

        - `9) SIGKILL` -- odmah prekida izvođenje procesa
        - `19) SIGSTOP` -- zaustavlja proces u izvođenju; stavlja proces u background

    - ostale signale proces hvata korištenjem funkcije `signal()` definirane u zaglavlju `signal.h` iz standardne biblioteke jezika C (C++ varijanta je `csignal`); detaljnije na [Wikipedijinoj stranici o signal.h](https://en.wikipedia.org/wiki/C_signal_handling)

- `kill -<n> PID` je naredba za slanje signala `n` procesu PID

    - zadan je signal broj 15, SIGTERM, koji traži od procesa da prekine s izvođenjem

- `killall -<n> ime_naredbe` šalje signal `n` svim pokrenutim instancama naredbe sa zadanim imenom (koje može biti i regularni izraz)

!!! admonition "Zadatak"
    - Pokrenite dva terminala u kojima ste povezani na poslužitelj.
    - U jednom terminalu pokrenite `less .bashrc`, a iz drugog pošaljite signal 15 tom procesu.
    - Ponovno pokrenite `less .bashrc`, ali mu sada pošaljite signal 9. Uočite razliku. Objasnite zašto ne možete koristiti PID iz prethodnog dijela zadatka.
    - Pokušajte poslati signal 9 ili 15 procesu `python` iz prethodnog zadatka, a onda objasnite zašto to ne možete.

!!! admonition "Zadatak"
    - Pokrenite tri terminala. U dva terminala pokrenite `emacs`. U barem jednom od njih počnite pisati nešto, ali nemojte to spremiti.
    - Pošaljite svim pokrenutim `emacs`-ima signal 15. Objasnite zbog čega javlja da nekima od njih to nije moguće učiniti.
    - Usporedite to sa situacijom kada pošaljete signal 9. Što javlja `emacs` kod prekida izvođenja u jednom, a što u drugom slučaju?

## Niceness i prioritet izvođenja

- [niceness](https://en.wikipedia.org/wiki/Nice_(Unix)) određuje koliko će procesi često doći na red za izvođenje (mali vremenski intervali)

    - vrijednost se kreće od -20 (češće dolazi na red) do 19 (rjeđe dolazi na red)
    - korisnici osim `root` korisnika mogu postaviti vrijednosti od 0 do 19 (zadana postavka, regulira je [PAM](https://en.wikipedia.org/wiki/Pluggable_authentication_module) u `/etc/security/limits.conf`)
    - niceness != prioritet; sustav dodjeljuje prioritet na temelju nicenessa koji zadaje korisnik; najčešće tako da pribraja niceness na zadani prioritet procesa, ali *ne mora biti tako*

- `renice` mijenja niceness u odnosu na trenutni, radi na već pokrenutim procesima
- `nice` mijenja niceness u odnosu na zadani, koristi se kod pokretanja procesa
- `ionice` za razliku od nicenessa, koji kontrolira prioritet kod redanja za obradu od strane procesora, ioniceness kontrolira prioritet redanja za korištenje ulaza i izlaza (primjerice, čitanje i zapisivanje na diskove)

!!! admonition "Zadatak"
    Pokrenite dva terminala.

    - U jednom terminalu pokrenite `sleep 30s` s nicenessom postavljenim na 10.
    - Promijenite niceness tog procesa na 17.

    **Napomena:** Ovisno o sigurnosnim ograničenjima sustava na kojem radite, obični korisnici ne mogu ni postavljati niceness na nižu vrijednost od one koju su prethodno postavili. Ovaj zadatak je zadan tako da uvijek bude rješiv.

!!! note
    Prema [Wikipediji](https://en.wikipedia.org/wiki/Nice_(Unix)#Etymology):

    > The name "nice" comes from the fact that the program's purpose is to modify a process niceness value. The true priority, used to decide how much cpu time to concede to each process, is calculated by the kernel process scheduler from a combination of the different processes niceness values and other data, such as the amount of I/O done by each process.
    >
    > The name "niceness" originates from the idea that a process with a higher niceness value is "nicer" to other processes in the system, as it allows the other processes more cpu time, by having a lower priority (and therefore a higher niceness) itself.

- `top` služi za nadgledanje procesa u realnom vremenu koji se izvode; na vrhu popisa prikazuje procese koji troše najviše procesorskog vremena (odatle i naziv)

    - funkcionalnost `ps`-a, `kill`-a i `renice`-a
    - kontrolira se uz pomoć tipkovnice (slično kao `less`)
    - `k` kill; traži se PID i broj ili ime signala
    - `r` renice; traži se PID i niceness
    - `u` prikaži samo procese navedenog korisnika
    - `h` prikaz pomoći
    - `z` uključuje boju kod prikaza
    - `B` uključuje bold kod prikaza; unosi se sa ++shift+b++

- `htop` je `top` na steroidima

    - na većini distribucija nije u zadanoj instalaciji, ali ima kultni status među tzv. hardcore geekovima (spominjemo radi potpunosti)

!!! admonition "Zadatak"
    Pokrenite tri terminala.

    - U jednom terminalu pokrenite `cat`, a u drugom `grep` s jednim argumentom.
    - U `top`-u, ograničite pogled samo na svoje procese. Poredajte ih po zauzeću memorije.
    - Pokušajte iz `top`-a poslati signal 9 ili 15 procesu `python` iz prethodnog zadatka. Ima li razlike?

!!! admonition "Ponovimo!"
    - Postoji li razlika između poslova i procesa?
    - Kako dijelimo naredbe s obzirom na vrijeme njihova izvođenja? Dajte po jedan primjer za svaku skupinu.
    - Čime su identificirani poslovi, a čime procesi?
    - Prisjetite se naredbi za manipulaciju poslovima.
    - Kako i zbog čega se procesi "bacaju" na izvođenje u pozadinu?
    - Prisjetite se parametara koji se koriste sa naredbom `ps` i njihova značenja.
    - Što je stablo procesa i kako ono dokazuje da je `init` zaista proces koji pokreće sve ostale procese?
    - Objasnite pojam signala i kako funkcioniraju.
    - Prisjetite se signala koje smo radili, pa objasnite razliku između onih koji se daju i onih koji se ne daju uloviti.
    - Objasnite funkcionalnost naredbe `killall`.
    - Što je niceness, kada kažemo da je neki proces više "nice" nego neki drugi?
    - Obrazložite razliku između niceness-a i prioriteta nekog procesa.
    - Zbog čega je `top` vrlo moćan alat, koje su njegove mogućnosti?
