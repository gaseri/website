---
author: Vedran Miletić
---

# Python: međuprocesna komunikacija: utičnice

- [BSD utičnice](https://en.wikipedia.org/wiki/Berkeley_sockets) ili [POSIX](https://en.wikipedia.org/wiki/POSIX) utičnice (engl. *sockets*), poznate i pod nazivima *Berkeley sockets* i *Unix sockets*, su *de facto* standardno sučelje za međuprocesnu komunikaciju lokalno i putem računalne mreže

    - koriste model komunikacije klijent-poslužitelj bez obzira izvodi li se komunikacija izvodi lokalno ili putem računalne mreže
    - koriste se na operacijskim sustavima sličnim Unixu i [na Windowsima](https://docs.microsoft.com/en-us/windows/win32/winsock/windows-sockets-start-page-2)
    - nastaju kao aplikacijsko programsko sučelje za pristup TCP-u i IP-u 1983. godine kad [Bill Joy](https://engineering.berkeley.edu/bill-joy-co-founder-of-sun-microsystems/) i kolege iz [Computer Systems Research Group](https://en.wikipedia.org/wiki/Computer_Systems_Research_Group) s Kalifornijskog sveučilišta u Berkeleyu izbacuju [4.2BSD](https://en.wikipedia.org/wiki/History_of_the_Berkeley_Software_Distribution#4.2BSD); [BSD](https://en.wikipedia.org/wiki/Berkeley_Software_Distribution) je u to doba razvojna platforma za TCP/IP, o čemu detaljnije govori [Marshall Kirk McKusick](https://www.mckusick.com/) u svojim izlaganjima o povijesti razvoja Unixa na Berkeleyu:

        - u članku [Twenty Years of Berkeley Unix: From AT&T-Owned to Freely Redistributable (Open Sources: Voices from the Open Source Revolution)](https://www.oreilly.com/openbook/opensources/book/kirkmck.html)
        - u predavanju [MeetBSD 2018: Dr. Kirk McKusick - A Narrative History of BSD](https://youtu.be/pKLq6jenQvY)
        - u predavanju [A Narrative History of BSD](https://youtu.be/bVSXXeiFLgk)
        - u predavanju [Kirk McKusick - A Narrative History of BSD](https://youtu.be/DEEr6dT-4uQ)
        - u predavanju [A Narrative History of BSD, Dr. Kirk McKusick](https://youtu.be/ds77e3aO9nA)

    - standardizirane su 2008. godine uz male razlike u imenovanju funkcija i preimenovane u POSIX utičnice

- podjela prema adresiranju (`AF_*`)

    - *Unix domain sockets* koriste datotečni sustav (obitelj adresa `AF_UNIX`)
    - *Unix network sockets* koriste IPv4 (obitelj adresa `AF_INET`) i IPv6 (obitelj adresa `AF_INET6`)

- podjela prema pouzdanosti (`SOCK_*`)

    - datagramski (tip `SOCK_DGRAM`), kod TCP/IP-a koriste se za UDP
    - tokovni (tip `SOCK_STREAM`), kod TCP/IP-a koriste se za TCP

- `module socket` ([službena dokumentacija](https://docs.python.org/3/library/socket.html)) nudi pristup BSD socket sučelju

    - koristi objekte tipa `bytes` u komunikaciji; kad šaljete objekte tipa `str`, za konverziju između tipova `str` i `bytes` koriste se `str.encode()` i `bytes.decode()`

        ``` python
        s1 = 'Bill Joy'
        b1 = s1.encode() # b1 ima vrijednost b'Bill Joy'
        b2 = b'uti\xc4\x8dnica' # u kodiranju UTF-8 0xc4 0x8d (c48d) je 'č'
        s2 = b2.decode() # s2 ima vrijednost 'utičnica'
        ```

- `socket.AF_UNIX` mrežne utičnice formalno koriste datotečni sustav kao prostor adresa (interna implementacija je efikasnija od pisanja i čitanja iz datoteka)

    - procesi međusobno mogu slati podatke i opisnike datoteka; mi ćemo se ograničiti na slanje podataka, a [Keith Packard](https://keithp.com/) u [članku fd-passing](https://keithp.com/blogs/fd-passing/) objašnjava kako se opisnici datoteka šalju putem utičnica i koja je primjena tih tehnika u X Window Systemu

- `socket.AF_INET` mrežne utičnice koriste IPv4, a `socket.AF_INET6` mrežne utičnice koriste IPv6 kao prostor adresa

    - primjer para IPv4 adrese i TCP/UDP vrata je oblika `('127.0.0.1', 5000)`
    - primjer para IPv6 adrese i TCP/UDP vrata je oblika `('::1', 5000)`
    - za IPv4 i IPv6 možemo koristiti i domene umjesto adresa, npr. `('localhost', 5000)`

- `socket.SOCK_DGRAM` su datagramske utičnice

    - omogućuju jednosmjernu komunikaciju kod koje klijent šalje, a poslužitelj prima prima podatke
    - nema mogućnosti baratanja s više klijenata, svi podaci stižu na jednu utičnicu bez obzira koji od klijenata ih šalje
    - nema osiguranja da će poslani podaci stići

- `socket.bind(address)` se povezuje na adresu `address` (poslužiteljska strana)
- `socket.recv(size)` čita podatke s utičnice do veličine `size` i njih vraća
- `socket.recvfrom(size)` čita podatke s utičnice do veličine `size` i vraća uređeni par `(data, address)`
- `socket.close()` zatvara utičnicu; potrebno je napraviti nakon završetka rada s utičnicom, slično kao kod datoteka

    ``` python
    # poslužiteljska strana, pokreće se prva
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('localhost', 5000))
    podaci = sock.recv(1024)
    sock.close()
    pozdrav = podaci.decode()
    print(pozdrav)
    ```

- ako želimo koristiti datoteke umjesto IP adresa, u primjeru poslužitelja iznad treba promijeniti dvije linije

    ``` python
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind('socket1') # ovu datoteku će biti potrebno ručno izbrisati nakon završetka izvođenja programa
    ```

- `socket.connect(address)` se povezuje na adresu `address` (klijentska strana)
- `socket.send(data)` šalje podatke `data` na adresu na koju je utičnica povezana i vraća veličinu poslanih podataka
- `socket.sendto(data, address)` šalje podatke `data` na adresu `address` i vraća veličinu poslanih podataka

    ``` python
    # klijentska strana, pokreće se nakon pokretanja poslužitelja
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(('localhost', 5000))
    pozdrav = "Pozdrav svijetu!"
    podaci = pozdrav.encode()
    sock.send(podaci)
    sock.close()
    ```

- ako želimo koristiti datoteke umjesto IP adresa, u primjeru klijenta iznad treba promijeniti dvije linije

    ``` python
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind('socket1') # ovu datoteku će biti potrebno ručno izbrisati nakon završetka izvođenja programa
    ```

!!! admonition "Zadatak"
    - Napišite komunikaciju klijenta i poslužitelja tako da klijent pošalje podatak koji korisnik unese (umjesto fiksnog niza znakova pokazanog u primjeru).
    - Dodajte zatim još jedan unos podataka i izvedite dva slanja na strani klijenta i dva primanja na strani poslužitelja.

- `socket.SOCK_STREAM` su tokovne utičnice

    - omogućuju dvosmjernu komunikaciju
    - garantiraju dostavu poruka
    - postoji konekcija dvaju strana

- poslužiteljska utičnica stvara utičnice na strani poslužitelja za klijente koji se povezuju
- `socket.listen(backlog)` poslužiteljska utičnica osluškuje za povezivanja od strane klijenata; prima klijente koji se žele povezati dok broj ne naraste do `backlog` klijenata, a nakon toga odbija nova povezivanja sve do prihvaćanja dotad primljenih klijenata korištenjem funkcije `socket.accept()`
- `socket.accept()` prihvaća klijenta i vraća uređeni par `(socket object, address info)` za svakog klijenta; `socket object` se koristi za komunikaciju s točno tim klijentom; kad se koristi `socket.AF_UNIX`, `address info` je prazan

    ``` python
    # poslužiteljska strana, pokreće se prva
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 5000))
    sock.listen(1)
    clisock, addr = sock.accept()
    while True:
        podaci = clisock.recv(1024)
        if not podaci:
            break
        pozdrav = podaci.decode()
        print(pozdrav)
        poslani_podaci = pozdrav.encode()
        clisock.send(poslani_podaci)
    clisock.close()
    sock.close()
    ```

- klijentska utičnica se povezuje na poslužitelj isto kao kod datagramskih

    ``` python
    # klijentska strana, pokreće se nakon pokretanja poslužitelja
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5000))
    pozdrav = "Pozdrav svijetu!"
    podaci = pozdrav.encode()
    sock.send(podaci)
    primljeni_podaci = sock.recv(1024)
    primljeni_pozdrav = primljeni_podaci.decode()
    print(primljeni_pozdrav)
    sock.close()
    ```

!!! admonition "Zadatak"
    - Preradite kod tako da korisnik na strani klijenta unosi dva broja koja se zatim odvojeno šalju poslužitelju; poslužitelj ih prima i ispisuje njihov zbroj. Pripazite kojeg tipa su podaci kojima baratate i izvedite konverziju gdje je potrebno.
    - Preradite kod tako da poslužitelj klijentu šalje zbroj koji onda klijent ispisuje.

!!! admonition "Zadatak"
    Napišite poslužiteljsku i klijentsku stranu aplikacije za dvosmjernu komunikaciju koristeći datagramske utičnice. Na klijentskoj strani korisnik unosi niz znakova.

    - U slučaju da niz znakova počinje znakom `0`, poslužitelj kao odgovor vraća ostatak niza.
    - U slučaju da niz znakova počinje znakom `a`, poslužitelj kao odgovor vraća duljinu ostatka niza.

    U preostalim slučajevima niz se ne šalje. Klijentska aplikacija prekida nakon jednog slanja i primanja, a poslužiteljska nakon jednog primanja i slanja.

- poslužiteljska aplikacija može primiti više od jednog klijenta korištenjem poziva `select()` iz modula `select` za provjeru postoji li novi klijent ili novi podaci za čitanje na nekoj od postojećih klijentskih utičnica

    - obzirom da se za komunikaciju sa svakim od klijenata koristi posebna utičnica, vrlo je lako razlikovati poruke različitih klijenata

    ``` python
    # poslužiteljska strana
    import socket
    import select

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 5000))
    sockets = [sock] # ova lista će nam trebat ispod za select()
    while True:
        sock.listen(1)
        readready, writeready, exceptready = select.select(sockets, [], [])
        for readysock in readready:
            if readysock == sock:
                # klijent se želi povezati
                clisock, addr = readysock.accept()
                sockets.append(clisock)
            else:
                # neki od klijenata ima podatke za razmijeniti
                while True:
                    podaci = readysock.recv(1024)
                    if not podaci:
                        break
                    pozdrav = podaci.decode()
                    print(pozdrav)
                    poslani_podaci = pozdrav.encode()
                    readysock.send(poslani_podaci)
                readysock.close()
    sock.close()
    ```
