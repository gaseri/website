---
author: Dino Aljević, Vedran Miletić
---

# Python modul PyZMQ: osnove

Python modul PyZMQ ([repozitorij na GitHubu](https://github.com/zeromq/pyzmq), [službena dokumentacija](https://pyzmq.readthedocs.io/)) nudi Python sučelje za [ZeroMQ](https://zeromq.org/) za aktualnu stabilnu verziju ([4.3.4](https://github.com/zeromq/libzmq/releases/tag/v4.3.4)) i sve nasljedne stabilne verzije ([4.2.5](https://github.com/zeromq/libzmq/releases/tag/v4.2.5), [4.1.8](https://github.com/zeromq/zeromq4-1/releases/tag/v4.1.8), [4.0.10](https://github.com/zeromq/zeromq4-x/releases/tag/v4.0.10) i [3.2.5](https://github.com/zeromq/zeromq3-x/releases/tag/v3.2.5)).

U nastavku pratimo [prvo poglavlje](https://zguide.zeromq.org/docs/chapter1/) [zguidea](https://zguide.zeromq.org/).

!!! todo

    - Objasniti razlike između klasičnih utičnica i ZMQ-ovih utičnica.
    - Objasniti neke osnovne pojmove, npr. ZMQ kontekst.
    - Dodati dio *A Minor Note on Strings* i upotpuniti ga s razlikom između `bytes` i `str` objekta u Pythonu.
    - Objasniti sindrom *Slow joiner* kod PUB-SUB socketa i paralelne obrade.
    - Slike prikazati lokalno u slučaju da se promijeni URL.

## Primjer "Hello, World"

Započinjemo klasičnim primjerom "Hello, World". Kreiramo klijenta koji šalje poruku sadržaja `"Hello"` poslužitelju koji mu odgovara porukom `"World"`.

### Poslužiteljska strana komunikacije tipa zahtjev-odgovor

``` python
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    # Čekanje primitka zahtjeva
    message = socket.recv()
    print("Primljen zahtjev: {0}".format(message))

    # Simulacija obrade podataka od 1 sekunde
    time.sleep(1)

    # Slanje odgovora klijentu
    socket.send(b"World")
```

Poslužitelj stvara ZeroMQ kontekst i započinje slušanje na vratima 5555 koristeći utičnicu REP na svim dostupnim sučeljima(znak \*). Klijent i poslužitelj komuniciraju TCP protokolom. Po primitku poruke pozivom `recv()`, poslužitelj ispisuje poruku, pričeka sekundu i zatim pošalje odgovor klijentu pozivom `send()`. Nakon slanja odgovora, poslužitelj nastavlja s čekanjem novog zahtjeva i cijeli proces se ponavlja.

Ovdje je važno napomenuti da se kao poruka šalje niz bajtova, a ne niz znakova. U primjeru smo poslali niz bajtova koji kodirani ASCII standardom odgovaraju riječi `"World"`, međutim mogli smo poslati bilo kakav niz bajtova. Za slanje znakovnih nizova (Pythonov `str` objekt) koji su kodirani UTF-8 (ili nekim drugim) standardom moramo koristiti metodu `send_string`. Dakle, `send_string` metoda se koristi za slanje *znakovnih nizova*, a `send` za proizvoljan *niz bajtova*.

### Klijentska strana komunikacije tipa zahtjev-odgovor

``` python
import zmq

context = zmq.Context()

print("Spajanje na poslužitelj...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Sekvencijalno slanje 10 zahtjeva i čekanje na odgovor svakog
for request in range(10):
    print("Šaljem zahtjev {0} ...".format(request))
    socket.send(b"Hello")

    # Čekanje na pojedini odgovor
    message = socket.recv()
    print("Dobiveni odgovor {0} [ {1} ]".format(request, message))
```

Kao i poslužitelj, klijent započinje stvaranjem ZMQ konteksta. Utičnica se spaja na `localhost`, tj. loopback sučelje s IPv4 adresom 127.0.0.1 i utičnicu REQ. Klijent zatim izvodi slanje poruke `"Hello"`, čeka na odgovor i zatim ispisuje redni broj zahtjeva i odgovor. Cijeli proces se ponavlja 10 puta.

Bitno je spomenuti da se parovi utičnica REQ-REP uvijek pojavljuju jedan iza drugoga. Drugim riječima, REQ uvijek započinje komunikaciju slanjem poruke, a REP šalje odgovor na dobiveni zahtjev. Odstupanje od ovog pristupa rezultira greškom u pozivu `send()` ili `recv()`. Na primjer, nije moguće poslati dvije poruke za redom koristeći utičnicu REQ.

![REQ-REP komunikacija](https://zguide.zeromq.org/images/fig2.png)

Izvor: [zguide, poglavlje 1](https://zguide.zeromq.org/docs/chapter1/)

!!! admonition "Zadatak"

    1. Ako se poslužitelj sruši nakon što je primio zahtjev, ali prije nego li je poslao odgovor, hoće li komunikacija nastaviti ponovnim pokretanjem poslužitelja?
    2. Hoće li program raditi ako se spoji više klijenata? Pokrenite dva klijenta umjesto jednog.
    3. Izmjenite program tako da poslužitelj šalje klijentu istu poruku koju je dobio u zahtjevu.
    4. Izmjenite program tako da poslužitelj šalje odgovor `"Parno"` za parne zahtjeve i `"Neparno"` za neparne.

## Jednosmjerna distribucija podataka

Osim komunikacije tipa zahtjev-odgovor (engl. *request-reply*), u klasične primjere ubrajamo i jednosmjernu distribuciju podataka gdje poslužitelj šalje poruke povezanim klijentima, tzv. komunikacija tipa pretplata-objava (engl. *subscribe-publish*). U ovoj komunikaciji klijenti su pretplaćeni na poruke poslužitelja, međutim oni sami ne šalju nikakve poruke poslužitelju. Slijedi primjer poslužitelja koji šalje podatke o vremenu pretplaćenim klijentima.

![PUB-SUB komunikacija](https://zguide.zeromq.org/images/fig4.png)

Izvor: [zguide, poglavlje 1](https://zguide.zeromq.org/docs/chapter1/)

### Poslužiteljska strana komunikacije tipa pretplata-objava

``` python
import zmq
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

while True:
    zipcode = random.randrange(10000, 53297)
    temperature = random.randrange(-60, 60)
    relhumidity = random.randrange(10, 60)

    socket.send_string("{0} {1} {2}".format(zipcode, temperature, relhumidity))
```

Poslužitelj započinje kreiranjem konteksta i utičnice na vratima 5556. Za vrijeme izvođenja programa, poslužitelj nasumično generira poštanski broj i pripadajuću temperaturu i relativnu vlažnost. Podaci se šalju svim spojenim klijentima kao niz brojeva odvojenih razmakom, dakle `"poštanski_broj temperatura vlažnost"`.

### Klijentska strana komunikacije tipa pretplata-objava

``` python
import sys
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)

print("Dohvaćanje klimaskih podataka s poslužitelja...")
socket.connect("tcp://localhost:5556")

# pretplata na poštanski broj, zadana je Rijeka, 51000
zip_filter = sys.argv[1] if len(sys.argv) > 1 else "51000"
socket.setsockopt_string(zmq.SUBSCRIBE, zip_filter) # ili jednostavno socket.subscribe = zip_filter

# dohvaćanje vrijednosti
total_temp = 0
total_relhumidity = 0
for update_nbr in range(5):
    string = socket.recv_string()
    zipcode, temperature, relhumidity = string.split()
    total_temp += int(temperature)
    total_relhumidity += int(relhumidity)

print("Poštanski broj: {0}".format(zipcode))
print("Prosječna temperatura: {0}".format(total_temp / update_nbr))
print("Prosječna relativna vlažnost: {0}".format(total_relhumidity / update_nbr))
```

Klijent stvara utičnicu SUB i spaja se na utičnicu PUB, odnosno na pošiljatelja. *Izuzetno* važan korak za SUB utičnce je postavljanje filtera poruka koristeći `setsockopt` metodu utičnice. Utičnice SUB odbacuju sve dobivene poruke osim onih koje započinju vrijednosću koja je postavljena u filteru. Bez postavljanja filtera, klijenti će odbacivati *sve* poruke.

Pozivom `setsockopt_string` postavljamo `SUBSCRIBE` filter s vrijednošću poštanskog broja. Klijenti će odbaciti sve poruke koje ne započinju tim poštanskim brojem, odnosno klijenti će dobivati klimatske podatke samo za poštanske brojeve za koji su pretplaćeni. Utičnice SUB se mogu pretplatiti na više poruka višestrukim pozivom `setsockopt` i na sve poruke postavljenjem praznog niza `""` kao vrijednost filtera. Analogno, klijenti se mogu odjaviti postavljanjem vrijednosti `UNSUBSCRIBE`.

Klijent primi i pohrani podatke 5 puta i izračuna prosjek koji potom ispisuje.

PUB-SUB parovi utičnica su asinkroni. utičnica PUB može slati podatke pozivom `send()`, ali ne može primati podatke, pa poziv `recv()` rezultira greškom i obrnuto. U teoriji nije bitno koja utčnica poziva `bind()`, a koja `connect()`, ali u praksi postoje nedokumentirane razlike i preporučuje se pozivanje `bind()` s utičnicom PUB i `connect()` s utičnicom SUB.

Za utičnice PUB-SUB je važno spomenuti da nije moguće precizno odrediti kada će utičnica SUB početi primati podatke. Ako pokrenemo pretplatnika (SUB), pričekamo i zatim pokrenemo pošiljatelja (PUB), pretplanik će uvijek propustiti nekolicinu poruka koju je poslužitelj poslao. Ovo proizlazi iz činjenice da pretplatniku treba neko vrijeme da se spoji na pošiljatelja. To vrijeme ne traje dugo, ali u tom vremenskom periodu pošiljatelj može započeti slati poruke koje pretplatnik neće dobiti budući da još nije završio proces spajanja.

Kasnije ćemo se detaljnije pozabaviti sinkronizacijom, no nije na odmet spomenuti jako naivan način kojim možemo osigurati da pretplatnik ne propusti prvih par poruka. Pošiljatelj može "spavati" prije nego li se pretplatnik spoji, međutim ovo nije dobro rješenje u produkcijskim okruženjima, ali može biti korisno tijekom razvoja.

Drugi način rješavanja problema je da pretpostavimo da je tok podataka s pošiljatelja neograničen i da nema početka niti kraja. Uz to, možemo pretpostaviti da pretplatnika ne zanima što je došlo prije. Na ove dvije pretpostavke se oslanja primjer iznad.

Zadnjih nekoliko točaka o utičnicama PUB-SUB:

- Utičnica SUB se može spojiti na više različitih utičnica PUB višestrukim `connect` pozivima. Podaci utičnici SUB stižu mehanizmom pravednog redanja (engl. *fair queueing*) tako da ni jedan pošiljatelj ne dominira svojim porukama.
- Ukoliko utičnica PUB nema spojenih pretplatnika, ona neće slati poruke mrežom već će ih samo odbaciti.
- Ukoliko utičnice komuniciraju TCP protokolom i pretplatnik je spor, poruke će se gomilati u redu čekanja pošiljatelja.
- Od ZeroMQ verzije 3.x, filtriranje se odvija kod pošiljatelja kada se koristi TCP ili IPC protokol. Kod EPGM protokola, filtriranje se odvija kod pretplatnika. U verziji ZeroMQ 2.x filtriranje se uvijek odvijalo kod pretplatnika.

!!! admonition "Zadatak"

    1. Napravite poslužitelj koji će na vratima 5557 slati drukčije podatke, npr. stanje na cestama. Izmjenite klijent tako da se spaja na oba poslužitelja.
    2. Izmjenite program tako da klijenti pretplaćeni na poštanski broj 10000 uz prijašnje podatke dobivaju i količinu oborina. Pokrenite barem dva klijenta, jedan pretplaćen na poštanski broj 10000, a drugi ne.
    3. Zašto uspavljivanje pošiljatelja nije dobra metoda sinkronizacije?

## Parelelna obrada podataka

Primjer prikazuje paralelnu obradu podataka koristeći utičnice tipa PULL-PUSH. Komponente arhitekture su:

1. Ventilator koji producira zadatke koji se mogu obrađivati paralelno,
2. skup radnika koji paralelno obrađuju zadatke i
3. odvod, odnosno primatelj kojemu se šalju rezultati obrade.

![PUSH-PULL komunikacija](https://zguide.zeromq.org/images/fig5.png)

Izvor: [zguide, poglavlje 1](https://zguide.zeromq.org/docs/chapter1/)

## Ventilator

``` python
import zmq
import random

context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.bind("tcp://*:5557")

# Veza na primatelja (sink). Koristi se za signaliziranje početka obrade.
sink = context.socket(zmq.PUSH)
sink.connect("tcp://localhost:5558")

print("Pritisnite Enter kada su svi radnici spremni: ")
input()
print("Slanje zadataka radnicima...")

# Prva poruka je bajt 0 koji signalizira početak slanja zadataka
sink.send(b'0')

# Slanje 100 zadataka
total_msec = 0
for task_nbr in range(100):
    # Nasumično trajanje zadatka od 1 do 100 ms
    workload = random.randint(1, 101)
    total_msec += workload

    sender.send_string(str(workload))

print("Ukupno očekivano trajanje: {0} ms".format(total_msec))
```

Ventilator stvara dvije utičnice. Prva utičnica je utičnica PUSH preko koje će povezani radnici dobivati zadatke. Druga utičnica je također utičnica PUSH, ali ovom utičnicom će se ventilator povezati na primatelja i javiti mu kada započne obavljanje zadataka.

Nakon što ventilator signalizira početak obrade primatelju, on nasumično generira 100 brojeva od 1 do 100 koji označavaju koliko dugo će svaki radnik simulirati obradu.

## Radnik

``` python
import time
import zmq

context = zmq.Context()

# Utičnica na koji pristižu zadaci
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://localhost:5557")

# Utičnica na koju se šalje rezultat
sender = context.socket(zmq.PUSH)
sender.connect("tcp://localhost:5558")

while True:
    s = receiver.recv()

    # simulacija rada
    time.sleep(int(s) * 0.001)

    # slanje rezultata primatelju
    sender.send(b'')
```

Radnik stvara dvije utičnice, jedna je PULL na koji će mu ventilator slati podatke, a druga je PUSH preko koje će slati rezultat primatelju. Radnik čeka onoliko sekundi koliko je primio od ventilatora, a zatim šalje primatelju bajt 0 kao rezultat.

## Odvod

``` python
import time
import zmq

context = zmq.Context()

# Utičnica na koji pristižu zadaci
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

# čekanje primitka bajta 0, odnosno početka orbade
s = receiver.recv()

# početak mjerenja vremena
tstart = time.time()

# Primitak 100 rezultata
for task_nbr in range(100):
    s = receiver.recv()

# završetak mjerenja vremena
tend = time.time()
print("Ukupno vrijeme obrade: {0} ms".format((tend - tstart) * 1000))
```

Primatelj je najjednostavniji, a započinje stvaranjem konteksta i utičnice PULL na koju radnici šalju rezultat. Obzirom da radnici i ventilator oboje koriste utičnice PUSH za slanje podataka, nije potrebno stvarati novu utičnicu PULL na koju će se spojiti ventilator, već se on spaja na istu kao i radnici.

Nakon što dođe do primitka signala koji označava početak obrade, primatelj počinje mjeriti vrijeme, dohvaća 100 rezultata i zatim ispisuje trajanje obrade.

Nekoliko opservacija o utičnicama PUSH-PULL i paralelnoj obradi:

- Radnici se spajaju uzvodno od ventilatora i nizvodno od primatelja. Ovo nam omogućava da dinamički dodajemo i ukidamo radnike. Kada bi situacija bila obrnuta, tj. kada bi se ventilator i primatelj spajao na radnike, morali bi dodati više krajnjih točaka i morali bi mijenjati ventilator i primatelja svaki put kada dodamo nove radnike. Kažemo da su ventilator i primatelj *stabilni* dijelovi arhitekture, a radnici *dinamični*.
- Potrebno je signalizirati početak obrade kada su svi radnici spremni. To je čest slučaj u ZeroMQ i nema jednostavnog rješenja. Spajanje traje neko vrijeme, pa kada nebi bilo sinkronizacije, prvi spojeni radnik bi dobio najviše zadataka dok su drugi u procesu spajanja. To bi rezultiralo sekvencijalnom umjesto paralelnom obradom.
- Ventilatorova utičnica PUSH podjednako distribuira zadatke radnicima ako su oni spojeni prije nego li obrada započne. Drugim riječima, radnici su podjednako opterećeni.
- Primateljeva utičnica PULL prima rezultate po mehanizmu pravednog redanja.

!!! admonition "Zadatak"

    1. Napišite program koji izračunava zbroj kvadrata brojeva u rasponu od 1 do 500000 koristeći 2 radnika i to tako da ventilator šalje radnicima brojeve od 1 do 250000 i od 250000 do 500000 (respektivno). Radnici računaju zbroj kvadrata i primatelju javljaju rezultate koje su dobili. Primatelj prima oba rezultata i njihov zbroj ispisuje na ekran.
    2. Kako bi optimizirali zadatak 1. s ciljem smanjivanja prometa mrežom?
    3. Je li moguće ovu arhitekturu ostvariti koristeći utičnice REQ i REP? Ako je moguće, koje su potrebne promjene, a ako nije moguće, zašto takva arhitektura nije moguća?
