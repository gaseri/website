---
author: Alen Hamzić, Darian Žeko, Matea Turalija
---

# Bistabili i registri

Bistabili i registri su ključni građevni blokovi modernih računala koji nam omogućuju pohranjivanje i rukovanje digitalnim informacijama. Kako se, zapravo, zapisuju nule i jedinice u računalu te kako se informacije prenose, pitanja su koja se na prvi pogled čine jednostavnima, ali su ključna za razumijevanje rada računala. Saznanje o tome kako se informacije pohranjuju i prenose omogućuje nam bolje razumijevanje funkcioniranja računalnih uređaja i njihovih komponenti.

U ovom poglavlju upoznat ćemo se s bistabilima i registrima, građevnim blokovima modernih računala koji nam omogućuju pohranjivanje i rukovanje digitalnim podacima. Naučit ćemo kako se bistabili mogu koristiti za stvaranje memorijskih elemenata, a registri za čuvanje i manipuliranje podacima.

## Bistabil

Bistabil (eng. *flip-flop*) je temeljni element računalne memorije i logičkih sklopova u računalima. On nam omogućuje pohranu i čuvanje podataka u digitalnom obliku, što je ključno za rad računalnih sustava.

Bistabil je sekvencijalni sklop koji može zapamtiti podatke veličine jednog bita informacije u dva stabilna stanja. Stanja se obično opisuju kroz niži napon (oko $0$ V) koji odgovara znamenki $0$ i viši napon (oko $5$ V) koji odgovara znamenki $1$. Koristi se u računalnoj memoriji jer može zadržati podatke bez stalne primjene energije, čime se omogućuje trajno pohranjivanje podataka.

### Osnovne mjerne jedinice

Kako bismo mogli mjeriti količinu podataka koju pohranjujemo, prenosimo ili obrađujemo, korisno je uvesti mjerne jedinice koje nam omogućuju da lakše shvatimo količinu podataka. Bitovi (engl. *BInary digiT*) su osnovne jedinice digitalne informacije u računalima. To su jednobitni podaci koji imaju dvije moguće vrijednosti, $0$ ili $1$, a prikazuju najmanju količinu informacije koja se može pohraniti. Ove vrijednosti su ključne za sve digitalne računalne operacije i omogućuju nam da kodiramo, prenosimo i obrađujemo informacije. Skup od 8 bitova naziva se **bajt** (eng. *byte*), a zapise u računalu koji sadrže veću dužinu bajtova nazivamo *riječ*.

### Izvedene mjerne jedinice

Kako su računala postajala sve naprednija, tako su mogla obrađivati i pohranjivati sve veće količine podataka. Uz napredak tehnologije pohrane podataka bilo je nužno uvesti nove mjerne jedinice koje bi omogućile da se količina podataka jasnije izrazi. Tako je ubrzo uvedena kilobajt (kB) kao mjerna jedinica za $1000$ bajtova. To se pokazalo nepraktičnim s obzirom na to da su računala bila efikasnija u radu s brojevima koji su cjelobrojne potencije broja $2$. Zbog toga je ubrzo kilobajt definiran kao $1024$ bajtova ($2^{10}$). Uz to, uvedene su i druge mjerne jedinice poput megabajta ($1$ MB = $2^{20}$ bajtova), gigabajta ($1$ GB = $2^{30}$ bajtova) itd.

Krajem 20. stoljeća, s porastom popularnosti računala, tradicionalne mjerne jedinice za količinu podataka postale su zbunjujuće za sve više ljudi. Kao što je poznato, prefiks "kilo" označava tisuću ili $10^3$, "mega" milijun ili $10^6$, "giga" milijardu ili $10^9$, itd.

Međutim, kako računala rade s potencijama broja $2$ prirodnije bi bilo da kilobajt označava točno $2^{10}$ bajtova ($1024$ bajta), a ne $1000$ bajtova. Stoga je u siječnju 1999. godine [Međunarodna elektrotehnička komisija](https://en.wikipedia.org/wiki/International_Electrotechnical_Commission) (IEC) uvela binarne prefikse za količine podataka. Prema tome, kilobajt i dalje označava $1000$ bajtova, a $1024$ bajta označava kibibajt (oznaka KiB). Slijedi pregled izvedenih mjernih jedinica za količinu podataka:

| Binarna jedinica |     |   | Dekadska jedinica |     |   |
| :--------------: | :-: | - | :---------------: | :-: | - |
| *Ime* | *Simbol* | *Količina podataka* | *Ime* | *Simbol* | *Količina podataka* |
| kibibajt | KiB | $1,024$ B $= 2^{10}$ B | kilobajt | kB | $10^3$ B |
| mebibajt | MiB | $1,048,576$ B $= 2^{20}$ B | megabajt | MB | $10^6$ B |
| gibibajt | GiB | $1,073,741,824$ B $= 2^{30}$ B | gigabajt | GB | $10^9$ B |
| tebibajt | TiB | $1,099,511,627,776$ B $= 2^{40}$ B | terabajt | TB | $10^{12}$ B |
| petibajt | PiB | $1,125,899,906,842,624$ B $= 2^{50}$ B | pentabajt | PB | $10^{15}$ B |
| eksbibajt | EiB | $1,152,921,504,606,846,976$ B $= 2^{60}$ B | eksabajt | EB | $10^{18}$ B |
| zebibajt | ZiB | $1,180,591,620,717,411,303,424$ B $= 2^{70}$ B | zetabajt | ZB | $10^{21}$ B |
| jobibajt | YiB | $1,208,925,819,614,629,174,706,176$ B $=2^{80}$ B | jotabajt | YB | $10^{24}$ B |

## Registar

Registar je sklop koji se sastoji od više povezanih bistabila i omogućuje pohranjivanje i manipuliranje većeg broja bitova informacije. Koriste se u računalima za brzo pohranjivanje i prijenos podataka. Različiti registri se razlikuju po veličini i funkcionalnosti, a mogu se koristiti za pohranjivanje brojeva, adresa, kontrolnih signala i drugih vrsta podataka.

Registri se označavaju kao $n$-bitni registar, gdje $n$ predstavlja broj bitova koje registar može pohraniti. Na primjer, $16$-bitni registar može pohraniti $16$ bitova podataka, a to znači da je broj mogućih stanja registra:

$$m = B^n.$$

$B$ označava bazu sustava u kojem se zapisuje broj pa je za binarni sustav baza $2$: $m = 2^{16}$. Dakle, s $16$ povezanih bistabila možemo pohraniti $65536$ različitih vrijednosti podataka.

Najveći broj koji se može zapisati u registru je:

$$W = B^n−1.$$

Prema tome, najveći broj koji možemo zapisati u $16$-bitnom registru je $65535$.

!!! example "Zadatak"
    1. Što je bit, a što bajt?
    2. Što je bistabil, a što registar?
    3. Koji je dio računala izgrađen od bistabila?
    4. Koliko različitih podataka možemo zapisati pomoću $8$ povezanih bistabila?
    5. Koliko je bitova potrebno za prikaz $128$ različitih podataka?
    6. Koji je najveći broj koji možemo zapisati u $4$-bitnom registru?
