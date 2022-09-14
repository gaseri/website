---
author: Vedran Miletić, Matea Turalija
---

# Stvaranje i konfiguracija podmreža

Prisjetimo se da, kako bismo mogli razlikovati računala spojena na Internet i na taj način usmjeravati pakete s izvornog računala na odredišno računalo, potrebno je računalima dodijeliti jedinstvene adrese.

Internet Protocol (IP) adresa je adresa mrežnog sloja. To je logička adresa dodijeljena uređaju koji želimo spojiti na mrežu. Svako računalo koje je povezano na Internet mora imati jednoznačno dodijeljenu IP adresu. Budući da je IP adresa logička adresa koja se može mijenjati i često se dodjeljuje dinamički, ne može se reći da jedinstveno identificira određeni mrežni uređaj. Umjesto toga, samo omogućuje lociranje uređaja i prosljeđivanje toka podataka do njega.

Trenutno su u upotrebi dvije verzije IP protokola: verzija 4 (IPv4) i verzija 6 (IPv6). IPv4 adresa se sastoji od 32 bita. Maksimalni broj različitih adresa je 2^32 što je približno 4,3 milijarde adresa. Broj 2 se uzima kao baza kalkulacije jer jedan bit može imati 2 stanja: 0 ili 1. Iako se ovaj broj čini prilično velikim, širenje Interneta i porast potražnje za novim IP adresama učinili su ovaj adresni prostor daleko premalim za sve potrebe. Doista, nije daleko dan kada će svaka osoba na svijetu imati svoje računalo koje treba javnu IP adresu, a postoje i mnogi poslužitelji koji također trebaju IP adrese za svoj rad. Osim toga, mobilni i ostali elektronički uređaji (kućanski aparati, razni uređaji u industriji, transportu, turizmu...) vrlo brzo će se integrirati s Internetom u svrhu razmjene i prikupljanja informacija, komunikacije, daljinskog upravljanja i sl.

Stoga su predložena različita rješenja od kojih je i implementacija internetskog protokola IPv6 koji ima širi raspon dostupnih adresa. IPv6 koristi 128-bitnu IP adresu, stoga je maksimalan broj različitih adresa 2^128 što je približno 3400 trlijun trilijuna mogućih adresa. Ovaj je broj teško uopće pojmiti. Ovdje je dovoljno reći da se teško može zamisliti stvarna situacija u kojoj taj broj različitih adresa ne bi bio dovoljan za današnje primjene. Međutim, sustav se ne može odmah implementirati jer postoje određeni problemi u njegovom uvođenju. Stoga se preporuča postupno uvođenje sustava čime se i postupno unapređuje sama mrežna infrastruktura.

## Karakteristike IPv4 adrese

IPv4 adresa je 32 bitna logička adresa koja jasno određuje domaćina mreže. Zbog lakšeg rada sa adresama, one se bilježe brojevima decimalnog brojevnog sustava. 32 bita su podjeljena u 4 okteta po 8 bitova. Svaki od okteta je odjeljen od slijedećeg s točkom.

IP adresa je u osnovi binarni broj, međutim zbog lakšeg rada s adresama, one se bilježe brojevima decimalnog brojevnog sustava. Decimalne vrijednosti svakog od okteta mogu se kretati od 0 do 255, a u binarnom sustavu od 00000000 do 11111111. Od 8 bitova koji se nalaze unutar okteta moguće je dobiti 2ˆ8 = 256 različitih brojčanih vrijednosti, odnosno adresa. IPv6 verzija protokola predviđa 128-bitne adrese te se u tom slučaju može koristiti i heksadecimalni zapis, radi kraćeg oblika i jednostavnosti.

Primjer IPv4 adrese domene www.inf.uniri.hr je 193.198.209.68, odnosno u binarnom zapisu 11000001.11000110.11010001.1000100

## Mrežne klase i maska podmreže

Dio bitova unutar IP adrese definira adresu mreže, a ostatak adrese domaćina. Svaka IP adresa dolazi uz pripadajuću masku podmreže (engl. *subnet mask*). Uz pomoć maske podmreže možemo razlikovati koji je dio adrese dio mreže, a koji dio domaćina. Primjerice, ako imamo IP adresu 193.198.209.68 sa mrežnom podmaskom 255.255.255.0 (ili /24), to znači da će dio 193.198.209 biti adresa mreže, odnosno nepromjenjiva za sva računala u toj mreži. Preostali će dio .68 biti adresa čvora (računala ili usmjerivača) različita za svaki čvor te mreže.

Pomoću tehnike podmrežavanja možemo podijeliti velike mreže određene klase na manje podmreže. Najčešće se koriste klase adresa A, B i C. Svaka klasa prepoznaje se po svojoj masci podmreže. Kategorizacijom zadane maske podmreže možemo lako prepoznati klasu IP adrese mreže:

- Klasi A adresa pripadaju sve adrese kojima prvi oktet počinje sa brojem između 1 i 126. Zadana mrežna maska definira prvi oktet kao mrežni dio adrese, a ostatak je adresa domaćina (255.0.0.0 ili /8).
- Klasi B pripadaju sve adrese kojima prvi oktet počinje brojem između 128 i 191. Zadana mrežna maska definira prva dva okteta kao mrežni dio adrese, a ostatak je adresa domaćina (255.255.0.0 ili /16).
- Klasi C pripadaju sve adrese kojima prvi oktet počinje brojem između 192 i 223. Zadana mrežna maska definira prva tri okteta kao mrežni dio adrese, a ostatak je adresa domaćina (255.255.255.0 ili /24).
- Klasi D pripadaju sve adrese kojima prvi oktet počinje brojem između 224 i 239 (od 224.0.0.0 do 239.255.255.255). Ovo su višeodredišne, multicast adrese.
- Klasi E pripadaju sve adrese kojima prvi oktet počinje brojem između 240 i 247 (od 240.0.0.0 do 247.255.255.255). Ove adrese služe za istraživačke svrhe.

## Podmreže varijabilne duljine

U slučaju različitog broja računala u svakoj podmreži potrebno je koristiti varijabilne maske. To znači da neće sva računala koristiti istu mrežnu masku, nego će svaka podmreža imati različitu masku. Varijabilno segmentiranje mreže tako omogućuje najučinkovitiju podjelu mreže.

Recimo da na raspolaganju imamo raspon adresa 192.198.25.0/24. Ako trebamo podmreže koje mogu smjestiti 120 računala za goste, 30 za prodaju, 20 za marketing, 10 za upravu i dva računala za šefa neke tvrtke, ustanovit ćemo da nepotrebno gubimo IP adrese ako dodijelimo 8 bita svakoj mreži, što odgovara 256 mogućih adresa računala.

Stoga postoji mogućnost definiranja različitih mrežnih maski za svaku podmrežu. To znači da prvoj mreži dodijelimo 7 bitova jer je 2^6 = 64 < 120 < 2^7 = 128, sljedećoj 5 bitova (2^5 = 32 IP adrese), marketingu također 5 bitova, upravi 4 bita i šefu tvrtke 2 bita. Dok ostale neiskorištene bitove možemo pripojiti mrežnom dijelu.

!!! note
    U varijabilnim mrežama vrlo je bitno da podmrežavanje započnemo od najvećih mreža, jer će nam to kasnije olakšati dodavanje novih računala ukoliko se ukaže potreba za time.

- U našem primjeru mreža za goste imat ćemo raspon od 192.198.25.1 do 192.198.25.126 s mrežnom podmaskom /25. Kada stavimo 7 bitova za adrese domaćina, proširujemo mrežni dio na 25 bitova.
- Mreža za prodaju treba 5 bitova i krenut će od sljedeće dostupne IP adrese, a to je 192.198.25.129 pa do 192.198.25.158 s prefiksom duljine /27.
- Isto vrijedi vrijedi i za marketing, koji će dobiti sljedeći dostupni raspon od 192.198.25.161 do 192.198.25.190 s prefiksom duljine /27.
- Za upravu imat ćemo raspon od 192.198.25.193 do 192.198.25.206 s prefiksom duljine /28.
- A šef tvrtke će dobiti raspon od 192.198.25.209 do 192.198.25.210 s prefiksom duljine /30.

Primijetite ovdje da preskačemo dvije IP adrese između podmreža. Početna i krajnja adresa unutar podmreže imaju posebno značenje i općenito se ne koriste kao adrese pojedinačnih mrežnih uređaja. Početna adresa je adresa podmreže koja identificira cijelu podmrežu. Završna ili broadcast adresa je adresa na kojoj mrežni promet primaju sva računala unutar podmreže. Najmanja mreža koja nema drugih podmreža naziva se broadcast domena, to je u biti lokalna mreža (engl. *local area network*, kraće LAN). Unutar broadcast domene mrežni uređaji (računala, komunikacijska oprema, ...) međusobno izravno komuniciraju pomoću fizičkih Media Access Control (MAC) adresa.

MAC adresa je broj koji označava neku mrežnu karticu (engl. *network interface card*, kraće NIC) i jedinstvena je za svaki uređaj. Sastoji se od 48 bita po 6 okteta koji se zapisuje u heksadecimalnom brojevnom sustavu na više različitih načina grupiranja i odvajanja znamenki:

- 6 parova znamenki odvojenih crticom: 00-1A-4D-5B-05-AB
- 6 parova znamenki odvojenih dvotočkom: 00:1A:4D:5B:05:AB
- 3 skupine po 4 znamenke odvojene točkom: 001A.4D5B.05AB

MAC adresa je logično podijeljena na dva dijela. Prva 24 bita predstavljaju naziv proizvođača mrežne kartice i isti su za sve kartice tog proizvođača. Preostala 24 bita jedinstvena su za svaku karticu i dodjeljuje ih proizvođač. Iako je zamišljeno da MAC adresa predstavlja mrežni uređaj na potpuno jedinstven način, to nije tako, jer većina današnjih mrežnih kartica ima mogućnost promjene MAC adrese. Taj se proces naziva MAC spoofing.

## Postupak stvaranja i konfiguracija podmreža

Vidjeli smo već kako unutar alata CORE čvorovi tipa `ethernet switch` povezuju čvorove u jednu podmrežu, a čvorovi tipa `router` omogućuju povezivanje različitih podmreža. Konkretno, mreža oblika

```
          n3
           |
           |
           |
n2 ------ n1 ------ n4
           |
           |
           |
          n5
```

imat će četiri podmreže ako je n1 tipa `router`. Primjerice, te podmreže mogu imati adrese:

- 10.0.1.0/24 za vezu n2 -- n1 tako da n2 ima adresu 10.0.1.2, a n1 ima adresu 10.0.1.1
- 10.0.2.0/24 za vezu n3 -- n1 tako da n3 ima adresu 10.0.2.2, a n1 ima adresu 10.0.2.1
- 10.0.3.0/24 za vezu n4 -- n1 tako da n4 ima adresu 10.0.3.2, a n1 ima adresu 10.0.3.1
- 10.0.4.0/24 za vezu n5 -- n1 tako da n5 ima adresu 10.0.4.2, a n1 ima adresu 10.0.4.1

Za usporedbu, ako je n1 tipa `ethernet switch`, svi čvorovi i veze bit će jedna podmreža. Primjerice, ta podmreža može imati adresu 10.0.1.0/24 i tada n1 nema adresu, a n2, n3, n4 i n5 mogu imati redom adrese 10.0.1.1, 10.0.1.2, 10.0.1.3, 10.0.1.4.

!!! danger
    U mreži 10.0.1.0/24 adresa 10.0.1.0 je adresa mreže i zbog toga ne može biti adresa niti jednog čvora. Ukoliko je dodijelimo nekom čvoru, komunikacija s tim čvorom neće biti ostvariva.

CORE u zadanim postavkama bira podmreže s duljinom prefiksa 24 iz raspona adresa 10.0.0.0/8. Kako je 24 - 8 = 16 bitova proizvoljno, tih podmreža ima 2^16 = 65536. One su redom 10.0.0.0/24, 10.0.1.0/24, 10.0.2.0/24, ..., 10.0.255.0/24, 10.1.0.0/24, 10.1.1.0/24, ..., 10.2.0.0/24, ..., 10.255.0.0/24, 10.255.1.0/24, ..., 10.255.255.0/24. Moguće je koristiti i bilo koji drugi od privatnih raspona navedenih u [RFC-u 1918 pod naslovom Address Allocation for Private Internets](https://datatracker.ietf.org/doc/html/rfc1918) navođenjem tog raspona u `Session/Options...` ([dokumentacija](https://coreemu.github.io/core/gui.html)).

Proces stvaranja i konfiguracije podmreža omogućava odvajanje pojedinih domaćina, računala i radnih stanica u skupine kojima je moguće dodijeliti različita pravila komunikacije (npr. zabranu komunikacije između pojedinih skupina, garantiranu širinu pojasa za pojedinu skupinu, pravo pristupa internetu za neke skupine i sl.)

Stvaranje podmreža je trivijalan problem kada imamo praktički neograničen broj dostupnih IP adresa, što je slučaj kada imamo mreže veličine nekoliko desetaka čvorova unutar CORE-a i koristimo čitavu podmrežu 10.0.0.0/8. Stvaranje podmreža je složen problem kada nam je na raspolaganju ograničen adresni prostor.

## Primjeri stvaranja podmreža

### Primjer 1

Uzmimo da su nam na raspolaganju adrese u rasponu 10.0.5.0/24 i da je zadana mreža oblika

```
n3              n4
 \              /
  \            /
  n1 -------- n2
  /            \
 /              \
n5              n6
```

pri čemu su čvorovi n1 i n2 tipa `router`, a n3, n4, n5 i n6 tipa `ethernet switch` i na njih će biti vezano redom 14 domaćina, 28 domaćina, 7 domaćina i 28 domaćina. Stvorimo podmreže tako da je moguće podijeliti adrese domaćinima.

Uočimo prvo da zadani raspon ima mrežni prefiks duljine 24 bita i identifikator domaćina duljine 8 bita. Zbog toga imamo na raspolaganju 2^8 = 256 adresa, od kojih 254 možemo iskoristiti kao adrese domaćina (podsjetimo se da je prva adresa upravo adresa mreže, a posljednja adresa služi za broadcast slanje).

Najveća podmreža mora imati dovoljno adresa za 28 domaćina i jedan usmjerivač, dakle trebamo ukupno 29 adresa. Uzmemo li prefiks duljine 27, imat ćemo 2^5 = 32 adrese, od kojih 30 možemo iskoristiti za domaćine, što nam je dovoljno. Takvih podmreža ima 2^8 / 2^5 = 2^3 = 8, a dovoljno nam ih je 5. Jedna moguća dodjela podmreža je:

- podmreža oko n3: 10.0.5.0/27 (raspon adresa domaćina je od 10.0.5.1 do 10.0.5.30)
- podmreža oko n4: 10.0.5.32/27 (raspon adresa domaćina je od 10.0.5.33 do 10.0.5.62)
- podmreža oko n5: 10.0.5.64/27 (raspon adresa domaćina je od 10.0.5.65 do 10.0.5.94)
- podmreža oko n6: 10.0.5.96/27 (raspon adresa domaćina je od 10.0.5.97 do 10.0.5.126)
- podmreža n1 -- n2: 10.0.5.128/27

!!! note
    Ukoliko stvorimo ovakvu mrežu unutar CORE-a i zatim postavimo adrese i duljine prefiksa na n1 i n2, kod kasnijeg dodavanja domaćina i povezivanja istih na n3, n4, n5 i n6 CORE će automatski dodijeliti adrese domaćinima u rasponu odgovarajućih podmreža određenih prema postavljenim adresama na usmjerivačima.

### Primjer 2

Uzmimo opet da su nam na raspolaganju adrese u rasponu 10.0.5.0/24 i da je zadana mreža oblika

```
n3              n4
 \              /
  \            /
  n1 -------- n2
  /            \
 /              \
n5              n6
```

pri čemu su čvorovi n1 i n2 tipa `router`, a n3, n4, n5 i n6 tipa `ethernet switch` i na njih će biti vezano redom 74 domaćina, 39 domaćina, 22 domaćina i 8 domaćina. Stvorimo podmreže tako da je moguće podijeliti adrese domaćinima.

Najveća podmreža mora imati dovoljno adresa za 74 domaćina i jedan usmjerivač, dakle 75. Uzmemo li prefiks duljine 26, ostaju nam 2^6 - 2 = 62 adrese za domaćine, što nam nije dovoljno pa moramo uzeti prefiks duljine 25 koji nam daje 2^7 - 2 = 126 adresa za domaćine. U rasponu koji imamo na raspolaganju postoje samo dvije takve podmreže, pa moramo drugu dalje dijeliti.

Za podmrežu koja sadrži 39 domaćina i jedan usmjerivač, treba nam prefiks duljine 26, što nam daje 2^6 - 2 = 62 adresa.

Za podmrežu koja sadrži 22 domaćina i jedan usmjerivač, treba nam prefiks duljine 27, što nam daje 2^5 - 2 = 30 adresa.

Za podmrežu koja sadrži 8 domaćina i jedan usmjerivač, treba nam prefiks duljine 28, što nam daje 2^4 - 2 = 14 adresa.

Za podmrežu koja sadrži dva usmjerivača možemo iskoristiti posljednju preostalu podmrežu s prefiksom duljine 28, iako bi i manja bila dovoljna.

Jedna moguća dodjela podmreža je sada oblika:

- podmreža oko n3: 10.0.5.0/25 (raspon adresa domaćina je od 10.0.5.1 do 10.0.5.126)
- podmreža oko n4: 10.0.5.128/26 (raspon adresa domaćina je od 10.0.5.129 do 10.0.5.190)
- podmreža oko n5: 10.0.5.192/27 (raspon adresa domaćina je od 10.0.5.193 do 10.0.5.222)
- podmreža oko n6: 10.0.5.224/28 (raspon adresa domaćina je od 10.0.5.225 do 10.0.5.238)
- podmreža n1 -- n2: 10.0.5.240/28

!!! tip
    Za lakši izračun koje su adrese dio pojedine podmreže možete iskoristiti gotove alate. U naredbenom retku najčešće korišten alat je [ipcalc](http://jodies.de/ipcalc), a na webu je jedan od boljih alata [IP Subnet Calculator](https://www.calculator.net/ip-subnet-calculator.html).
