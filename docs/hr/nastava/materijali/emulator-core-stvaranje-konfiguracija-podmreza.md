---
author: Vedran Miletić
---

# Stvaranje i konfiguracija podmreža

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
