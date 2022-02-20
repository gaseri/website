---
author: Ivan Ivakić, Vedran Miletić
---

# Modularni usmjerivač Click

[Click](https://github.com/kohler/click) je softver koji omogućuje izgradnju usmjerivača korištenjem gotovih modula. Omogućuje veliku slobodu u konfiguraciji i specificiranju načina usmjeravanja i prosljeđivanja paketa. Click podržava realizaciju preklopnika, "običnog" usmjerivača, usmjerivača koji vrši prevođenje mrežnih adresa (Network Address Translation, NAT), vatrozida ili bilo kojeg drugog softverskog entiteta koji obrađuje pakete. S obzirom na dobru unutrašnju strukturu sloboda pri konfiguriranju je velika čime se pruža mogućnost detaljnog specificiranja raznih protokola te je pogodan za simuliranje standardnih, kao i eksperimentalnih konfiguracija.

Click je napisan u programskom jeziku C++ i dostupan je kao otvoreni kod pod modficiranom MIT licencom. Detaljni opis Clicka dan je u [doktoratu Eddieja Kohlera pod naslovom The Click Modular Router](https://read.seas.harvard.edu/~kohler/pubs/phdthesis.pdf). Mi se internom implementacijom Clicka nećemo detaljnije baviti, već ćemo se fokusirati na elemente koje on nudi i njihovu konfiguraciju. Programski jezik koji se koristi za definiranje same konfiguracije je vlastiti jezik Click, a datoteke koje sadrže konfiguraciju imaju ekstenziju `.click`.

Click pokrećemo naredbom `click` i navođenjem konfiguracijske datoteke. Primjerice, za konfiguracijsku datoteku `mojrouter.click` naredba bi bila oblika:

``` shell
$ click mojrouter.click
```

U nekim konfiguracijama Click će se nastaviti izvoditi sve dok ga ne prekinemo. Naime, radi se o usmjerivaču koji je osmišljen da nakon inicijalizacije konfiguracije radi usmjeravanje i prosljeđivanje paketa dokle god mu se ne kaže da radi drugačije. Prekid rada Clicka ćemo izvesti kombinacijom tipki ++control+c++, što terminali često prikazuju kao `^C`.

## Vlastiti jezik Click

[Jezik Click](https://github.com/kohler/click/wiki/Language) opisuje konfiguracije Click usmjerivača. Radi se o vlastitom jeziku koji nema veze s C++-om u kojem je Click implementiran. Dva su osnovna sintaktička elementa jezika Click: deklaracija i veza.

Deklaracija je oblika:

``` c
name :: class(config);
```

Ovim kodom uvodi se element imena `name` koji je klase elementa `class` i ima konfiguraciju `config`.

Veza je oblika:

``` c
name1 [port1] -> [port2] name2;
```

Ovim kodom povezuje se izlazna vrata `port1` elementa `name1` s ulaznim vratima `port2` elementa `name2`.

## Elementi

Za primjere elemenata uzet ćemo `RandomSource` ([dokumentacija](https://github.com/kohler/click/wiki/RandomSource)), koji generira slučajne pakete, i `Tee` ([dokumentacija](https://github.com/kohler/click/wiki/Tee)), koji duplicira pakete, analogno naredbi `tee` na operacijskim sustavima sličnim Unixu ([kopira standardni ulaz na standardni izlaz i u datoteku](https://shapeshed.com/unix-tee/)).

!!! note
    Ako ste ispravno instalirali Click, dokumentacija elemenata vam je dostupna i u man stranicama u sekciji 7. Primjerice, za dokumentaciju elementa Tee, naredba je `man 7 Tee`. Dokumentacija samog Clicka, kao i većine naredbi na operacijskim sustavima sličnim Unixu, dostupna je u sekciji 1 (naredba `man 1 click` ili samo `man click`).

Element `RandomSource` ima obavezan parametar u kojem se navodi duljina paketa. Element `Tee` ima obavezan parametar u kojem se navodi broj izlaza. Povezivanje izvora paketa duljine 64 bajta i duplikatora paketa s dva izlaza izvest ćemo kodom:

``` c
RandomSource(64) -> Tee(2);
```

Pokušamo li navedeni kod spremiti u datoteku `randomsrc-tee.click` i pokrenuti u Clicku, dobit ćemo poruku o grešci.

``` shell
$ click randomsrc-tee.click
foo.click:1: ‘Tee@2 :: Tee’ output 0 unused
```

Spojimo izlaze elementa `Tee` na elemente `ToDump` ([dokumentacija](https://github.com/kohler/click/wiki/ToDump)), koji zapisuje pakete u formatu pcap, i `ToIPSummaryDump` ([dokumentacija](https://github.com/kohler/click/wiki/ToIPSummaryDump)), koji zapisuje informacije o paketima u tekstualnom obliku. Također, ograničimo broj paketa koji će biti generirani od strane elementa `RandomSource` na 1000.

``` c
RandomSource(64, LIMIT 1000) -> t :: Tee(2);
t[0] -> ToDump(paketi.pcap);
t[1] -> ToIPSummaryDump(paketi.txt);
```

Obzirom da izlaz 0 možemo izravno spojiti, navedeni kod možemo zapisati i na način:

``` c
RandomSource(64, LIMIT 1000) -> t :: Tee(2) -> ToDump(paketi.pcap);
t[1] -> ToIPSummaryDump(paketi.txt);
```

Navedeni kod spremamo u datoteku `randomsrc-tee-dump.click` i pokrećemo naredbom:

``` shell
$ click randomsrc-tee-dump.click
```

Pokretanjem Clicka uočit ćemo da ovaj put nema grešaka te će obje izlazne datoteke biti stvorene i ispunjene sadržajem. Kada Click završi generiranje 1000 paketa i njihovo zapisivanje u obje datoteke, nastavit će se izvoditi (i čekati na nove pakete iz elementa `RandomSource`) sve dok ga ne prekinemo.

Možemo prekinuti izvođenje Clicka i datoteku `paketi.pcap` možemo pogledati u Wiresharku, a datoteku `paketi.txt` u bilo kojem uređivaču teksta ili `less`-u. Uočit ćemo da naši slučajno generirani paketi nisu IP paketi pa nismo dobili naročito mnogo informacija o njima sa zadanom konfiguracijom elementa `ToIPSummaryDump` (detaljni pregled informacija koje je moguće ispisati možemo pronaći u dokumentaciji elementa).

## Primjeri konfiguracija Click usmjerivača

!!! note
    Primjeri konfiguracija dani ispod preuzeti su s danas nedostupnih [službenih stranica Clicka](https://web.archive.org/web/20171005171702/http://www.read.cs.ucla.edu/click/click), specifično [službenog tutoriala](https://web.archive.org/web/20171008224301/http://www.read.cs.ucla.edu/click/tutorial1). Također nedostupna [dokumentacija elemenata](https://web.archive.org/web/20171003052722/http://www.read.cs.ucla.edu/click/elements) prebačena je na [Clickovom wiki na GitHubu](https://github.com/kohler/click/wiki/Elements).

### Primjer 1

Konfigurirajte usmjerivač koji čita pakete iz tcpdump datoteke `f1a.pcap` i zapišite te pakete u novu tcpdump datoteku `f1b.pcap`. Pakete pritom nemojte mijenjati ni na kakav način.

### Rješenje primjera 1

Element `FromDump` ([dokumentacija](https://github.com/kohler/click/wiki/FromDump)) prima dva parametra: prvi je ulazna datoteka, a drugi određenje da se čita do kraja datoteke.

Element `ToDump` ([dokumentacija](https://github.com/kohler/click/wiki/ToDump)) je ekvivalent prethodnoj funkciji samo u "obrnutom" smjeru, prvi parametar je izlazna datoteka, a drugi je tip enkapsulacije koja se koristi.

``` c
FromDump(f1a.pcap, STOP true) -> ToDump(f1b.pcap, ENCAP IP);
```

### Primjer 2

Konfigurirajte usmjerivač koji će usmjeravati pakete u izlazne datoteke s obzirom na danu tablicu usmjeravanja i pripadajućih odredišta (izlaznih datoteka).

| Odredište | Izlazne datoteke |
| --------- | ---------------- |
| 131.0.0.0/8 | `f2b.pcap` |
| 131.179.0.0/16 | `f2c.pcap` |
| 18.0.0.0/8 | `f2d.pcap` |
| Ostalo | `f2e.pcap` |

### Rješenje primjera 2

Element `RadixIPLookup` ([dokumentacija](https://github.com/kohler/click/wiki/RadixIPLookup)) prima uparene vrijednosti adrese i pripadajuće maske te porta kao parametre. Ponovno čitamo ulaznu datoteku elementom `FromDump` i zapisujemo pomoću elementa `ToDump`. Primijetimo kako povezujemo pojedine izlaze elementa `RadixIPLookup`, kojih ima četiri, s četiri različita elementa `ToDump`. Indeksi izlaza počinju od 0 te odgovaraju redoslijedu uparenih vrijednosti adrese i izlaznih vrata elementa.

``` c
FromDump(f2a.pcap, STOP true)
    -> r :: RadixIPLookup(131.0.0.0/8 0, 131.179.0.0/16 1, 18.0.0.0/8 2, 0/0 3);
r[0] -> ToDump(f2b.pcap, ENCAP IP);
r[1] -> ToDump(f2c.pcap, ENCAP IP);
r[2] -> ToDump(f2d.pcap, ENCAP IP);
r[3] -> ToDump(f2e.pcap, ENCAP IP);
```

### Primjer 3

Dogradite kreirani usmjerivač prethodnog zadatka koji će čitati `f3a.pcap` datoteku te iz nje kreirati `f3f.pcap` datoteku poštujući pravila dana u tablici. (**Napomena:** pripazite na redoslijed provjera te prethodno danu tablicu usmjeravanja.)

| Problem | Akcija |
| ------- | ------ |
| Neispravno IP zaglavlje ili checksum | Odbaciti paket |
| Neispravno TCP zaglavlje ili checksum | Odbaciti paket |
| Neispravno UDP zaglavlje ili checksum | Odbaciti paket |
| Neispravno ICMP zaglavlje ili checksum | Odbaciti paket |
| Istek TTL-a | Generirati pripadajuću ICMP poruku, poslati poruku u `f3f.pcap` datoteku |
| Paket duži od 1500 bajtova | Odbaciti paket |

### Rješenje primjera 3

Novi elementi koje ćemo koristiti su `CheckIPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckIPHeader)), `CheckTCPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckTCPHeader)), `CheckLength` ([dokumentacija](https://github.com/kohler/click/wiki/CheckLength)), `CheckUDPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckUDPHeader)), `CheckICMPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckICMPHeader)) i `ICMPError` ([dokumentacija](https://github.com/kohler/click/wiki/ICMPError)). Uočimo da je ovdje poredak provjera značajan.

``` c
FromDump(f3a.pcap, STOP true)
     -> CheckIPHeader
     -> i1 :: IPClassifier(tcp, udp, icmp, -)
     -> CheckTCPHeader
     -> ttl :: IPClassifier(ttl > 0, -)
     -> cl :: CheckLength(1500)
     -> ip :: IPClassifier(dst 131.179.0.0/16, dst 131.0.0.0/8, dst 18.0.0.0/8, -);
  i1[1] -> CheckUDPHeader -> ttl;
  i1[2] -> CheckICMPHeader -> ttl;
  i1[3] -> ttl;
  ttl[1] -> ICMPError(18.26.7.1, timeexceeded, transit)
     -> ToDump(f3f.pcap, ENCAP IP);
  ip[0] ->  ToDump(f3c.pcap, ENCAP IP);
  ip[1] ->  ToDump(f3b.pcap, ENCAP IP);
  ip[2] ->  ToDump(f3d.pcap, ENCAP IP);
  ip[3] ->  ToDump(f3e.pcap, ENCAP IP);
```

### Primjer 4

Nadogradite prethodni primjer tako da brojite koliko je paketa odbačeno iz pojedinih razloga. Odredite:

1. Broj paketa sa neispravnim IP zaglavljem ili kontrolnim zbrojem
1. Broj paketa sa neispravnim TCP zaglavljem ili kontrolnim zbrojem
1. Broj paketa sa neispravnim UDP zaglavljem ili kontrolnim zbrojem
1. Broj paketa sa neispravnim ICMP zaglavljem ili kontrolnim zbrojem
1. Broj paketa sa isteklim TTL poljem
1. Broj paketa duljih od 1500 bajtova

### Rješenje primjera 4

Novi elementi koje ćemo koristiti su `Counter` ([dokumentacija](https://github.com/kohler/click/wiki/Counter)), `Discard` ([dokumentacija](https://github.com/kohler/click/wiki/Discard)) i `DriverManager` ([dokumentacija](https://github.com/kohler/click/wiki/DriverManager)).

``` c
FromDump(f4a.pcap, STOP true)
   -> cip :: CheckIPHeader
   -> i1 :: IPClassifier(tcp, udp, icmp, -)
   -> ctcp :: CheckTCPHeader
   -> ttl :: IPClassifier(ttl 0, -) [1]
   -> cl :: CheckLength(1500)
   -> ip :: IPClassifier(dst 131.179.0.0/16, dst 131.0.0.0/8, dst 18.0.0.0/8, -);

i1[1] -> cudp :: CheckUDPHeader -> ttl;
i1[2] -> cicmp :: CheckICMPHeader -> ttl;
i1[3] -> ttl;

ttl[0] -> cttl :: Counter
   -> ICMPError(18.26.7.3, timeexceeded, transit)
   -> ToDump(f4f.pcap, ENCAP IP);

ip[0] -> ToDump(f4c.pcap,  ENCAP IP);
ip[1] -> ToDump(f4b.pcap,  ENCAP IP);
ip[2] -> ToDump(f4d.pcap,  ENCAP IP);
ip[3] -> ToDump(f4e.pcap,  ENCAP IP);

cl[1] -> ccl :: Counter -> Discard;

DriverManager(pause, print >f4drops.txt cip.drops,
   print >>f4drops.txt ctcp.drops,
   print >>f4drops.txt cudp.drops,
   print >>f4drops.txt cicmp.drops,
   print >>f4drops.txt cttl.count,
   print >>f4drops.txt ccl.count);
```

### Primjer 5

Konfigurirajte usmjerivač koji će raditi na isti način kao usmjerivač u primjeru 3, no umjesto da pakete bilježite u datoteke jednostavno ih odbacite. Definirajte element koji će izvršavati provjeru (umjesto simulacije Click usmjerivača koji vrši provjeru).

### Rješenje primjera 5

``` c
elementclass ErrorChecker {
   input -> cip :: CheckIPHeader
      -> i1 :: IPClassifier(tcp, udp, icmp, -)
      -> ctcp :: CheckTCPHeader
      -> ttl :: IPFilter(drop ttl 0, allow all)
      -> cl :: CheckLength(1500)
      -> output;
   i1[1] -> cudp :: CheckUDPHeader -> ttl
   i1[2] -> cicmp :: CheckICMPHeader -> ttl
   i1[3] -> ttl
}
```

Definirani element `ErrorChecker` sada ima jedan ulaz i jedan izlaz te ga možemo koristiti kao i bilo koji drugi element, primjerice:

``` c
FromDump(f5a.pcap, STOP true) -> ErrorChecker -> ToDump(f5b.pcap, ENCAP IP);
```
