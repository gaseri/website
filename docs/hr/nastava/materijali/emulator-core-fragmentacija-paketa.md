---
author: Vedran Miletić
---

# Fragmentacija IPv4 paketa

Mrežne veze na kojima radi IPv4 imaju definiranu maksimalnu prijenosnu jedinicu (engl. *maximum transmission unit*, kraće MTU) kao najveću veličinu podatkovne jedinice koju mogu poslati odjednom. Podaci koji se šalju na mrežnom sloju bit će kod predaje veznom sloju podijeljeni u jedinice odgovarajuće veličine.

Primjerice, MTU na Ethernet mrežama iznosi 1500 (bajta) u koje se stavljaju IP zaglavlje, TCP/UDP zaglavlje i podaci mrežne aplikacije. Ako imamo za poslati 3400 B podataka aplikacije, 20 B TCP zaglavlja i 20 B IP zaglavlja, Ethernetom će to ići u tri IP paketa redom:

- 20 B IP zaglavlja, 20 B TCP zaglavlja, 1460 B podataka aplikacije
- 20 B IP zaglavlja, 1480 B podataka aplikacije (pomak 1480 / 8 = 185)
- 20 B IP zaglavlja, 460 B podataka aplikacije (pomak (1480 + 1480) / 8 = 370)

Uočimo kako je nužno da svaki paket ima IP zaglavlje kako bi mogao proći internetom. Nadalje, zbog potrebe za računanjem pomaka (koji se u IPv4 zaglavlju izražava u 8-bajtnim jedinicama) veličine podatkovnog dijela paketa na mrežnom sloju (dakle, TCP/UDP zaglavlja i podataka aplikacije) moraju biti djeljive s 8.

U slučaju kad ovi IP paketi dođu na iduću mrežu koja ima manji MTU, primjerice 980 (bajta), prva dva će se dodatno fragmentirati, a treći će proći mrežom. Time ćemo imati pet IP paketa redom:

- 20 B IP zaglavlja, 20 B TCP zaglavlja, 940 B podataka aplikacije
- 20 B IP zaglavlja, 520 podataka aplikacije (pomak 960 / 8 = 120)
- 20 B IP zaglavlja, 960 B podataka aplikacije (pomak (960 + 520) / 8 = 185)
- 20 B IP zaglavlja, 520 podataka aplikacije (pomak (960 + 520 + 960) / 8 = 305)
- 20 B IP zaglavlja, 460 B podataka aplikacije (pomak (960 + 520 + 960 + 520) / 8 = 370)

Sastavljanje će izvesti domaćin na drugom kraju mreže.

## Postavljanje MTU-a mrežnog adaptera

Na stvarnim i emuliranim čvorovima MTU mrežnog adaptera možemo provjeriti i postaviti već ranije spomenutim alatom `ifconfig`. Primjerice, ako se na čvoru na kojem radimo, mrežni adapter zove `em0`, MTU ćemo očitati naredbom

``` shell
# ifconfig em0
em0: flags=8843<UP,BROADCAST,RUNNING,SIMPLEX,MULTICAST> metric 0 mtu 1500
        options=81249b<RXCSUM,TXCSUM,VLAN_MTU,VLAN_HWTAGGING,VLAN_HWCSUM,LRO,WOL_MAGIC,VLAN_HWFILTER>
        ether 64:00:6a:46:1f:4a
        inet 172.16.46.108 netmask 0xffffff00 broadcast 172.16.46.255
        media: Ethernet autoselect (1000baseT <full-duplex>)
        status: active
        nd6 options=29<PERFORMNUD,IFDISABLED,AUTO_LINKLOCAL>
```

Uočimo da MTU iznosi 1500. Postavljanje MTU-a na neku drugu vrijednost, primjerice 1200, izvodimo naredbom

``` shell
# ifconfig em0 mtu 1200
```

nakon čega možemo ponovno pokrenuti `ifconfig` i uvjeriti se da je postavljanje uspjelo. U primjeru radimo na stvarnom čvoru, ali postupak je identičan kada se radi o mrežnim adapterima na emuliranom čvoru.

## Stvaranje prometa alatom MGEN

Alat MGEN omogućuje nam da koristimo različite uzorke prema kojima će stvarati promet (konkretno, `PERIODIC`, `POISSON`, `BURST` i `JITTER`). Paketi koji se šalju variraju u terminima učestalosti slanja i veličine paketa. Mi ćemo se ovdje ograničiti na korištenje uzorka `PERIODIC`.

Uzorak `PERIODIC` očekuje parametre `rate` (učestalost slanja paketa u terminima broja poslanih paketa po sekundi) i `size` (veličina paketa u bajtovima). Učestalost slanja mora biti strogo veća od 0.0 paketa po sekundi. Ograničenje veličine paketa koja se može postaviti ne postoji kada se koristi TCP, a kada se koristi UDP ono iznosi 8192 bajta.

Primjerice, naredba `PERIODIC [10.0 1024]` slati će 10 paketa veličine 1024 bajta po sekundi, a naredba `PERIODIC [0.2 3400]` poslat će po jedan paket veličine 3400 bajta svakih 5 sekundi.

!!! note
    Kod odabira predloška prometa unutar CORE-a se zapravo samo koriste različite predefinirane naredbe MGEN-a i zatim je moguće te naredbe dodatno prilagoditi svojim potrebama.
