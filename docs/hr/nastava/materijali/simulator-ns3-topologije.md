---
author: Ivan Ivakić, Vedran Miletić
---

# Slaganje složenijih topologija u simulaciji računalnih mreža

## Topologija zvijezde i proširene zvijezde

Topologije mreža koje srećemo u praksi najčešće nisu jednostavne kao ovdje navedene na kojima učimo načela i metode simulacije. Međutim, ns-3 je vrlo fleksibilan, odnosno ne ograničava korisnika u slaganju topologije, i skalabilan, odnosno omogućuje slaganje simulacija reda veličine tisuću čvorova i više.

Najjednostavnija topologija zvijezde koja nije istovremeno i linearna je oblika "zvijezde s tri kraka", odnosno

```
           n3
           /
          /
n1 ----- n2
          \
           \
           n4
```

i u ns-3-u se slaže istom logikom kao i linearna topologija: prvo stvorimo kontejner `allNodes` sa 4 čvora. Zatim u kontejner `nodes12` stavljamo čvorove `n1` i `n2`, u kontejner `nodes23` čvorove `n2` i `n3`, i naposlijetku u `nodes24` čvorove `n2` i `n4`. Sve čvorove pritom dohvaćamo iz `allNodes`.

``` c++
NodeContainer allNodes;
allNodes.Create (4);
NodeContainer nodes12 (allNodes.Get (0), allNodes.Get (1));
NodeContainer nodes23 (allNodes.Get (1), allNodes.Get (2));
NodeContainer nodes24 (allNodes.Get (1), allNodes.Get (3));
```

Nešto složeniji primjer bila bi proširena zvijezda oblika

```
                    n4 ----- n5
                    /
                   /
n1 ----- n2 ----- n3
                   \
                    \
                    n6 ----- n7
```

i u ns-3-u je za nju potrebno imati šest kontejnera (obzirom da nam kontejner služi za jednostavniju instalaciju point-to-point veze među čvorovima). Dobivamo je analogno prethodnom primjeru, s time da imamo

- u kontejneru `nodes12` čvorove `n1` i `n2`,
- u kontejneru `nodes23` čvorove `n2` i `n3`,
- u kontejneru `nodes34` čvorove `n3` i `n4`,
- u kontejneru `nodes45` čvorove `n4` i `n5`,
- u kontejneru `nodes36` čvorove `n3` i `n6`,
- u kontejneru `nodes67` čvorove `n6` i `n7`.

``` c++
NodeContainer allNodes;
allNodes.Create (7);
NodeContainer nodes12 (allNodes.Get (0), allNodes.Get (1));
NodeContainer nodes23 (allNodes.Get (1), allNodes.Get (2));
NodeContainer nodes34 (allNodes.Get (2), allNodes.Get (3));
NodeContainer nodes45 (allNodes.Get (3), allNodes.Get (4));
NodeContainer nodes36 (allNodes.Get (2), allNodes.Get (5));
NodeContainer nodes67 (allNodes.Get (5), allNodes.Get (6));
```

Još jedan vrlo zanimlji primjer topologije zvijezde je

```
    n4  n3
     \  /
      \/
n5 -- n1 -- n2
      /\
     /  \
    n6  n7
```

U programskom kodu to je

``` c++
NodeContainer allNodes;
allNodes.Create (7);
NodeContainer nodes12 (allNodes.Get (0), allNodes.Get (1));
NodeContainer nodes13 (allNodes.Get (0), allNodes.Get (2));
NodeContainer nodes14 (allNodes.Get (0), allNodes.Get (3));
NodeContainer nodes15 (allNodes.Get (0), allNodes.Get (4));
NodeContainer nodes16 (allNodes.Get (0), allNodes.Get (5));
NodeContainer nodes17 (allNodes.Get (0), allNodes.Get (6));
```

Instalacija point-to-point veza, postavljanje IP adresa i instalacija aplikacija ide potpuno analogno kao u prethodnom primjeru te je ostavljena kao vježba za čitatelja.

## Topologija bučice

Topologija bučice (engl. *dumbell*) je vrlo često korištena podvrsta zvjezdaste topologije, zbog čega ćemo je detaljnije analizirati. U općenitom slučaju ona se sastoji od dva usmjerivača povezana vezom tipa točka-do-točke na koje su zatim povezani domaćini. Najjednostavniji primjer topologije bučice je ona koja ima dva usmjerivača (označenih s `n2` i `n3`) s po dva domaćina.

```
n1 ----\              /---- n5
        \            /
        n3 -------- n4
        /            \
n2 ----/              \---- n6
```

``` c++
NodeContainer allNodes;
allNodes.Create (6);
NodeContainer nodes13 (allNodes.Get (0), allNodes.Get (2));
NodeContainer nodes23 (allNodes.Get (1), allNodes.Get (2));
NodeContainer nodes34 (allNodes.Get (2), allNodes.Get (3));
NodeContainer nodes45 (allNodes.Get (3), allNodes.Get (4));
NodeContainer nodes46 (allNodes.Get (3), allNodes.Get (5));

PointToPointHelper pointToPoint;
pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
pointToPoint.SetChannelAttribute ("Delay", StringValue ("50ms"));

NetDeviceContainer devices13, devices23, devices34, devices45, devices46;
devices13 = pointToPoint.Install (nodes13);
devices23 = pointToPoint.Install (nodes23);
devices45 = pointToPoint.Install (nodes45);
devices46 = pointToPoint.Install (nodes46);

pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("2Mbps"));
devices34 = pointToPoint.Install (nodes34);
```

Dodjela adresa vrši se na isti način kao i ranije.

``` c++
InternetStackHelper stack;
stack.Install (allNodes);

Ipv4InterfaceContainer interfaces13, interfaces23, interfaces34, interfaces45, interfaces46;

Ipv4AddressHelper address;
address.SetBase ("10.1.1.0", "255.255.255.0");
interfaces13 = address.Assign (devices13);

address.SetBase ("10.1.2.0", "255.255.255.0");
interfaces23 = address.Assign (devices23);

address.SetBase ("10.1.3.0", "255.255.255.0");
interfaces34 = address.Assign (devices34);

address.SetBase ("10.1.4.0", "255.255.255.0");
interfaces45 = address.Assign (devices45);

address.SetBase ("10.1.5.0", "255.255.255.0");
interfaces46 = address.Assign (devices46);
```

## Dodatak: slaganje složenijih topologija većih razmjera

Modul `point-to-point-layout` pomaže nam u stvaranju topologija sa više čvorova raspoređenih u određenom obliku. Ns-3 implementira tri takve topologije:

- Dumbbell (hrv. *bučica*) topologiju sa `PointToPointDumbbellHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d3/dd4/classns3_1_1_point_to_point_dumbbell_helper.html))
- Grid (hrv. *rešetka*) topologiju sa `PointToPointGridHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/dd/d58/classns3_1_1_point_to_point_grid_helper.html)) te
- Star (hrv. *zvijezda*) topologiju sa `PointToPointStarHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d3/d7b/classns3_1_1_point_to_point_star_helper.html))

Kao što smo već vidjeli, topologija bučice ima jednu središnju vezu, a na svakom od krajnjih čvorova određeni broj čvorova povezanih na krajnji čvor. Broj čvorova da desne i lijeve strane bučice ne mora biti isti.

```
    n4  n3       n8   n9
     \  /         \   /
      \/           \ /
n5 -- n1 ---------- n2 -- n10
      /\           / \
     /  \         /   \
    n6  n7      n11   n12
```

Kod toplogije rešetke susjedni čvorovi povezani su tako da tvore rešetkastu strukturu.

```
n11 -- n12 -- n13 -- n14
 |      |      |      |
 |      |      |      |
n21 -- n22 -- n23 -- n24
 |      |      |      |
 |      |      |      |
n31 -- n32 -- n33 -- n34
```

Topologija zvijezde stvara se na način koji smo već spomenuli; na jedan središnji čvor povežemo najmanje tri čvora tako da izgledom podsjećaju na zvijezdu.

```
    n4  n3
     \  /
      \/
n5 -- n1 -- n2
      /\
     /  \
    n6  n7
```

Zaglavlje u kojem se nalaze pomoćnici za kreiranje ovakvih složenih topologija je `point-to-point-layout-module.h` te ga je potrebno uključiti u naš programski kod linijom:

``` c++
#include <ns3/point-to-point-layout-module.h>
```

Nakon što smo to učiniti spremni smo za kreiranje bilo koje od tri prethodno navedene topologije.

### Složenija topologija bučice

Kreiranje topologije bučice radimo na način da odredimo broj čvorova sa lijeve i broj čvorova sa desne strane središnja dva čvora, a da bismo olakšali kasniju izmjenu samog broja čvorova spremimo vrijednosti u varijable.

``` c++
uint32_t nLeftLeaf = 3;
uint32_t nRightLeaf = 7;
```

Zatim kreiramo point-to-point veze već poznatim pomoćnikom `PointToPointHelper`.

``` c++
PointToPointHelper pointToPointRouter;
pointToPointRouter.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
pointToPointRouter.SetChannelAttribute ("Delay", StringValue ("1ms"));

PointToPointHelper pointToPointLeaf;
pointToPointLeaf.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
pointToPointLeaf.SetChannelAttribute ("Delay", StringValue ("2ms"));
```

Na kraju je samo preostalo pozvati pomoćnik `PointToPointDumbbellHelper` sa odgovarajućim parametrima te dodijeliti kreiranim čvorovima adrese. U dodjeljivanju adresa pomaže nam metoda `AssignIpv4Addresses` koja prima tri parametra, prvi parametar je adresni prostor lijevog dijela bučice, drugi desnog dijela, a treći središnjeg. Za kreiranje adresnog prostora koristimo već poznati pomoćnik `Ipv4AddressHelper`.

``` c++
PointToPointDumbbellHelper d (nLeftLeaf, pointToPointLeaf, nRightLeaf, pointToPointLeaf, pointToPointRouter);

InternetStackHelper stack;
d.InstallStack (stack);

d.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                       Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"),
                       Ipv4AddressHelper ("10.3.1.0", "255.255.255.0"));
```

Instalacija aplikacija vrši se kao i u prethodnim primjerima uz preporuku da se koriste petlje ukoliko želimo aplikacije na više čvorova.

### Topologija rešetke

Kod kreiranja topologije rešetke polazimo od određivanja broja redaka i broja stupaca (primjetite da se zapravo kreira matrica čvorova).

``` c++
uint32_t nRows = 4;
uint32_t nCols = 5;
```

Potom je potrebno kreirati pomoćnik za stvaranje point-to-point veza koji će biti između svaka dva čvora kao što smo kreirali i kod topologije bučice.

``` c++
PointToPointHelper pointToPointGrid;
pointToPointGrid.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
pointToPointGrid.SetChannelAttribute ("Delay", StringValue ("5ms"));
```

Sada pozivamo pomoćnik za kreiranje grid topologije. Nakon kreiranja same topologije dodjeljujemo joj adresni prostor na način da pozivamo metodu `AssignIpv4Addresses`. Prvi parametar te metode je adresni prostor redaka, a drugi adresni prostor stupaca.

``` c++
PointToPointGridHelper g (nRows, nCols, pointToPointGrid);

InternetStackHelper stack;
g.InstallStack (stack);

g.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                       Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"));
```

### Topologija zvijezde

Najjednostavnije kreiranje ima topologija zvijezde s obzirom da pomoćnik prima samo dva parametra, prvi je broj krakova koje kreirana zvijezda ima, a drugi naravno veza koja će biti između pojedinih čvorova. Jedino što moramo paziti da se kreiranjem `n` krakova zvijezde zapravo kreira `n + 1` čvorova ukupno u topologiji iz razloga što se središnji čvor nužno kreira. U primjeru kreiramo 6 krakova, dakle 7 čvorova ukupno.

``` c++
uint32_t numSpokes = 6;

PointToPointHelper pointToPointStar;
pointToPointStar.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
pointToPointStar.SetChannelAttribute ("Delay", StringValue ("1ms"));

PointToPointStarHelper s (numSpokes, pointToPointStar);

InternetStackHelper stack;
s.InstallStack (stack);

s.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"));
```

Ovaj smo primjer već stvorili i ručno, te je moguće usporediti programski kod obaju varijanti.
