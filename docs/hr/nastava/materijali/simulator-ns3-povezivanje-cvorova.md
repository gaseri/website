---
author: Ivan Ivakić, Vedran Miletić
---

# Povezivanje čvorova u simulaciji računalnih mreža

## Jednostavna simulacija s izravno povezanim čvorovima

U ovom primjeru stvorit ćemo dva čvora; instalirat ćemo komunikacijske uređaje i podršku za potrebne protokole i uspostaviti komunikaciju korištenjem UDP-a između klijentske aplikacije na prvom čvoru i poslužiteljske na drugom.

Na početku programa uključujemo zaglavlja pojedinih modula simulatora ns-3 naredbom `#include<>`. Svi simboli koje ns-3 koristi su unutar imenika `ns3`; zbog toga je potrebno uključiti korištenje tog imenika kako bi pojednostavili pisanje koda (u protivnom bilo bi potrebno pisati, primjerice, `ns3::NodeContainer` umjesto `NodeContainer` i `ns3::PointToPointHelper` umjesto `PointToPointHelper`).

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>
```

using namespace ns3;

Cjelokupan program nalazi se unutar funkcije `main()`, što je i realno za očekivati zbog njegove jednostavnosti. Zanemarimo za sada linije

``` c++
LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
```

koje se tiču logginga. Za sada je dovoljno znati da su one potrebne da bi simulacija dala ikakav izlaz na ekran. Na počeku simulacije stvaramo objekt `nodes` koji je instanca klase `NodeContainer` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d7/db2/classns3_1_1_node_container.html)). Zatim u tom objektu stvaramo dva čvora metodom `Create()`.

``` c++
NodeContainer nodes;
nodes.Create (2);
```

`PointToPointHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d3/d84/classns3_1_1_point_to_point_helper.html)) je pomoćnik (ili čarobnjak kako možemo vidjeti u mnogim programima) koji stvara sve što je potrebno za rad veze tipa točka-do-točke između dva čvora (stvara mreže uređaje, dodaje ih čvorovima i povezuje ih kanalom). Stvaramo instancu te klase i nazivamo ju pointToPoint, a zatim metodom `SetDeviceAttribute()` postavljamo atribute budućih uređaja. Vrijednosti parametara su tipa znakovni niz i zapravo eksplicitno govore o čemu se radi: konkretno, atribut `DataRate` je brzina prijenosa podataka i iznosi `"5Mbps"`.

Metoda `SetChannelAttribute()` postavlja atribute same veze. `Delay` je zadržavanje veze i ono iznosi `"2ms"`. Funkciju StringValue koristimo da bismo pretvorili eventualni varijable tipa `const char *` (podaci `"5Mbps"` i `"2ms"` su tog tipa, kao i svi ostali znakovni nizovi navedeni u izvornom kodu) u tip `StringValue` koji metode SetDeviceAttribute i SetChannelAttribute primaju kao drugi parametar.

``` c++
PointToPointHelper pointToPoint;
pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
```

`NetDeviceContainer` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d6/ddf/classns3_1_1_net_device_container.html)) je klasa koja služi kao kontejner za mrežne uređaja. Kreiramo instancu `devices`, a zatim metodom `Install()` na objekt `nodes` instaliramo uređaje na čvorove i istovremeno ih pohranjujemo u kontejner da lakše s njima baratamo. (Uočite da je u dokumentaciji navedena metoda `Get()` koja nam olakšava dohvaćanje elementa kontejnera.)

``` c++
NetDeviceContainer devices;
devices = pointToPoint.Install (nodes);
```

`InternetStackHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/db/df3/classns3_1_1_internet_stack_helper.html)) je još jedan u nizu pomoćnika koje koristimo. Metodom `Install()` na objekt `nodes` instaliramo TCP/UDP i IP funkcionalnost na čvorove koji su unutar tog kontejnera.

``` c++
InternetStackHelper stack;
stack.Install (nodes);
```

`Ipv4AddressHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d5/d4f/classns3_1_1_ipv4_address_helper.html)) je pomoćnik koji dodjeljuje IPv4 adrese mrežnim uređajima (podsjetite se zašto su IPv4 adrese vezane za mrežne uređaje, a ne za čvorove). `SetBase()` je metoda koja prima dva parametra: prvi parametar je adresa, a drugi maska podmreže.

``` c++
Ipv4AddressHelper address;
address.SetBase ("10.1.1.0", "255.255.255.0");
```

`Ipv4InterfaceContainer` ([dokumentacija](https://www.nsnam.org/docs/doxygen/db/d86/classns3_1_1_ipv4_interface_container.html)) je kontejner za IPv4 sučelja na mrežnim uređajima. Metoda `Assign()` klase `Ipv4AddressHelper` dodjeljuje adrese uređajima unutar kontejnera `devices`.

``` c++
Ipv4InterfaceContainer interfaces;
interfaces = address.Assign (devices);
```

`UdpEchoServerHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d6/d64/classns3_1_1_udp_echo_server_helper.html)) je pomoćnik. Kod stvaranja instance `echoServer` prosljeđuje se jedan parametar; iz dokumentacije se može iščitati da se radi o broju vrata (u ovom slučaju 9) na kojima će komunikacija biti omogućena.

``` c++
UdpEchoServerHelper echoServer (9);
```

Zatim kreiramo serversku aplikaciju. `ApplicationContainer` je još jedan u nizu kontejnera u koje spremamo određene elemente simulacije; instancu nazivamo `serverApps`. Metoda `Install()` dobiva kao parametar objekt tipa `Node`, jer je to povratni tip metode `Get()` objekta `nodes`. Pritom pripazimo na indeksiranje; radi se o *drugom* elementu, obzirom da brojanje počinje od 0.

`Start()` i `Stop()` su metode koje zakazuju vrijeme pokretanja i zaustavljanja rada serverske aplikacije, a kao parametre primaju sekunde. Ono na što treba obratiti pozornost je da se serverska aplikacija pokreće prije i gasi nakon pokretanja klijentske aplikacije, osim u slučajevima gdje baš želimo testirati specifično ponašanje.

``` c++
ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
serverApps.Start (Seconds (1.0));
serverApps.Stop (Seconds (10.0));
```

`UdpEchoClientHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d5/d2f/classns3_1_1_udp_echo_client_helper.html)) imena `echoClient` koristimo na sličan način za kreiranje klijentske aplikacije. Parametri koje prima kod stvaranja su adresa i vrata poslužiteljske aplikacije. Ovdje je to IPv4 sučelje drugog čvora (`interfaces.GetAddress(1)`) i vrata 9. Zatim postavljamo razne atribute klijentske aplikacije.

- `MaxPackets` je broj paketa koje šalje klijentska aplikacija i očekuje vrijednost tipa `UintegerValue` (vrijednost tipa nepredznačeni cijeli broj) te je potrebno napraviti pretvorbu.
- `Interval` je razmak između slanja svakog pojedinog paketa, zapravo frekvencija slanja, vrijednost tipa `TimeValue`. Ovdje je vrijednost postavljena na 1 sekundu što znači da će se paketi slati u pravilnom vremenskom intervalu od jedne sekunde.
- `PacketSize` je veličina paketa, također vrijednost tipa `UintegerValue`.

``` c++
UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
echoClient.SetAttribute ("PacketSize", UintegerValue (1024));
```

Alternativno, mogli smo IPv4 adresu poslužitelja navesti i eksplicitno.

``` c++
UdpEchoClientHelper echoClient (Ipv4Address ("10.1.1.2"), 9);
```

Kao i serversku aplikaciju i klijentsku instaliramo na određeni čvor. Ovdje je to čvor sa indeksom `0` odnosno prvi čvor unutar kontejnera `nodes`. Za klijentsku aplikaciju je potrebno zakazati pokretanje i zaustavljanje u određenim trenutcima metodama `Start()` i `Stop()`.

``` c++
ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
clientApps.Start (Seconds (2.0));
clientApps.Stop (Seconds (10.0));
```

Na kraju ne preostaje ništa drugo nego pokrenuti simulator sa metodom `Run()`.

``` c++
Simulator::Run ();
```

Uočite da se ova metoda pokreće na drugačiji način, odnosno da nema instancu klase. Radi se o takozvanoj *statičkoj klasi* koja se ne može instancirati; time se nećemo ovdje nećemo više baviti (uzimati ćemo da to jednostavno radi tako), obzirom da se o tome više govori na kolegiju Objektno orijentirano programiranje.

Nakon što je simulator završio sa radom valja počistiti memoriju, a to činimo metodom `Destroy()`. Dvije potonje linije koda zapravo slijede jedna drugu iz razloga što u samom simulatoru određujemo vrijeme trajanja simulacije definicijom topologije i zakazivanjem događaja. Simulacija se u cijelosti izvršava pozivom metode `Run()` pa tek nakon završetka izvođenje programa dolazi do linije u kojoj se poziva metoda `Destroy()`.

``` c++
Simulator::Destroy ();
```

Cjelokupni kod primjera je

``` c++
/*
 * UdpEchoClient                            UdpEchoServer
 *            10.1.1.0               10.1.2.0
 *      n1 ------------------n2------------------n3
 *         point-to-point       point-to-point
 */

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

int main ()
{
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer allNodes, nodes12, nodes23;
  allNodes.Create (3);
  nodes12.Add (allNodes.Get (0));
  nodes12.Add (allNodes.Get (1));
  nodes23.Add (allNodes.Get (1));
  nodes23.Add (allNodes.Get (2));

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices12, devices23;
  devices12 = pointToPoint.Install (nodes12);
  devices23 = pointToPoint.Install (nodes23);
  // pointToPoint.EnablePcapAll ("vjezba-udp-echo-neizravna-veza");

  InternetStackHelper stack;
  stack.Install (allNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (allNodes.Get (2));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (interfaces23.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (allNodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Praćenje i analiza simuliranog mrežnog prometa

Sve dosad opisano je dovoljno da bi aplikacija radila. Međutim, iskoristit ćemo još jednu metodu objekta tipa `PointToPointHelper`, a to je `EnablePcapAll()` koja čini da se stvaraju snimke mrežnog prometa (engl. *packet capture*, kraće [pcap](https://en.wikipedia.org/wiki/Pcap)), odnosno datoteke koje sadrže simulirane pakete. Ta metoda prima niz znakova koji će biti prefiks imena datoteka koje će stvoriti.

Važno je da se ova metoda pozove **nakon** svih poziva `Install()` metode nad objektom `pointToPoint`, što je najlakše napraviti tako da pozovete metodu neposredno prije samog pokretanja simulacije sa `Simulator::Run()`. Time se izbjegava mogućnost nepravovremenog poziva metode koja u tom slučaju neće generirati ništa ili će pak generirati pogrešan izvještaj.

Datoteka koju generira ova metoda bit će smještena u direktoriju gdje se događa kompajliranje vašeg programskog koda. U slučaju da ste kao odredište spremanja datoteka projekta odabrali vaš kućni direktorij (u virtualnoj mašini to je `/home/student`), a projekt nazvali ga `rm2-vj1-primjer1i`, direktorij u kojem se nalazi pcap datoteke zove se `rm2-vj1-primjer1i-build-<verzija Qt biblioteke i program prevoditelja>`. Taj direktorij, sadrži izvršnu datoteku vašeg projekta, datoteke s objektnim kodom (`*.o`) i pcap datoteke koje imaju prefiks u imenu koji ste zadali. Naravno, pretpostavka je da ste prethodno pokrenuli izvršnu datoteku koja će stvoriti pcap datoteke (upravo zbog toga što ih stvara izvršna datoteka simulacije pcap datoteke se i nalaze u tom direktoriju).

Imena pcap datoteka generirana su prema uzorku `<odabrani niz znakova>-X-Y.pcap`, pri čemu je `X` je identifikator čvora, a `Y` identifikator mrežnog sučelja na čvoru. Sadržaj pcap datoteka moguće je pregledati korištenjem [Wiresharka](https://en.wikipedia.org/wiki/Wireshark) ili [tcpdumpa](https://en.wikipedia.org/wiki/Tcpdump).

``` c++
pointToPoint.EnablePcapAll ("vjezba-udp-echo");
```

S ciljem optimizacije izvođenja simulacije, kontrolni zbroj paketa se u zadanim postavkama ne računa i postavlja se uvijek na vrijednost `0`. Posljedica toga je da su svi paketi u Wiresharku označeni crvenom bojom, odnosno označeni su kao iskrivljeni. U slučaju da želite pregledavati pakete, uključite računanje kontrolnog zbroja funkcijom

``` c++
Config::SetGlobal ("ChecksumEnabled", BooleanValue (true));
```

koju je potrebno pokrenuti **prije stvaranja objekata simulacije**. Iako to ne pravi razliku, radi konzistentnosti uzet ćemo dogovorno da se funkcije iz `Config` imenika pokreću prije funkcija `LogComponentEnable()`, i to prvo funkcije `Config::SetGlobal()`, a zatim `Config::SetDefault()`.

Cjelokupni kod primjera je

``` c++
/*
 * UdpEchoClient     UdpEchoServer
 *   10.1.1.1          10.1.1.2
 *      n1 -------------- n2
 *         point-to-point
 */

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

int main ()
{
  Config::SetGlobal ("ChecksumEnabled", BooleanValue (true));

  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);
  pointToPoint.EnablePcapAll ("vjezba-udp-echo");

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces;
  interfaces = address.Assign (devices);

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Neizravno povezivanje čvorova

Želimo stvoriti linearnu topologiju od tri čvora oblika

```
n1 ------ n2 ------ n3
```

pri čemu su, radi jednostavnosti, postavke obaju veza jednake. Uočimo da ovdje postoje dvije podmreže, prva koja sadrži čvorove `n1` i `n2`, i druga koja sadrži `n2` i `n3`. Kako je komunikacija između različitih podmreža neizravna, potrebno će biti uključiti usmjeravanje da bi `n1` mogao komunicirati sa `n3`.

Instancirajmo tri objekta klase `NodeContainer`; to su redom `allNodes`, `nodes12` i `nodes23`. Ideja je prva dva čvora iz kontejnera svih čvorova pospremiti u prvi kontejner, a posljednja dva čvora u drugi kontejner. To će nam olakšati instalaciju mrežnih u uređaja i dodjelu IPv4 adresa.

Metodom `Create()` stvaramo 3 nova čvora samo u kontejneru `allNodes`. Metoda Add() koju nismo do sada koristi ima za cilj dodati neki od prethodno stvorenih čvorova; metodom `Get()` dohvaćamo određeni čvor iz `NodeContainer`-a, u ovom slučaju prvi i treći jednom, a drugi dvaput. Nemojmo zaboraviti da `NodeContainer` indeksira kao polja u C/C++-u, što konkretno znači: prvi čvor svakog `NodeContainer`-a ima indeks 0, drugi čvor ima indeks 1, treći indeks 2.

``` c++
NodeContainer allNodes, nodes12, nodes23;
allNodes.Create (3);
nodes12.Add (allNodes.Get (0));
nodes12.Add (allNodes.Get (1));
nodes23.Add (allNodes.Get (1));
nodes23.Add (allNodes.Get (2));
```

Pomoćnik za stvaranje veza tipa točka-do-točke možemo iskoristiti dvaput, obzirom da obje veze imaju ista svojstva. Uočite da sada postoje četiri mrežna uređaja, od kojih prvi i posljednji čvor imaju po jedan, a srednji čvor dva.

``` c++
PointToPointHelper pointToPoint;
pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

NetDeviceContainer devices12, devices23;
devices12 = pointToPoint.Install (nodes12);
devices23 = pointToPoint.Install (nodes23);
```

Kod instalacije složaja internetskih protokola treba paziti da ne instaliramo prokole na neki čvor više od jednom; zbog toga je najpraktičnije pokrenuti instalaciju na kontejner `allNodes`.

``` c++
InternetStackHelper stack;
stack.Install (allNodes);
```

Alternativno, mogli smo to napraviti na način da prvo instaliramo na prvi i drugi čvor, a zatim na treći.

``` c++
InternetStackHelper stack;
stack.Install (nodes12);
stack.Install (nodes23.Get (1));
```

Dodjela IPv4 adresa radi se analogno prvom primjeru, osim što treba paziti da se dodijele različiti rasponi dvjema podmrežama kako bi algoritmi usmjeravanja mogli raditi. Time je omogućena komunikacija prvog i drugog čvora, te drugog i trećeg. Komunikacija prvog i trećeg čvora preko drugog čvora još nije omogućena.

``` c++
Ipv4AddressHelper address;
address.SetBase ("10.1.1.0", "255.255.255.0");
Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

address.SetBase ("10.1.2.0", "255.255.255.0");
Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);
```

Obzirom da imamo više od jedne podmreže koje su vezane neizravno, potrebno je ispuniti tablice usmjeravanja čvorova. Statička funkcija `PopulateRoutingTables()` iz klase `Ipv4GlobalRoutingHelper` puni tablice usmjeravanja svih čvorova u simulaciji. Ovime je omogućena komunikacija prvog i trećeg čvora preko drugog čvora.

``` c++
Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
```

Instalacija aplikacija vrši se na isti način kao u prvom primjeru, jedino je potrebno pripaziti na adresu poslužitelja kod stvaranja klijenta i identifikatore čvorova na koje ćemo instalirati aplikacije.

``` c++
UdpEchoServerHelper echoServer (9);

ApplicationContainer serverApps = echoServer.Install (allNodes.Get (2));
serverApps.Start (Seconds (1.0));
serverApps.Stop (Seconds (10.0));
```

Ovim postavkama dobiti ćemo klijentsku na prvom čvoru koja komunicira s poslužiteljskom na trećem čvoru.

``` c++
UdpEchoClientHelper echoClient (interfaces23.GetAddress (1), 9);
echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

ApplicationContainer clientApps = echoClient.Install (allNodes.Get (0));
clientApps.Start (Seconds (2.0));
clientApps.Stop (Seconds (10.0));
```

Cjelokupni kod primjera je

``` c++
/*
 * UdpEchoClient                            UdpEchoServer
 *            10.1.1.0               10.1.2.0
 *      n1 ------------------n2------------------n3
 *         point-to-point       point-to-point
 */

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

int main ()
{
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer allNodes, nodes12, nodes23;
  allNodes.Create (3);
  nodes12.Add (allNodes.Get (0));
  nodes12.Add (allNodes.Get (1));
  nodes23.Add (allNodes.Get (1));
  nodes23.Add (allNodes.Get (2));

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices12, devices23;
  devices12 = pointToPoint.Install (nodes12);
  devices23 = pointToPoint.Install (nodes23);
  // pointToPoint.EnablePcapAll ("vjezba-udp-echo-neizravna-veza");

  InternetStackHelper stack;
  stack.Install (allNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (allNodes.Get (2));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (interfaces23.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (allNodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Dodatak: optimizirano usmjeravanje

Nix-Vector je protokol usmjeravanja specifičan za simulaciju i namijenjen za veće mrežne topologije. On radi na zahtjev, odnosno ne računa tablice usmjeravanja unaprijed, te ima manje zauzeće memorije i time bolje performanse (odnosno kraće vrijeme izvođenja simulacije) u odnosu na GlobalRouting, što se naročito dobro vidi kod simulacija koje imaju nekoliko stotina ili čak tisuća čvorova. Nix-Vector usmjeravanje rute računa korištenjem breadth-first pretraživanja, a izračunate rute se spremaju u efikasnu strukturu koja se zove nix-vector.

!!! tip
    Da bi vidjeli kako usmjeravanje može imati problema sa skaliranjem na veliki broj čvorova, razmotrimo sljedeće. Algoritam usmjeravanja koji GlobalRouting koristi radi tako da svaki čvor pamti izlazni mrežni uređaj na koji mora poslati paket za svaku od IPv4 adresa koje postoje u simulaciji.

    Simulacija koja ima `n` čvorova s po 2 adrese, ima ukupno `2 * n` adresa. Kada se koristi `GlobalRouting`, to znači da tablica usmjeravanja *na svakom čvoru* ima `2 * n` unosa, odnosno da ukupno ima `2 * n * n` unosa u tablicama usmjeravanja u simulaciji.

    Za simulacije koje imaju relativno mali broj čvorova (relativno mali `n`) i broj `2 * n * n` nije velik, pa je usmjeravanje relativno brzo. Međutim, za simulacije koje imaju nekoliko tisuća čvorova `2 * n * n` je reda veličine nekoliko milijuna, što je mnogo.

    Kako će sve naše simulacije biti relativno malene, za nas ovo neće predstavljati problem.

Kada se paket stvori na čvoru i spreman je za slanje, ruta se računa i gradi se nix-vector; nix-vector sprema indeks susjeda za svaki skok po putu. Taj se indeks koristi da se odredi koji će se mrežni uređaj koristiti za slanje podataka. Da bi mogao usmjeriti paket, nix-vector mora biti poslan zajedno s paketom. Na svakom skoku, trenutni čvor uzima prikladni indeks susjeda iz nix-vectora i šalje paket kroz odgovarajući mrežni uređaj. To se nastavlja sve dok paket ne dođe do odredišta.

Trenutna implementacija u simulatoru ns-3 podržava IPv4 na point-to-point i CSMA vezama. IPv6 za sada nije podržan. Pored toga, u slučaju kvara neke od veza, vrši se pražnjenje postojećih tablica i algoritam kreće ispočetka.

Kreiranje čvorova i kontejnera koji ih sadrže jednako je kao u prethodnim primjerima. Ono što je potrebno napraviti da bi `nix-vector-routing` protokol radio ispravno je nakon kreiranja svih čvorova smjestiti ih u jedan kontejner. Prvo uključujemo zaglavlje `nix-vector-routing-module.h` u našu simulaciju.

``` c++
#include <ns3/nix-vector-routing-module.h>
```

Zatim sve čvorove smještamo u jedan kontejner za čvorove kao i ranije.

``` c++
NodeContainer allNodes;
allNodes.Create (4);
NodeContainer nodes01 = NodeContainer(nodes01.Get (0), nodes01.Get (1));
NodeContainer nodes12 = NodeContainer(nodes12.Get (0), nodes12.Get (1));
NodeContainer nodes23 = NodeContainer(nodes23.Get (0), nodes23.Get (1));
```

Nakon kreiranja kontejnera incijaliziramo pomoćnika za Nix-Vector, a zatim u popis protokola usmjeravanja dodajemo pomoćnika za Nix-Vector protokol. Zbog trenutne implementacije potrebno je uključiti i statičko usmjeravanje ranije korištenim pomoćnikom `Ipv4StaticRoutingHelper`. Za to kreiramo varijablu `list` koja će pohraniti popis protokola usmjeravanja, i metodom `Add` dodajemo u popis protokole usmjeravanja:

- prvi parametar je pomoćnik protokola usmjeravanja,
- drugi parametar je prioritet protokola usmjeravanja i određuje koji će protokol biti preferiran nad kojim u slučaju da se oba mogu koristiti, i tu što je veći broj to je veći prioritet. Ustaljena praksa je dodijeliti statičkom usmjeravanju prioritet 0, a nix-vector protokolu 10.

``` c++
Ipv4NixVectorHelper nixHelper;
Ipv4StaticRoutingHelper staticRouting;

Ipv4ListRoutingHelper list;
list.Add (staticRouting, 0);
list.Add (nixHelper, 10);
```

Potom postavljamo stog kao u prethodnim primjerima uz izmjenu da moramo specifično odrediti popis protokola usmjeravanja koje koristimo. To činimo metodom `SetRoutingHelper` koja prima za parametar popis koji je prethodno kreiran i pohranjen u varijabli `list`.

``` c++
InternetStackHelper stack;
stack.SetRoutingHelper (list);
stack.Install (allNodes);
```

Ostale postavke simulacije ostaju nepromijenjene.
