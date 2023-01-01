---
author: Vedran Miletić, Ivan Ivakić
---

# Simulacijski modeli grešaka paketa

Obzirom da mreže nisu apsolutno pouzdani sustavi i da su ograničenog kapaciteta, moguće je da dođe do gubitaka određenih dijelova podataka koji se šalju. Tu se najčešće događaju dvije situacije:

- **gubitak paketa zbog grešaka**, odnosno jedan ili više bitova paketa su iskrivljeni i kod primanja mrežna kartica to prepoznaje (na temelju vrijednosti CRC-a) i odbacuje paket,
- **gubitak paketa zbog prepunjenog reda čekanja**, odnosno na nekom od usmjerivača je ukupan kapacitet reda čekanja prepunjen i neki od paketa je morao biti odbačen.

Gubitkom paketa zbog grešaka bavimo se u ovoj vježbi; u idućoj vježbi baviti ćemo se gubitkom zbog prepunjenog reda čekanja.

Model grešaka paketa (ns-3 klasa `ErrorModel`) opisuje način na koji će biti odlučeno koji će paketa u određenom uzorku biti označeni kao iskrivljeni te biti odbačeni prilikom primitka.

!!! note
    Implementacija grešaka paketa je donekle specifična. `ErrorModel` radi tako da paket označava kao iskrivljen umjesto da mijenja vrijednosti bitova, s ciljem povećanja efikasnosti i time brzine izvođenja simulacije.

    Kao što je već rečeno, također s ciljem povećanja efikasnosti, ns-3 prilikom simulacije u zadanim postavkama uopće ne računa CRC paketa, već se svim paketima CRC postavlja na vrijednost 0, a provjera ispravnosti se vrši na temelju `ErrorModel`-a. U slučaju da `ErrorModel` nije dio simulacije, svi se paketi smatraju ispravnima.

## Programska podloga

Da bi mogli razumijeti način korištenja modela grešaka (i kasnije redova čekanja), moramo prvo proširiti znanje o programiranju.

### Pametni pokazivači i predlošci

Prisjetimo se da u C/C++-u razlikujemo statičku i dinamičku alokaciju memorije. Kod statičke alokacije, tip `int` i klasu `Node` alocirali bi i koristili na način

``` c++
int broj = 5;
Node node0;
node0.GetDevice (0);
```

i ne bi bilo potrebno izvršavati brisanje jer to program odrađuje sam. Kod dinamičke alokacije i dealokacije objektima pristupamo pomoću pokazivača, što je za tip `int` i klasu `Node` oblika

``` c++
int *broj = new int (5);
Node *node0 = new Node ();
node0->GetDevice (0); // ekvivalentno (*node0).GetDevice (0);
delete broj;
delete node0;
```

Naravno, u C-u bi koristili `malloc()` i `free()` umjesto `new` i `delete` (respektivno).

Uočimo da metodi `GetDevice()` pristupamo na drugačiji način nego u statički alociranoj varijanti, obzirom da je `node0` pokazivač na objekt tipa `Node` te je prvo potrebno napraviti dereferenciranje. Operator `->` je samo kraći zapis za dereferenciranje pokazivača i pristup metodi objekta.

Uočimo također da klasa `Node` ne zahtijeva navođenje parametara kod stvaranja; općenito za klase to ne mora biti slučaj. Jedini problem ovog načina rada je dosta složeno baratanje objektima u situaciji kada imate više od jednog pokazivača na isti objekt; postavlja se pitanje kada pozvati `delete`.

Odgovor na to pitanje ns-3 nudi u vidu **pametnih pokazivača** (engl. *smart pointers*), koji su implementirani u klasi `Ptr`. Za `Node` je pametni pokazivač oblika:

``` c++
Ptr<Node> node0 = CreateObject<Node> ();
// metode objekta se koriste se na isti način kao kod i običnih pokazivača
node0->GetDevice (0);
// pokazivači se dereferenciraju na isti način kao i obični pokazivači
(*node0).GetId ();
```

U ovom slučaju funkcija `CreateObject<Node>()` preuzima ulogu naredbe `new`; naime, ona će stvoriti objekt tipa `Node` i vratiti pokazivač na njega tipa `Ptr<Node>` koji će biti pohranjen u varijablu `node0`. S tim na umu, uočite kako ovdje nije napravljen `delete`; naime pametni pokazivač učinit će da program će u izvođenju sam izvodi dealokaciju memorije za objekte na koje više niti jedan pokazivač ne pokazuje (u tome je njegova "pametnost"). Način rada koji se ovime postiže naziva se sakupljanjem smeća (engl. *garbage collection*) i vrlo je čest u praksi.

Pored toga, uočimo još nešto: "špičaste" zagrade su oznaka za [predloške](https://en.wikipedia.org/wiki/Template_(C++)) (engl. *templates*). Mi se ovdje definiranjem funkcija s predlošcima nećemo baviti, obzirom da ćete se s njima sresti na kolegiju Objektno orijentirano programiranje, već ćemo samo ukratko objasniti ideju i primjene.

Kada bi implementirali pametne pokazivače bez korištenja predložaka, to bi zahtijevalo da za svaku klasu koja postoji u vašem kodu (npr. `Node`) imate definiranu dodatnu klasu (npr. `PtrNode`), što nije problem kada je tih klasa malo, ali je kada ih program ima nekoliko stotina. Naime, svaka promjena u načinu rada pokazivača zahtijeva nekoliko stotina promjena u kodu (koje su pored toga trivijalne i mukotrpne jer svi pametni pokazivači rade na istom načelu). Stoga se zajednički dio koda koji implementira načelo rada pametnog pokazivača apstrahira i definira tako da se može koristiti s bilo kojom klasom, strukturom ili tipom podataka, te se tek kod korištenja dobiva specijalizirani pokazivač na tu klasu, strukturu ili tip podataka.

Potpuno je analogna stvar sa funkcijama; za svaku definiranu klasu (npr. `Node`) bilo bi potrebno definirati posebnu funkciju (npr. `CreateNodeObject()`) koja bi stvarala objekt i vraćala pokazivač na njega. Ponovno, moguće je definirati funkciju koja implementira načelo rada neovisno o klasi, strukturi ili tipu podataka, a onda se kod korištenja dobiva specijalizirana funkcija za tu klasu, strukturu ili tip podataka.

!!! note
    Više informacija o ovim temama i poveznice na dodatnu literaturu možete naći Wikipedijinim stranicama o [metaprogramiranju korištenjem predložaka](https://en.wikipedia.org/wiki/Template_metaprogramming), [pametnim pokazivačima](https://en.wikipedia.org/wiki/Smart_pointer) i [sakupljanju smeća](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)).

### Povratni poziv funkcije

[Povratni poziv funkcije](https://en.wikipedia.org/wiki/Callback_(computer_programming)) (engl. *callback*) je referenca na funkciju koja se prosljeđuje funkciji kao argument na sličan način kao i obično korišteni tipovi podataka. Primjerice, u kodu

``` c++
#include <iostream>
#include <cstdlib>

void IspisDvaBroja (double (*izvorBrojeva) (int), int a, int b)
{
  std::cout << izvorBrojeva (a) << " i " << izvorBrojeva (b) << std::endl;
}

double prviIzvorBrojeva (int x)
{
  return 2.7 + x;
}

double drugiIzvorBrojeva (int x)
{
  return rand() % 10 + 0.5 * x;
}

int main()
{
  IspisDvaBroja (prviIzvorBrojeva, 4, 5);
  IspisDvaBroja (drugiIzvorBrojeva, 4, 5);
  return 0;
}
```

funkcija `IspisDvaBroja()` prima kao prvi argument fukciju koja prima jedan parametar tipa `int` i vraća rezultat tipa `double`. Preciznije rečeno, ona kao prvi argument prima referencu na funkciju, odnosno *povratni poziv*; moguće funkcije za taj povratni poziv su `prviIzvorBrojeva()` i `drugiIzvorBrojeva()` jer su prikladnog tipa (primaju jedan parametar tipa `int` i vraćaju rezultat tipa `double`).

Ovaj primjer je prilično jednostavan pa nije sasvim očita prednost ovog načina rada, ali uočimo da općenito daje mogućnost da se funkcija za ispis implementira odvojeno od algoritma koji rješava problem, te je moguće kombinirati različite prikladno definirane funkcije jedne i druge skupine.

## Praćenje događaja simulacije povezivanjem izvora i odvoda

Korištenjem povratnog poziva funkcije ns-3 na apstraktan način implementira ispis informacija o simulaciji (primjerice, informacija o paketima odbačenim kod primanja). Korisnik u simulaciji definira na koji način želi taj ispis primiti i u kojem obliku. U konkretnom primjeru to može biti

- sa strane izvora za praćenje događaja simulacije možda bi htjeli znati u kojoj sekundi je odbačen paket i koji je on bio po redu, ili koja mu je veličina, ili koja je zaglavlja imao na sebi, dok
- sa strane odvoda za praćenje događaja simulacije možda želite da dio ispisa bude na ekran, a dio u neku datoteku.

**Izvor za praćenje događaja simulacije** (engl. *trace source*) je jednoznačno određen referencom na objekt i svojim imenom koje je znakonvni niz (primjerice, to bi mogli biti `"CongestionWindowSize"` na TCP mrežnoj utičnici ili `"PacketDropBecauseFragmentationNeeeded"` na IP sučelju). Iako je to dosta jednostavno, mi se ovdje nećemo baviti stvaranjem novih izvora, već ćemo samo koristiti postojeće.

**Odvod za praćenje događaja simulacije** (engl. *trace sink*) je funkcija koju sami definiramo u našoj simulaciji. Ona je uvijek tipa `void`, a argumenti koje prima ovise o izvoru na koji će se spajati.

Da bi povezali izvor i odvod iskoristit ćemo metodu `TraceConnectWithoutContext()` koji ima svaki objekt i koja služi za povezivanje izvora za praćenje događaja simulacije (koji je dio objekta) sa željenim odvodom (koji mora biti prikladnog oblika). Postoji i funkcija `TraceConnect()` koja uzima kontekst u obzir, međutim, zasad ćemo zanemariti taj dio u vezi s kontekstom jer nam nije bitan.

Konkretno, ukoliko želimo pratiti slanje koje vrši neka mrežna kartica, tada ćemo iz `NetDeviceContainer`-a `devices` dohvatiti tu mrežnu karticu i na njen izvor za praćenje početka slanja paketa na fizički medij `PhyTxBegin` povezati funkciju odgovarajućeg tipa kodom

``` c++
devices.Get (0)->TraceConnectWithoutContext ("PhyTxBegin", MakeCallback (&PhyTxLogging));
```

Ovdje funkcija `MakeCallback()` radi povratni poziv od reference na funkciju koja joj se proslijedi kao argument.

!!! note
    Funkcija `TraceConnectWithoutContext()` **vrši** provjeru je li niz znakova koji prima kao prvi argument (u našem slučaju `"PhyTxBegin"`) ispravan, i korisniku to signalizira na način da vraća `true` ako je, i `false` ako nije. Zbog toga program neće prekinuti izvođenje ako je uneseni niz znakova neispravan. Dakle, ako želite biti sigurni da se povezivanje zaista dogodilo, možete to provjeriti na način:

``` c++
bool success = devices.Get (0)->TraceConnectWithoutContext ("PhyTxBegin", MakeCallback (&PhyTxLogging));
std::cout << "Povezivanje uspjelo: " << success << std::endl;
```

Radi čistoće koda to ovdje nećemo raditi.

Zbog načina na koji je definiran izvor `PhyTxBegin`, funkcija `PhyTxLogging` koja služi kao odvod za praćenje mora biti tipa `void` i kao argument primati pokazivač na nepromjenjivi paket. Za ispis na ekran ćemo iskoristiti uobičajeni `std::cout` definiran u zaglavlju `iostream` (koje je već uključeno unutar ns-3-ovog zaglavlja `core-module.h`).

Želimo znati kada su počela slanja paketa. Informaciju o sadašnjem trenutku u simuliranom vremenu moguće je dobiti pozivom funkcije `Simulator::Now()` koja vraća objekt tipa `Time`. Objekti tipa `Time` imaju definiranu metodu `GetSeconds()` koja izražava simulirano vrijeme sadašnjeg trenutka u sekundama.

``` c++
void
PhyTxLogging (Ptr<const Packet> p)
{
  std::cout << "Begging physical transmission of a packet at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}
```

Na sličan način kao `PhyTxBegin` rade i `PhyTxEnd`, koja prati događaje završetka prijenosa na fizički medij i `PhyTxDrop`, koja prati odbacivanja paketa prije slanja. Potpuno analogno njima rade i izvori za praćenje primanja paketa

- `PhyRxEnd`, kraj primanja paketa na fizičkom sloju,
- `PhyRxDrop`, odbacivanje paketa kod primanja na fizičkom sloju zbog grešaka, i
- `MacRx`, početak primanja paketa na sloju veze podataka.

Svi navedeni izvori opisani su u [dokumentaciji klase PointToPointNetDevice](https://www.nsnam.org/docs/doxygen/dc/d89/classns3_1_1_point_to_point_net_device.html).

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

void
PhyTxLogging (Ptr<const Packet> p)
{
  std::cout << "Begging physical transmission of a packet at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

int main ()
{
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  devices.Get (0)->TraceConnectWithoutContext ("PhyTxBegin", MakeCallback (&PhyTxLogging));

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
  echoClient.SetAttribute ("MaxPackets", UintegerValue (4));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (2.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Dodatak: korištenje ns-3 logging podsustava

Kada bi htjeli biti konzistentni s ostatkom ns-3-a, mogli bi koristiti makro funkciju `NS_LOG_UNCOND()` koja bezuvjetno (engl. *unconditionally*) ispisuje na *standardni izlaz za greške* vrijednost danog argumenta, i koristi `std::cerr`. Da bi makro funkcija `NS_LOG_UNCOND()` mogla raditi, potrebno je definirati ime naše simulacije koje će se koristiti za praćenje. Za to ćemo iskoristiti makro funkciju `NS_LOG_COMPONENT_DEFINE()` na početku našeg simulacijskog programa, nakon `#include<>` naredbi i prije funkcije `main()`. Ona kao argument prima upravo to ime, koje može biti proizvoljno ali mora biti jedinstveno u ns-3-u.

``` c++
NS_LOG_COMPONENT_DEFINE ("VjezbaTracing");
```

Zatim unutar funkcije `main()` možemo koristiti `NS_LOG_UNCOND()` na način

``` c++
int main ()
{
  // ...
  int x = 3;
  NS_LOG_UNCOND ("Varijabla x ima vrijednost " << x);
  // ekvivalentno std::cerr << "Varijabla x ima vrijednost " << x << std::endl;
  // ...
}
```

Postoji nekoliko nivoa logginga, od kojih smo već vidjeli `LOG_LEVEL_INFO` koji uključuje prikazivanje nivoa informativnih poruka i viših. Specifično, to znači da su poruke koje se koriste u debugiranju skrivene; za prikaz istih morali bi umjesto `LOG_LEVEL_INFO` koristiti `LOG_LEVEL_DEBUG`. Svakom nivou logginga pripada makro funkcija; primjerice, `LOG_LEVEL_INFO` pripada makro funkcija `NS_LOG_INFO()`, a `LOG_LEVEL_DEBUG` pripada makro funkcija `NS_LOG_DEBUG()`.

Implikacije ovog pristupa su vrlo zanimljive. U kodu

``` c++
int main ()
{
  LogComponentEnable ("VjezbaTracing", LOG_LEVEL_INFO);
  // ...
  int x = 3;
  NS_LOG_DEBUG ("Varijabla x ima vrijednost " << x);
  // ekvivalentno std::cerr << "Varijabla x ima vrijednost " << x << std::endl;
  // ...
}
```

na ekranu neće biti prikazana poruka `Varijabla x ima vrijednost` i vrijednost varijable `x`, dok će u kodu

``` c++
int main ()
{
  LogComponentEnable ("VjezbaTracing", LOG_LEVEL_DEBUG);
  // ...
  int x = 3;
  NS_LOG_DEBUG ("Varijabla x ima vrijednost " << x);
  // ekvivalentno std::cerr << "Varijabla x ima vrijednost " << x << std::endl;
  // ...
}
```

ta poruka biti prikazana. Uočimo da razliku čini količina logginga uključena pozivom funkcije `LogComponentEnable()`. Ovo omogućuje fino razdvajanje poruka koje pojedina komponenta ispisuje na ekran, što olakšava rad s većim simulacijama.

!!! note
    Više o loggingu možete pronaći u [službenoj dokumentaciji](https://www.nsnam.org/docs/manual/html/logging.html).

## Model učestalosti grešaka paketa

Najjednostavniji model grešaka je `RateErrorModel` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d6/de9/classns3_1_1_rate_error_model.html)) koji određeni broj paketa u uzorku označava iskrivljenim.

Kao i do sada, u primjeru korištenja objasnit ćemo samo one dijelove koji su novi u odnosu na prethodnu vježbu. Ukoliko ima kakvih nejasnoća u dijelu koda koji ovdje nije objašnjen preporuka je da ponovno proučite prethodnu vježbu.

Prvo inicijaliziramo pokazivač `em` na objekt tipa `RateErrorModel` koji stvaramo.

``` c++
Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
```

Postavljamo atribute:

- `ErrorUnit`, koji određuje jedinice u kojima će se mjeriti učestalost grešaka, moguće vrijednosti su

    - `ERROR_UNIT_BIT`: bitovi,
    - `ERROR_UNIT_BYTE`: bajtovi (zadana vrijednost),
    - `ERROR_UNIT_PACKET`: paketi,

- `ErrorRate`, koji određuje učestalost pojavljivanja grešaka, na vrijednost `0.01`,
- `RanVar`, koji je slučajna varijabla na temelju čije vrijednosti će se odlučivati o tome hoće li paket biti označen kao iskrivljen ili ne, na vrijednost uobičajeno korištene uniformne slučajne variajble.

``` c++
em->SetAttribute ("ErrorUnit", EnumValue (RateErrorModel::ERROR_UNIT_PACKET));
em->SetAttribute ("ErrorRate", DoubleValue (0.01));
em->SetAttribute ("RanVar", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"));
```

Zatim iz `NetDeviceContainer`-a `devices` dohvaćamo karticu čvora koji ima odvod za pakete i postavljamo na njen atribut `ReceiveErrorModel` vrijednost upravo stvorenog modela grešaka, koja je u sustavu atributa tipa `PointerValue`.

``` c++
devices.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (em));
```

Ovo će učiniti da će neki paketi biti odbačeni kod primitka (u prosjeku 1 na njih 100). Da bi to vidjeli, definirat ćemo funkciju `PrintPhyRxPacketDrop()` koja će služiti kao odvod za praćenje. Ime funkcije je proizvoljno, a zadano je da je tipa `void` i da prima jedan argument tipa `Ptr<const Packet>`.

``` c++
void
PrintPhyRxPacketDrop (Ptr<const Packet> p)
{
  std::cout << "Packet with UID " << p->GetUid () << " dropped on reception at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}
```

Zatim je potrebno spojiti funkciju koja je povratni poziv na `PhyRxDrop` izvor mrežne kartice drugog čvora.

``` c++
devices.Get (1)->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&PrintPhyRxPacketDrop));
```

Nakon pokretanja simulacije, na ekran će se ispisati u kojim su vremenskim trenucima odbačeni paketi kod primanja. Za razliku od paketa u stvarnosti, svaki paket stvoren u simulaciji ima identifikator (UID) koji služi za međusobno razlikovanje paketa i nije dio podataka koje paket nosi.

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

void
PrintPhyRxPacketDrop (Ptr<const Packet> p)
{
  std::cout << "Packet with UID " << p->GetUid () << " dropped on reception at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

int main ()
{
  LogComponentEnable ("ErrorModel", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);
  pointToPoint.EnableAsciiAll ("vjezba-rate-error-model");

  Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
  em->SetAttribute ("ErrorUnit", EnumValue (RateErrorModel::ERROR_UNIT_PACKET));
  em->SetAttribute ("ErrorRate", DoubleValue (0.01));
  em->SetAttribute ("RanVar", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"));
  devices.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (em));

  devices.Get (1)->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&PrintPhyRxPacketDrop));

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces;
  interfaces = address.Assign (devices);

  PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (interfaces.GetAddress(1), 9));
  ApplicationContainer apps = sink.Install (nodes.Get (1));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (11.0));

  OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces.GetAddress (1), 9));
  onOffApp.SetAttribute ("DataRate", StringValue ("1Mbps"));
  onOffApp.SetAttribute ("PacketSize", UintegerValue (2048));
  onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
  onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));

  ApplicationContainer clientApps = onOffApp.Install (nodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Dodatak: model praskavih grešaka paketa

Nešto složeniji model grešaka je `BurstErrorModel` koji...

!!! todo
    Ovaj dio treba napisati u cijelosti.
