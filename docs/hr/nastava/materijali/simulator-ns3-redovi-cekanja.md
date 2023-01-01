---
author: Ivan Ivakić, Vedran Miletić
---

# Simulacijski modeli redova čekanja

U računalnim mrežama, osim zbog grešaka paketa, gubici paketa na putu obično nastaju kad paketi dospiju do nekog usmjerivača čiji je memorijski prostor za primanje dolazećih paketa sasvim ispunjen, tako da taj usmjerivač odbacuje dolazeće pakete. Zbog velikog kašnjenja paketa pošiljatelj koji koristi TCP doživljava istek vremena i ponavlja slanje paketa. Ponavljanjem slanja paketa pokušava se otkloniti posljedice zagušenja usmjerivača (konkretno, gubitak paketa), ali se time ne otklanja uzrok tog zagušenja. Štoviše, ponavljanjem slanja paketa dodatno se opterećuje preopterećene usmjerivače, i time se povećava vjerojatnost da neki paketi budu (ponovno) odbačeni.

Osnovni uzrok zagušenja usmjerivača i time odbacivanja paketa je u tome što prevelik broj pošiljatelja šalje u mrežu pakete prevelikim intenzitetom u odnosu na trenutne mogućnosti prijenosa onih usmjerivača i veza preko kojih trebaju biti prenijeti ti paketi na svom putu od izvora do odredišta. Problem zagušenja usmjerivača u mreži treba rješavati na taj način da se otklanja uzrok toga problema.

Redovi čekanja su u ns-3-u implementirani u klasi `Queue`.

## Redovi čekanja s odbacivanjem repa

Najjednostavnija varijanta reda čekanja je `DropTailQueue` ([dokumentacija](https://www.nsnam.org/docs/doxygen/dc/dc6/classns3_1_1_drop_tail_queue.html)) koja implementira red čekanja s odbacivanjem repa.

Da bi imalo smisla uopće promatrati redove čekanja s odbacivanjem repa, potrebno je stvoriti situaciju u kojoj dolazi do prepunjavanja reda čekanja. Razmatrat ćemo mrežu koja ima već razmatranu linearnu topologiju koja se sastoji od tri čvora i dvije veze između njih; pritom prvi čvor šalje pakete trećem čvoru preko drugog. Želimo simulirati situaciju u kojoj drugi čvor ima red čekanja paketa za slanje na kojem se događaju odbacivanja repa. Kako je u ovom primjeru on-off aplikacija na prvom čvoru, a šalje pakete u odvod na trećem čvoru, drugi čvor koji radi kao usmjerivač pakete prima s prve mrežne kartice, a šalje ih na drugu. U slučaju da su obje veze jednake širine frekventnog pojasa, do zagušenja (i time odbacivanja repa) neće doći. Stoga ćemo staviti da je prva veza veće širine frekventnog pojasa nego druga. Zadržavanje nam nije naročito značajno pa ćemo staviti da obje veze imaju jednako.

``` c++
NodeContainer allNodes, nodes12, nodes23;
allNodes.Create (3);
nodes12.Add (allNodes.Get (0));
nodes12.Add (allNodes.Get (1));
nodes23.Add (allNodes.Get (1));
nodes23.Add (allNodes.Get (2));

PointToPointHelper pointToPoint;
pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
pointToPoint.SetChannelAttribute ("Delay", StringValue ("1ms"));

NetDeviceContainer devices12, devices23;
devices12 = pointToPoint.Install (nodes12);

pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("2Mbps"));
devices23 = pointToPoint.Install (nodes23);
```

Objekt tipa `DropTailQueue` stvaramo na uobičajen način uz navođenje tipa elemenata reda čekanja, a to je `Packet`.

``` c++
Ptr<DropTailQueue<Packet> > queue = CreateObject<DropTailQueue<Packet> > ();
```

Zatim postavljamo vrijednost atributa `MaxSize`, koja je maksimalni broj paketa ili bajtova koje taj red čekanja može u sebi imati u nekom trenutku, na vrijednost `20p` koja znači 20 paketa (alternativno, mogli smo postaviti i vrijednost u bajtovima, primjerice `512KiB` ili `1MiB`).

``` c++
queue->SetAttribute ("MaxSize", QueueSizeValue (QueueSize ("20p")));
```

Na drugu karticu drugog čvora postavljamo red čekanja.

``` c++
devices23.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));
```

Potpuno analogno modelu grešaka, definiramo funkciju `QueueTailDrop()` koja će služiti kao odvod za praćenje. Kao i kod `ErrorModel`-a, funkcija je tipa `void`, a prima jedan argument tipa `Ptr<const Packet>`.

``` c++
void
QueueTailDrop (Ptr<const Packet> p)
{
  std::cout << "Queue dropped tail at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}
```

Zatim spajamo povratni poziv na `Drop` izvor reda čekanja.

``` c++
queue->TraceConnectWithoutContext ("Drop", MakeCallback (&QueueTailDrop));
```

Pored povećanja širine frekventnog pojasa veze, potrebno je povećati brzinu kojom on-off aplikacija šalje pakete na vrijednost koja je veća od širine frekventnog pojasa druge veze.

``` c++
OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress (1), 9));
onOffApp.SetAttribute ("DataRate", StringValue ("4Mbps"));
onOffApp.SetAttribute ("PacketSize", UintegerValue (2048));
onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));
```

Nakon pokretanja simulacije, na ekran će se ispisati u kojim su vremenskim trenucima odbačeni paketi zbog prepunjenog reda čekanja.

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

void QueueTailDrop (Ptr<const Packet> p)
{
  std::cout << "Queue dropped tail at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

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
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("1ms"));

  NetDeviceContainer devices12, devices23;
  devices12 = pointToPoint.Install (nodes12);

  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
  devices23 = pointToPoint.Install (nodes23);

  Ptr<DropTailQueue<Packet> > queue = CreateObject<DropTailQueue<Packet> > ();
  queue->SetAttribute ("MaxSize", QueueSizeValue (QueueSize ("20p")));
  queue->TraceConnectWithoutContext ("Drop", MakeCallback (&QueueTailDrop));

  devices23.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));

  InternetStackHelper stack;
  stack.Install (allNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress(1), 9));
  ApplicationContainer apps = sink.Install (allNodes.Get (2));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (20.0));

  OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress (1), 9));
  onOffApp.SetAttribute ("DataRate", StringValue ("5Mbps"));
  onOffApp.SetAttribute ("PacketSize", UintegerValue (4096));
  onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
  onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));

  ApplicationContainer clientApps = onOffApp.Install (allNodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (19.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Topologija bučice i redovi čekanja

Razmotrimo već ranije spomenutu topologiju bučice oblika

```
n1 ----\              /---- n5
        \            /
        n3 -------- n4
        /            \
n2 ----/              \---- n6
```

Uzmimo da su on-off aplikacije instalirane na domaćinima `n1` i `n2`, a njihovi odvodi na domaćinima `n5` i `n6`. Ukoliko je širina frekventnog pojasa veze usmjerivača `n3` i `n4` znatno manji od sume širina frekventnog pojasa veza `n1 -- n3` i `n2 -- n3` i ako aplikacije na domaćinima `n1` i `n2` šalju pakete intenzitetom koji koristi velik postotak širine frekventnog pojasa, ograničeni kapacitet veze `n3 -- n4` dovesti će do povećanja duljine reda čekanja na `n3` i vremenom do odbacivanja paketa. Ovo je donekle pojednostavljen model stvarne mreže, ali dovoljno precizan da na temelju njega možemo promatrati način rada TCP algoritama za upravljanje zagušenjem u situaciji kada se gubici paketa događaju odbacivanjem paketa zbog prepunjenog reda čekanja.

Navedenu topologiju s redom čekanja veličine 20 paketa na mrežnoj kartici čvora `n3` u vezi `n3 -- n4` stvorili bi na način kako je objašnjeno u prethodnim vježbama. Postavljanje reda čekanja na željenu mrežnu karticu vrši se idućim kodom. Ponovno uočimo da postavljamo red čekanja na mrežnu karticu čvora `n3` kojim ostvaruje vezu prema `n4` i da ona nije jedina mrežna kartica koju `n3` ima na sebi.

``` c++
Ptr<DropTailQueue<Packet> > queue = CreateObject<DropTailQueue<Packet> > ();
queue->SetAttribute ("MaxSize", QueueSizeValue (QueueSize ("20p")));
devices34.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));
```

## Dodatak: redovi čekanja s ranim otkrivanjem/odbacivanjem

Slučajno rano otkrivanje/odbacivanje (engl. *Random Early Detection/Drop*, RED) je algoritam izbjegavanja zagušenja te aktivne kontrole reda čekanja na usmjerivaču. RED algoritam prati trenutnu veličinu reda čekanja i broj odbačenih paketa za svaki od domaćina koji šalje te na osnovu toga statističkim metodama računa vjerojatnost da odbaci iduće pakete određenog domaćina koji šalje. Kada je red čekanja prazan vjerojatnost da se odbaci paket približava se vrijednosti 0, a kako se red čekanja povećava uslijed broja paketa koji pristižu vjerojatnost raste prema vrijednosti 1 i time se povećava mogućnost da paket bude odbačen. (Naravno, ukoliko vjerojatnost dosegne vrijednost 1 taj paket je odbačen s obzirom da vjerojatnost 1 označava siguran događaj.)

Simulaciju kreiramo kao i za redove čekanja tipa `DropTailQueue`. Promjena u kodu je vezana za promjenu tipa reda čekanja, a osnovni je način prikazan u nastavku.

Kreiramo novi objekt tipa `RedQueue` ([dokumentacija](https://www.nsnam.org/docs/doxygen/df/dae/classns3_1_1_red_queue_disc.html)) i spremamo pokazivač na njega u varijablu `queue`.

``` c++
Ptr<RedQueue> queue = CreateObject<RedQueue> ();
```

Postavljamo način rada mjerenja ograničenja u paketima. Prvo odabiremo način rada metodom `SetMode()` kojoj prosljeđujemo parametar `RedQueue::QUEUE_MODE_PACKETS` (druga mogućnost je `RedQueue::QUEUE_MODE_BYTES`).

``` c++
queue->SetMode(RedQueue::QUEUE_MODE_PACKETS );
```

Metodi `SetQueueLimit()` dajemo parametar koji ima proizvoljnu cjelobrojnu vrijednost čime postavljamo ograničenje u broju paketa (ili bajtova) kreiranog reda čekanja.

``` c++
queue->SetQueueLimit(5);
```

``` c++
devices2.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));
```

Red čekanja tipa `RedQueue` moguće je postaviti i ranije, kod pomoćnika. Ukoliko želimo da red čekanja na svakom uređaju koji pomoćnik stvori bude tipa RED, to možemo učiniti na način da nad `pointToPoint` instancom `PointToPointHelper`-a pozovemo metodu `SetQueue()` koja prima kao parametar ime tipa reda čekanja.

``` c++
pointToPoint.SetQueue ("ns3::RedQueue");
```

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

void QueueTailDrop (Ptr<const Packet> p)
{
  std::cout << "Queue dropped tail at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

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
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("1ms"));

  NetDeviceContainer devices12, devices23;
  devices12 = pointToPoint.Install (nodes12);

  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
  devices23 = pointToPoint.Install (nodes23);

  InternetStackHelper stack;
  stack.Install (allNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);

  Ptr<RedQueue> queue = CreateObject<RedQueue> ();
  queue->SetMode (RedQueue::QUEUE_MODE_PACKETS);
  queue->SetQueueLimit(5);

  devices23.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));

  queue->TraceConnectWithoutContext ("Drop", MakeCallback (&QueueTailDrop));

  PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress(1), 9));
  ApplicationContainer apps = sink.Install (allNodes.Get (2));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (20.0));

  OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress (1), 9));
  onOffApp.SetAttribute ("DataRate", StringValue ("5Mbps"));
  onOffApp.SetAttribute ("PacketSize", UintegerValue (4096));
  onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
  onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));

  ApplicationContainer clientApps = onOffApp.Install (allNodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (19.0));

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Dodatak: nadgledanje tokova paketa

Modul `flow-monitor` možemo koristiti za nadgledanje toka/ova paketa. Radi jednostavnosti koristit ćemo pomagač `FlowMonitorHelper` ([dokumentacija](https://www.nsnam.org/docs/doxygen/db/dbb/classns3_1_1_flow_monitor_helper.html)).

``` c++
FlowMonitorHelper flow;
```

U klasi `FlowMonitorHelper` postoji mogućnost nadgledanja jednog, više ili svih čvorova. Za prva dva slučaja koristimo metodu `Install()` koja prima parametar tipa `Ptr<Node>` za jedan čvor ili `NodeContainer` za više čvorova. Ukoliko pak želimo nadgledati sve čvorove to činimo pozivanjem metode `InstallAll()` koja ne prima parametre. Za primjer, stvoriti ćemo objekt tipa `Ptr<FlowMonitor>` koji nadgleda tokove svih čvorova.

``` c++
Ptr<FlowMonitor> flow_nodes = flow.InstallAll ();
```

Alternativno, to smo mogli napraviti samo na prvom čvoru kodom oblika

``` c++
Ptr<FlowMonitor> flow_nodes = flow.Install (allNodes.Get (0));
```

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>
#include <ns3/flow-monitor-module.h>

using namespace ns3;

void QueueTailDrop (Ptr<const Packet> p)
{
  std::cout << "Queue dropped tail at " << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

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
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("1ms"));

  NetDeviceContainer devices12, devices23;
  devices12 = pointToPoint.Install (nodes12);

  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
  devices23 = pointToPoint.Install (nodes23);

  InternetStackHelper stack;
  stack.Install (allNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces12 = address.Assign (devices12);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces23 = address.Assign (devices23);

  Ptr<DropTailQueue<Packet> > queue = CreateObject<DropTailQueue<Packet> > ();
  queue->SetAttribute ("MaxSize", QueueSizeValue (QueueSize ("20p")));

  devices23.Get (0)->SetAttribute ("TxQueue", PointerValue (queue));

  queue->TraceConnectWithoutContext ("Drop", MakeCallback (&QueueTailDrop));

  PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress(1), 9));
  ApplicationContainer apps = sink.Install (allNodes.Get (2));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (11.0));

  OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces23.GetAddress (1), 9));
  onOffApp.SetAttribute ("DataRate", StringValue ("5Mbps"));
  onOffApp.SetAttribute ("PacketSize", UintegerValue (4096));
  onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
  onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));

  ApplicationContainer clientApps = onOffApp.Install (allNodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper flow;
  Ptr<FlowMonitor> flowmonitor = flow.InstallAll();

  Simulator::Run ();
  flowmonitor->CheckForLostPackets ();
  flowmonitor->SerializeToXmlFile ("flow.xml", false, false);
  Simulator::Destroy ();

  return 0;
}
```
