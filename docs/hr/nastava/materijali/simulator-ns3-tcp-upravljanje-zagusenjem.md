---
author: Vedran Miletić, Ivan Ivakić
---

# Simulacijski modeli TCP-ovih algoritama za upravljanje zagušenjem

Prisjetimo se da je jedna od prednosti TCP-a u odnosu na UDP kontrola zagušenja. Zagušenje je pojava kada čvor prima ili šalje toliku količinu podataka da se javljaju gubici paketa, blokiranje novih veza ili pak stvaranje velikog reda čekanja. TCP se brine o tome da do ovakvih situacija ne dođe time što ima ugrađenu kontrolu zagušenja. Kontrola zagušenja će u slučaju da se ovakvo ponašanje mreže primijeti poduzeti potrebne mjere da zagušenje smanji ili otkloni.

TCP algoritmi za kontrolu zagušenja koje opisujemo u nastavku razvijeni s ciljem smanjenja intenziteta kojim pošiljatelji šalju pakete u mrežu u slučaju da uoče veće opterećenje mreže. Ovdje ćemo samo "zagrebati po površini"; problematika izbjegavanja zagušenja mreže, uz optimalno iskorištenje njenih kapaciteta, spada među najvažnije i najaktivnije teme u području istraživanja i razvoja računalnih mreža.

## Varijanta TCP RFC793

[RFC 793](https://datatracker.ietf.org/doc/html/rfc793) definira da TCP vrši ponovni prijenos podataka u slučaju isteka vremena. Veličina prozora pošiljatelja se pritom ne mijenja, već ovisi samo o oglašenom prozoru primatelja. To uzrokuje česta zagušenja mreže, te su tijekom 80-tih godina predloženi algoritmi za upravljanje zagušenjem koji mijenjaju veličinu prozora pošiljatelja, koji se naziva i [prozor zagušenja](https://en.wikipedia.org/wiki/Congestion_window) (engl. *congestion window*).

## Varijante TCP Tahoe i TCP Reno

Marshall Kirk McKusick, jedan od voditelja razvoja *Berkeley Software Distribution* (kraće BSD) Unixa na Sveučilištu u Berkeleyu tijekom 80-tih godina, u predavanju [A Narrative History of BSD](https://youtu.be/ds77e3aO9nA) opisuje detaljno način na koji se razvijao TCP unutar BSD Unixa i ideje koje su prethodile. Za TCP algoritme za upravljanje zagušenjem značajne su dvije verzije BSD Unixa:

- 4.3BSD-Tahoe izdana 1988. godine, čija implementacija TCP-a se očekivano naziva **TCP Tahoe**, te
- 4.3BSD-Reno izdana 1990. godine, čija implementacija TCP-a se naziva **TCP Reno**.

Osnovni algoritmi koje TCP koristi za upravljanje zagušenjem su:

- spori početak (engl. *slow-start*),
- izbjegavanje zagušenja (engl. *congestion avoidance*),
- brzo ponovno slanje (engl. *fast retransmit*),
- brzi oporavak (engl. *fast recovery*, koristi ga samo TCP Reno).

[Spori početak](https://en.wikipedia.org/wiki/Slow-start) (engl. *slow start*) je jedan od algoritama koji TCP koristi za kontrolu zagušenja u mreži. On povećava `cwnd` za 1 MSS za svaki primljeni ACK, što ima za posljedicu udvostručenje `cwnd`-a svaki RTT, odnosno eksponencijalni porast. To se događa sve dok se ne dogodi gubitak paketa ili dok se ne dosegne unaprijed zadana maksimalna vrijednost veličine prozora `ssthresh` (prag sporog početka (engl. *slow-start threshold*), veličina je dana u bajtovima, većina implementacija ima inicijalnu vrijednost 65536), a zatim se prelazi u izbjegavanje zagušenja.

Ako se dogodi gubitak paketa za vrijeme sporog početka, TCP pretpostavlja da je to zbog zagušenja mreže i poduzima korake da smanji opterećenje, odnosno prelazi u fazu [izbjegavanja zagušenja](https://en.wikipedia.org/wiki/TCP_congestion-avoidance_algorithm) (engl. *congestion avoidance*), u kojoj povećava `cwnd` za 1 MSS svaki RTT, što ima za posljedice linearni (aditivni) porast veličine prozora. TCP u toj fazi radi po metodi [aditivnog porasta i multiplikativnog pada](https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease). Preciznije, veličina prozora se računa po formuli

$$
cwnd \gets cwnd + MSS \times \frac{MSS}{cwnd}.
$$

Kada dođe do gubitka paketa (što se indicira istekom vremena ili primitkom dupliciranih ACK-ova), polovina trenutne veličine prozora `cwnd` sprema se u `ssthresh`. U slučaju da je došlo do isteka vremena, pored navedenog se `cwnd` postavlja na veličinu 1 MSS, i do `ssthresh` koristi spori početak, a od `ssthresh` koristi izbjegavanje zagušenja.

[Brzo ponovno slanje](https://en.wikipedia.org/wiki/Fast_retransmit) (engl. *fast retransmit*) smanjuje vrijeme koje pošiljatelj čeka prije ponovnog slanja izgubljenog segmenta, a radi tako da kada TCP pošiljatelj dobije tri duplikata ACK-a (tj. četiri identična ACK-a) s istim sekventnim brojem, pošiljatelj smatra da je paket s idućim višim sekventnim brojem izgubljen, i da neće naknadno stići. Pošiljatelj će tada ponovno slati taj paket bez da čeka da njegov istek vremena. Tu postoji razlika kod TCP varijanti Tahoe i Reno.

- Kod TCP varijante Tahoe tri duplikata ACK-a tretiraju se kao istek vremena, `cwnd` se smanjuje na 1 MSS, segment se šalje, i prelazi se u spori početak.
- Kod TCP varijante Reno tri duplikata ACK-a tretiraju čine da se `cwnd` smanjuje na pola, segment se šalje, i prelazi se u brzi oporavak.

U fazi [brzog oporavka](https://en.wikipedia.org/wiki/Slow-start#Fast_recovery) (engl. *fast recovery*) TCP Reno šalje segment koji je označen od strane tri duplikata ACK-a i očekuje potvrdu cijelog prozora prije prelaska u izbjegavanje zagušenja. Ako potvrda stigne, vrijednost `cwnd` postavlja na `ssthresh`, time se preskače spori početak i nastavlja upravljanje veličinom prozora korištenjem izbjegavanja zagušenja. Ako ne stigne potvrda cijelog prozora, TCP prelazi u spori početak.

## Dodatak: varijanta TCP NewReno

TCP NewReno je poboljšana verzija Reno varijante TCP protokola. Eksperimentalno pokazuje bolje rezultate od Reno varijante u slučaju velikog broja grešaka. Kod Reno protokola brzo ponovno slanje događa se nakon primitka 3 ponovljena ACK te se čeka potvrda cijelog prozora slanja da bi se prozor pomaknuo na iduće pakete, dok NewReno za svaki ponovljeni ACK šalje još jedan novi (prethodno ne poslani) paket ne čekajući potvrdu cijelog prozora ubrzavajući time popunjavanje praznina u redosljedu paketa. Također za svaki ACK koji stigne smatra da idući paket u nizu fali te ga počinje slati (u fazi brzog ponovnog slanja). NewReno ima i jedan nedostatak, naime u slučaju da paketi stignu u pogrešnom redosljedu uz pomak veći od 3 paketa (u slijednom broju) smatra se da je paket izgubljen i ulazi se u fazu brzog ponovnog slanja. Pošto je takvo ponašanje uočeno dorađen je sa sustavom potvrđivanja paketa nakon što je primljen ponovljeni paket koji je bio van redosljeda tako što se sekventni brojevi uvećavaju te se šalju ACK paketi bez da je ponovljeno slanje svih ostalih paketa ukoliko je samo jedan paket van redosljeda.

!!! note
    Na [Wikipedijinoj stranici s taksonomijom algoritama za upravljanje zagušenjem](https://en.wikipedia.org/wiki/Taxonomy_of_congestion_control) možete pročitati više o varijantama TCP-a koje postoje i njihovim svojstvima.

## Praćenje promjene veličine prozora zagušenja

Cilj nam je pratiti promjenu veličine prozora zagušenja TCP-a. Rezultate, pored ispisa na ekran, želimo zapisati u datoteku za kasniju analizu i crtanje grafova. Primjer koji opisujemo u nastavku ima istu topologiju mreže i iste aplikacije kao prethodni primjer.

Nakon stvaranja čvorova i instalacije mrežnih kartica i veze tipa točka-do-točke, instalacije TCP/IP protokolarnog stoga i dodjele IP adresa, potrebno je postaviti on-off aplikaciju i odvod za pakete. Međutim, kako bi simulacija stvorila TCP mrežne utičnice koje koriste željeni algoritam za upravljanje zagušenjem (Tahoe ili Reno), potrebno je unaprijed postaviti parametre tvornice TCP mrežnih utičnica. Postoji nekoliko načina za to, a najjednostavniji i najčešće korišten je pomoću sustava konfiguracije implementiranog u klasi `Config`. Način rada s klasom `Config` je vrlo sličan načinu rada s klasom `Simulator`, obzirom da se također radi o statičkoj klasi, te načinu rada sa sustavom atributa. Konkretno, da bi promijenili algoritam za upravljanje zagušenjem, treba promijeniti vrijednost ključa `"ns3::TcpL4Protocol::SocketType"`.

``` c++
Config::SetDefault ("ns3::TcpL4Protocol::SocketType", StringValue ("ns3::TcpTahoe"));
```

Podsjetimo se da je `Config::SetGlobal()` i `Config::SetDefault()` potrebno izvesti prije stvaranja objekata simulacije. Za TCP Reno bi stavili `"ns3::TcpReno"` kao vrijednost atributa umjesto `"ns3::TcpTahoe"`.

Sada stvaramo on-off aplikaciju i odvod za pakete na isti način kao i ranije. Informacija o veličini prozora zagušenja nalazi se u TCP mrežnoj utičnici koju će aplikacija koristiti. Da bi dosegli tu utičnicu, prvo pomoću metode `GetObject<>()` iz pametnog pokazivača `app` dohvaćamo objekt tipa `OnOffApplication`; zatim pomoću metode `GetSocket()` tog objekta dohvaćamo pokazivač na odgovarajuću mrežnu utičnicu koji pohranjujemo u varijablu `appSocket`. Ovo dohvaćanje objekta odgovarajućeg tipa potrebno je zato što `ApplicationContainer` sadrži pokazivače na objekte tipa `Application` kako bi omogućio spremanje različitih aplikacija u isti kontejner, a objekti tipa `Application` nemaju metodu `GetSocket()`. Pored toga, povezujemo povratni poziv na izvor `CongestionWindow`.

``` c++
Ptr<Socket> appSocket = app->GetObject<OnOffApplication> ()->GetSocket ();
appSocket->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));
```

Prikladna funkcija `CwndChange()` za taj povratni poziv je tipa `void` i prima dva argumenta tipa `uint32_t` (što je na većini operacijskih sustava i program-prevoditelja samo *fancy* naziv za tip `unsigned int`).

``` c++
void
CwndChange (uint32_t oldCwnd, uint32_t newCwnd)
{
  std::cout << "Congestion window has changed at time " << Simulator::Now ().GetSeconds () "s from " << oldCwnd << " to " << newCwnd;
}
```

Međutim, ovdje imamo dva problema:

- kao prvo, ovo će nam ispisati podatke na ekran, a mi želimo da podaci budu zapisani u datoteku, i
- kao drugo, dohvaćanje mrežne utičnice neće raditi ono što bi htjeli, već će rezultat biti nul-pokazivač. Naime, u trenutku pokretanja simulacije on-off aplikacija još nije pokrenuta i njena TCP mrežna utičnica nije stvorena, te je pokazivač na nju nul-pokazivač.

Da bi riješili prvi problem, iskoristit ćemo `AsciiTraceHelper`. Njegova metoda `CreateFileStream()` prima kao argument ime datoteke i stvara stream tipa `OutputStreamWrapper` koji služi za zapisivanje u datoteku traženog imena, a vraća pametni pokazivač na njega.

``` c++
AsciiTraceHelper asciiTraceHelper;
Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream ("vjezba-tcp-tahoe.cwnd");
```

Sada je potrebno pokazivač `stream` proslijediti kod povezivanja sustava za praćenje. Za to se koristi tzv. povezani povratni poziv (engl. *bound callback*). Funkcija `MakeBoundCallback()` koja stvara povezani povratni poziv slična je funkciji `MakeCallback()`, ali je proširuje na način da omogućuje prosljeđivanje argumenata funkciji koja se koristit kao povratni poziv. Preciznije, svi argumenti od drugog nadalje postaju argumenti funkcije povratnog poziva od prvog nadalje.

``` c++
appSocket->TraceConnectWithoutContext ("CongestionWindow", MakeBoundCallback (&CwndChangeWriteFile, stream));
```

Posljednja dva argumenta funkcije `CwndChangeFileWrite()` su tipa `uint32_t`, kao i kod funkcije `CwndChange()`. Njihove će vrijednosti proslijediti `appSocket` kod poziva, odnosno onda kad se promijeni veličina prozora zagušenja; uočite da je `MakeBoundCallback()` u gornjem primjeru ne prosljeđuje vrijednosti ta dva argumenta.

Funkcija `CwndChangeFileWrite()` pored ispisa na ekran vrši i ispis u datoteku. Sam način na koji to radi nije nam značajan; dovoljno je znati da je sučelje vrlo slično standardnom streamu `std::ofstream`. Obzirom nam je dovoljna funkcionalnost koja je ovdje navedena, načinom rada i dubljim razumijevanjem sučelja `OutputStreamWrapper`-a ovdje se nećemo baviti.

``` c++
void
CwndChangeWriteFile (Ptr<OutputStreamWrapper> stream, uint32_t oldCwnd, uint32_t newCwnd)
{
  std:cout << "Congestion window has changed at time " << Simulator::Now ().GetSeconds () "s from " << oldCwnd << " to " << newCwnd;
  *stream->GetStream () << Simulator::Now ().GetSeconds () << "\t" << newCwnd << std::endl;
}
```

Riješimo sada drugi problem. Želimo da se on-off metoda `GetSocket()` koja služi za dohvaćanje mrežne utičnice izvede na on-off aplikaciji točno nakon što ona započne s radom umjesto na početku simulacije. Kako on-off aplikacija započinje s radom u trenutku `Seconds (2.0)`, odnosno `t = 2.0s`, poziv možemo napraviti u trenutku `Seconds (2.0) + NanoSeconds (1.0)`, odnosno `t = 2.0s + 1.0ns`. To radimo iz dva razloga:

- U trenutku `t = 2.0s` ne možemo biti sigurni je li se dogodilo stvaranje utičnice i povezivanje aplikacije na utičnicu. Naime, to ovisi o načinu na koji će simulator poredati događaje zakazane za isti vremenski trenutak, te se može dogoditi da se dohvaćanje mrežne utičnice zakazano za `t = 2.0s` dogodi prije stvaranja mrežne utičnice zakazanog za `t = 2.0s`.
- U zadanim postavkama ns-3 koristi preciznost na nanosekundu, što znači da u simuliranom vremenu, koje je diskretno s intervalom od jedne nanosekunde, prvi trenutak nakon `t = 2.0s` je `t = 2.0s + 1.0ns`.

Kako bi to napravili, definirat ćemo funkciju `DoCwndTraceConnectOnSocketAfterAppStarts()` koja je tipa void i prima kao argumente pokazivač na aplikaciju i stream koji ćemo koristit za zapisivanje promjene veličine prozora zagušenja. Koristimo metodu `Simulator::Schedule()` da bi zakazali poziv funkcije željene funkcije sa željenim argumentima u željenom trenutku nakon pokretanja simulacije. Postoji velika sličnost u načinu rada povezanih povratnih poziva i ove metode, i ona nipošto nije slučajna. Za razliku od povratnih poziva i povezanih povratnih poziva, *svi* argumenti navedeni nakon reference na funkciju koju pozivamo prosljeđuju se funkciji kao argumenti.

``` c++
Simulator::Schedule (Seconds (2.0) + NanoSeconds (1.0), &DoCwndTraceConnectOnSocketAfterAppStarts, nodes.Get (0)->GetApplication (0), stream);
```

Unutar funkcije `DoCwndTraceConnectOnSocketAfterAppStarts()` prvo provjeravamo je li proslijeđeni pokazivač na aplikaciju različit od nul-pokazivača, da bi izbjegli dereferenciranje nul-pokazivača. Zatim deklariramo varijablu `appSocket` tipa `Ptr<Socket>` i u nju dohvaćamo pokazivač na mrežnu utičnicu, za koji također provjeravamo je li različit od nul-pokazivača. Naposlijetku na toj utičnici izvodimo povezivanje izvora za praćenje sa odgovarajućim povezanim povratnim pozivom i streamom.

``` c++
void
DoCwndTraceConnectOnSocketAfterAppStarts (Ptr<Application> app, Ptr<OutputStreamWrapper> stream)
{
  NS_ASSERT_MSG (app != 0, "OnOffApplication pointer can't be null");
  Ptr<Socket> appSocket = app->GetObject<OnOffApplication> ()->GetSocket ();
  NS_ASSERT_MSG (appSocket != 0, "OnOffApplication socket can't be null, is app started?");
  appSocket->TraceConnectWithoutContext ("CongestionWindow", MakeBoundCallback (&CwndChangeWriteFile, stream));
}
```

Kao rezultat izvođenja dobivamo tekstualnu datoteku s podacima o veličini prozora zagušenja u određenim vremenskim trenucima, na temelju koje možemo nacrtati graf, izračunati maksimalnu i prosječnu veličinu prozora zagušenja itd.

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

void
CwndChangeWriteFile (Ptr<OutputStreamWrapper> stream, uint32_t oldCwnd, uint32_t newCwnd)
{
  std::cout << "Congestion window has changed at time " << Simulator::Now ().GetSeconds () << "s from " << oldCwnd << " to " << newCwnd << std::endl;
  *stream->GetStream () << Simulator::Now ().GetSeconds () << "\t" << newCwnd << std::endl;
}

void
DoCwndTraceConnectOnSocketAfterAppStarts (Ptr<Application> app, Ptr<OutputStreamWrapper> stream)
{
  NS_ASSERT_MSG (app != 0, "OnOffApplication pointer can't be null");
  Ptr<Socket> appSocket = app->GetObject<OnOffApplication> ()->GetSocket ();
  NS_ASSERT_MSG (appSocket != 0, "OnOffApplication socket can't be null, is app started?");
  appSocket->TraceConnectWithoutContext ("CongestionWindow", MakeBoundCallback (&CwndChangeWriteFile, stream));
}

int main ()
{
  Config::SetDefault ("ns3::TcpL4Protocol::SocketType", StringValue ("ns3::TcpTahoe"));

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

  AsciiTraceHelper asciiTraceHelper;
  Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream ("vjezba-tcp-tahoe-cwnd.txt");

  Simulator::Schedule (Seconds (2.0) + NanoSeconds (1.0), &DoCwndTraceConnectOnSocketAfterAppStarts, nodes.Get (0)->GetApplication (0), stream);

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
```

## Crtanje grafova promjene prozora zagušenja u ovisnosti o vremenu

[Gnuplot](https://en.wikipedia.org/wiki/Gnuplot) je često korišten komandnolinijski alat za crtanje 2D i 3D grafova funkcija i skupova podataka. Gnuplot podržava velik broj izlaznih formata, od kojih se najčešće koriste PNG, EPS, PDF, SVG i JPEG. Mi ćemo koristiti PNG.

Gnuplot pokrećemo naredbom `gnuplot` u komandnoj liniji.

``` shell
$ gnuplot

        G N U P L O T
(...)
```

Nakon pokretanja gnuplota omogućen nam je unos naredbi. Pretpostavimo da su podaci o veličini prozora spremljeni u datoteku `cwnd-graph.txt`, a želimo kao izlaz dobiti datoteku `cwnd-graph.png`. To možemo ostvariti idućim nizom naredbi.

```
gnuplot> set terminal png size 640,480
gnuplot> set output "cwnd-graph.png"
gnuplot> plot "cwnd-data.txt" using 1:2 title 'Congestion Window' with linespoints
gnuplot> exit
```

## Dodatak: korištenje funkcionalnosti simulatora za crtanje grafova

Simulator ns-3 ima modul `stats` koji između ostalog omogućuje generiranje gnuplotovih ulaznih `.plt` datoteka na temelju nekih podataka simulacije. Da bi ga mogli koristiti, moramo uključiti zaglavlje `stats-module.h`.

``` c++
#include <ns3/stats-module.h>
```

Zatim stvaramo objekt `plot` tipa `Gnuplot` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d4/d96/classns3_1_1_gnuplot.html)) koji će biti korišten za stvaranje datoteke s uputama za crtanje; pritom navodimo željeno ime izlazne datoteke s `.png` ekstenzijom. Metodom `SetTitle()` postavljamo naslov grafa, a metodom `SetTerminal()` postavljamo tip izlazne datoteke (terminal u gnuplot terminologiji) koja će biti stvorena nakon pokretanja gnuplota na `.plt` datoteku.

``` c++
Gnuplot plot ("cwnd-plot.png");
plot.SetTitle ("Congestion Window Plot");
plot.SetTerminal ("png");
```

``` c++
plot.SetLegend ("Vrijeme u sekundama", "Veličina prozora zagušenja u bajtovima");
plot.AppendExtra ("set xrange [0:12]");
```

``` c++
Gnuplot2dDataset dataset;
dataset.SetTitle ("Congestion Window Size");
dataset.SetStyle (Gnuplot2dDataset::LINES_POINTS);
```

``` c++
// TODO zamijeni ovo
double x;
double y;

// Create the 2-D dataset.
for (x = -5.0; x <= +5.0; x += 1.0)
  {
    // Calculate the 2-D curve
    //
    //            2
    //     y  =  x   .
    //
    y = x* x;

    // Add this point.
    dataset.Add (x, y);
  }
```

``` c++
plot.AddDataset (dataset);
```

``` c++
std::ofstream plotFile ("plot2d.plt");
plot.GenerateOutput (plotFile);
plotFile.close ();
```

Gnuplot kao ulaz prima `.plt` datoteku koja definira koji će objekti biti nacrtani i na koji način. *U komandnoj liniji* u direktoriju gdje se nalazi `plot2d.plt` pokrećemo naredbu

``` shell
$ gnuplot plot2d.plt
```

čime u istom direktoriju nastaje datoteka `plot2d.png`, u kojoj se nalazi graf koji prikazuje promjenu veličine prozora zagušenja u ovisnosti o vremenu.
