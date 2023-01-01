---
author: Ivan Ivakić, Vedran Miletić
---

# Simulacijski modeli mrežnih aplikacija kao generatora prometa

U ovom poglavlju koristit ćemo jednostavan model stvarnog ponašanja mrežnih aplikacija koji je poznat pod nazivom on-off aplikacija. Uočimo da većina mrežnih aplikacija radi tako da određeni vremenski period šalje podatke, a onda određeni vremenski period miruje. Primjerice, aplikacija pregledavanje web stranica kao što je [Firefox](https://en.wikipedia.org/wiki/Firefox) može učitati web stranicu (*on* period), a zatim "mirovati" (barem što se mrežnog prometa tiče) dok korisnik pregledava web stranicu (*off* period). Ti periodi zatim alterniraju, a njihova vremena trajanja mogu varirati slučajno ili prema nekom pravilu.

Stvarne aplikacije koje rade po ovom pravilu mogu koristiti UDP ili TCP. User Datagram Protocol (UDP) nije pouzdan te se stoga koristi u onim prijenosima u kojima nije od presudne važnosti primiti svaki pojedini paket. Pakete koje UDP šalje nazivamo datagramima. U zaglavlju sadrže sve informacije potrebne da paket bude dostavljen procesu na odredištu, ali pritom ne postoji mehanizam kojim se potvrđuje primitak već je taj dio ostavljen višim slojevima za implementaciju u slučaju da ga trebaju. UDP koristi mehanizam slanja koji ne uspostavlja konekciju niti provjerava prije slanja je li odredišni čvor aktivan. Prvenstvena namjena UDP-a je osigurati što brži prijenos uz eventualnu nepouzdanost samog prijenosa. UDP podržava načine slanja multicast i broadcast; ovaj prvi koristi se, primjerice, kod emitranja televizije putem Interneta.

Transmission Control Protocol (TCP) ostvaruje pouzdan prijenos sadržaja time što se potvrđuje primitak samo onih segmenata koji su stigli na odredište u ispravnom stanju. Drugim riječima, ne potvrđuje primitak paketa koji su na putu iskrivljeni (ili izgubljeni) te se time zahtjeva da se ti segmenti ponovno pošalju. TCP time ispravlja greške nastale u prijenosu. U slučaju da je paket uspješno primljen šalje se potvrda (acknowledgment, kraće ACK) pošiljatelju paketa koji tada pouzdano zna da je sadržaj koji je poslao uspješno primljen točno u tom obliku u kojem je poslan. Ovaj protokol koristi se za sve prijenose u kojima je od presudne važnosti da poslana poruka stigne u cijelosti i točno takva kakva je poslana na odredište. Pouzdan prijenos nužan je primjerice za slanje e-maila, preuzimanje sadržaja sa web stranica, rad na udaljenom računalu, preuzimanje datoteka i slično. Bitno je naglasiti da neki od prijenosa podataka koriste vlastite protokole viših slojeva za preciznije definiranje prijenosa i načina prikaza sadržaja, kao što su primjerice HTTP (za web stranice) ili SMTP (za e-mail), no to ne mijenja činjenicu da se na nižem sloju prijenos odvija TCP-om.

Klasa `OnOffApplication` koja je dio simulatora ns-3 može se koristiti UDP ili TCP kao protokol transportnog sloja. Za određivanje trajanja on i off perioda on-off aplikacije koriste se slučajne varijable.

U području teorije vjerojatnosti jedan od osnovnih pojmova je slučajna varijabla. Kako u trenutnom studijskom programu kolegij Vjerojatnost i statistika slijedi nakon Računalnih mreža, ovdje ćemo definirati i koristiti slučajne varijable na vrlo ograničen način dovoljan za naše potrebe. Možemo reći da je slučajna varijabla ona varijabla čija vrijednost ovisi o slučajnosti.

Ono što određuje vrste slučajnih varijabli je *učestalost pojavljivanja* određenih brojeva. Ograničit ćemo se na korištenje dvije najjednostavnije vrste slučajnih varijabli, konkretno:

- `ConstantRandomVariable`, koja vraća uvijek isti "slučajan" broj (ima atribut `Constant` kojim se taj broj postavlja),
- `UniformRandomVariable`, koja vraća slučajan broj iz zadanog intervala tako da se u prosjeku svi brojevi iz tog intervala pojavljuju podjednako često (ima atribute `Min` i `Max` kojim se ograničava interval).

## UDP on-off aplikacija

Ponovno ćemo analizirati kod liniju po liniju. Zaglavlja

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;
```

identična su kao u prethodnom primjeru.

S druge strane, uočimo da su linije

``` c++
LogComponentEnable ("OnOffApplication", LOG_LEVEL_INFO);
LogComponentEnable ("PacketSink", LOG_LEVEL_INFO);
```

različite; naime, kako ćemo ovdje koristiti on-off aplikaciju i odvod za pakete umjesto UDP echo klijenta i poslužitelja, linije su promijenjene te uključujemo loging za `OnOffApplication` te `PacketSink`. Općenito, ovaj mehanizam služi da bi ispisivao poruke određenih komponenata simulatora ns-3; kada bi ns-3 ispisivao *sve* poruke koje simulacija stvara, čak i najjednostavnija simulacija imala bi izlaz reda veličine nekoliko tisuća redova.

Stvaramo jednostavnu topologiju kao i ranije.

``` c++
NodeContainer nodes;
nodes.Create (2);

PointToPointHelper pointToPoint;
pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

NetDeviceContainer devices;
devices = pointToPoint.Install (nodes);

InternetStackHelper stack;
stack.Install (nodes);

Ipv4AddressHelper address;
address.SetBase ("10.1.1.0", "255.255.255.0");

Ipv4InterfaceContainer interfaces;
interfaces = address.Assign (devices);
```

`PacketSinkHelper` je pomoćnik koji stvara "odvod" (engl. sink) za pakete, odnosno stvara aplikaciju koja će primati pakete i biti svojevrsni "poslužitelj" za "klijentsku" on-off aplikaciju. Za razliku od poslužiteljske aplikacije, odvod neće slati nikakav odgovor on-off aplikaciji već će sve što stigne na odgovarajuću adresu i vrata biti odbačeno ("pušteno u odvod").

`PacketSinkHelper` imena `sink` kod stvaranja prima 2 parametra; prvi parametar je znakovni niz, u ovom slučaju "ns3::UdpSocketFactory", koji označava protokol koji on-off aplikacija koristi. Preciznije, definira na koji će način biti stvorena mrežna utičnica (engl. *network socket*) te aplikacije, odnosno koja će se "tvornica" koristiti (odakle i *factory* u nazivu).

Drugi parametar je mrežna adresa na kojoj će paketi biti pušteni u odvod. Obzirom da se očekuje adresa procesa, ona se sastoji od IPv4 adrese i vrata; u ns-3-u je ta kombinacija tip `InetSocketAddress`. Prvi dio dohvaćamo metodom `GetAddress()` iz objekta interfaces koji smo ranije kreirali, odnosno dohvaćamo IPv4 adresu drugog čvora, a drugi dio je broj vrata na kojima će aplikacija primati pakete (u našem slučaju je to 9).

Nakon toga u kontejner za aplikacije imena `apps` instaliramo taj odvod, i definiramo vremena početka i kraja rada. Ovdje smo stavili da odvod počinje s radom sekundu prije pripadne on-off aplikacije i završava sekundu nakon da stignu i budu primljeni svi paketi koji su eventualno poslani u posljednjem trenutku.

``` c++
PacketSinkHelper sink ("ns3::UdpSocketFactory", InetSocketAddress (interfaces.GetAddress (1), 9));
ApplicationContainer apps = sink.Install (nodes.Get (1));
apps.Start (Seconds (1.0));
apps.Stop (Seconds (11.0));
```

`OnOffHelper` je pomoćnik koji služi za stvaranje on-off aplikacije. Kao i `PacketSinkHelper` prima dva parametra od kojih je prvi protokol, a drugi mrežna adresa tipa `InetSocketAddress` **na koju će aplikacija slati podatke**, dakle adresa odredišta.

Pomoću metode `SetAttribute()` kao i do sada postavljamo različite atribute. Atributi koje postavljamo su redom

- `DataRate`, odnosno količina prometa u jednici vremena koju aplikacija stvara kada je aktivna,
- `PacketSize`, odnosno veličina paketa koju aplikacija stvara izražena u bajtovima,
- `OnTime` i `OffTime`, odnosno trajanje vremena (izraženo u sekundama) u kojem je aplikacija aktivna i trajanje vremena u kojem je neaktivna (respektivno).

`OnTime` i `OffTime` su atributi koji imaju donekle specifičnu sintaksu, koja ovisi od vrste slučajne varijable koju će on-off aplikacija koristiti. Primjer za `ConstantRandomVariable` i `UniformRandomVariable` demonstriramo u nastavku.

``` c++
OnOffHelper onOffApp ("ns3::UdpSocketFactory", InetSocketAddress (interfaces.GetAddress (1), 9));
onOffApp.SetAttribute ("DataRate", StringValue ("1Mbps"));
onOffApp.SetAttribute ("PacketSize", UintegerValue (2048));
onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));
```

Dakle, ovim kodom stvorena je aplikacija koja u vremenu kada je aktivna šalje pakete veličine 2048 bajta brzinom 1 Mbit/s. Period aktivnosti traje uvijek točno 2 sekunde, nakon čega slijedi period neaktivnosti koje traje najmanje 1 sekundu, a najviše 3 sekunde.

Instalaciju on-off aplikacije vršimo na isti način kao i instalaciju UDP echo aplikacije.

``` c++
ApplicationContainer clientApps = onOffApp.Install (nodes.Get (0));
clientApps.Start (Seconds (2.0));
clientApps.Stop (Seconds (10.0));
```

Važno je spomenuti da on-off aplikacija nakon početka rada ima prvo period neaktivnosti.

Ovo je dovoljno da bi aplikacija radila. Međutim, iskoristit ćemo još jednu metodu objekta tipa `PointToPointHelper`, a to je `EnableAsciiAll()` koja čini da se stvaraju tekstualne datoteke koje sadrže izvještaj o paketima u simulaciji. Ta metoda prima niz znakova koji će biti prefiks imena datoteka koje će stvoriti.

Važno je da se ova metoda pozove **nakon** svih poziva `Install()` metode nad objektom `pointToPoint`, što je najlakše napraviti tako da pozovete metodu neposredno prije samog pokretanja simulacije sa `Simulator::Run()`. Time se izbjegava mogućnost nepravovremenog poziva metode koja u tom slučaju neće generirati ništa ili će pak generirati pogrešan izvještaj.

Datoteka koju generira ova metoda bit će smještena u direktoriju gdje se događa kompajliranje vašeg programskog koda. U slučaju da ste kao odredište spremanja datoteka projekta odabrali vaš kućni direktorij (u virtualnoj mašini to je `/home/student`), a projekt nazvali ga `rm2-vj2-primjer1`, direktorij u kojem se nalazi AsciiTracing datoteka zove se `rm2-vj2-primjer1-build-<verzija Qt biblioteke i program prevoditelja>`. Taj direktorij, sadrži izvršnu datoteku vašeg projekta, datoteke s objektnim kodom (`*.o`) i AsciiTracing datoteke koje imaju prefiks u imenu koji ste zadali. Naravno, pretpostavka je da ste prethodno pokrenuli izvršnu datoteku koja će stvoriti AsciiTracing datoteke (upravo zbog toga što ih stvara izvršna datoteka simulacije AsciiTracing datoteke se i nalaze u tom direktoriju).

Imena AsciiTracing datoteka generirana su prema uzorku `<odabrani niz znakova>-X-Y.tr`, pri čemu je `X` je identifikator čvora, a `Y` identifikator mrežnog sučelja na čvoru.

U sadržaju datoteke svaki redak predstavlja jedan paket. Stupci su standardizirani i redom su:

- Prvi stupac sastoji se od jednog znaka koji označava što se dogodilo sa paketom. Mogućnosti su:

    - `r`: Receive (primljen)
    - `+`: Enqueue (dodavanje u red čekanja)
    - `-`: Dequeue (uklanjanje iz reda čekanja)

- Drugi stupac je vrijeme u kojem se promatrani događaj nad paketom dogodio i izraženo je u sekundama.
- Treći stupac je putanja do događaja koji se dogodio, koja uključuje identifikatore i tipove uređaja. Ova putanja može se koristiti i na drugim mjestima u ns-3-u, ali se time detaljnije nećemo baviti na ovom kolegiju. Svi nazivi su prilično intuitivni, a više detalja možete pronaći u [dijelu ns-3 Manuala koji govori o atributima](https://www.nsnam.org/docs/manual/html/attributes.html).

- Četvrti i ostali stupci su niz tipova zaglavlja (primjerice, `ns3::PppHeader`, `ns3::Ipv4Header` i `ns3::UdpHeader`), korisnog dijela (`Payload`) i eventualno repova za pakete koji ih imaju. Svaki od tipova ima u zagradama navedene neke od informacija koje su specifične za protokol.

``` c++
pointToPoint.EnableAsciiAll ("vjezba-udp-on-off-app");
```

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

int main ()
{
  LogComponentEnable ("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("PacketSink", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);
  pointToPoint.EnableAsciiAll ("vjezba-udp-on-off-app");

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces;
  interfaces = address.Assign (devices);

  PacketSinkHelper sink ("ns3::UdpSocketFactory", InetSocketAddress (interfaces.GetAddress(1), 9));
  ApplicationContainer apps = sink.Install (nodes.Get (1));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (11.0));

  OnOffHelper onOffApp ("ns3::UdpSocketFactory", InetSocketAddress (interfaces.GetAddress (1), 9));
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

## TCP on-off aplikacija

TCP on-off aplikacija radi na vrlo sličan način kao UDP on-off aplikacija. Ovdje ćemo pojasniti dio koda koji različit.

`PacketSinkHelper` ima kao prvi parametar znakovni niz "ns3::TcpSocketFactory". Ostalo je isto kao kod UDP aplikacije.

``` c++
PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (interfaces.GetAddress(1), 9));
ApplicationContainer apps = sink.Install (nodes.Get (1));
apps.Start (Seconds (1.0));
apps.Stop (Seconds (11.0));
```

`OnOffHelper` ima kao prvi parametar "ns3::TcpSocketFactory". Ostalo je isto kao kod UDP aplikacije.

``` c++
OnOffHelper onOffApp ("ns3::TcpSocketFactory", InetSocketAddress (interfaces.GetAddress (1), 9));
onOffApp.SetAttribute ("DataRate", StringValue ("1Mbps"));
onOffApp.SetAttribute ("PacketSize", UintegerValue (2048));
onOffApp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=2.0]"));
onOffApp.SetAttribute ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));
```

Uzgred budi rečeno da kada bi sami pisali programsku logiku aplikacije, odnosno kada bi radili na nižoj razini i sami implementirali povezivanje putem TCP-a, korištenje UDP-a ili TCP-a uzrokovalo bi da postoje određene razlike. Međutim, kako ovdje radimo na relativno visokoj razini apstrakcije ns-3 simulator se umjesto nas brine o tom dijelu.

Radi preciznosti, promijenit ćemo i ime datoteke za AsciiTrace izlaz. To nije bilo potrebno, obzirom da je ime proizvoljno.

``` c++
pointToPoint.EnableAsciiAll ("vjezba-tcp-on-off-app");
```

Cjelokupan kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

int main ()
{
  LogComponentEnable ("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("PacketSink", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);
  pointToPoint.EnableAsciiAll ("vjezba-tcp-on-off-app");

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

## Dodatak: usporedba AsciiTrace datoteka koje stvaraju UDP i TCP on-off aplikacije

Usporedimo li AsciiTrace datoteke koje stvaraju UDP i TCP on-off aplikacije (dakle, datoteke `vjezba-udp-on-off-app-0-0.tr` i `vjezba-tcp-on-off-app-0-0.tr` te `vjezba-udp-on-off-app-1-0.tr` i `vjezba-tcp-on-off-app-1-0.tr`), vidjet ćemo da među njima postoji razlika.

!!! todo
    Ovdje je potrebno opisati razliku i prikazati na primjeru kako izgleda.

## Dodatak: složenije vjerojatnosne razdiobe i aplikacijsko programsko sučelje slučajne varijable

!!! warning
    Za razumijevanje ovog dijela potrebno je poznavati gradivo kolegija Vjerojatnost i statistika.

Pored klasa `ConstantRandomVariable` i `UniformRandomVariable`, ns-3 definira još nekoliko klasa koje implementiraju slučajne varijable:

- `DeterministicRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d9/d77/classns3_1_1_deterministic_random_variable.html)), koja vraća unaprijed brojeve iz niza koji je zadan kod stvaranja,
- `ErlangRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d3/d6d/classns3_1_1_erlang_random_variable.html)), koja vraća brojeve prema [Erlangovoj razdiobi](https://en.wikipedia.org/wiki/Erlang_distribution),
- `ExponentialRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/da/d69/classns3_1_1_exponential_random_variable.html)), koja vraća brojeve prema [eksponencijalnoj razdiobi](https://en.wikipedia.org/wiki/Exponential_distribution),
- `GammaRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d9/dcd/classns3_1_1_gamma_random_variable.html)), koja vraća brojeve prema [gama razdiobi](https://en.wikipedia.org/wiki/Gamma_distribution),
- `LogNormalRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d1/d92/classns3_1_1_log_normal_random_variable.html)), koja vraća brojeve prema [logaritmiranoj normalnoj razdiobi](https://en.wikipedia.org/wiki/Log-normal_distribution),
- `NormalRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/da/d67/classns3_1_1_normal_random_variable.html)), koja vraća brojeve prema [normalnoj (Gaussovoj) razdiobi](https://en.wikipedia.org/wiki/Normal_distribution),
- `ParetoRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d0/df0/classns3_1_1_pareto_random_variable.html)), koja vraća brojeve prema [Paretovoj razdiobi](https://en.wikipedia.org/wiki/Pareto_distribution) (često se koristi u simulaciji stvarnog mrežnog prometa),
- `SequentialRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/df/d31/classns3_1_1_sequential_random_variable.html)), koja vraća brojeve iz niza koji monotono raste za dani period počevši od dane početne vrijednosti, a kada vrijednost postane veća od ograničenja počinje ponovno od dane početne vrijednosti,
- `TriangularRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/d8/d9b/classns3_1_1_triangular_random_variable.html)), koja vraća brojeve prema [trokutastoj razdiobi](https://en.wikipedia.org/wiki/Triangular_distribution),
- `WeibullRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/dd/d52/classns3_1_1_weibull_random_variable.html)), koja vraća brojeve prema [Weibullovoj razdiobi](https://en.wikipedia.org/wiki/Weibull_distribution) (često se koristi u proračunima pouzdanosti mrežnih sustava),
- `ZetaRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/dd/d78/classns3_1_1_zeta_random_variable.html)), koja vraća brojeve prema [zeta razdiobi](https://en.wikipedia.org/wiki/Zeta_distribution),
- `ZipfRandomVariable` ([dokumentacija](https://www.nsnam.org/docs/doxygen/db/d00/classns3_1_1_zipf_random_variable.html)), koja vraća brojeve prema [Zipfovoj razdiobi](https://en.wikipedia.org/wiki/Zipf's_law).

Sve one nasljeđuju klasu `RandomVariableStream` i imaju isto sučelje kao `ConstantRandomVariable` i `UniformRandomVariable` osim parametara koje je potrebno dati kod stvaranja obzirom da oni ovise o samoj vjerojatnosnoj razdiobi. Zbog toga možete jednostavno zamijeniti dosad korištene slučajne varijable za `OnTime` i `OffTime` on-off aplikacije nekim drugim po vlastitom izboru i vidjeti na koji način one utječu na broj poslanih paketa.

U simulatoru ns-3, svaka slučajna varijabla ima metodu `GetValue()` koja kod poziva vraća slučajan realan broj i metodu `GetInteger()` koja kod poziva vraća slučajan cijeli broj. Iako metode `GetValue()` i `GetInteger()` nećemo koristiti, radi potpunosti je ovdje dan kod koji definira dvije slučajne varijable `var1` i `var2`. Vrijedi spomenuti da će kod dati isti rezultat kod višestrukog pokretanja zbog načina na koji je "slučajnost" implementirana u simulatoru ns-3, o čemu više možete pročitati u [dijelu koji govori o slučajnim varijabla u ns-3 Manualu](https://www.nsnam.org/docs/manual/html/random-variables.html).

``` c++
#include <ns3/core-module.h>
#include <iostream>

using namespace ns3;
using namespace std;

int main()
{
  Ptr<RandomVariableStream> var1 = CreateObject<ConstantRandomVariable> ();
  var1->SetAttribute ("Constant", DoubleValue (5.0));
  cout << var1->GetValue () << endl; // vraća 5.0
  cout << var1->GetInteger () << endl; // vraća 5.0
  Ptr<RandomVariableStream> var2 = CreateObject<UniformRandomVariable> ();
  var2->SetAttribute ("Min", DoubleValue (0.0));
  var2->SetAttribute ("Max", DoubleValue (10.0));
  cout << var2->GetValue () << endl;  // vraća realan broj čija je vrijednost između 0.0 i 10.0
  cout << var2->GetInteger () << endl; // vraća cijeli broj čija je vrijednost između 0.0 i 10.0
  return 0;
}
```

## Dodatak: pojmovi nasljeđivanja i apstraktne klase

U objektno orijentiranom programiranju postoji koncept nasljeđivanja; drugim riječima, omogućuje se da klasa (koju nazivamo *podklasom*) dobije sve ili dio atributa i metoda druge klase (koju nazivamo *nadklasom*). To je vrlo korisno, jer smanjuje duplikaciju koda.

Nadgradnja toga je uključuje koncept *apstraktne klase*; pod tim nazivom podrazumijevamo klasu koja će biti korištena isključivo kao nadklasa, i deklarira koje funkcije sve podklase koje je nasljeđuju moraju definirati. Na taj način se omogućuje da se sa svim podklasama određene apstraktne klase radi na isti način  (obzirom da one imaju metode koje primaju i vraćaju iste tipove podataka), što bitno olakšava rad kod većeg broja klasa. O svemu tome više ćete čuti na kolegiju Objektno orijentirano programiranje.

U konkretnom slučaju, `RandomVariableStream` je primjer apstraktne klase, a `ConstantRandomVariable` i `UniformRandomVariable` su njene podklase.

## Dodatak: programiranje vlastite mrežne aplikacije u simulatoru

!!! warning
    Za razumijevanje ovog dijela potrebno je poznavati gradivo kolegija Objektno orijentirano programiranje.

U ovom dijelu ćemo napraviti vlastitu aplikaciju, u dosadašnjoj terminologiji "on" aplikaciju. Za razliku od on-off aplikacije, ova nema "off" period pa je nešto jednostavnija. Bez obzira, demonstirati će neka od sučelja koja je potrebno koristiti za programiranje vlastitih aplikacija. Ono što je olakotna okolnost je da su sučelja u simulatoru vrlo slična onima u stvarnim operacijskim sutavima, a sama programska logika mrežne aplikacije potpuno ista, te je lako znanje stečeno u jednom području primijeniti na ono drugo.

Naša jednostavna aplikacija raditi će tako da će slati određen broj paketa određene veličine, na određenu adresu. Kada završi s time, neće zakazivati dodatne događaje i čekati će kraj simulacije. Kako je ns-3 sustav zasnovan na C++ klasama, za stvaranje takve aplikacije potrebno je poznavati osnovne objektno orijentiranog programiranja.

Simulator ns-3 implementira osnovne značajke aplikacije u klasi `Application`. Naša aplikacija, nazvana jednostavno `MojaAplikacija` ,nasljeđuje klasu `Application`.

``` c++
class MojaAplikacija : public Application
{
```

U dijelu klase koji je `public` deklariramo konstruktor, destruktor i funkciju `Setup()` tipa void koja podešava aplikaciju, a kao argumente prima redom socket koji će aplikacija koristiti, adresu na koju će slati podatke, veličinu paketa koji će slati, broj paketa i brzinu slanja.

``` c++
public:

  MojaAplikacija ();
  virtual ~MojaAplikacija ();

  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);
```

U dijelu klase koji je `private` deklariramo virtualne metode `StartApplication()` i `StopApplication()`, koje služe za pokretanje i zaustavljanje aplikacije (respektivno). Zatim deklariramo metodu `ScheduleTx()`, koja se pokreće nakon svakog slanja i zakazuje iduće slanje, i metodu `SendPacket()`, koja šalje paket.

Pored metoda, deklarirani su i idući atributi:

- `m_socket`, socket koji aplikacija koristi,
- `m_peer`, adresa na koju se paketi šalju,
- `m_packetSize`, veličina paketa koji se šalju,
- `m_nPackets`, broj paketa koje će aplikacija poslati,
- `m_dataRate`, brzina kojom će aplikacija slati pakete,
- `m_sendEvent`, identifikator događaja koji opisuje iduće slanje,
- `m_running`, je li aplikacija pokrenuta,
- `m_packetsSent`, broj poslanih paketa.

``` c++
private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);

  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
};
```

U konstruktoru postavljamo vrijednosti svih atributa na zadane.

``` c++
MojaAplikacija::MojaAplikacija ()
  : m_socket (0),
    m_peer (),
    m_packetSize (0),
    m_nPackets (0),
    m_dataRate (0),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0)
{
}
```

U destruktoru pokazivač na socket postavljamo na vrijednost 0, kako bi se dealokacija memorije koju je taj socket objekt zauzeo mogla uspješno izvršiti kad bude potrebno.

``` c++
MojaAplikacija::~MojaAplikacija ()
{
  m_socket = 0;
}
```

Metoda `Setup()` postavlja vrijednosti atributa `m_socket`, `m_peer`, `m_packetSize`, `m_nPackets` i `m_dataRate` na vrijednosti koje joj se proslijede.

``` c++
void
MojaAplikacija::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
}
```

Metoda `StartApplication()` pokreće aplikaciju: postavlja vrijednost atributa koji kaže je li aplikacija pokrenuta na `true`, broj poslanih paketa na 0, radi `Bind()` na socketu (dobiva adresu i vrata s kojih će slati pakete), radi `Connect()` na socketu (povezuje se s drugom stranom), i zatim pokreće slanje paketa metodom `SendPacket()`.

``` c++
void
MojaAplikacija::StartApplication (void)
{
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind ();
  m_socket->Connect (m_peer);
  SendPacket ();
}
```

Metoda `StopApplication()` zaustavlja aplijkaciju: postavlja vrijednost atributa koji kaže je li aplikacija pokrenuta na `false`, a zatim otkazuje zadnji događaj slanja paketa ako je još aktivan, i zatvara socket ako postoji.

``` c++
void
MojaAplikacija::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}
```

Metoda `SendPacket()` šalje paket; prvo stvara paket veličine koja je određena kod podešavanja aplikacije, a zatim na socketu pokreće metodu `Send()`. Ako je tada broj poslanih paketa manji od ukupnog broja paketa koji će se poslati, zakazuje iduće slanje.

``` c++
void
MojaAplikacija::SendPacket (void)
{
  Ptr<Packet> packet = Create<Packet> (m_packetSize);
  m_socket->Send (packet);

  if (++m_packetsSent < m_nPackets)
    {
      ScheduleTx ();
    }
}
```

Metoda `ScheduleTx()` zakazuje slanje idućeg paketa. U slučaju da je aplikacija pokrenuta, vrijeme slanja idućeg paketa $t_{next}$ određuje se po formuli

$$
t_{next} = \frac{packetSize \times 8}{dataRate},
$$

pri čemu je $packetSize$ veličina paketa u bajtovima, i množimo je s 8 kako bi dobili veličinu u bitovima, a $dataRate$ brzina slanja izražena u bitovima po sekundi. Zatim se metodom `Simulator::Schedule()` o kojoj ćemo više reći u idućj vježbi za taj trenutak zakazuje slanje paketa, odnosno pokretanje metode `MojaAplikacija::SendPacket()` na objektu `this`, odnosno samoj toj aplikaciji. Vrijednost koju metoda `Simulator::Schedule()` vraća je identifikator zakazanog događaja i sprema se u varijablu `m_sendEvent`.

``` c++
void
MojaAplikacija::ScheduleTx (void)
{
  if (m_running)
    {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &MojaAplikacija::SendPacket, this);
    }
}
```

Time je implementirana cjelokupna aplikacijska logika. Unutar `main()` funkcije, `PacketSink` za našu aplikaciju napraviti ćemo na isti način kao za `OnOffApplication`.

``` c++
uint16_t sinkPort = 8080;
PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
ApplicationContainer sinkApps = packetSinkHelper.Install (nodes.Get (1));
sinkApps.Start (Seconds (0.));
sinkApps.Stop (Seconds (20.));
```

Zatim stvaramo socket (koji može biti TCP ili UDP socket, ali mora biti istog tipa kao socket za pripadni `PacketSink`), i stvaramo aplikaciju tipa `MojaAplikacija`. Na njoj pokrećemo metodu `Setup()`, a zatim je dodajemo na prvi čvor i podešavamo vrijeme početka i završetka.

``` c++
Ptr<Socket> ns3TcpSocket = Socket::CreateSocket (nodes.Get (0), TcpSocketFactory::GetTypeId ());
Ptr<MojaAplikacija> app = CreateObject<MojaAplikacija> ();
app->Setup (ns3TcpSocket, InetSocketAddress (interfaces.GetAddress (1), sinkPort), 1024, 500, DataRate ("1Mbps"));
nodes.Get (0)->AddApplication (app);
app->SetStartTime (Seconds (1.));
app->SetStopTime (Seconds (20.));
```

Cjelokupni kod primjera je

``` c++
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>

using namespace ns3;

class MojaAplikacija : public Application
{
public:

  MojaAplikacija ();
  virtual ~MojaAplikacija ();

  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);

  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
};

MojaAplikacija::MojaAplikacija ()
  : m_socket (0),
    m_peer (),
    m_packetSize (0),
    m_nPackets (0),
    m_dataRate (0),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0)
{
}

MojaAplikacija::~MojaAplikacija ()
{
  m_socket = 0;
}

void
MojaAplikacija::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
}

void
MojaAplikacija::StartApplication (void)
{
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind ();
  m_socket->Connect (m_peer);
  SendPacket ();
}

void
MojaAplikacija::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}

void
MojaAplikacija::SendPacket (void)
{
  Ptr<Packet> packet = Create<Packet> (m_packetSize);
  m_socket->Send (packet);

  if (++m_packetsSent < m_nPackets)
    {
      ScheduleTx ();
    }
}

void
MojaAplikacija::ScheduleTx (void)
{
  if (m_running)
    {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &MojaAplikacija::SendPacket, this);
    }
}

int main ()
{
  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.252");
  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  uint16_t sinkPort = 8080;
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
  ApplicationContainer sinkApps = packetSinkHelper.Install (nodes.Get (1));
  sinkApps.Start (Seconds (0.));
  sinkApps.Stop (Seconds (20.));

  Ptr<Socket> ns3TcpSocket = Socket::CreateSocket (nodes.Get (0), TcpSocketFactory::GetTypeId ());
  Ptr<MojaAplikacija> app = CreateObject<MojaAplikacija> ();
  app->Setup (ns3TcpSocket, InetSocketAddress (interfaces.GetAddress (1), sinkPort), 1024, 500, DataRate ("1Mbps"));
  nodes.Get (0)->AddApplication (app);
  app->SetStartTime (Seconds (1.));
  app->SetStopTime (Seconds (20.));

  pointToPoint.EnableAsciiAll ("vjezba-vlastita-tcp-app");

  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
```
