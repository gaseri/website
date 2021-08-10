---
author: Ivan Ivakić, Vedran Miletić
---

# Programska osnova usmjerivača

## Arhitektura alata Click

Sama struktura Clicka bazira se na klasi `Master` koja sadrži, među ostalim dvije, nama najvažnije klase: `Router` i `RouterThread`. Klasa `Router` sadrži informacije o samoj konfiguraciji koja se dohvaća iz *.click* datoteka, a `RouterThread` stvara niti za svaku od pojedinih instanci klase `Router`. Pojednostavljeno rečeno, za svaki usmjerivač koji konfiguriramo i kreiramo stvara se programska nit koja upravlja njime (u slučaju kreiranja samo jednog usmjerivača stvara se samo jedna nit). Sastavne dijelove Clicka nazivamo elementima te ih možemo smatrati manjim jedinicama aplikacije koji imaju točno određeni zadatak.

U trenutku pokretanja simulacije događa se nekoliko stvari koje možemo kategorizirati u tri skupine:

- Inicijalizacija -- kreiranje osnovnih elemenata
- Konfiguriranje -- postavljanje specifičnih parametara
- Pokretanje -- stvaranje i pokretanje niti koja upravlja usmjerivačem

Programski kod koji to izvodi je oblika:

``` c++
static Router *router;

int
main(int argc, char **argv)
{
  click_static_initialize();// Inicijalizacija
  //...
  router = parse_configuration(router_file, file_is_expr, false, errh);// Konfiguriranje
  //...
  router->master()->thread(0)->driver(); // Pokretanje
}
```

Pogledajmo sada detaljnije svaku od točaka.

### Inicijalizacija

Pri pozivanju `click_static_initialize()` funkcije definiraju se tipovi parametara za elemente, parametri usmjerivača te se stvara podatkovna struktura sa nazivima i ostalim potrebnim informacijama o svim dostupnim elementima.

Kod koji ovaj redoslijed definira je vrlo jednostavan:

``` c++
cp_va_static_initialize();

Router::static_initialize();

click_export_elements();
```

Dakako ovo nije potpuni kod, ali je dovoljan za razumijevanje strukture i redoslijeda odvijanja događaja.

Detaljniji pogled na `cp_va_static_initialize()` i `Router::static_initialize()` nije potreban, no zanimljivo je pogledati što se događa pozivanjem `click_export_elements()` funkcije.

``` c++
#include <click/config.h>
#include <click/package.hh>
#include "../elements/standard/delayshaper.hh"
#include "../elements/threads/spinlockrelease.hh"
#include "../elements/test/listtest.hh"
//... (nastavak popisa elemenata) ...

beetlemonkey(uintptr_t heywood)
{
  switch (heywood) {
    case 0: return new AdjustTimestamp;
    case 1: return new AggregateCounter;
    case 2: return new AggregatePacketCounter;
    //... (ostali elementi) ...
    case 259: return new ToRawSocket;
    case 260: return new ToSocket;
    case 261: return new UMLSwitch;
    default: return 0;
  }
}

void
click_export_elements()
{
  (void) click_add_element_type_stable("AdjustTimestamp", beetlemonkey, 0);
  //... (ostali elementi) ...
  (void) click_add_element_type_stable("ToRawSocket", beetlemonkey, 259);
  (void) click_add_element_type_stable("ToSocket", beetlemonkey, 260);
  (void) click_add_element_type_stable("UMLSwitch", beetlemonkey, 261);
  CLICK_DMALLOC_REG("nXXX");
}
```

Definiramo dostupne elemente, uključujemo datoteke zaglavlja i dodjeljujemo redni broj svakom od elemenata. Funkcija `click_add_element_type_stable()` je zadužena da dodijeli ime svakom od elemenata i kreira ga, a ako pobliže pogledamo njenu realizaciju možemo vidjeti da se zapravo dodijeljeni parametri parsiraju u svrhu utvrđivanja postojanja i stablizacije samog koda.

``` c++
extern "C" int
click_add_element_type_stable(const char *ename,
                              Element *(*func)(uintptr_t),
                              uintptr_t thunk)
{
    assert(ename);
    if (Lexer *l = click_lexer())
        return l->add_element_type(String::make_stable(ename),
                                   func,
                                   thunk);
    else
        return -99;
}
```

Naime, funkcija `click_lexer()` (konstruktor objekta `l` tipa `Lexer`) dodavajući element u simulaciju ujedno provjerava postoji li i koji mu je redni broj.

### Konfiguracija

Konfiguriranje je bitan dio svake simulacije pa tako i simulacije usmjerivača. Zapravo sve bitno što možemo podešavati radimo upravo u dijelu koji zovemo konfiguriranje. Stoga, pogledajmo pobliže kod koji osigurava da je konfiguracija valjana te ju kreira/mijenja na usmjerivaču.

``` c++
static Router *
parse_configuration(const String &text, bool text_is_expr, bool hotswap,
                    ErrorHandler *errh)
{
  Master *master = (router ? router->master() : new Master(nthreads));
  Router *r = click_read_router(text, text_is_expr, errh, false, master);
  if (!r)
    return 0;

  //... (programski kod) ...

  return r;
}
```

Ova funkcija stvara konfigurirani usmjerivač na osnovu nekoliko parametara. Prvi parametar je uređeni niz znakova tipa `String` koji može biti ili ime datoteke u kojoj su konfiguracijski parametri ili sama konfiguracija, drugi parametar određuje kako će se interpretirati prvi parametar (na jedan od dva prethodno navedena načina), treći parametar označava mijenjamo li konfiguraciju, a četvrti je pokazivač na eventualno nastale greške.

Nakon toga provjeravamo postoji li definiran `router` te ukoliko postoji dohvaćamo, u prvom poglavlju spomenut, `master` element, a ukoliko ne postoji stvaramo novu instancu `Master` klase kojoj dodjeljujemo željeni broj niti.

Posljednje što je potrebno učiniti je pozvati funkciju `click_read_router()`. Sama ta funkcija izgleda ovako:

``` c++
Router *
click_read_router(String filename,
                  bool is_expr,
                  ErrorHandler *errh,
                  bool initialize,
                  Master *master)
{
    //... (1. dio koda) ...
    String config_str;
    if (is_expr) {
        config_str = filename;
        filename = "config";
    } else {
        config_str = file_string(filename, errh);
        if (!filename || filename == "-")
            filename = "<stdin>";
    }

    //... (2. dio koda) ...

    RequireLexerExtra lextra(&archive);
    int cookie = l->begin_parse(config_str, filename, &lextra, errh);
    while (l->ystatement())
        /* ne čini ništa */;
    Router *router = l->create_router(master ? master : new Master(1));
    l->end_parse(cookie);

    //... (3. dio koda)...

    if (initialize)
        if (errh->nerrors() > before || router->initialize(errh) < 0) {
            delete router;
            return 0;
        }

    return router;
}
```

Prvi dio koda osigurava da se ukoliko je zastavica `is_expr` postavljena zabilježi konfiguracijski string te postavi zadano ime datoteke na `config`, a ukoliko nije postavljena znači da je putanja do datoteke koja sadrži konfiguraciju pa ju je potrebno samo pročitati te osigurati da konfiguracija nije prazna (ukoliko je prazna koristi se predefinirana konfiguracija).

Drugi dio koda parsira samu konfiguraciju i kreira usmjerivač sa tim postavkama.

Treći dio koda provjerava je li sve prošlo u redu u drugom dijelu koda. Ukoliko je došlo do bilo kakve greške ili nepredviđenih rezultata usmjerivač se briše te se izlazi iz programa, a ukoliko je sve prošlo u redu u glavni program se vraća sam konfigurirani usmjerivač.

### Pokretanje

Ovo poglavlje pojasnit će što se događa u trenutku pokretanja našeg usmjerivača. Kada u glavnom kodu pozovemo metodu `driver()` na način da ju zovemo nad instancom klase `Router` kao što slijedi: `router->master()->thread(0)->driver()` zapravo pokrećemo sam aplikativni softver na usmjerivaču čime ga "uključujemo". Kod koji realizira to je:

``` c++
void
RouterThread::driver()
{
  const volatile int * const stopper = _master->stopper_ptr();
  int iter = 0;

  driver_loop:

    if (*stopper == 0) {

        iter++;

        _master->run_signals(this);

        if ((iter % _iters_per_os) == 0)
            run_os();

        bool run_timers = (iter % _master->timer_stride()) == 0;
        if (run_timers) {
            _master->run_timers(this);
        }
    }

    // izvrši prvi zadatak (1)
    if (_pending_head)
        process_pending();

    run_tasks(_tasks_per_iter);

    if (*stopper > 0) {
        driver_unlock_tasks();
        bool b = _master->check_driver();
        driver_lock_tasks();
        if (!b)
            goto finish_driver;
    }

    goto driver_loop;

  finish_driver:
    driver_unlock_tasks();
}
```

Ne objašnjavajući detaljno svaku od pojedinih linija koda recimo samo da se ovime pokreće usmjerivač te se zadaci koje obavlja vrte u beskonačnoj petlji simulirajući time stvarno ponašanje usmjerivača. Dakako postoji i mogućnost gašenja usmjerivača što činimo izlaskom iz petlje.

Samo izvršavanje određenog zadatka vrši se funkcijom `run_tasks()` čija je realizacija dana kako slijedi:

``` c++
inline void
RouterThread::run_tasks(int ntasks)
{
  Task *t;
  for (; ntasks >= 0; --ntasks) {
    t = task_begin();
    if (t == this)
      break;

      t->fast_remove_from_scheduled_list();

      _pass = t->_pass;

      t->_status.is_scheduled = false;
      t->fire();
  }
}
```

Izvršavanje određenog zadatka te pomicanje na sljedeći uz brisanje prethodnog zadatka iz niza zadataka koje usmjerivač treba obaviti. Još jedna važna metoda korištena u prethodnom bloku koda je `fire()` kojom zapravo "zakačimo" izvršavanje tog određenog zadatka za točno određeni element Clicka (podsjetimo se elementi su sastavnice koje obavljaju određenu funkciju).

## Dodatak: sučelje prema mrežnom simulatoru ns-3

Korištenje Click usmjerivača unutar ns-3 simulacije je prilično jednostavno zbog načina na koji su oba simulatora realizirana. Razlog zašto se Click implementira u ns-3 simulacije je mogućnost preciznijeg određivanja protokola na temelju kojeg će usmjerivač obrađivati pakete. Simulator ns-3 poznaje nekoliko osnovnih, standardnih modela te uglavnom ne dopušta izmjene njihove strukture bez izmjene programskog koda same implementacije, dok Click isključivo simulira usmjerivače te dopušta puno veću slobodu pri kreiranju protokola kojim će usmjeravati pakete.

### Način korištenja

Korištenje Click usmjerivača unutar ns-3 simulacije svodi se zapravo na pozivanje pomoćnika `ClickInternetStackHelper` te upućivanje simulacije na datoteku koja sadrži programski kod željenog Click usmjerivača

``` c++
ClickInternetStackHelper click;
click.SetClickFile (node, "mojUsmjerivac.click");
click.SetRoutingTableElement (node, "u/rt");
click.Install (nodes);
```

pri čemu je `nodes` tipa `NodeContainer`. Usporedimo li ovaj kod s uobičajenim korištenjem `InternetStackHelper`-a oblika

``` c++
InternetStackHelper stack;
stack.Install (nodes);
```

vidimo da je osnova ideja vrlo slična. Ostatak simulacije je identičan uobičajenoj ns-3 simulaciji.

### Primjeri simulacija

!!! todo
    Dodaj poveznice na dokumentaciju funkcija koje se koriste.

Simulacije kreiramo uobičajeno te je jedina razlika integracija nekog od Click usmjerivača unutar kreirane simulacije.  Ponovno koristimo jedan u nizu pomagača, ovaj puta je to `ClickInternetStackHelper` te koristimo tri metode nad instancom tog pomagača.

Prvo metodom `SetClickFile()` postavljamo određeni konfigurirani usmjerivač na određeni čvor. Ona prima dva parametra gdje je prvi parametar čvor na koji želimo postaviti usmjerivač, a drugi parametar je putanja do datoteke `.click` koja sadrži sam kod kojim je realiziran usmjerivač.

Potom metodom `SetRoutingTableElement()` postavljamo metodu rada, prvi parametar je ponovno čvor na koji želimo postaviti usmjerivač, a drugi je metoda rada gdje može biti kernel (`kernel/rt`) ili samostalni model rada (`u/rt`).

Nakon postavljanja pozivamo metodu `Install()` kojoj dodjeljujemo parametar kontejnera čvorova koje smo instalirali te time dovršavamo postavljanje našeg konfiguriranog Click usmjerivača na čvorove.

Primjerice, ako ponovno uzmemo da se `NodeContainer` zove `nodes` tada za instalaciju na prvi čvor imamo kod oblika

``` c++
ClickInternetStackHelper clickinternet;
clickinternet.SetClickFile (nodes.Get (0), "src/click/examples/nsclick-routing-node0.click");
clickinternet.SetRoutingTableElement (nodes.Get (0), "kernel/rt");
clickinternet.Install (nodes);
```

U širem kontekstu taj je kod oblika

``` c++
// Topologija
//
//              172.16.1.0/24
//        (1.1)  (1.2)  (2.1)  (2.2)
//
//         eth0   eth0  eth1    eth0
//       n0 ========= n1 ========= n2
//            LAN 1       LAN 2
//
// - UDP tokovi n0 do n2 preko n1.
// - Svi čvorovi koriste Click model usmjeravanja.
//

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/ipv4-l3-click-protocol.h"
#include "ns3/click-internet-stack-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NsclickRouting");

int
main (int argc, char *argv[])
{

//
// Uključivanje loginga za Click klijenta i servera
//
  LogComponentEnable ("NsclickRoutingClient", LOG_LEVEL_INFO);
  LogComponentEnable ("NsclickRoutingServer", LOG_LEVEL_INFO);

//
// Kreiranje čvorova
//
  NS_LOG_INFO ("Stvaranje čvorova.");
  NodeContainer n;
  n.Create (3);

//
// Instalacija Clicka na čvorove
//
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (n.Get (0), "src/click/examples/nsclick-routing-node0.click");
// datoteke su proizvoljne, ove su iz primjera unutar samog ns-3-a
  clickinternet.SetClickFile (n.Get (1), "src/click/examples/nsclick-ip-router.click");
  clickinternet.SetClickFile (n.Get (2), "src/click/examples/nsclick-routing-node2.click");
  clickinternet.SetRoutingTableElement (n.Get (0), "kernel/rt");
  clickinternet.SetRoutingTableElement (n.Get (1), "u/rt");
  clickinternet.SetRoutingTableElement (n.Get (2), "kernel/rt");
  clickinternet.Install (n);

  NS_LOG_INFO ("Stvaranje kanala.");
//
// Stvaranje kanala
//
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1400));
  NetDeviceContainer d01 = csma.Install (NodeContainer (n.Get (0), n.Get (1)));
  NetDeviceContainer d12 = csma.Install (NodeContainer (n.Get (1), n.Get (2)));

  Ipv4AddressHelper ipv4;
//
// Postavljanje IP adresa
//
  NS_LOG_INFO ("Dodjeljivanje IP adresa");
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i01 = ipv4.Assign (d01);

  ipv4.SetBase ("172.16.2.0", "255.255.255.0");
  Ipv4InterfaceContainer i12 = ipv4.Assign (d12);

  NS_LOG_INFO ("Stvaranje aplikacija.");
//
// Kreiranje UDP serverske aplikacije na čvoru 2
//
  uint16_t port = 4000;
  UdpServerHelper server (port);
  ApplicationContainer apps = server.Install (n.Get (2));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

//
// Kreiranje klijentske aplikacije na čvoru 0
//
  uint32_t MaxPacketSize = 1024;
  Time interPacketInterval = Seconds (0.05);
  uint32_t maxPacketCount = 320;
  UdpClientHelper client (i12.GetAddress (1), port);
  client.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  client.SetAttribute ("Interval", TimeValue (interPacketInterval));
  client.SetAttribute ("PacketSize", UintegerValue (MaxPacketSize));
  apps = client.Install (NodeContainer (n.Get (0)));
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

//
// Omogućavanje Pcap-a
//
  csma.EnablePcap ("nsclick-routing", d01, false);
  csma.EnablePcap ("nsclick-routing", d12, false);

//
// Pokretanje simulacije
//
  NS_LOG_INFO ("Pokreni simulaciju.");
  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Kraj.");
}
```

Ovaj primjer je složen prema datoteci `src/click/examples/nsclick-routing.cc` koja je dio izvornog koda mrežnog simulatora ns-3.
