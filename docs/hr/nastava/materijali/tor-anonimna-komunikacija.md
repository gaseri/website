---
author: Jan Božić, Vedran Miletić
---

# Anonimna komunikacija alatom Tor

Slabo je poznato da je danas na internetu putem poznatijih internet tražilica poput Google-a dohvatljivo samo 5% internetskog sadržaja. Ostatak internet sadržaja nalazi se u takozvanom dubokom webu (engl. *deep web*). Duboki web je dio interneta čiji sadržaj nije indeksiran internetskim tražilicama. Poseban dio dubokog weba je tamni web (engl. *dark web*), web nedostupan putem konvencionalnih mrežnih (web) pretraživača poput Chromea, Firefoxa, Internet Explorera itd. Kako bi pristupili tom "tamnome" dijelu interneta potreban nam je pretraživač Tor, kojega ćemo detaljnije objasniti u nastavku.

Osim samoga pristupa dubokome web-u, Tor nam pruža i zaštitu anonimnosti na čitavom webu kao i dodatnu sigurnost na tom djelu weba. Također treba napomenuti da postoji i tamni web kojega ne treba miješati sa dubokim web-om. Tamni web je šifrirana mreža koja postoji između Tor poslužitelja i njihovih klijenata, dok kod dubokoga web-a jednostavno sadržaj baza podataka i drugih web usluga iz nekoga razloga ne može biti indeksirana od strane konvencionalnih tražilicama.

Živimo u doba slobodnog toka podataka, gdje svaka osoba sa internetskom vezom ima naizgled sve informacije na dohvat ruke. Iako je internet uvelike proširio mogućnost dijeljenja znanja, također je donio sa sobom i velike pobleme u vezi privatnosti i sigurnosti na njemu. Mnogi su korisnici interneta zabrinuti što ih mogu pratiti ne samo vladine agencije, već i razne korporacije s ciljem isporučivanja raznih reklama i oglasa vezanih za interese korisnika. Može se reći da su "nevidljive oči" posvuda, te da se svaki pokret danas na internetu prati. U takvom ozračju razvio se preglednik Tor koji je brzo postao popularan i predmet rasprave. Poput mnogih novih pojava na internetu, ljudi ga ne razumiju dovoljno te je široj javnosti predstavljen kao vrsta tehnološke mistike. Uglavnom se povezuje sa hakerima i tamnim webom prepunim ilegalnim stvarima. Iako je Tor odličan preglednik za zaštitu anonimnosti te se može koristiti npr. kod otvorenih Wi-Fi mreža, nije ga preporučljivo koristiti za pristupanje zabranjenim sadržajima na tamnom webu. Neke su stvari sa razlogom cenzurirane i sakrivene od šire javnosti.

U nastavku ćemo objasniti što su duboki i tamni web, kako im pristupiti te kako se zaštiti na njima. Objasniti će se i preglednik Tor, kako je nastao, za što nam služi i što možemo sa njim učiniti.

## Površinski, duboki i tamni web

S obzirom na dostupnost sadržaja, web možemo podijeliti na dva dijela: [površinski web i duboki web](https://kompjuteras.com/sta-je-deep-dark-web-trebam-li-se-brinuti-kako-da-se-zastitim/). Duboki web je dio interneta čiji sadržaj nije dostupan putem konvencionalnih mrežnih (web) pretraživača. Površinski web je pojam koji označava sadržaj na Webu koji je lako pretraživ putem konvencionalnih Web pretraživača, kao što su Google, Bing, Yahoo! i sl. Konvencionalni Web pretraživači stvaraju svoje indekse pomoću robota (tzv. pauka) koji pretražuju Web u potrazi za novim informacijama. Roboti putuju od jedne do druge Web stranice koristeći se poveznicama među tim stranicama. Oni pritom indeksiraju sadržaj svake Web stranice na koju naiđu te pomoću tih indeksa poslije korisnici mogu pretraživati i pregledavati te stranice.

Michael K. Bergman smatra se da se 99% informacija nalazi u dubokom webu (detaljniji podaci se nalaze u radu [The Deep Web: Surfacing Hidden Value](https://quod.lib.umich.edu/j/jep/3336451.0007.104?view=text;rgn=main)), dok je samo 1% informacija dostupno putem konvencionalnih web pretraživača (površinski web). Anand Rajaraman je u svom radu [Kosmix: High-Performance Topic Exploration using the Deep Web](https://dl.acm.org/doi/abs/10.14778/1687553.1687581) to slikovito opisao izjavom da je ono što nam nude konvencionalni Web pretraživači samo vrh ledene sante.

Suprotno mišljenju, duboki web nije neko mistično mjesto kojega koriste black hat hakeri i ostali stručnjaci za ilegalne poslove. Sadržaj dubokoga weba može biti nešto trivijalno sa čime smo se svi susreli, poput komentara na forumu kojima mogu prisustvovati samo prijavljeni korisnici, Facebook objava koje su postavljene tako da ih mogu vidjeti samo prijatelji, privatni YouTube sadržaji kojima mogu pristupiti samo oni kojima je proslijeđen link i slično. Google kao najpoznatiji servis za pretraživanje interneta u svoju pretragu [uključuje samo 16% površinskog (surface) web-a i 0,03% ukupnog sadržaja dostupnog na internetu](https://www.tportal.hr/tehno/clanak/nevidljivi-internet-skriva-gomilu-zanimljivih-stvari-20131018).

## Tamni web

Tamni web je dio dubokog weba nedostupan putem konvencionalnog DNS-a, a ponekad se koristi za promociju ili distribuciju ilegalnih aktivnosti. Termin "tamni web" odnosi se na web stranice koje su javno dostupne ali imaju skrivene IP adrese servera na kojima se nalaze, zbog toga je teško pronaći tko je vlasnik pojedine stranice. Skoro sve web stranice na tamnom webu koriste Tor enkripciju, pa se tako tim stranicama može pristupiti samo preko Tor web preglednika. Svaka ta stranica ima specifičnu domenu `.onion` kojoj se može pristupiti samo preko Tor preglednika. .onion domenama, za razliku od [.com](https://hr.wikipedia.org/wiki/.com), [.org](https://hr.wikipedia.org/wiki/.org), [.net](https://hr.wikipedia.org/wiki/.net), [.hr](https://hr.wikipedia.org/wiki/.hr) i ostalih poznatijih domena, je teško saznati tko je vlasnik, a zbog načina na koji Tor radi, teško je otkriti i gdje se server točno nalazi.

Razlika tamnoga weba i dubokoga weba je u tome što vlasnici stranica u tamnom webu [namjerno sakrivaju svoje stranice od javnosti](https://www.techadvisor.com/how-to/internet/dark-web-3593569/). Najpoznatiji primjer web stranice na tamnom webu je [Silk Road](https://www.wired.com/2015/04/silk-road-1/), aukcijska stranica poput eBaya za prodaju ilegalnih stvari poput droga, oružja, itd. [Studija Dr. Garetha Owena sa Sveučilišta u Portsmouthu iz 2014. godine](https://www.cigionline.org/publications/tor-dark-net) pokazala je da je većina sadržaja na tamnom webu dječja pornografija, aukcije za crno tržište te botnet serveri. Zbog same prirode takvih tog sadržaja, većina je i namjerno sakrivena od javnosti kako bi zaštitili obične korisnike koje takvi sadržaji ne zanimaju.

## Povijest Tora

Tor je slobodni softver, internet pretraživač koji omogućavanje anonimnu komunikaciju. Samo ime Tor je skraćenica izvedena iz imena izvornoga softverskog projekta zvanog [The Onion Router](https://support.torproject.org/about/why-is-it-called-tor/).

Prilikom bilo kakve vrste komunikacije na internetu, informacija o poslužitelju i klijentu poznata je svim prijenosnim uređajima na putu između poslužitelja i klijenta, odnosno njihovim vlasnicima. Kako je internet nastao na ideji da se komunikacija ostvaruje putem paketa koji od polazišta do odredišta mogu putovati različitim putevima, tako se višestruko multipliciraju uređaji i njihovi vlasnici koji imaju informaciju o komunikaciji između klijenta i poslužitelja. Kako se ove informacije odnose na mrežni i transportni sloj, jednostavno je odrediti o kojoj se osobi radi.

Povijest Tor-a započinje 1998. godine kada [Ratna mornarica SAD-a](https://www.navy.mil/) (engl. *US Navy*) razvija Onion routing protocol koji štiti privatnost komunikacije između pošiljatelja i primatelja.
Iz toga razloga razvijen je [protokol Onion routing](https://www.onion-router.net/) koji obavlja enkripciju i deskripciju paketa na svakom pojedinom prijenosnom uređaju čime se u potpunosti gubi informacija o primatelju, pošiljatelju i sadržaju samog paketa, tj. informacija je zaštićena i u potpunosti anonimna.

Sam razvoj Tor preglednika započela je tvrtka Syverson sa računalnim znanstvenicima Rogerom Dingledinom i Nickom Mathewsonom koji su [objavili prvu alfa verziju 20. srpnja 2002. godine](https://yashalevine.com/files/tor-spooks-by-yasha-levine-pando-quarterly-summer-2014.pdf). Službeno je Tor predstavljen javnosti [13. kolovoza 2004. godine na konferenciji USENIX Security](https://www.usenix.org/legacy/events/sec04/tech/dingledine.html). Iste godine Tor je pušten pod licencom otvorenoga koda te ga je počeo financirati Dingledine & Mathewson zaklada kako bi se Tor nastavio razvijati.

Zanimljivost je da se Tor projekt danas uglavnom financira iz sredstava Vlade SAD-a. TechCrunch je u travnju 2017. godine objavio kako je [financiranje projekta Tora u 2015. godini naraslo na 3.3 milijuna dolara, od čega je 86% iz raznih vladinih izvora](https://techcrunch.com/2017/04/25/tor-project-funding-3-3-million-2015/). U ranije dvije godine taj udio bio je 89% (od 2.5 milijuna dolara u 2014. godini) i 95% (u 2013. godini).

## Način rada Tora

[Dva su ključna aspekta](https://www.digitaltrends.com/computing/a-beginners-guide-to-tor-how-to-navigate-through-the-underground-internet/) koji utječu na način kako funkcionira proces Onion routing usmjeravanja. Kao prvo, Tor mreža se sastoji od korisnika-volontera koji su pristali poslužiti svoja računala kao čvorove u mreži. Na internetu pri normalnom surfanju informacije putuju u paketima od čvora do čvora odnosno servera. Za razliku od običnih korisnika, korisnicima sa Tor preglednikom paketi ne putuju izravno do određeno servera, već Tor stvara sam put nasumičnih čvorova koji će paket slijediti prije nego li dođe do odredišnog servera.

Drugi bitni aspekt je način kako su paketi konstruirani. Normalni paket sadrži adresu pošiljatelja i odredište, dok paket stvoren putem Tora je "umotan" u slojeve paketa. Kada korisnik šalje paket, sloj na vrhu usmjerava ga da ide na router A koji je ujedno i prva destinacija u mreži. Sljedeći sloj kaže routeru A da ga šalje na sljedeći router B. Router A ne zna konačnu destiniaciju, samo da paket dolazi od korisnika i da mora otići na router B. Potom router B "ljušti" sljedeći sloj te vidi da je sljedeće odredište paketa router C. Proces se nastavlja sve dok paket ne stigne na svoje odredište. Na svakoj "stanici" paketa, čvor zna samo posljednje mjesto gdje je paket bio te gdje će sljedeći paket biti, te nitko ne zna cijelu putanju paketa.

## Konfiguracija Tor čvora

Tor čvor (engl. Tor relay) je server u Tor mreži koji prima i šalje pakete unutar nje. Svaki korisnik Tor preglednika može podesiti da njegovo računalo bude jedan od tih čvorova. Važno je napomenuti da brzina pregledavanja sadržaja putem Tor-a je uglavnom puno sporija od standardnih pregleda iz razloga što brzina ovisi od brzini ostalih servera. Postoje [tri vrste Tor relaya](https://community.torproject.org/relay/types-of-relays/), a to su: middle relay, exit relay i bridge.

U Tor mreži promet paketa prolazi kroz najmanje tri čvora prije nego li stignu na odredište. Middle relay ili središnji čvor se sastoji od prva dva čvora te on prenosi promet samo između njih. Iako je middle relay vidljiv svim korisnicima u Tor mreži kako bi se što lakše i brže spojili na njega, vrlo je siguran te se nikako ne može vidjeti server i IP adresa sa kojeg je hostan. On je relativno siguran za hostanje od strane svi korisnika iz privatnih mreža pa je baš iz toga razloga on odbran kao praktični dio koji će se pokazati u ovome radu. S druge strane exit relay ili izlazni čvor je posljednji čvor u Tor mreži prije nego li paketi stignu na svoje odredište odnosno zadnji server u nizu. Također kao i middle relay, exit relay je vidljiv svim korisnicima unutar Tor mreže kako bi mu mogli pristupiti. Pošto je to izlazni čvor njegova IP adresa je prikazana kao izvor. Tu dolazi do opasnosti zbog korisnika koji rade ilegalne stvari te onda IP exit relay-a može poslužiti kao dokaz za tužbu i pravne sankcije. Iz tog razloga nije preporučljivo za obične korisnike imai otvoren exit relay, već se on uglavnom nalazi na serverima koji služe isključivo za to.

Most (engl. *bridge*) je čvor koji [nije javno naveden kao dio Tor mreže](https://support.torproject.org/censorship/censorship-7/). Bridgevi se uglavnom koriste u zemljama se koje imaju cenzure i blokiraju razne IP adrese poput Kine. Kao takva Kina je jedna od zemalja u kojima je Tor dosta popularan, ali ne za pristupanje dubokom i tamnom webu već za obične stranice poput Facebook-a koji je zabranjen od strane države. Korištenjem bridgeva još je teže otkriti tko je iza Tor preglednika.

Tor radi kao usluga operacijskog sutava. Pokrećemo ga naredbom:

``` shell
$ sudo systemctl start tor.service
```

Tor će tada započeti sakupljati podatke o relejima, što možemo vidjeti u izvještaju statusa:

``` shell
$ systemctl status tor.service
I learned some more directory information, but not enough to build a circuit: We have no usable consensus.
Bootstrapped 10%: Loading relay descriptors
Bootstrapped 20%: Loading relay descriptors
...
Bootstrapped 80%: Connecting to the Tor network
Bootstrapped 85%: Finishing handshake with first hop
Bootstrapped 90%: Establishing a Tor circuit
Tor has successfully opened a circuit. Looks like client functionality is working.
Bootstrapped 100%: Done
```

Kada *bootstrapping* dođe do [100 posto](https://youtu.be/pZerJ_egEmY) moguće je ostvarivati veze korištenjem `torsocks`-a.

Postupak postavljanja middle relay-a je vrlo jednostavan, potrebno je:

1. Preuzeti i instalirati Tor pomoću naredbe: apt-get install tor
1. Locirati i editirati torrc datoteku koja se nalazi na /etc/tor/torrc
1. Potrebno je dodati ORPort 9001 i ExitPolicy reject *:*
1. Nakon toga je još potrebno samo restartati Tor naredbom: sudo service tor restart
1. Za provjeru da li proces uspješno obavljen potrebno je pronaći log datoteku u /var/log/tor/ te ukoliko u datoteci piše:

    ```
    Self-testing indicates your ORPort is reachable from the outside. Excellent.
    ```

    znači da je sve u redu.

## Anonimnost Tora

U konačnici, obzirom na veliku količinu računalnih i ljudskih resursa koje su vlada SAD-a i drugih država ulažu u sigurnost, razumno je sumnjati u anonimnost Tora.

- Istraživanje završeno u lipnju 2015. godine [propituje povjerenje u izlazne čvorove Tora](https://nakedsecurity.sophos.com/2015/06/25/can-you-trust-tors-exit-nodes/).
- Tadašnji direktor FBI-a James Comey je u rujnu 2015. godine [izjavio sljedeće o korisnicima Tora koji putem istoga gledaju dječju pornografiju](https://theintercept.com/2015/09/10/comey-asserts-tors-dark-web-longer-dark-fbi/): "They'll use the onion router to hide their communications. They think that if they go to the dark web … that they can hide from us. They're kidding themselves, because of the effort that's been put in by all of us in the government over the last five years or so, that they are out of our view."
- [Yasha Levine](https://yashalevine.com/) je u veljači 2018. godine [provjerio poveznice između projekta Tor i vlade SAD-a](https://surveillancevalley.com/blog/fact-checking-the-tor-projects-government-ties) i zaključio: "The Tor Project, a private non-profit that underpins the dark web and enjoys cult status among privacy activists, is almost 100% funded by the US government." Levine otvara detaljnu analizu s: ["CLAIM #1: Tor does not provide backdoors to the U.S. government. RATING: Moderately true."](https://surveillancevalley.com/blog/claim-tor-does-not-provide-backdoors-to-the-u-s-government)
