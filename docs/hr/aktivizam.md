---
author: Vedran Miletić
---

# Aktivizam

> The world is open, keep it open; it's up to us.

-- [Paul Cormier, 2012](https://youtu.be/tkz8WQ_a6Yk?t=31m33s)

U nastavku su detaljno opisane teme koje su relevantne za naš sadašnji aktivistički rad prema većoj slobodi u tehnologiji (uključujući slobodni softver otvorenog izvora i otvoreni hardver). Informiranje o temama uključuje praćenje glavnih, alternativnih i specijaliziranih izvora vijesti koji pokrivaju tehnologiju, društvo, digitalnu privatnost, autorska prava i slično.

## Uvjerenja i vrijednosti

Ideje, uvjerenja, pretpostavke i vrijednosti su važni jer utječu na to kako se bavimo znanošću i kako razvijamo tehnologiju. [Peter Thiel](http://zerotoonebook.com/) navodi u [The Education of a Libertarian](https://www.cato-unbound.org/2009/04/13/peter-thiel/education-libertarian/):

> The future of technology is not pre-determined, and we must resist the temptation of technological utopianism -- the notion that technology has a momentum or will of its own, that it will guarantee a more free future, and therefore that we can ignore the terrible arc of the political in our world.
>
> A better metaphor is that we are in a deadly race between politics and technology. The future will be much better or much worse, but the question of the future remains very open indeed. We do not know exactly how close this race is, but I suspect that it may be very close, even down to the wire. Unlike the world of politics, in the world of technology the choices of individuals may still be paramount.

Kao i [Red Hat](https://www.redhat.com/en/about/company), vjerujemo da je naša misija biti dio zajednice koja stvara bolju tehnologiju na [način otvorenog izvora](https://opensource.com/open-source-way). I znanje i izvorni kod softvera (koji je sam po sebi oblik znanja) su sami po sebi neoskudni resursi: jednom stvoreni, mogu se beskonačno kopirati po želji vrlo jeftino ili gotovo besplatno. Vjerujemo da bi ovo svojstvo znanja trebalo koristiti za maksimiziranje koristi koje ono donosi društvu.

Snažno podržavamo različite inicijative u otvorenom licenciranju, otvorenom pristupu znanstvenim rezultatima, otvorenim standardima i otvorenim patentima. Često govorimo o [problemima koji okružuju autorska prava](https://falkvinge.net/topic/era/old-world/copyright-monopoly/), uključujući [Digital Millenium Copyright Act](https://www.eff.org/issues/dmca) i njegove europske ekvivalente, i [ograničenjima softverskih patenata](https://www.eff.org/issues/patents). Za ilustraciju, u članku [What is the price of open-source fear, uncertainty, and doubt?](../en/blog/2015-09-14-what-is-the-price-of-open-source-fear-uncertainty-and-doubt.md), Vedran Miletić je dao odgovor na članak [What Is the Price of Open-Source Software?](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b01258), stajalište koje su objavili Anna I. Krylov, John M. Herbert, Filipp Furche, Martin Head-Gordon, Peter J. Knowles, Roland Lindh, Frederick R. Manby, Peter Pulay, Chris-Kriton Skylaris i Hans-Joachim Werner u časopisu *The Journal of Physical Chemistry Letters*. Slične ideje nudi i [Dr. Christoph R. Jacob](https://blog.christophjacob.eu/about/), profesor s Tehničkog sveučilišta u Braunschweigu, u svom članku [How Open Is Commercial Scientific Software?](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b02609) također objavljenom u časopisu *The Journal of Physical Chemistry Letters*.

## Teme

### Promocija paketa uredskih alata LibreOffice i otvorenog formata dokumenata OpenDocument

#### Slobodni i neslobodni paketi uredskih alata

Paket uredskih alata (tekst procesor, tablični kalkulator i alat za izradu prezentacija) spada među najkorištenije softverske paketa na računalu. Vladine organizacije, tvrtke, neprofitne organizacije, akademska zajednica, zdravstvene ustanove i mnogi drugi sustavi ovise o paketima uredskih alata za svakodnevno izvršavanje vlastitih zadaća; možemo reći da su ti alati toliko korišteni da se o njima ne razmišlja, štoviše uzima ih se zdravo za gotovo. Najpopularniji paket uredskih alata za desktop računala je [Microsoft Office](https://en.wikipedia.org/wiki/Microsoft_Office), a uz njega vrijedi spomenuti još dva alata: [Apple iWork](https://en.wikipedia.org/wiki/IWork) (Pages, Numbers, Keynote), namijenjen korisnicima macOS-a, i [LibreOffice](https://www.libreoffice.org/), namijenjen korisnicima Windowsa, Linuxa i macOS-a. Microsoft Office i Apple iWork su vlasnički softveri, a LibreOffice je slobodni softver otvorenog koda čijim razvojem upravlja [The Document Foundation](https://www.documentfoundation.org/).

Kako raširenost interneta omogućuje suradnju velikog broja programera, njihov rad tijekom sredine 90-ih i kasnije dovodi u mainstream softvere kao što su [Mozilla Firefox](https://www.mozilla.org/firefox/), LibreOffice, Linux, [HTTP poslužitelj Apache](https://httpd.apache.org/), [VLC media player](https://www.videolan.org/vlc/) i brojne druge slobodne softvere otvorenog koda. Zajednička razlika svih ovih softvera sa softverima koje nudi Microsoft je ta da ih stvatko može bez ograničenja dijeliti, mijenjati i prilagođavati vlastitim potrebama, te dijeliti prilagođene verzije drugima. Tom otvorenošću se izbjegava ovisnost korisnika softvera o samo jednom proizvođaču softvera, takozvani *vendor lock-in*. Kako je kod otvoren, kompanije se ne mogu natjecati na temelju vlasništva nad softverom, već isključivo na temelju kvalitete podrške i prilagodbe softvera zahtjevima korisnika.

Specifično u domeni paketa uredskih alata Microsoft ne uspijeva razumjeti i iskoristiti potrebu programera i korisnika za otvorenošću. Steve Ballmer, CEO Microsofta od 2000. do 2014. godine, pokazuje nerazumijevanje svojom izjavom da je Linux ["rak koji se veže u smislu intelektualnog vlasništva na sve što takne"](https://www.theregister.com/2001/06/02/ballmer_linux_is_a_cancer/). Ono što on opisuje je licencija GNU General Public License (GPL), koju među ostalim projektima otvorenog koda koristi i Linux. GPL je viralna licencija, koja zahtijeva od svakog novog softvera koji iskorištava postojeći softver pod GPL-om da i sam bude pod GPL-om. Vremenom tako sve veći broj softvera postaje slobodni softver otvorenog koda, odnosno postoji sve više tehnologije dostupne svima za korištenje i prilagodbu u skladu s vlastitim potrebama.

Ključan alat u procesu zamjene svih neslobodnih softvera slobodnim je uredski paket LibreOffice, nasljednik projekta OpenOffice.org. U javnoj upravi navode se brojni primjeri [migracije s uredskog paketa Microsoft Office na OpenOffice.org ili LibreOffice](https://wiki.documentfoundation.org/LibreOffice_Migrations). Neki od primjera su grad Limerick u Irskoj, bolnice u gradu Kopenhagenu u Danskoj, grad Las Palmas u Španjolskoj, grad Largo u saveznoj državi Florida u SAD-u, grad München u Njemačkoj, grad Toulouse u Francuskoj i Ministarstvo obrane u Italiji. Procjene ušteda variraju od nekoliko desetaka tisuća do nekoliko desetaka milijuna eura, ovisno o broju računala i specifičnostima ugovora s Microsoftom koji je prethodio migraciji.

#### Situacija u Hrvatskoj

U Republici Hrvatskoj ukupan trošak Microsoft licencija koji se ugovara putem Državnog ureda za središnju javnu nabavu iznosi [približno 200 milijuna kuna za tri godine](https://www.linuxzasve.com/potpisivanje-200-milijuna-kuna-vrijednog-sporazuma-s-microsoftom-evo-sto-kazu-najkompetentniji-ljudi-na-tom-podrucju). Implementacija slobodnih softvera otvorenog koda u državnoj upravi i javnim institucijama razmatrana je u nekoliko navrata, međutim zbog brojnih prepreka zasad nije došlo do realizacije. Jedna od većih prepreka su vlastite aplikacije koji rade isključivo u Windowsima i/ili Internet Exploreru. Svakako veseli činjenica da je format OpenDocument, otvoreni format koji konkurira vlasničkim Microsoft Office formatima, [postao hrvatska norma s oznakom HRN ISO/IEC 26300:2008](https://www.linuxzasve.com/odf-postao-nacionalna-norma).

Za razliku od implementacije u državnoj upravi, implementacija slobodnih softvera otvorenog koda u školama i u akademskoj zajednici ovisi o entuzijazmu nastavnika. Brojni fakulteti u Republici Hrvatskoj već koriste Linux samostalno ili u kombinaciji s Windowsima, od kojih su neki PMF -- Matematički odsjek, Tehnički fakultet, Odjel za informatiku, Odjel za fiziku i Odjel za biotehnologiju Sveučilišta u Rijeci. Znanstveni softveri koje ove institucije koriste u istraživanju i nastavi su uglavnom specifični za granu znanosti kojom se one bave, a ponekad i uvjetuju korištenje određenog operacijskog sustava.

Što se tiče osnovnih i srednjih škola, vrijedi spomenuti Prirodoslovnu i grafičku školu Rijeka gdje se nastava predmeta Baze podataka, Programiranje za web i Internetske tehnologije izvodi na Linuxu, zahvaljujući inicijativi [Barbare Smilović](http://barbara-smilovic.from.hr/), prof. mat. i inf. Korištenje uredskih paketa ne obrađuje se posebno na tim predmetima, ali se LibreOffice koristi usputno. Ova praksa slijedi ideju da se učenike ne treba učiti na specifično sučelje, već na određenu funkcionalnost koju može očekivati od softvera i snalaženje u sučeljima koja su izgledom različita, ali idejom vrlo slična.

Voditeljima promocije za sada nije poznat primjer srednje škole koja je zamijenila Microsoft Office za LibreOffice na predmetima na kojima se obrađuje rad u paketima uredskih alata.

#### Ciljevi i postupak promocije

[Primjer školskog okruga Penn Manor iz savezne države Pennsylvania u SAD-u](https://youtu.be/Nj3dGK3c4nY) pokazuje da inicijativa može ići i odozdo. Taj okrug već dvije godine studentima daje laptope s Ubuntu Linuxom i LibreOfficeom, te ih uči radu u navedenom softveru. Korištenjem slobodnog softvera štede na licencijama i -- što nije manje važno -- podižu svijest o slobodnom korištenju softvera i potiču individualnu kreativnost u rješavanju problema. Učenik se dovodi u poziciju da je dio zajednice koja stvara softver jer može pomoći poboljšanju softvera koji koristi, bez obzira na znanje programiranja. Primjerice, može poboljšati dokumentaciju ili dizajnirati novi skup ikona. Učenik više nije samo konzument softvera koji nalazi u školi, već može postati aktivni sudionik u njegovu razvoju.

Često se ova praksa propituje s obzirom na to da još uvijek na desktop računalima Microsoft Windows i Office koristi najveći broj korisnika. Međutim, učenik koji danas kreće u školu nakon završetka svog obrazovanja koristit će softvere koji će izgledom biti bitno drugačiji od onoga što je učio. No ti će softveri funkcionalnošću biti vrlo slični, vjerojatno i bogatiji. Učenik stoga treba biti spreman snaći se za koji god softver da sjedne; grafička sučelja na Linuxu, macOS-u i Windowsima funkcionalnošću su danas ionako vrlo slična. Razlike u izgledu (dizajnu) koje postoje među njima ne smiju biti prepreka korištenju.

LibreOffice i Linux nastavit će napredovati, a način i načela razvoja slobodnog softvera širit će se u druge vrste softvera i izvan softvera. Želimo li naše učenike pripremiti za takav svijet, alati su nam dostupni, a sve što nam treba je inicijativa odozdo. S tim na umu, i [poučeni dosadašnjim iskustvima](http://zsv-inf.skole.hr/?news_hk=1&news_id=86&mshow=290), voditelji promocije planiraju sljedeće:

- održavanje prezentacija LibreOfficea i načela slobodnog softvera otvorenog koda na Županijskim stručnim vijećima profesora informatike i računalstva,
- individualne konzultacije putem e-maila, društvenih mreža i/ili IRC-a s nastavnicima i učenicima zainteresiranim za LibreOffice i općenito slobodni softver otvorenog koda,
- izrada kratkih video materijala na hrvatskom jeziku koji predstavljaju koncepte slobodnog softvera otvorenog koda,
- izrada dužih video i/ili tekstualnih uputa na hrvatskom jeziku za netrivijalne operacije u LibreOfficeu koje se obrađuju u srednjoškolskim programima,
- održavanje radionica za nastavnike putem interneta ili u realnom svijetu koje bi služile za razmjenu iskustava i prikupljanje ideja za poboljšanje prethodne četiri točke.

### Linux kao igraća platforma

Uvjerenja smo da je najvažniji razlog što Linux nije najrašireniji operacijski sustav nedovoljna promidžba Linux rješenja. Na tome radimo kroz čitav niz predavanja i prezentacija rješenja zasnovanih na slobodnom softveru.

Tehnički, Linux je superioran svemu drugome što postoji na tržištu u većini primjena. U zadnje vrijeme, kroz distribucije prilagođene krajnjim korisnicima, postao je dovoljno jednostavan za svakoga. To želimo pokazati, prije svega budućim profesorima, inženjerima i voditeljima informatičkih odjela u tvrtkama koji su danas studenti fakulteta Sveučilišta u Rijeci, ali i svim drugim zainteresiranima.

Relativno je lako doći u poziciju da koristite Linux za sve osim za igranje igara. Zahvaljujući zajednicama kao što su [GamingOnLinux](https://www.gamingonlinux.com/) i [/r/linux_gaming](https://www.reddit.com/r/linux_gaming/), bazama kao što su [SteamDB](https://steamdb.info/) i [ProtonDB](https://www.protondb.com/) te popularnosti [Valve Steam Decka](https://www.steamdeck.com/), igranje igara na Linuxu postaje realna mogućnost. Stoga je legitimno postaviti pitanje postoji li uopće potreba za korištenjem neslobodnih operacijskih sustava kao što je Microsoft Windows.

#### Windows ni na jednom osobnom računalu

(Ideja je kopirana i prilagođena sa [Stop Disabling SELinux](https://stopdisablingselinux.com/), javne usluge [Majora Haydena](https://major.io/).)

Microsoftova originalna vizija *osobnog računala na svakom stolu, Windowsa na svakom osobnom računalu*, u velikoj se mjeri ostvarila. Bez sumnje, uspjeh osobnih računala i Windowsa omogućio je da se dogode mnoge velike stvari, ali to vrijeme je prošlo. Sada je vrijeme da se ide dalje.

Danas je Windows, uključujući tehnologije poput DirectX-a, samo još jedan naslijeđeni vlasnički softver koji živi zahvaljujući [zaključavanju dobavljača, uključujući Microsoftovu kontrolu nad ekosustavom Secure Boot](../en/blog/2016-01-30-i-am-still-not-buying-the-new-open-source-friendly-microsoft-narrative.md). Uz [moderne DirectX igre koje su ekskluzivne za Windows Store](https://www.theguardian.com/technology/2016/mar/04/microsoft-monopolise-pc-games-development-epic-games-gears-of-war) i [Microsoft uništava zajednice oko Androida i Linuxa patentnim licencama](https://www.infoworld.com/article/3042699/microsoft-loves-open-source-only-when-its-convenient.html), situacija će se u budućnosti samo pogoršavati.

Svaki put kada koristite Windows za igranje računalnih igara, rasplačete [Gabea](https://gaben.tv/) [Newella](http://gabenewell.org/). On je naš veliki gospodar i to sigurno ne zaslužuje.

Zaustavimo dominaciju Windowsa zajedno. Koristite Linux ili [SteamOS](https://store.steampowered.com/steamos/), prijavite pogreške u igrama koje igrate i upravljačkim programima hardvera koje koristite, popravite stvari koje znate i pomozite drugima da učine isto. Odbijte kupiti igre koje ne rade na Linuxu ili SteamOS-u.

Otvorimo dalje osobna računala. Neka vizija budućnosti osobnog računarstva bude **Windows ni na jednom osobnom računalu**.

### Programski jezici i program prevoditelji otvorenog koda

Intenzivno promoviramo Python i njegove dodatke. Poštujemo individualni izbor programskog jezika, ali uvjereni smo da je Python odlična alternativa zastarjelim programskim jezicima kao što su Pascal i Visual Basic, i zato se može koristiti u edukaciji. Pored toga, Python može biti alternativa i Fortranu i raznom znanstvenom softveru (uz MatPlotLib, SciPy, NumPy), C#-u, Javi i drugim jezicima.

### Slobodni softveri za slovoslagarstvo

Smatramo da je TeX (LaTeX, ConTeXt) bolji nego bilo koji drugi program iste namjene. Jedan od pokazatelja te činjenice je i to što je to jedini rašireni program koji je kompatibilan sa samim sobom već gotovo 30 godina, a da se pritom aktivno razvija i proširuje.

Želimo prije svega potaknuti primjenu TeX-a kod studenata, za pisanje seminarskih, završnih i diplomskih radova zbog njegovih prednosti kod slaganja teksta i formula. Iako nam to nije primarni cilj, želimo promovirati TeX i kao alternativu DTP programima, programima za vektorsku grafiku i izradu prezentacija.

### Načela otvorenog izvora van softvera

Laboratoriji kao što je [Bradner Lab](https://web.archive.org/web/20161024182501/http://www.bradnerlab.org/), projekti kao što je [Open Source Malaria](http://opensourcemalaria.org/) i [Open Source Biotechnology](https://opensourcebiotech.anu.edu.au/) čine avangardu u ovom kretanju prema otvorenosti izvora izvan softvera. Lijepo je vidjeti proširenje principa besplatnog softvera otvorenog koda na farmaciju i biotehnologiju. Naravno, to je tek početak i ima još mnogo posla.

Izvan softvera, patenti su puno važniji nego u softveru (gdje je autorsko pravo primarno sredstvo zaštite intelektualnog vlasništva). Srećom, već postoji praksa pod nazivom [Patentleft](https://en.wikipedia.org/wiki/Patentleft) opisana kao:

> Patentleft (also patent left, copyleft-style patent license) is the practice of licensing patents (especially biological patents) for royalty-free use, on the condition that adopters license related improvements they develop under the same terms. Copyleft-style licensors seek "continuous growth of a universally accessible technology commons" from which they, and others, will benefit.
>
> Patentleft is analogous to copyleft, a license which allows distribution of a copyrighted work and derived works, but only under the same terms.

Ove ideje su zakonski implementirane u [Defensive Patent License](https://www.defensivepatentlicense.org/), a u praksi ih koristi [Tesla](https://www.tesla.com/). Elon Musk, izvršni direktor Tesle, [izjavio je sljedeće](https://www.tesla.com/blog/all-our-patent-are-belong-you):

> Technology leadership is not defined by patents, which history has repeatedly shown to be small protection indeed against a determined competitor, but rather by the ability of a company to attract and motivate the world's most talented engineers. We believe that applying the open source philosophy to our patents will strengthen rather than diminish Tesla's position in this regard.

## Aktivnosti

Naše aktivnosti možete pratiti na stranicama [Riječke podružnice Hrvatske udruge Linux korisnika (HULK Rijeka)](https://www.ri.linux.hr/tag/openclass/) i [Odjela za informatiku Sveučilišta u Rijeci](https://www.inf.uniri.hr/znanstveni-i-strucni-rad/predavanja-i-radionice/open-class).

### Predavanje: Od Aleksandrijske knjižnice do programskih knjižnica na GitHubu

**Predavač:** v. pred. dr.sc. Vedran Miletić, Odjel za informatiku Sveučilišta u Rijeci

**Mjesto:** O-403, [zgrada sveučilišnih odjela Sveučilišta u Rijeci](https://www.openstreetmap.org/way/436306129) i [YouTube kanal Festival Znanosti](https://www.youtube.com/channel/UCtl8E53TxHXJpFqJDrpcIdA)

**Vrijeme:** 13. listopada 2020. u 14:00

**Događaj:** [Otvoreni dan sveučilišnih odjela 2020](https://www.inf.uniri.hr/znanstveni-i-strucni-rad/dogadanja-info/otvoreni-dan)

#### Sažetak predavanja "Od Aleksandrijske knjižnice do programskih knjižnica na GitHubu"

Aleksandrijska knjižnica bila je jedna od najvećih i najvažnijih knjižnica antičkog svijeta. Točan broj svitaka papirusa koje je knjižnica imala nije poznat, ali procjenjuje se na desetke pa čak i stotine tisuća u doba kad je knjižnični fond bio najveći. Uništenje Aleksandrijske knjižnice često se pogrešno doživljava kao iznenadan događaj, dok je zapravo u pitanju bilo višestoljetno polagano propadanje. Već za dinastije Ptolemejevića provedena je čistka intelektualaca iz Aleksandrije, a zatim pod vlašću Rima opada financiranje i podrška knjižnici te s vremenom i brojnost njenih članova. Uništena je i naposlijetku srušena odredbom pape Teofila I. Aleksandrijskog 391. godine, no današnja Bibliotheca Alexandrina služi istovremeno kao spomen i kao pokušaj obnove. Stoljeće nakon uništenja Aleksandrijske knjižnice slijedi pad Zapadnog Rimskog Carstva i pogoršanje stanja u Zapadnoj Europi u demografskim, kulturološkim i ekonomskim terminima, doba koje Francesco Petrarca naziva mračnim, nasuprot svijetlom dobu klasične antike.

Današnje knjižnice pohranjuju podatke u digitalnom obliku pa naizgled postoji neograničena mogućnost pohrane rezervnih kopija i time manja bojazan da će pohranjeni podaci biti uništeni. Ipak, digitalizacija nosi svoje probleme: kroz vrijeme se mijenjaju formati pohranjenih dokumenata, softver koji te formate čita i zapisuje te hardver na kojem se taj softver izvodi, a promjene nisu uvijek takve da zadržavaju kompatibilnost s tehnološkim nasljeđem formata, softvera i hardvera. Pored toga, sami podaci se mogu korumpirati, bilo zbog nenamjernih oštećenja medija na kojem su zapisani ili grešaka softvera koji na njima radi, bilo zbog ciljanog djelovanja zlonamjernih aktera. Uzimajući u obzir te probleme digitalizacije, teoretičari digitalnog knjižničarstva uvode pojam digitalnog mračnog doba koje bi uslijedilo u slučaju gubitka kulturološkog nasljeđa koje je digitalizirano. Prilikom govora o podacima pohranjenim u knjižnicama često se ograničavamo na multimedijski sadržaj: tekst, slike, zvuk i video. Međutim, softver koji svakodnevno koristimo također je skup podataka, konkretno podataka u kojima zapisani postupci kao što su obrada slika u digitalnom obliku, prikaz web stranica, emitiranje komprimiranog audiovizualnog sadržaja visoke rezolucije, izračun energije vezanja molekula, predviđanje putanje planeta oko Sunca, abecedno nizanje popisa učenika i slično. Štoviše, u računalnim se znanostima datoteke u koje se spremaju implementacije tih različitih postupaka nazivaju (programskim) knjižnicama. Zahvaljujući uspjehu slobodnog softvera otvorenog kôda posljednjih desetljeća, izvorni je kôd velikog broja računalnih programa i programskih knjižnica dostupan bilo kome za proučavanje i izmjenu, a razvijen je i komercijalno uspješan sustav za pohranu, dijeljenje, označavanje i pretragu izvornog kôda poznat pod nazivom GitHub. Kako znanost teži biti otvorena, značajan dio tih programa su programi koje znanstvenici koriste u istraživačkom radu i onda dijele s ostalima kako bi primili povratnu informaciju i na temelju nje u budućnosti poboljšali postupke koje koriste. GitHub tako sprema znanje o poznatim postupcima u raznim domenama znanosti i zbog toga služi kao knjižnica suvremenog doba.

Predavanje će govoriti o načinima pohrane znanja o tehnologiji koja podupire suvremenu kulturu i civilizaciju te rizicima koje takva pohrana povlači, kao i postupcima koji bi mogli dovesti do digitalnog mračnog doba.

#### Životopis predavača "Od Aleksandrijske knjižnice do programskih knjižnica na GitHubu"

Vedran Miletić radi kao viši predavač na Odjelu za informatiku Sveučilišta u Rijeci, gdje je nositelj više kolegija iz područja računalnih mreža. Član je Povjerenstva za superračunalne resurse Sveučilišta u Rijeci. Doktorirao je računarstvo na FER-u u Zagrebu, a poslijedoktorsko usavršavanje u području računalne biokemije proveo je na Heidelberškom institutu za teorijske studije u Heidelbergu, Njemačka. Doprinio je nekolicini slobodnosoftverskih projekata razvojem otvorenog koda, a voditelj je razvoja slobodnog softvera otvorenog koda za pristajanje, visokoprotočni virtualni probir i predviđanje načina vezivanja biomolekula RxDock.

### Predavanje: Fotorealistična multimedija i široki raspon boja

**Predavač:** dr.sc. Vedran Miletić, Odjel za informatiku Sveučilišta u Rijeci

**Koautor predavanja:** prof. dr. sc. Nataša Hoić-Božić, Odjel za informatiku Sveučilišta u Rijeci

**Mjesto:** O-028, [zgrada sveučilišnih odjela Sveučilišta u Rijeci](https://www.openstreetmap.org/way/436306129)

**Vrijeme:** 9. travnja 2019. u 14:00

**Događaj:** [Otvoreni dan sveučilišnih odjela 2019](https://www.inf.uniri.hr/11-hr/naslovnica/87-otvoreni-dan-sveucilisnih-odjela-9-4-2019)

#### Sažetak predavanja "Fotorealistična multimedija i široki raspon boja"

Računalna grafika, jedna od grana informatike, uči nas da raspon boja otisnut na papiru nastaje miješanjem tirkizne, purpurne, žute, dok raspon boja na zaslonu nastaje miješanjem crvene, zelene i plave. Osim kroz udio pojedinih osnovnih boja, pojedina boja ponekad se opisuje i nijansom, zasićenjem i osvjetljenjem. Model miješanja crvene, zelene i plave koji zasloni koriste već desetljećima iznenađujuće je ograničavajući u doba kad nam razvoj tehnologije, konkretno one koju vidimo u današnjim 4K televizorima i HD monitorima, omogućuje prikaz velike varijacije u osvjetljenju.

Zasloni koji podržavaju tehnologiju High dynamic range (HDR) donose prikaz deset do sto puta veće svjetline u odnosu na dosadašnje zaslone. Tako danas pojedine Sonyjeve videokamere, ponajbolji Philipsovi monitori i odabrani LG-evi televizori, da spomenemo tek neke od onih koji se mogu pronaći u trgovinama elektroničkom robom diljem Lijepe naše, vrlo rado naglašavaju podršku za HDR. Dosadašnji zasloni omogućuju prikaz tek oko trećine vidljivih boja koje navodi Međunarodna komisija za rasvjetu, a tehnologija Wide color gamut koja uglavnom ide u paru s HDR-om, omogućuje prikaz gotovo dvostruko toliko boja.

Fotografi, filmaši i dizajneri, ali i gledatelji filmova i igrači računalnih igara pozdravili su široku dostupnost HDR-a pa je Adobe izdao softvere Photoshop i Lightroom u novim verzijama s podrškom za HDR, Netflix snimio niz originalnih serija u većem rasponu boja, a računalne su igre Far Cry 5 i Hitman isti taj filmski ugođaj ponudile igračima. Predavanje će govoriti o filmovima, igrama, virtualnoj stvarnosti i drugoj multimediji širokog raspona boja te tehnologiji koja stoji iza njihovog fotorealizma.

#### Životopis predavača "Fotorealistična multimedija i široki raspon boja"

Vedran Miletić radi kao poslijedoktorand na Odjelu za informatiku Sveučilišta u Rijeci, gdje izvodi vježbe na kolegijima u području računalnih mreža i operacijskih sustava. Doktorirao je računarstvo na FER-u u Zagrebu, a poslijedoktorsko usavršavanje u području računalne biokemije proveo je na Heidelberškom institutu za teorijske studije u Heidelbergu, Njemačka. Ostali znanstveni i stručni interesi uključuju superračunala, znanstveno računanje na grafičkim procesorima i slobodni softver otvorenog koda.

### Razvoj upravljačkih programa otvorenog koda za grafičke procesore AMD Radeon

U lipnju 2014. godine na predavanju *NVIDIA CUDA ekosustav: što je tu open, a što baš i ne?* u okviru konferencije konferenciji [DORS/CLUC](https://www.dorscluc.org/) [2014](https://2014.dorscluc.org/) u Zagrebu, Vedran Miletić je rekao da AMD treba shvatiti kako im otvorenost OpenCL-a daje prednost pred NVIDIA-om i da treba zbog toga jako poraditi na svojoj podršci za OpenCL u terminima upravljačkih programa, biblioteka i alata koji bi također bili otvorenog koda.

U prosincu 2015. godine, AMD je zauzeo stav za otvoreni kod i objavio da pokreće [GPUOpen](https://gpuopen.com/). Odlučivši temeljiti svoju strategiju za Linux i računarstvo visokih performansi na softveru otvorenog koda, AMD je napravio veliki korak za slobodni softver u domeni grafike ([detaljni osvrt iz pera Vedrana Miletića](../en/blog/2016-01-17-amd-and-the-open-source-community-are-writing-history.md)). Taj je događaj intenzivno popraćen u medijima:

- [AnandTech](https://www.anandtech.com/show/9853/amd-gpuopen-linux-open-source)
- [Ars Technica](https://arstechnica.com/information-technology/2015/12/amd-embraces-open-source-to-take-on-nvidias-gameworks/)
- [ExtremeTech](https://www.extremetech.com/gaming/219434-amd-finally-unveils-an-open-source-answer-to-nvidias-gameworks)
- [HotHardware](https://hothardware.com/news/amd-goes-open-source-announces-gpuopen-initiative-new-compiler-and-drivers-for-lunix-and-hpc)
- [InfoWorld](https://www.infoworld.com/article/3015782/amd-announces-open-source-initiative-gpuopen.html)
- [PCWorld](https://www.pcworld.com/article/418766/watch-out-gameworks-amds-gpuopen-will-offer-developers-deeper-access-to-its-chips.html)
- [Maximum PC](https://www.maximumpc.com/amd-rtg-summit-gpuopen-and-software/)
- [Phoronix](https://www.phoronix.com/scan.php?page=news_item&px=AMD-GPUOpen)
- [Softpedia](https://news.softpedia.com/news/amd-going-open-source-with-amdgpu-linux-driver-and-gpuopen-tools-497663.shtml)
- [Wccf tech](https://wccftech.com/amds-answer-to-nvidias-gameworks-gpuopen-announced-open-source-tools-graphics-effects-and-libraries/)

AMD-ov radu u okviru GPUOpena podrazumijeva uključivanje zajednice entuzijasta u proces razvoja otvorenih upravljačkih programa za grafičke procesore. Kako bismo omogućili korišenje [OpenCL](https://www.khronos.org/opencl/)-a na Radeonima, radimo na poboljšanju [Mesa3D](https://www.mesa3d.org/)/Gallium upravljačkih programa [r600](https://dri.freedesktop.org/wiki/R600ToDo/) i posebno [radeonsi](https://dri.freedesktop.org/wiki/RadeonsiToDo/). Specifično, dodajemo podršku za različite značajke OpenCL-a s ciljem podrške svih značajki OpenCL-a verzije 1.2 i starijih vezija te ispravaka odstupanja od standarda.

Cilj je omogućiti da grafički procesori AMD Radeon mogu pokretati [GROMACS](https://www.gromacs.org/), [LAMMPS](https://www.lammps.org/) i [CP2K](https://www.cp2k.org/). Kako bi se to ostvarilo, poboljšanja će se prvo dogoditi u [Radeonovom OpenCL upravljačkom programu](https://dri.freedesktop.org/wiki/GalliumCompute/), a tek onda u OpenCL aplikacijama; ako su aplikacije u skladu sa standardom, neće se raditi nikakve promjene na njima. Daljnje informacije:

- [tstellar dev blog](https://www.stellard.net/tom/blog/)
- [Bug 99553 -- Tracker bug for runnning OpenCL applications on Clover](https://bugs.freedesktop.org/show_bug.cgi?id=99553)
- [XDC2013: Tom Stellard - Clover Status Update](https://youtu.be/UTaRlmsCro4)
- [FSOSS 2014 Day 2 Tom Stellard AMD Open Source GPU Drivers](https://youtu.be/JZ-EEgXYzUk)

Prezentacije na konferencijama:

- \[1\] Miletić, V., Páll, S. & Gräter, F. [LLVM AMDGPU for High Performance Computing: are we competitive yet?](https://www.llvm.org/devmtg/2017-03/2017/02/20/accepted-sessions.html#31) in 2017 European LLVM Developers' Meeting, Saarbrücken, Germany (2017). ([video](https://youtu.be/r2Chmg85Xik?list=PL_R5A0lGi1AD12EbUChEnD3s51oqfZLe3))
- \[2\] Miletić, V., Páll, S. & Gräter, F. [Towards fully open source GPU accelerated molecular dynamics simulation.](https://www.llvm.org/devmtg/2016-03/#lightning6) in 2016 European LLVM Developers' Meeting, Barcelona, Spain (2016). ([video](https://youtu.be/TkanbGAG_Fo?t=23m47s&list=PL_R5A0lGi1ADuZKWUJOVOgXr2dRW06e55))

### Predavanje: Ratovi web preglednika

**Predavač:** Vedran Miletić, Odjel za informatiku Sveučilišta u Rijeci

**Mjesto:** O-028, [zgrada sveučilišnih odjela Sveučilišta u Rijeci](https://www.openstreetmap.org/way/436306129)

**Vrijeme:** 21. travnja 2015. u 14:00

**Događaj:** [Otvoreni dan sveučilišnih odjela 2015](https://web.archive.org/web/20150320153120/https://www.inf.uniri.hr/hr/student-info/589-otvoreni-dan-sveucilisnih-odjela-21-4-2015.html)

#### Sažetak predavanja "Ratovi web preglednika"

Pod pojmom rata web preglednika podrazumijeva se natjecanje više web preglednika za tržišnim udjelom, odnosno za korisnicima. Ratovi web preglednika oblikovali su web kakav danas znamo. On je rezultat pobjede preglednika zasnovanih na web standardima nakon drugog rata web preglednika između [Microsoft Internet Explorera](https://en.wikipedia.org/wiki/Internet_Explorer) i [Mozilla Firefoxa](https://www.mozilla.org/firefox/).

Tijekom 90-tih godina [Netscape](https://en.wikipedia.org/wiki/Netscape) i [Microsoft](https://www.microsoft.com/) ratovali su za korisnike Interneta. Krajem 90-tih sve je bilo izvjesnije da je Microsoft dobio prvi rat web preglednika. Kako se Netscape tada povukao, Microsoftov Internet Explorer je relativno dugo imao monopol. Taj monopol doveo je do stagnacije razvoja weba; iako su nove tehnologije bile razvijene i dostupne za korištenje svima, njihova široka primjena nije bila moguća. Osnovni razlog nemogućnosti primjene novih tehnologija bio je nedostatak interesa od strane Microsofta za razvoj i implementaciju istih u Internet Exploreru, dominantnom web pregledniku.

Međutim, povučeni Netscape nije spavao. U siječnju 1998. obznanio je da će vlastiti preglednik dati svima na korištenje potpuno besplatno te da će izvorni kod preglednika također biti besplatno dostupan. Otvoreni kod Netscape-ovog preglednika postao je dostupan 31. ožujka 1998. i oko njega je okupljen projekt [Mozilla](https://www.mozilla.org/). Kroz kratko vrijeme [nastao je Mozilla Firefox](https://wiki.mozilla.org/Timeline), a drugi rat preglednika koji je uslijedio je povijest.

Predavanje će govoriti kako su prvi i drugi rat preglednika učinili da [otvoreni web standardi](https://www.w3.org/standards/) prevladaju nad vlasničkim "standardima". Prevladavanje otvorenih standarda omogućilo je da se Apple Safari i [Google Chrome](https://www.google.com/chrome/) ravnopravno natječu s Mozilla Firefoxom i Microsoft Internet Explorerom. Web današnjice oblikovan je kroz suradnju velikog broja neovisnih organizacija i pojedinaca, o čemu će također biti govora.

#### Životopis predavača "Ratovi web preglednika"

Vedran Miletić radi kao asistent na [Odjelu za informatiku](https://www.inf.uniri.hr/) i [Tehničkom fakultetu](http://www.riteh.uniri.hr/) [Sveučilišta u Rijeci](https://uniri.hr/). Student je [doktorskog studija elektrotehnike i računarstva](https://www.fer.unizg.hr/studiji/doktorski_studij) na [FER-u](https://www.fer.unizg.hr/) u Zagrebu, s temom [u području pouzdanosti i raspoloživosti optičkih telekomunikacijskih mreža](https://www.fer.unizg.hr/ztel). Ostali znanstveni interesi uključuju računarstvo visokih performansi, otvorene tehnologije za distribuirane računalne sustave i korištenje grafičkih procesora u domeni znanstvenog računanja.

### Predavanje: Savršena oluja

**Predavač:** Vedran Miletić, Odjel za informatiku Sveučilišta u Rijeci

**Mjesto:** O-028, [zgrada sveučilišnih odjela Sveučilišta u Rijeci](https://www.openstreetmap.org/way/436306129)

**Vrijeme:** 8. travnja 2014. u 14:00

**Događaj:** Otvoreni dan sveučilišnih odjela 2014

#### Sažetak predavanja "Savršena oluja"

[Savršena oluja](https://www.publishersweekly.com/978-0-393-04016-6) je kombinacija okolnosti koja drastično otežava situaciju.

Razvoj znanosti i tehnologije u posljednjih 250 godina iz temelja je promijenio društvo u kojem živimo. U doba industrijske revolucije, a zatim i informacijske revolucije, dolazi do drastičnog povećanja standarda i produljenja očekivanog trajanja života čovjeka. Osim toga, pad cijena usluga povećava dostupnost tehnologije svakom pojedincu, a time i njegove mogućnosti profesionalnog razvoja.

Svjetska financijska kriza 2007-2008 dovela je do preispitivanja rastrošnosti u mnogim područjima, i time otvorila prostor za nagli rast korištenja softvera otvorenog koda. Iako korišten već od 90-tih godina, tek je krajem 00-tih postao mainstream. Otvoreni kod danas prožima čitavu vertikalu; od milijuna samostalnih malenih pametnih telefona i tableta zasnovanih na Androidu do glomaznih koherentnih poslužiteljskih sustava pokretanih [Red Hat Enterprise Linuxom](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux) koji svakodnevno odrađuju milijarde dolara transakcija na [Newyorškoj burzi](https://www.nyse.com/), serviraju milijardu korisnika Facebooka, održavaju internacionalnu svemirsku postaju i istražuju postanak svemira u [CERN-u](https://home.cern/).

Kod industrijske revolucije mogućnost jeftine masovne proizvodnje komponenata dovela do razvoja brojnih proizvoda koji su prije bili nezamislivi. Isto tako, kod informacijske revolucije komodifikacija komponenata otvorenog koda i široka dostupnost dovest će do razvoja brojnih danas nezamislivih proizvoda. Sve što danas imamo, koliko god fascinantno izgledalo, tek je početak.

Predavanje će govoriti kako je svjetska financijska kriza ustvari bila ključan faktor koji je izazvao savršenu oluju u sektoru informacijske i komunikacijske tehnologije, kako je isti pod pritiskom krize doživio revoluciju, kako je informacijska revolucija utjecala na svakodnevni život, naročito u domeni zabave, te o ključnom utjecaju koji je imala na omogućavanje novih otkrića u znanosti.

#### Životopis predavača "Savršena oluja"

Vedran Miletić radi kao asistent na [Odjelu za informatiku](https://www.inf.uniri.hr/) i [Tehničkom fakultetu](http://www.riteh.uniri.hr/) [Sveučilišta u Rijeci](https://uniri.hr/). Student je [doktorskog studija elektrotehnike i računarstva](https://www.fer.unizg.hr/studiji/doktorski_studij) na [FER-u](https://www.fer.unizg.hr/) u Zagrebu, s temom [u području pouzdanosti i raspoloživosti optičkih telekomunikacijskih mreža](https://www.fer.unizg.hr/ztel). Ostali znanstveni interesi uključuju računarstvo visokih performansi, otvorene tehnologije za distribuirane računalne sustave i korištenje grafičkih procesora u domeni znanstvenog računanja.
