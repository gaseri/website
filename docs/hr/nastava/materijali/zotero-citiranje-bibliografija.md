---
author: Luka Vretenar, Vedran Miletić
---

# Upute za upravljanje bibliografijom korištenjem alata Zotero

Zotero je alat za upravljanje bibliografskom bazom, umetanje referenci u tekst i generiranje popisa referenci na temelju referenciranih bibliografskih jedinica.

Zotero može generirati bibliografiju u različitim stilovima, uključujući Chicago i IEEE. Ove upute objasnit će postupak instalacije i način korištenja alata Zotero u kombinaciji s uređivačima teksta Microsoft Word i LibreOffice Writer te web preglednicima Mozilla Firefox i Google Chrome.

## Instalacija Zotera

Alat Zotero se sastoji od tri dijela, to su:

- samostalna aplikacija koja se također zove Zotero koja radi na macOS-u, Windowsima i Linuxu,
- dodatak Zotero Connector za web preglednike Firefox, Chrome, Safari i Edge,
- Zoterov dodatak za uređivače teksta [Microsoft Word](https://www.microsoft.com/en-us/microsoft-365/word) i [LibreOffice Writer](https://www.libreoffice.org/discover/writer/).

Da bi započeli raditi sa alatom prvo nam je potrebno instalirati njegovu samostalnu aplikaciju koju je potrebno preuzeti sa [Zoterove web stranice](https://www.zotero.org/). Nakon što odaberemo poveznicu [Download](https://www.zotero.org/download/), potrebno je preuzeti pod Zotero instalacijsku datoteku za samu aplikaciju, te je instalirati. Kod instalacije aplikacije potrebno je prihvatiti sve ponuđene opcije i potvrdno odgovoriti na sve upite.

Prilikom instalacije aplikacije instaliraju se i dodaci za uređivače teksta Microsoft Word i/ili LibreOffice Writer. U slučaju da instalacija dodataka nije bila uspješna, moguće je nakon pokretanja Zotera u prozoru `Preferences...` u dijelu `Cite` na kartici `Word Processors` izvršiti instaciju jednog ili obaju dodataka.

Zatim je potrebno instalirati dodatak za web preglednik koji koristite, čiju poveznicu Također možemo pronaći na stranici [Download](https://www.zotero.org/download/). Kada instalacija završi, imate mogućnost pokretanja aplikacije sa radne površine vašeg računala i mogućnost sakupljanja bibliografskih jedinica iz web preglednika.

## Korištenje samostalne aplikacije Zotero

!!! note
    Zotero ima i službeni [Quick Start Guide](https://www.zotero.org/support/quick_start_guide).

Kako bi mogli upravljati našom bazom citata potrebno je prvo pokrenuti aplikaciju Zotero sa naše radne površine.

Glavni prozor služi nam za pregled, upravljanje i izmjenu bibliografije. Sastoji se od tri stupca koji odgovaraju pogledima:

- popisa biblioteka, koje u sebi sadrže skupove bibliografskih jedinica,
- popisa bibliografskih jedinica u odabranom skupu,
- detalja o odabranoj bibliografskoj jedinici.

Iznad svakog od navedenih pogleda postoje kontrole za upravljanje pogledima. Na taj način možemo dodavati nove skupove bibliografskih jedinica i same jedinice koristeći simbole u alatnoj traci ili preko menija.

Prije početka prikupljanja bibliografskih jedinica za dokument na kojem trenutno radimo, potrebno je izraditi skup koji će povezati sve bibliografske jedinice vezane za rad. To možemo učiniti na način da odaberemo iz menija `File\New Collection...` i odaberemo naziv našeg skupa.

Bibliografske jedinice dodajemo tako da odaberemo skup u koji ih želimo dodati i potom odaberemo iz menija `File\New Item\<vrsta>`, gdje se pod `<vrsta>` bira vrsta bibliografske jedinice koju izrađujemo (npr. `Book`, `Journal Article`, `Report`). Pri izradi bibliografske jedinice važno je da odaberemo odgovarajući tip za tu jedinicu jer nemaju sve vrste iste parametre za opis referenci.

Nakon što izradimo novu bibliografsku jedinicu odabranog tipa, pojavit će nam se u popisu bibliografskih jedinica prazna stavka za koju moramo popuniti sve podatke. Podatke za praznu bibliografsku jedinicu popunjavamo u trećem stupcu programa. Važno je prikupiti i popuniti što više informacija za bibliografsku jedinicu koju definiramo, jer će oni biti iskorišteni kod generiranja popisa referenci našeg rada.

Reference možemo, pored pojedinačnog dodavanja i ručnog unošenja podataka, dodati i korištenjem dodatka web preglednika.

## Korištenje Zotero Connectora u web pregledniku

Dodatak za web preglednik nam omogućuje da izravno dodajemo materijale koje pronađemo na internetu kao reference u naše biblioteke. Za web stranice kao što su online baze članaka (npr. [Google Scholar](https://scholar.google.com), [IEEE Xplore](https://ieeexplore.ieee.org/), [Web of Science](https://www.webofknowledge.com/), [Scopus](https://www.scopus.com/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [bioRxiv](https://www.biorxiv.org/), [Hrčak](https://hrcak.srce.hr/)) i popularna sjedišta (npr. [Amazon](https://www.amazon.com/), [YouTube](https://www.youtube.com/), [Wikipedia](https://www.wikipedia.org/)) postoji mogućnost automatskog preuzimanja informacija o izvoru kojeg citirate.

U slučaju da se nalazimo na stranici sa koje Zotero može sam preuzeti informacije o izvoru i stvoriti bibliografsku jedinicu na temelju njih, Zoterova ikona u alatnoj traci web preglednika će se promijeniti ovisno o vrsti izvora.

Ako nam ta operacija nije moguća, i dalje možemo dodati bibliografsku jedinicu koja se odnosi na web stranicu na kojoj se nalazimo odabirom Zotero mogućnosti iz menija kojeg dobijemo desnim klikom miša.

## Korištenje Zoterovog dodatka u uređivaču teksta

Kada definiramo i/ili preuzmemo sve potrebne bibliografske jedinice, možemo ih dohvatiti izravno u uređivaču teksta koji koristimo. U slučaju da koristite Microsoft Word, naredbe za dodavanje citata i bibliografije će nam se nalaziti u kartici `Add-Ins`. U slučaju da koristite LibreOffice Writer, Zoterova traka s alatima trebala bi biti vidljiva u zadanim postavkama, a može se isključiti i uključiti pod `View\Toolbars\Zotero`. U oba slučaja gumbi koje ćemo koristiti su redom:

- `Insert Citation` kojim možemo dodati u tekst referencu na neku od prethodno definiranih bibliografskih jedinica pomoću dijaloga za pretragu,
- `Insert Bibliography` kojim možemo umetnuti popis literature koji uključuje one bibliografske jedinice na koje postoji referenca u radu,
- `Refresh` kojim se osvježava se popis literature,
- `Set Document Preferences` kojim moguće je promijeniti stil kojim se navode reference u tekstu i literaturi.
