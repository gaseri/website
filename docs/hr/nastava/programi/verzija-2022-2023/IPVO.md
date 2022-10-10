# Infrastruktura za podatke velikog obujma

## Opće informacije

Nositelj predmeta: doc. dr. sc. Vedran Miletić  
Naziv predmeta: Infrastruktura za podatke velikog obujma  
Studijski program: Diplomski studij Informatika  
Status predmeta: **obvezatan** za modul IIS  
Godina: 1.  
Bodovna vrijednost i način izvođenja nastave:

- ECTS koeficijent opterećenosti studenata: 6
- Broj sati (P+V+S): 30+30+0

## Opis predmeta

### Ciljevi predmeta

Cilj je predmeta usvajanje znanja o infrastrukturi u pozadini aplikacija i usluga inteligentnih informacijskih sustava koji rade s podacima velikog obujma te stjecanje vještina implementacije i održavanja takve infrastrukture u računalnom oblaku.

### Uvjeti za upis predmeta

Nema uvjeta za upis predmeta.

### Očekivani ishodi učenja za predmet

Očekuje se da će nakon uspješno ispunjenih svih programom predviđenih obveza na predmetu student biti sposoban:

I1. Odabrati distribuirane arhitekture za rad s podacima velikog obujma (npr. lambda, kappa, delta i sl.) i odgovarajuće alate za takve arhitekture.  
I2. Predvidjeti potrebe inteligentnog informacijskog sustava za infrastrukturom u oblaku uz povezivanje na odgovarajuća sučelja repozitorija podataka, informacija i znanja s pripadnim metapodacima.  
I3. Oblikovati model upravljanja podacima, koordinacije, razmjene poruka i interakcije u inteligentnom informacijskom sustavu koristeći odgovarajuće metode i tehnike (npr. distribuirane baze podataka, sustavi za predmemoriju, sustavi razmjene poruka, sustavi strujanja podataka i sl.) te pripadni model distribuirane baze podataka koristeći odgovarajuće jezike za modeliranje podataka i uzimajući u obzir specifičnosti arhitekture sustava.  
I4. Preporučiti tehnologije za implementaciju integracije podataka, informacija i znanja iz heterogenih i distribuiranih podatkovnih sustava koje zadovoljavaju zahtjeve postavljenog problema.  
I5. Odabrati odgovarajući skup tehnologija u oblaku (npr. monolitne i mikrouslužne arhitekture, kontejneri, virtualni strojevi i sl.) za implementaciju inteligentnog informacijskog sustava.  
I6. Razviti inteligentne usluge u oblaku temeljene na analitici podataka i umjetnoj inteligenciji te pripadna sučelja i odgovarajuću dokumentaciju.  
I7. Razviti komponente inteligentnih informacijskih sustava i pripadne procedure automatiziranog testiranja koristeći platforme, biblioteke, okvire i usluge u oblaku kao infrastrukturu.  
I8. Implementirati inteligentnog agenta koji rješava postavljeni problem koristeći zadana sučelja, usluge, aplikacije, mehanizame interakcije i vrste ponašanja prikladne za postavljeni problem te agentni model sustava koji će se iskoristiti za simulaciju ponašanja sustava.

### Sadržaj predmeta

Sadržaj predmeta čine teme:

- Pouzdanost, skalabilnost i održivost aplikacija. Podatkovni modeli. Pohrana i dohvaćanje podataka. Kodiranje podataka za pohranu i slanje.
- Replikacija i particioniranje podataka. Transakcije. Izazovi distribuiranih sustava: pogreške, nepouzdanost, garancija konzistentnosti i konsenzus.
- Razvoj i implementacija oblaku urođenih aplikacija. Operacije nad podacima u oblaku. Prenosivost između različitih oblaka. Evolucija monolitnih aplikacija u mikrouslužne.
- Infrastruktura i usluge za serijsku i tokovnu obradu podataka. Potporne usluge inteligentnog informacijskog sustava i agenta.
- Tehnološki trendovi i budućnost sustava za obradu podataka velikog obujma.

### Vrsta izvođenja nastave

- [x] predavanja
- [ ] seminari i radionice
- [x] vježbe
- [x] obrazovanje na daljinu
- [ ] terenska nastava
- [x] samostalni zadaci
- [ ] multimedija i mreža
- [x] laboratorij
- [ ] mentorski rad
- [ ] ostalo ___________________

### Komentari

Nastava će se izvoditi kombinirajući rad u učionici i samostalni rad izvan učionice, uz korištenje sustava za e-učenje.

### Obveze studenata

Obveze studenata na predmetu su:

- Redovito pohađati nastavu i sudjelovati u svim aktivnostima predmeta te pratiti obavijesti vezane uz nastavu u sustavu za e-učenje.
- Pristupiti kontinuiranim provjerama znanja (teorijskim i praktičnim kolokvijima) i uspješno ih položiti.
- Izraditi praktične radove (individualne ili timske projekte) na zadane teme i obraniti ih.
- Pristupiti završnom ispitu i na njemu postići barem 50% bodova.

Detaljan način razrade bodovanja na predmetu te pragovi prolaza za pojedine aktivnosti vrednovanja bit će navedeni u detaljnom izvedbenom nastavnom planu predmeta.

### Praćenje[^1] rada studenata

- Pohađanje nastave: 2
- Aktivnost u nastavi:
- Seminarski rad:
- Eksperimentalni rad: 1
- Pismeni ispit: 1
- Usmeni ispit: 1
- Esej:
- Istraživanje:
- Projekt:
- Kontinuirana provjera znanja:
- Referat:
- Praktični rad: 1
- Portfolio:

### Postupak i primjeri vrednovanja pojedinog ishoda učenja tijekom nastave i na završnom ispitu

- Pisana ili online provjera u kojoj će student pokazati razumijevanje te sposobnost analize i sinteze teorijskih koncepata distribuiranih sustava, heterogenih podatkovnih sustava, arhitektura za rad s podacima velikog obujma, infrastrukture inteligentnih informacijskih sustava i tehnologija u oblaku (I1, I2, I4, I5).
- Eksperimentalni rad s različitim arhitekturama za rad s podacima velikog obujma i odgovarajućima alatima (npr. Hadoop, Spark, Kafka, HBase i sl.) s ciljem prikupljanja analitičkih metrika nužnih za predviđanje potreba za infrastrukturom od strane inteligentnog informacijskog sustava temeljenog na toj arhitekturi (I1, I2). U skladu s predviđenom infrastruktrom student će oblikovati model upravljanja podacima, koordinacije, razmjene poruka i interakcije te preporučiti tehnologije za implementaciju heterogenog i distribuiranog podatkovnog sustava (poput distribuiranih relacijskih i ne-relacijskih (NoSQL) baza podataka, baza podataka temeljenih na strujanju podataka (npr. Kafka), tehnologija lanca blokova (engl. *blockchain*) i/ili poopćenih baza podataka, baza podataka temeljenih na dokumentima te medijskih i objektno-orijentiranih baza podataka) (I3, I4).
- Praktični rad obranjen usmenim putem u okviru kojeg će student odabrati odgovarajući skup tehnologija u oblaku (poput AWS, Azure, Google Cloud, IBM Cloud, Scaleway, DigitalOcean, Watson, Wit.ai, Botpress i sl.) i iskoristiti ga za razvoj inteligentne usluge (npr. inteligentnog agenta ili komponente inteligentnog informacijskog sustava) temeljene na analitici podataka i umjetnoj inteligenciji te pripadnih sučelja (npr. REST, WebSocket, TCP/UDP, ZMTP, AMQP, XMPP i sl.), uz odgovarajuću dokumentaciju (I5, I6, I8). U okviru razvoja implementirat će i procedure automatiziranog testiranja servisa u oblaku koristeći odgovarajuće tehnologije (npr. jedinično testiranje, testiranje s kraja na kraj, penetracijsko testiranje, etičko hakiranje i sl.) (I7).

### Obvezna literatura (u trenutku prijave prijedloga studijskog programa)

1. Takada, M. *Distributed systems: for fun and profit*. (Mixu, 2013). Dostupno online: [book.mixu.net/distsys/](http://book.mixu.net/distsys/)
2. Beyer, B., Jones, C., Petoff, J. & Murphy, N. R. *Site Reliability Engineering: How Google Runs Production Systems*. Dostupno online: [sre.google/sre-book/table-of-contents/](https://sre.google/sre-book/table-of-contents/)
3. Kleppmann, M. *Designing data-intensive applications: The big ideas behind reliable, scalable, and maintainable systems*. (O'Reilly Media, 2017).
4. Scholl, B., Swanson, T. & Jausovec, P. *Cloud Native: Using Containers, Functions, and Data to Build Next-Generation Applications*. (O'Reilly Media, 2019).
5. Aspnes, J. *Notes on Theory of Distributed Systems*. (Aspnes, 2021). Dostupno online: [cs-www.cs.yale.edu/homes/aspnes/classes/465/notes.pdf](http://cs-www.cs.yale.edu/homes/aspnes/classes/465/notes.pdf)
6. Sadržaji pripremljeni za učenje putem sustava za učenje.

### Dopunska literatura (u trenutku prijave prijedloga studijskog programa)

1. Raman, A., Hoder, C., Bisson, S. & Branscombe, M. *Azure AI Services at Scale for Cloud, Mobile, and Edge: Building Intelligent Apps with Azure Cognitive Services and Machine Learning*. (O'Reilly Media, 2022).
2. Fregly, C. & Barth, A. *Data Science on AWS: Implementing End-to-End, Continuous AI and Machine Learning Pipelines*. (O'Reilly Media, 2021).
3. Winder, P. *Reinforcement Learning: Industrial Applications of Intelligent Agents*. (O'Reilly Media, 2020).
4. Adkins, H., Beyer, B., Blankinship, P., Oprea, A., Lewandowski, P. & Stubblefield, A. *Building Secure and Reliable Systems: Best Practices for Designing, Implementing, and Maintaining Systems*. (O'Reilly Media, 2020). Dostupno online: [sre.google/static/pdf/building_secure_and_reliable_systems.pdf](https://sre.google/static/pdf/building_secure_and_reliable_systems.pdf)
5. Reznik, P., Dobson, J. & Glenow, M. *Cloud Native Transformation: Practical Patterns for Innovation*. (O'Reilly Media, 2019).
6. Arundel, J. & Domingus, J. *Cloud Native DevOps with Kubernetes: Building, Deploying, and Scaling Modern Applications in the Cloud*. (O'Reilly Media, 2019).
7. Newman, S. *Monolith to Microservices: Evolutionary Patterns to Transform Your Monolith*. (O'Reilly Media, 2019).
8. Sridharan, C. *Distributed Systems Observability*. (O'Reilly Media, 2018).
9. Burns, B. *Designing Distributed Systems*. (O'Reilly Media, 2018).
10. Beyer, B., Murphy, N. R., Rensin, D., Kawahara, K. & Thorne, S. *The Site Reliability Workbook: Practical Ways to Implement SRE*. (O'Reilly Media, 2018). Dostupno online: [sre.google/workbook/table-of-contents/](https://sre.google/workbook/table-of-contents/)

### Broj primjeraka obavezne literature u odnosu na broj studenata koji trenutno pohađaju nastavu na predmetu

| Naslov | Broj primjeraka | Broj studenata |
| ------ | --------------- | -------------- |
| Distributed systems: for fun and profit | [Dostupno online](http://book.mixu.net/distsys/) | 20 |
| Site Reliability Engineering: How Google Runs Production Systems | [Dostupno online](https://sre.google/sre-book/table-of-contents/) | 20 |
| Designing data-intensive applications: The big ideas behind reliable, scalable, and maintainable systems | 1 | 20 |
| Cloud Native: Using Containers, Functions, and Data to Build Next-Generation Applications | 1 | 20 |
| Notes on Theory of Distributed Systems | [Dostupno online](http://cs-www.cs.yale.edu/homes/aspnes/classes/465/notes.pdf) | 20 |

### Načini praćenja kvalitete koji osiguravaju stjecanje izlaznih znanja, vještina i kompetencija

Predviđa se periodičko provođenje evaluacije s ciljem osiguranja i kontinuiranog unapređenja kvalitete nastave i studijskog programa (u okviru aktivnosti Odbora za upravljanje i unapređenje kvalitete Fakulteta informatike i digitalnih tehnologija). U zadnjem tjednu nastave provodit će se anonimna evaluacija kvalitete održane nastave od strane studenata. Provest će se i analiza uspješnosti studenata na predmetu (postotak studenata koji su položili predmet i prosjek njihovih ocjena).

[^1]: **Važno:** Uz svaki od načina praćenja rada studenata unijeti odgovarajući udio u ECTS bodovima pojedinih aktivnosti tako da ukupni broj ECTS bodova odgovara bodovnoj vrijednosti predmeta. Prazna polja upotrijebiti za dodatne aktivnosti.
